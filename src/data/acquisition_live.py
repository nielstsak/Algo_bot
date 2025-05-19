import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, TYPE_CHECKING

import pandas as pd
import numpy as np 
import requests

if TYPE_CHECKING:
    from src.config.loader import AppConfig, LiveConfig 

logger = logging.getLogger(__name__)

BASE_COLUMNS_RAW = [
    'timestamp', 'kline_close_time', 'open', 'high', 'low', 'close', 'volume',
    'quote_asset_volume', 'number_of_trades',
    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
    'taker_sell_base_asset_volume', 'taker_sell_quote_asset_volume'
]

BINANCE_KLINE_COLUMNS = [
    'kline_open_time', 'open', 'high', 'low', 'close', 'volume',
    'kline_close_time', 'quote_asset_volume', 'number_of_trades',
    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
]

# Limite typique de l'API Binance pour les klines (peut varier entre SPOT/MARGIN et FUTURES)
API_BATCH_LIMIT_SPOT_MARGIN = 1000
API_BATCH_LIMIT_FUTURES = 1500


try:
    from binance.client import Client as BinanceClient
    KLINE_INTERVAL_1MINUTE = BinanceClient.KLINE_INTERVAL_1MINUTE
except ImportError:
    KLINE_INTERVAL_1MINUTE = "1m" 
    logger.warning("python-binance Client not found, using '1m' string for KLINE_INTERVAL_1MINUTE.")


def get_binance_klines(
    symbol: str, 
    config_interval_context: str, # Intervalle de la config pour le contexte (nom de fichier, etc.)
    limit: int = 100, 
    account_type: str = "MARGIN",
    end_timestamp_ms: Optional[int] = None # Pour récupérer les klines *avant* ce timestamp
) -> Optional[pd.DataFrame]:
    """
    Récupère les klines de Binance via l'API REST.
    Peut récupérer les plus récentes ou celles avant un timestamp donné.
    """
    actual_fetch_interval = KLINE_INTERVAL_1MINUTE 
    log_ctx = f"{symbol} ({actual_fetch_interval}, context: {config_interval_context}, Acc: {account_type.upper()})"
    
    if end_timestamp_ms:
        logger.debug(f"Requesting {limit} klines for {log_ctx} ending before {pd.to_datetime(end_timestamp_ms, unit='ms', utc=True)} via REST API")
    else:
        logger.debug(f"Requesting latest {limit} klines for {log_ctx} via REST API")
    
    account_type_upper = account_type.upper()
    api_batch_limit: int
    if account_type_upper in ["SPOT", "MARGIN", "BINANCE_MARGIN"]:
        base_url = "https://api.binance.com"
        endpoint = "/api/v3/klines"
        api_batch_limit = API_BATCH_LIMIT_SPOT_MARGIN
    elif account_type_upper == "FUTURES": 
        base_url = "https://fapi.binance.com" 
        endpoint = "/fapi/v1/klines"
        api_batch_limit = API_BATCH_LIMIT_FUTURES
    else:
        logger.error(f"Unsupported account_type for kline fetching: {account_type}")
        return None

    # S'assurer que la limite demandée ne dépasse pas la limite de l'API par requête
    # Si limit > api_batch_limit, cela sera géré par la logique de chunking dans initialize_pair_data.
    # Ici, on s'assure que chaque *appel individuel* respecte la limite.
    current_fetch_limit = min(limit, api_batch_limit)


    url = base_url + endpoint
    params: Dict[str, Any] = {"symbol": symbol, "interval": actual_fetch_interval, "limit": current_fetch_limit}
    if end_timestamp_ms:
        params["endTime"] = end_timestamp_ms
        # Si endTime est utilisé, Binance retourne les klines *avant* ce temps.
        # Pas besoin de startTime si on veut juste les N dernières avant endTime.

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status() 
        data = response.json()

        if not data:
            logger.warning(f"No kline data received from Binance API for {log_ctx}")
            return None

        df = pd.DataFrame(data, columns=BINANCE_KLINE_COLUMNS)
        
        df.rename(columns={'kline_open_time': 'timestamp'}, inplace=True)

        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
        if 'kline_close_time' in df.columns: 
            df['kline_close_time'] = pd.to_datetime(df['kline_close_time'], unit='ms', utc=True)
        else:
            df['kline_close_time'] = pd.NaT 

        cols_to_numeric = [
            'open', 'high', 'low', 'close', 'volume',
            'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume'
        ]
        logger.debug(f"Converting columns to numeric: {cols_to_numeric}")
        for col in cols_to_numeric:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            else:
                logger.warning(f"Column '{col}' not found in DataFrame from API for {log_ctx}. Will be NaN.")
                df[col] = np.nan 

        if 'volume' in df.columns and 'taker_buy_base_asset_volume' in df.columns:
            df['taker_sell_base_asset_volume'] = df['volume'] - df['taker_buy_base_asset_volume']
        else:
            logger.warning(f"Cannot calculate 'taker_sell_base_asset_volume' for {log_ctx} due to missing source columns.")
            df['taker_sell_base_asset_volume'] = np.nan
            
        if 'quote_asset_volume' in df.columns and 'taker_buy_quote_asset_volume' in df.columns:
            df['taker_sell_quote_asset_volume'] = df['quote_asset_volume'] - df['taker_buy_quote_asset_volume']
        else:
            logger.warning(f"Cannot calculate 'taker_sell_quote_asset_volume' for {log_ctx} due to missing source columns.")
            df['taker_sell_quote_asset_volume'] = np.nan

        for col in BASE_COLUMNS_RAW:
            if col not in df.columns:
                df[col] = np.nan
                logger.debug(f"Ensuring column '{col}' exists for {log_ctx}, added with NaNs.")
        
        essential_cols_for_dropna = ['timestamp', 'open', 'high', 'low', 'close']
        if 'kline_close_time' in df.columns:
             essential_cols_for_dropna.append('kline_close_time')
        df.dropna(subset=essential_cols_for_dropna, inplace=True)
        
        df = df[BASE_COLUMNS_RAW]

        logger.info(f"Fetched and processed {len(df)} klines for {log_ctx}")
        return df

    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching klines for {log_ctx} via REST API: {e}")
        return None
    except Exception as e: 
        logger.error(f"Unexpected error processing klines for {log_ctx}: {e}", exc_info=True)
        return None

def ensure_data_files(pair: str, interval_context: str, raw_dir: str, processed_dir: str) -> tuple[Path, Path]:
    raw_path = Path(raw_dir) / f"{pair}_{interval_context}.csv" 
    processed_path = Path(processed_dir) / f"{pair}_{interval_context}_processed_live.csv" # Nom de fichier cohérent

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    processed_path.parent.mkdir(parents=True, exist_ok=True)

    if not raw_path.exists() or raw_path.stat().st_size == 0:
        pd.DataFrame(columns=BASE_COLUMNS_RAW).to_csv(raw_path, index=False)
        logger.info(f"Created/Ensured empty raw data file with headers: {raw_path}")
    
    if not processed_path.exists() or processed_path.stat().st_size == 0:
         pd.DataFrame().to_csv(processed_path, index=False) 
         logger.info(f"Created/Ensured empty processed data file: {processed_path}")

    return raw_path, processed_path

def initialize_pair_data(pair: str, config_interval_context: str, raw_path: Path, total_klines_to_fetch: int, account_type: str):
    """
    Initialise le fichier de données brutes pour une paire, en récupérant par lots si nécessaire.
    """
    log_ctx = f"{pair} (context: {config_interval_context})"
    try:
        num_lines_existing = 0
        df_existing = pd.DataFrame(columns=BASE_COLUMNS_RAW) 
        if raw_path.exists() and raw_path.stat().st_size > 0:
            try:
                df_existing = pd.read_csv(raw_path, low_memory=False)
                if not all(col.lower() in [c.lower() for c in df_existing.columns] for col in BASE_COLUMNS_RAW if col != 'timestamp'): # timestamp sera index
                    logger.warning(f"Raw file {raw_path} missing expected columns. Recreating with headers.")
                    df_existing = pd.DataFrame(columns=BASE_COLUMNS_RAW)
                num_lines_existing = len(df_existing)
            except pd.errors.EmptyDataError:
                logger.warning(f"Raw file {raw_path} is empty despite existing. Treating as 0 lines.")
                df_existing = pd.DataFrame(columns=BASE_COLUMNS_RAW)
            except Exception as e:
                logger.error(f"Error reading existing raw file {raw_path}: {e}. Assuming 0 lines and attempting recreation.")
                df_existing = pd.DataFrame(columns=BASE_COLUMNS_RAW)
        else: 
            logger.info(f"Raw file {raw_path} was not found or empty. ensure_data_files created it with headers.")

        logger.info(f"File {raw_path} check: Found {num_lines_existing} data rows (excluding header).")
        min_historical_rows_threshold = 100 # Seuil arbitraire pour considérer l'historique comme "suffisant" pour ne pas re-télécharger agressivement.
                                        # total_klines_to_fetch est le vrai objectif.

        if num_lines_existing < total_klines_to_fetch:
            klines_needed = total_klines_to_fetch - num_lines_existing
            logger.info(f"Fetching initial {klines_needed} (target total: {total_klines_to_fetch}) klines (1-minute) for {log_ctx} as row count ({num_lines_existing}) < target.")
            
            all_new_klines_list = []
            current_end_timestamp_ms: Optional[int] = None # Commencer par les plus récentes et remonter
            
            # Déterminer la limite par batch de l'API
            api_batch_limit = API_BATCH_LIMIT_SPOT_MARGIN if account_type.upper() in ["SPOT", "MARGIN"] else API_BATCH_LIMIT_FUTURES

            while klines_needed > 0:
                fetch_this_batch = min(klines_needed, api_batch_limit)
                logger.info(f"Fetching batch of {fetch_this_batch} klines for {log_ctx} (remaining needed: {klines_needed}). End_ms: {current_end_timestamp_ms}")
                
                df_batch = get_binance_klines(
                    symbol=pair, 
                    config_interval_context=config_interval_context, 
                    limit=fetch_this_batch, 
                    account_type=account_type,
                    end_timestamp_ms=current_end_timestamp_ms
                )

                if df_batch is not None and not df_batch.empty:
                    all_new_klines_list.append(df_batch)
                    klines_needed -= len(df_batch)
                    # Le timestamp de la plus ancienne kline de ce batch devient le endTime pour le prochain batch
                    # On soustrait 1 ms pour éviter de récupérer la même kline
                    oldest_timestamp_in_batch_ms = int(df_batch['timestamp'].iloc[0].timestamp() * 1000)
                    current_end_timestamp_ms = oldest_timestamp_in_batch_ms - 1 
                    
                    if len(df_batch) < fetch_this_batch:
                        logger.info(f"API returned fewer klines ({len(df_batch)}) than requested ({fetch_this_batch}) for {log_ctx}. Assuming end of available history in this direction.")
                        break # Fin de l'historique disponible
                    time.sleep(0.5) # Petit délai pour respecter les limites de l'API
                else:
                    logger.warning(f"Failed to fetch a batch or batch empty for {log_ctx}. Stopping history fetch for this cycle.")
                    break # Arrêter si un batch échoue ou est vide

            if all_new_klines_list:
                df_new_total = pd.concat(all_new_klines_list, ignore_index=True)
                
                # Assurer que la colonne timestamp est au format datetime pour la fusion
                if not df_existing.empty and 'timestamp' in df_existing.columns:
                    df_existing['timestamp'] = pd.to_datetime(df_existing['timestamp'], utc=True, errors='coerce')
                    df_existing.dropna(subset=['timestamp'], inplace=True)

                df_combined = pd.concat([df_existing, df_new_total], ignore_index=True)
                if 'timestamp' in df_combined.columns:
                    df_combined['timestamp'] = pd.to_datetime(df_combined['timestamp'], utc=True, errors='coerce')
                    df_combined.dropna(subset=['timestamp'], inplace=True)
                    df_combined = df_combined.sort_values(by='timestamp')
                    df_combined = df_combined.drop_duplicates(subset=['timestamp'], keep='last')
                
                for col in BASE_COLUMNS_RAW: # S'assurer que toutes les colonnes sont présentes
                    if col not in df_combined.columns:
                        df_combined[col] = np.nan
                df_combined[BASE_COLUMNS_RAW].to_csv(raw_path, index=False) 
                logger.info(f"Saved initial/updated {len(df_combined)} rows (1-minute data) to {raw_path} for {log_ctx}")
            elif num_lines_existing == 0: 
                logger.warning(f"Initial fetch failed for {log_ctx}, saving empty file with headers as it was initially empty: {raw_path}")
                pd.DataFrame(columns=BASE_COLUMNS_RAW).to_csv(raw_path, index=False)
            else: 
                logger.warning(f"Initial fetch failed for {log_ctx}, keeping existing {num_lines_existing} rows in {raw_path}.")
        else:
            logger.info(f"Sufficient data ({num_lines_existing} rows >= target {total_klines_to_fetch}) found in {raw_path}. No initial fetch needed for {log_ctx}.")

    except Exception as e:
        logger.exception(f"Error during initialization for {log_ctx} at {raw_path}: {e}")


def run_initialization(live_config: 'LiveConfig'):
    logger.info("--- Running Live Data Initialization (fetching 1-minute klines for history via REST API) ---")
    live_fetch_config = live_config.live_fetch
    global_settings = live_config.global_live_settings 

    pairs = getattr(live_fetch_config, 'crypto_pairs', [])
    config_intervals_for_context = getattr(live_fetch_config, 'intervals', [])
    total_klines_to_fetch_init = getattr(live_fetch_config, 'limit_init_history', 1000) # Ex: 20000
    account_type = getattr(global_settings, 'account_type', 'MARGIN')

    if not hasattr(live_config, 'global_config') or not hasattr(live_config.global_config, 'paths'): 
        logger.critical("Path configuration missing in live_config.global_config.paths. Cannot determine data directories.")
        return

    paths_config = live_config.global_config.paths 
    raw_dir = Path(paths_config.data_live_raw)
    processed_dir = Path(paths_config.data_live_processed)

    if not pairs or not config_intervals_for_context:
        logger.error("No crypto_pairs or intervals defined in live_config.live_fetch. Cannot initialize.")
        return

    for pair in pairs:
        # Pour l'initialisation, on utilise généralement l'intervalle "1min" comme contexte pour le nom de fichier brut,
        # car c'est la granularité des données que nous allons stocker.
        # Les fichiers traités pour différents contextes (5m, 1h) seront générés à partir de ce fichier brut 1min.
        # Cependant, la logique actuelle utilise config_interval_context pour nommer le fichier brut.
        # Pour simplifier, on va supposer que le premier intervalle dans config_intervals_for_context
        # est le "contexte de base" pour le fichier brut 1min, ou on utilise "1min" explicitement.
        # Ici, on va se baser sur la logique existante qui crée un fichier brut par (pair, interval_context).
        # Si plusieurs interval_context sont définis, cela créera plusieurs fichiers bruts pour la même paire,
        # ce qui n'est pas idéal. Idéalement, un seul fichier brut 1min par paire.

        # Correction: Utiliser un nom de fichier brut standardisé par paire (ex: PAIR_1min.csv)
        # et s'assurer que `initialize_pair_data` est appelé une seule fois par paire pour ce fichier brut.
        # La boucle sur config_intervals_for_context est plus pour `ensure_data_files` des fichiers traités.

        raw_file_name_base = f"{pair}_1min.csv" # Nom standard pour le fichier brut 1min
        raw_path_for_pair = raw_dir / raw_file_name_base
        
        # S'assurer que le répertoire pour le fichier brut existe
        raw_path_for_pair.parent.mkdir(parents=True, exist_ok=True)
        if not raw_path_for_pair.exists() or raw_path_for_pair.stat().st_size == 0:
            pd.DataFrame(columns=BASE_COLUMNS_RAW).to_csv(raw_path_for_pair, index=False)
            logger.info(f"Created/Ensured empty raw data file with headers: {raw_path_for_pair}")

        logger.info(f"Initializing 1-minute data for {pair} into {raw_path_for_pair}...")
        try:
            # L'intervalle de contexte ici est "1min" car on initialise le fichier brut 1min.
            initialize_pair_data(pair, "1min", raw_path_for_pair, total_klines_to_fetch_init, account_type)
        except Exception as e:
            logger.error(f"Failed to initialize {pair} (raw file: {raw_path_for_pair}): {e}", exc_info=True)

        # S'assurer que les fichiers pour les autres contextes (traités) existent
        for cfg_interval_ctx in config_intervals_for_context: 
            _, processed_path = ensure_data_files(pair, cfg_interval_ctx, str(raw_dir), str(processed_dir))
            # Le fichier traité sera rempli par preprocessing_live.py, on s'assure juste qu'il existe.
            if not processed_path.exists() or processed_path.stat().st_size == 0:
                 pd.DataFrame().to_csv(processed_path, index=False)
                 logger.info(f"Ensured empty processed data file for context {cfg_interval_ctx}: {processed_path}")


    logger.info("--- Live Data Initialization Complete ---")


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s')
    logger.info("Running acquisition_live.py directly for testing (REST API only)...")

    from dataclasses import dataclass, field

    @dataclass
    class MockLiveFetchConfig:
        crypto_pairs: List[str] = field(default_factory=lambda: ["BTCUSDT"])
        intervals: List[str] = field(default_factory=lambda: ["5m", "1min"]) 
        limit_init_history: int = 2500 # Test avec un nombre > API_BATCH_LIMIT
        limit_per_fetch: int = 5 
        max_retries: int = 3 
        retry_backoff: float = 1.5 
    
    @dataclass
    class MockGlobalLiveSettings:
        account_type: str = "SPOT" 
    
    @dataclass
    class MockPathsConfig: 
        data_historical_raw: str = "data/historical/raw_test_rest"
        data_historical_processed_cleaned: str = "data/historical/processed/cleaned_test_rest"
        data_historical_processed_enriched: str = "data/historical/processed/enriched_test_rest" 
        logs_backtest_optimization: str = "logs/backtest_optimization_test_rest"
        logs_live: str = "logs/live_trading_test_rest"
        results: str = "results_test_rest"
        data_live_raw: str = "data/live/raw_test_rest_init" 
        data_live_processed: str = "data/live/processed_test_rest_init" 
        live_state: str = "data/live_state_test_rest_init"

    @dataclass
    class MockGlobalConfig: 
        project_name: str = "test_project_rest"
        paths: MockPathsConfig = field(default_factory=MockPathsConfig)
        logging: Any = None 
        simulation_defaults: Any = None
        wfo_settings: Any = None
        optuna_settings: Any = None
        optimization_profiles: Any = None # Ajouté pour la complétude

    @dataclass
    class MockLiveConfig: 
         live_fetch: MockLiveFetchConfig = field(default_factory=MockLiveFetchConfig)
         global_live_settings: MockGlobalLiveSettings = field(default_factory=MockGlobalLiveSettings)
         global_config: MockGlobalConfig = field(default_factory=MockGlobalConfig) 
         strategy_deployments: List[Any] = field(default_factory=list) # Ajouté
         live_logging: Any = None # Ajouté

    test_config_obj = MockLiveConfig() 
    
    test_project_root = Path(__file__).resolve().parent.parent.parent 
    Path(test_project_root / test_config_obj.global_config.paths.data_live_raw).mkdir(parents=True, exist_ok=True)
    Path(test_project_root / test_config_obj.global_config.paths.data_live_processed).mkdir(parents=True, exist_ok=True)

    try:
        run_initialization(test_config_obj) 
        logger.info("Live data initialization (REST only) completed for testing.")

    except ImportError as e:
         logger.critical(f"Missing library: {e}. Please install required packages.")
    except Exception as main_err:
         logger.critical(f"Critical error in main execution: {main_err}", exc_info=True)

