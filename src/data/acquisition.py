import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Dict, Optional, Any, List, Union, TYPE_CHECKING

import pandas as pd
import numpy as np # Ajout de numpy pour la gestion des NaN si besoin
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException

if TYPE_CHECKING:
    from src.config.loader import AppConfig

logger = logging.getLogger(__name__)

# Colonnes telles que reçues de l'API Binance pour les klines
BINANCE_KLINES_COLS = [
    'kline_open_time', 'open', 'high', 'low', 'close', 'volume',
    'kline_close_time', 'quote_asset_volume', 'number_of_trades',
    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
]

# Colonnes OHLCV de base que nous voulons conserver et utiliser
OUTPUT_OHLCV_COLS = ['open', 'high', 'low', 'close', 'volume']

# Colonnes finales dans les fichiers de sortie, incluant le timestamp et les données Taker
FINAL_OUTPUT_COLS = [
    'timestamp', 'open', 'high', 'low', 'close', 'volume',
    'quote_asset_volume', 'number_of_trades',
    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
    'taker_sell_base_asset_volume', 'taker_sell_quote_asset_volume' # Nouvelles colonnes Taker
]

KLINE_INTERVAL_1MINUTE = Client.KLINE_INTERVAL_1MINUTE

def _parse_and_clean_binance_klines(klines_data: List[List[Any]], pair_symbol: str) -> pd.DataFrame:
    """
    Parse les données brutes des klines de Binance, les nettoie, et ajoute les volumes taker vendeurs.

    Args:
        klines_data (List[List[Any]]): Données brutes de l'API Binance.
        pair_symbol (str): Symbole de la paire (pour logging).

    Returns:
        pd.DataFrame: DataFrame nettoyé avec les colonnes OHLCV et Taker.
    """
    if not klines_data:
        logger.warning(f"[{pair_symbol}] Aucune donnée kline fournie à _parse_and_clean_binance_klines.")
        return pd.DataFrame(columns=FINAL_OUTPUT_COLS)

    df = pd.DataFrame(klines_data, columns=BINANCE_KLINES_COLS)

    # Conversion du timestamp d'ouverture en datetime UTC
    df['timestamp'] = pd.to_datetime(df['kline_open_time'], unit='ms', utc=True)
    
    # Conversion des colonnes numériques
    numeric_cols = OUTPUT_OHLCV_COLS + [
        'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume'
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            logger.warning(f"[{pair_symbol}] Colonne attendue '{col}' non trouvée dans les données brutes. Sera remplie de NaN.")
            df[col] = np.nan


    # Calcul des volumes taker vendeurs
    # Assurer que les colonnes sources existent et sont numériques avant le calcul
    if 'volume' in df.columns and 'taker_buy_base_asset_volume' in df.columns:
        df['taker_sell_base_asset_volume'] = df['volume'] - df['taker_buy_base_asset_volume']
    else:
        logger.warning(f"[{pair_symbol}] Impossible de calculer 'taker_sell_base_asset_volume' en raison de colonnes manquantes ('volume' ou 'taker_buy_base_asset_volume').")
        df['taker_sell_base_asset_volume'] = np.nan

    if 'quote_asset_volume' in df.columns and 'taker_buy_quote_asset_volume' in df.columns:
        df['taker_sell_quote_asset_volume'] = df['quote_asset_volume'] - df['taker_buy_quote_asset_volume']
    else:
        logger.warning(f"[{pair_symbol}] Impossible de calculer 'taker_sell_quote_asset_volume' en raison de colonnes manquantes ('quote_asset_volume' ou 'taker_buy_quote_asset_volume').")
        df['taker_sell_quote_asset_volume'] = np.nan

    # Sélection et ordonnancement des colonnes finales
    # S'assurer que toutes les colonnes de FINAL_OUTPUT_COLS existent, sinon les créer avec NaN
    for col in FINAL_OUTPUT_COLS:
        if col not in df.columns:
            df[col] = np.nan
            logger.debug(f"[{pair_symbol}] Colonne finale '{col}' ajoutée avec NaN car non présente initialement.")
            
    df = df[FINAL_OUTPUT_COLS]
    
    # Suppression des lignes avec NaN pour les colonnes essentielles (timestamp + OHLCV)
    # Les volumes Taker peuvent parfois être NaN pour certaines bougies anciennes, on les garde pour l'instant
    # et on gérera les NaN plus tard si nécessaire lors de l'utilisation.
    essential_cols_for_dropna = ['timestamp'] + OUTPUT_OHLCV_COLS
    df.dropna(subset=essential_cols_for_dropna, how='any', inplace=True)
    
    df.sort_values(by='timestamp', ascending=True, inplace=True)
    
    initial_rows = len(df)
    df.drop_duplicates(subset=['timestamp'], keep='last', inplace=True)
    if initial_rows > len(df):
        logger.debug(f"[{pair_symbol}] Suppression de {initial_rows - len(df)} entrées de timestamp dupliquées.")

    # Vérification finale des NaN dans les colonnes OHLCV après nettoyage
    if df[OUTPUT_OHLCV_COLS].isnull().any().any():
        logger.warning(f"[{pair_symbol}] Des valeurs NaN subsistent dans les colonnes OHLCV après le chargement initial. Application de ffill et bfill.")
        for col in OUTPUT_OHLCV_COLS: # Appliquer uniquement sur OHLCV pour ne pas masquer des NaN légitimes ailleurs
            df[col] = df[col].ffill().bfill()
        # Nouvelle suppression des lignes si OHLCV sont encore NaN (ne devrait pas arriver après ffill/bfill si données existent)
        df.dropna(subset=OUTPUT_OHLCV_COLS, how='any', inplace=True)

    df = df.reset_index(drop=True)
    logger.debug(f"[{pair_symbol}] Données parsées et nettoyées. Shape: {df.shape}")
    return df

def _fetch_single_pair_1min_history_and_clean(
    client: Client,
    pair: str,
    start_str: str,
    end_str: Optional[str] = None,
    asset_type: str = "Margin",
    limit: int = 1000
) -> Optional[pd.DataFrame]:
    """
    Récupère l'historique des klines 1-minute pour une seule paire et les nettoie.
    """
    logger.info(f"Récupération et nettoyage des données 1-minute pour {pair} de {start_str} à {end_str or 'maintenant'}...")
    all_klines_raw: List[List[Any]] = []
    current_start_str = start_str
    max_retries = 3
    retry_delay_seconds = 5

    while True:
        klines_batch = []
        for attempt in range(max_retries):
            try:
                logger.debug(f"Requête des klines {KLINE_INTERVAL_1MINUTE} pour {pair} commençant à {current_start_str}, limite {limit}")
                if asset_type.upper() in ["MARGIN", "SPOT"]:
                    klines_batch = client.get_historical_klines(
                        pair, KLINE_INTERVAL_1MINUTE, current_start_str, end_str, limit=limit
                    )
                # Note: Pour les Futures, l'endpoint et potentiellement le client seraient différents.
                # Cette fonction est actuellement orientée SPOT/MARGIN.
                else:
                    logger.error(f"Type d'actif non supporté '{asset_type}' pour la récupération de données historiques.")
                    return None
                logger.debug(f"Reçu {len(klines_batch)} klines pour {pair} commençant à {current_start_str}.")
                break 
            except BinanceAPIException as e:
                logger.error(f"Erreur API Binance lors de la récupération de {pair}: {e} (Statut: {e.status_code}, Code: {e.code})")
                if e.status_code in [429, 418] and attempt < max_retries - 1: # Codes pour rate limit
                    sleep_time = retry_delay_seconds * (2**attempt) # Backoff exponentiel
                    logger.warning(f"Rate limit atteint. Nouvelle tentative dans {sleep_time} secondes...")
                    time.sleep(sleep_time)
                elif attempt == max_retries -1:
                    logger.error(f"Nombre maximal de tentatives atteint pour {pair} après erreur API.")
                    return None # Abandonner après max_retries
                else: # Autres erreurs API, nouvelle tentative simple
                    time.sleep(retry_delay_seconds)
            except BinanceRequestException as e: # Erreurs liées à la requête elle-même (ex: mauvais paramètres)
                logger.error(f"Erreur de requête Binance lors de la récupération de {pair}: {e}")
                return None # Probablement inutile de réessayer
            except Exception as e: # Autres exceptions (réseau, etc.)
                logger.exception(f"Erreur inattendue lors de la récupération de {pair}: {e}", exc_info=True)
                if attempt < max_retries - 1:
                    time.sleep(retry_delay_seconds)
                else:
                    logger.error(f"Nombre maximal de tentatives atteint pour {pair} après erreur inattendue.")
                    return None
        
        if not klines_batch: # Plus de données à récupérer ou erreur persistante
            break

        all_klines_raw.extend(klines_batch)
        
        # Préparer pour la prochaine requête
        try:
            # Le timestamp de la dernière kline + 1 minute (en millisecondes)
            next_start_time_ms = int(klines_batch[-1][0]) + 60000 
            current_start_str = str(next_start_time_ms)
        except (IndexError, TypeError, ValueError) as e_time:
            logger.error(f"Erreur lors du traitement du timestamp pour le prochain lot de {pair}: {e_time}. Lot: {klines_batch[-1] if klines_batch else 'vide'}")
            break # Impossible de continuer si le timestamp est invalide


        # Vérifier si la date de fin est atteinte (si fournie)
        if end_str:
            try:
                # Convertir end_str en timestamp millisecondes pour comparaison
                end_dt_ms = pd.to_datetime(end_str, utc=True).timestamp() * 1000
                if int(klines_batch[-1][0]) >= end_dt_ms:
                    logger.debug(f"Date de fin atteinte pour {pair}.")
                    break
            except ValueError:
                 logger.error(f"Impossible de parser end_str '{end_str}' comme date pour {pair}.")
                 break # Arrêter si la date de fin est mal formatée
        
        time.sleep(0.2) # Petit délai pour respecter les limites de l'API

    if not all_klines_raw:
        logger.warning(f"Aucune donnée historique 1-minute récupérée pour {pair}.")
        return pd.DataFrame(columns=FINAL_OUTPUT_COLS)

    logger.info(f"Récupération réussie de {len(all_klines_raw)} klines 1-minute brutes pour {pair}. Parsing et nettoyage...")
    df_cleaned = _parse_and_clean_binance_klines(all_klines_raw, pair)

    # Filtrage final par date de fin si nécessaire (pour s'assurer de ne pas dépasser)
    if end_str and not df_cleaned.empty:
        try:
            end_dt_filter = pd.to_datetime(end_str, utc=True)
            df_cleaned = df_cleaned[df_cleaned['timestamp'] < end_dt_filter]
        except ValueError:
            logger.error(f"Impossible de parser end_str '{end_str}' pour le filtrage final de {pair}.")


    logger.info(f"Récupération et nettoyage des données 1-minute terminés pour {pair}. Lignes finales: {len(df_cleaned)}")
    return df_cleaned

def fetch_all_historical_data(config: 'AppConfig') -> str:
    """
    Orchestre la récupération des données historiques 1-minute pour toutes les paires configurées.
    Sauvegarde les données brutes (CSV) et les données nettoyées (Parquet).
    """
    logger.info("Démarrage du processus de récupération et de nettoyage des données historiques (klines 1-minute par paire)...")

    api_key = config.api_keys.binance_api_key
    api_secret = config.api_keys.binance_secret_key
    if not api_key or not api_secret:
        logger.error("Clé API ou secrète Binance manquante. Impossible de récupérer les données historiques.")
        raise ValueError("Les identifiants API Binance n'ont pas été trouvés dans la configuration/environnement.")

    try:
        client = Client(api_key, api_secret)
        logger.info("Client Binance initialisé.")
    except Exception as client_err:
        logger.error(f"Échec de l'initialisation du client Binance: {client_err}", exc_info=True)
        raise RuntimeError("Échec de l'initialisation du client Binance") from client_err

    pairs = config.data_config.assets_and_timeframes.pairs
    start_date_str = config.data_config.historical_period.start_date
    end_date_config_str = config.data_config.historical_period.end_date
    
    end_date_to_use_str: Optional[str]
    if end_date_config_str is None:
        end_date_to_use_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        logger.info(f"Date de fin non spécifiée, utilisation de la date/heure actuelle: {end_date_to_use_str}")
    else:
        end_date_to_use_str = end_date_config_str

    asset_type = config.data_config.source_details.asset_type
    max_workers = config.data_config.fetching_options.max_workers
    batch_limit = config.data_config.fetching_options.batch_size # Limite par requête API

    # Création des répertoires de sortie
    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    
    raw_data_base_dir = config.global_config.paths.data_historical_raw
    # Les données brutes sont versionnées par exécution pour traçabilité
    raw_run_output_dir = os.path.join(raw_data_base_dir, run_timestamp)
    
    # Les données nettoyées écrasent la version précédente pour avoir toujours la dernière version propre
    cleaned_data_output_dir = config.global_config.paths.data_historical_processed_cleaned

    try:
        os.makedirs(raw_run_output_dir, exist_ok=True)
        logger.info(f"Répertoire de cycle créé pour les données brutes: {raw_run_output_dir}")
        os.makedirs(cleaned_data_output_dir, exist_ok=True)
        logger.info(f"Répertoire assuré pour les données Parquet 1-minute nettoyées: {cleaned_data_output_dir}")
    except OSError as e:
        logger.error(f"Échec de la création des répertoires de sortie: {e}")
        raise

    tasks: List[str] = [pair for pair in pairs if pair] # Filtrer les paires vides/None
    results_dfs: Dict[str, Optional[pd.DataFrame]] = {}

    logger.info(f"Démarrage du téléchargement parallèle et du nettoyage des klines 1-minute pour {len(tasks)} paires...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_pair = {
            executor.submit(
                _fetch_single_pair_1min_history_and_clean,
                client, pair, start_date_str, end_date_to_use_str, asset_type, batch_limit
            ): pair
            for pair in tasks
        }

        for future in as_completed(future_to_pair):
            pair_symbol = future_to_pair[future]
            try:
                result_df = future.result()
                results_dfs[pair_symbol] = result_df
                
                if result_df is not None and not result_df.empty:
                    # Sauvegarde des données brutes (mais parsées et avec colonnes taker) pour audit
                    raw_file_name = f"{pair_symbol}_1min_raw_with_taker.csv" # Nom de fichier mis à jour
                    raw_file_path = os.path.join(raw_run_output_dir, raw_file_name)
                    
                    # Sauvegarde des données nettoyées finales au format Parquet
                    cleaned_parquet_file_name = f"{pair_symbol}_1min_cleaned_with_taker.parquet" # Nom de fichier mis à jour
                    cleaned_parquet_file_path = os.path.join(cleaned_data_output_dir, cleaned_parquet_file_name)
                    
                    try:
                        result_df.to_csv(raw_file_path, index=False)
                        logger.info(f"Données brutes 1-minute (avec taker) sauvegardées pour {pair_symbol} vers: {raw_file_path}")
                        
                        result_df.to_parquet(cleaned_parquet_file_path, index=False, engine='pyarrow')
                        logger.info(f"Données 1-minute nettoyées (avec taker) sauvegardées pour {pair_symbol} vers: {cleaned_parquet_file_path}")
                    except IOError as e:
                        logger.error(f"Échec de la sauvegarde du fichier de données pour {pair_symbol}: {e}")
                elif result_df is not None and result_df.empty:
                    logger.warning(f"Aucune donnée 1-minute retournée/traitée pour {pair_symbol}, fichiers non sauvegardés.")
                else: # result_df is None
                    logger.error(f"La tâche de récupération et de nettoyage a échoué pour {pair_symbol}, fichiers non sauvegardés.")
            except Exception as exc:
                logger.error(f"La tâche pour {pair_symbol} a généré une exception: {exc}", exc_info=True)
                results_dfs[pair_symbol] = None # Marquer comme échoué

    successful_fetches = sum(1 for df in results_dfs.values() if df is not None and not df.empty)
    failed_fetches = len(tasks) - successful_fetches
    logger.info(f"Récupération et nettoyage des données historiques 1-minute terminés. Paires réussies: {successful_fetches}, Paires échouées/vides: {failed_fetches}.")

    if successful_fetches == 0 and tasks: # Si aucune donnée n'a été récupérée pour aucune paire
        logger.error("Aucune donnée historique 1-minute n'a pu être récupérée et nettoyée avec succès pour aucune paire configurée.")
        raise RuntimeError("La récupération et le nettoyage des données historiques 1-minute ont complètement échoué.")

    return raw_run_output_dir # Retourne le chemin du répertoire de cycle pour les données brutes
