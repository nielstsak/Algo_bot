import logging
import time
from pathlib import Path
from typing import Dict, Optional, Any, List, Union
import re
import pandas as pd
import numpy as np
import pandas_ta as ta
from src.data import data_utils # Importation correcte

logger = logging.getLogger(__name__)

REQUIRED_INPUT_COLS_LOWER = {
    'timestamp', 'open', 'high', 'low', 'close', 'volume',
    'quote_asset_volume', 'number_of_trades',
    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume',
    'taker_sell_base_asset_volume', 'taker_sell_quote_asset_volume',
    'kline_close_time'
}

def _ensure_required_columns(df: pd.DataFrame, required_cols: List[str], log_ctx: str = "") -> bool:
    df_cols_lower = {str(c).lower() for c in df.columns}
    required_lower = {str(c).lower() for c in required_cols}
    
    logger.debug(f"[_ensure_required_columns][{log_ctx}] Vérification. Colonnes DataFrame (minuscules): {df_cols_lower}. Colonnes requises (minuscules): {required_lower}")

    if not required_lower.issubset(df_cols_lower):
        missing_cols = required_lower - df_cols_lower
        logger.error(f"[_ensure_required_columns][{log_ctx}] DataFrame manquant des colonnes requises (insensible à la casse): {missing_cols}")
        return False
    return True

def _calculate_ema_rolling(series: pd.Series, period: int, series_name_debug: str = "Series") -> pd.Series:
    log_prefix_calc = f"[EMA_Calc for {series_name_debug}({period})]"
    if not isinstance(series, pd.Series):
        logger.warning(f"{log_prefix_calc} Entrée n'est pas une Series pandas. Type: {type(series)}")
        return pd.Series(np.nan, name=f"EMA_{period}", index=series.index if isinstance(series, pd.Series) else None)
    if series.empty:
        logger.warning(f"{log_prefix_calc} Série d'entrée vide.")
        return pd.Series(np.nan, index=series.index, name=f"EMA_{period}")
    if series.isnull().all():
        logger.warning(f"{log_prefix_calc} Série d'entrée entièrement NaN.")
        return pd.Series(np.nan, index=series.index, name=f"EMA_{period}")
    if period <= 0:
        logger.warning(f"{log_prefix_calc} Période <= 0.")
        return pd.Series(np.nan, index=series.index, name=f"EMA_{period}")
    
    ema_result = series.ewm(span=period, adjust=False, min_periods=max(1,period)).mean()
    return ema_result

def _calculate_atr_rolling(high_series: pd.Series, low_series: pd.Series, close_series: pd.Series, period: int, series_name_debug: str = "Series") -> pd.Series:
    log_prefix_calc = f"[ATR_Calc for {series_name_debug}({period})]"
    # Determine a valid index for returning empty series if inputs are problematic
    valid_index = None
    if isinstance(close_series, pd.Series) and not close_series.empty:
        valid_index = close_series.index
    elif isinstance(high_series, pd.Series) and not high_series.empty:
        valid_index = high_series.index
    elif isinstance(low_series, pd.Series) and not low_series.empty:
        valid_index = low_series.index
        
    if not all(isinstance(s, pd.Series) for s in [high_series, low_series, close_series]):
        logger.warning(f"{log_prefix_calc} Une ou plusieurs entrées ne sont pas des Series pandas.")
        return pd.Series(np.nan, name=f"ATR_{period}", index=valid_index)
    if high_series.empty or low_series.empty or close_series.empty:
        logger.warning(f"{log_prefix_calc} Une ou plusieurs séries d'entrée (H,L,C) est vide.")
        return pd.Series(np.nan, index=valid_index, name=f"ATR_{period}")
    if high_series.isnull().all() or low_series.isnull().all() or close_series.isnull().all():
        logger.warning(f"{log_prefix_calc} Une ou plusieurs séries d'entrée (H,L,C) est entièrement NaN.")
        return pd.Series(np.nan, index=valid_index, name=f"ATR_{period}")
    if period <= 0:
        logger.warning(f"{log_prefix_calc} Période <= 0.")
        return pd.Series(np.nan, index=valid_index, name=f"ATR_{period}")

    prev_close = close_series.shift(1)
    tr1 = high_series - low_series
    tr2 = abs(high_series - prev_close)
    tr3 = abs(low_series - prev_close)
    
    true_range = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
    if true_range.isnull().all():
        logger.warning(f"{log_prefix_calc} La série True Range est entièrement NaN.")
        return pd.Series(np.nan, index=valid_index, name=f"ATR_{period}")
        
    atr = _calculate_ema_rolling(true_range, period, f"TR_for_{series_name_debug}") 
    atr.name = f"ATR_{period}" 
    return atr

def preprocess_live_data_for_strategy(
    raw_data_path: Path,
    processed_output_path: Path,
    strategy_params: Dict[str, Any], 
    strategy_name: str 
) -> Optional[pd.DataFrame]:
    
    pair_symbol_raw_log = raw_data_path.name.split('_')[0] 
    log_prefix_main = f"[{pair_symbol_raw_log} for {strategy_name}]" 

    logger.info(f"{log_prefix_main} Démarrage du prétraitement live. Sortie: {processed_output_path.name}")
    logger.debug(f"{log_prefix_main} Paramètres de stratégie reçus: {strategy_params}")
    start_time = time.time()

    if not raw_data_path.exists() or raw_data_path.stat().st_size == 0:
        logger.warning(f"{log_prefix_main} Fichier de données 1-minute brutes non trouvé ou vide: {raw_data_path}.")
        return None
    
    df_1m_raw: pd.DataFrame
    try:
        logger.debug(f"{log_prefix_main} Lecture du CSV: {raw_data_path}")
        temp_df = pd.read_csv(raw_data_path, low_memory=False)
        
        if temp_df is None or temp_df.empty: 
            logger.error(f"{log_prefix_main} pd.read_csv a retourné None ou un DataFrame vide pour {raw_data_path}.")
            return None
        
        logger.debug(f"{log_prefix_main} Colonnes brutes du CSV: {temp_df.columns.tolist()}")
        temp_df.columns = [str(col).lower() for col in temp_df.columns]
        logger.debug(f"{log_prefix_main} Colonnes après lowercasing: {temp_df.columns.tolist()}")
        
        if 'timestamp' not in temp_df.columns:
            logger.error(f"{log_prefix_main} Colonne 'timestamp' manquante dans les données 1-minute brutes {raw_data_path.name} après lowercasing.")
            return None
        logger.debug(f"{log_prefix_main} Colonne 'timestamp' trouvée. Aperçu: {temp_df['timestamp'].head(2).to_string() if not temp_df.empty else 'DF Vide'}")

        temp_df['timestamp'] = pd.to_datetime(temp_df['timestamp'], errors='coerce', utc=True)
        temp_df.dropna(subset=['timestamp'], inplace=True)
        if temp_df.empty: 
            logger.warning(f"{log_prefix_main} DataFrame vide après conversion/dropna du timestamp pour {raw_data_path.name}.")
            return None

        logger.debug(f"{log_prefix_main} Colonnes de temp_df avant set_index: {temp_df.columns.tolist()}")
        temp_df = temp_df.set_index('timestamp').sort_index()
        logger.debug(f"{log_prefix_main} Colonnes de temp_df APRES set_index (devrait être les colonnes de df_1m_raw): {temp_df.columns.tolist()}")
        
        if not temp_df.index.is_unique:
            temp_df = temp_df[~temp_df.index.duplicated(keep='last')]
        
        df_1m_raw = temp_df.copy() 
        
        cols_to_check_in_df_1m_raw = list(REQUIRED_INPUT_COLS_LOWER - {'timestamp'})
        logger.debug(f"{log_prefix_main} Colonnes à vérifier dans df_1m_raw (excluant l'index 'timestamp'): {cols_to_check_in_df_1m_raw}")
        if not _ensure_required_columns(df_1m_raw, cols_to_check_in_df_1m_raw, f"{pair_symbol_raw_log}_df_1m_raw_check"):
            logger.error(f"{log_prefix_main} Vérification des colonnes de df_1m_raw (post set_index) échouée.")
            return None
        
        for col in cols_to_check_in_df_1m_raw: 
            if col in df_1m_raw.columns and col != 'timestamp': 
                df_1m_raw[col] = pd.to_numeric(df_1m_raw[col], errors='coerce')

        ohlcv_cols_for_cleaning = ['open', 'high', 'low', 'close', 'volume']
        nan_ohlcv_before = df_1m_raw[ohlcv_cols_for_cleaning].isnull().sum().sum()
        if nan_ohlcv_before > 0:
            df_1m_raw[ohlcv_cols_for_cleaning] = df_1m_raw[ohlcv_cols_for_cleaning].ffill().bfill() 
            df_1m_raw.dropna(subset=ohlcv_cols_for_cleaning, inplace=True) 
        
        if df_1m_raw.empty: 
            logger.warning(f"{log_prefix_main} DataFrame 1-minute vide après nettoyage des NaNs OHLCV.")
            return None
        
        base_ohlcv_cols_for_processed = ['open', 'high', 'low', 'close', 'volume']
        if not _ensure_required_columns(df_1m_raw, base_ohlcv_cols_for_processed, f"{pair_symbol_raw_log}_base_ohlcv_check_for_df_processed"):
            logger.error(f"{log_prefix_main} Colonnes OHLCV de base manquantes dans df_1m_raw pour initialiser df_processed.")
            return None
        df_processed = df_1m_raw[base_ohlcv_cols_for_processed].copy() 
        logger.info(f"{log_prefix_main} Données OHLCV 1-minute initiales pour traitement. Shape: {df_processed.shape}")

    except Exception as e: 
        logger.error(f"{log_prefix_main} Erreur lors du chargement/nettoyage initial des données 1-minute brutes de {raw_data_path}: {e}", exc_info=True)
        return None

    freq_map_for_rolling: Dict[str, List[str]] = {} 
    for param_key, param_value in strategy_params.items():
        if "indicateur_frequence" in param_key and isinstance(param_value, str) and param_value:
            context_name = param_key.replace("indicateur_frequence_", "").upper()
            if param_value not in freq_map_for_rolling:
                freq_map_for_rolling[param_value] = []
            if context_name not in freq_map_for_rolling[param_value]: 
                freq_map_for_rolling[param_value].append(context_name)
    
    atr_freq_sl_tp_raw = strategy_params.get('atr_base_frequency_sl_tp', strategy_params.get('atr_base_frequency')) 
    if isinstance(atr_freq_sl_tp_raw, str) and atr_freq_sl_tp_raw:
        if atr_freq_sl_tp_raw not in freq_map_for_rolling:
            freq_map_for_rolling[atr_freq_sl_tp_raw] = []
        if "ATR_SL_TP" not in freq_map_for_rolling[atr_freq_sl_tp_raw]:
            freq_map_for_rolling[atr_freq_sl_tp_raw].append("ATR_SL_TP")
    
    logger.info(f"{log_prefix_main} Carte des fréquences construite pour les indicateurs: {freq_map_for_rolling}")

    rolling_windows_to_calculate: Dict[int, str] = {} 
    for freq_str_config, contexts in freq_map_for_rolling.items(): 
        match = re.fullmatch(r"(\d+)(min|m|h|d)", freq_str_config.lower().strip())
        if match:
            num = int(match.group(1))
            unit = match.group(2)
            window_minutes = 0
            if unit in ["min", "m"]: window_minutes = num
            elif unit == "h": window_minutes = num * 60
            elif unit == "d": window_minutes = num * 60 * 24
            
            if window_minutes > 0:
                if window_minutes not in rolling_windows_to_calculate:
                    rolling_windows_to_calculate[window_minutes] = freq_str_config 
                elif rolling_windows_to_calculate[window_minutes] != freq_str_config:
                    logger.debug(f"{log_prefix_main} Fenêtre de {window_minutes} min déjà mappée à '{rolling_windows_to_calculate[window_minutes]}'. '{freq_str_config}' (pour {contexts}) utilisera les mêmes K-lines agrégées mais le nom de colonne sera basé sur la première fréquence mappée.")
            else:
                logger.warning(f"{log_prefix_main} window_minutes <= 0 pour freq_str='{freq_str_config}'. Ignoré.")
        else:
            logger.warning(f"{log_prefix_main} Impossible de parser la taille de la fenêtre depuis la chaîne de fréquence: {freq_str_config} (contextes: {contexts})")

    logger.info(f"{log_prefix_main} Fenêtres glissantes uniques à calculer (taille_min: label_freq_config): {rolling_windows_to_calculate}")

    if not isinstance(df_1m_raw.index, pd.DatetimeIndex):
        logger.critical(f"{log_prefix_main} L'index de df_1m_raw n'est pas un DatetimeIndex avant les calculs de fenêtres glissantes. Type: {type(df_1m_raw.index)}. Abandon.")
        return None

    taker_vol_cols_to_agg = [
        'taker_buy_base_asset_volume', 'taker_sell_base_asset_volume',
        'taker_buy_quote_asset_volume', 'taker_sell_quote_asset_volume',
        'number_of_trades', 'quote_asset_volume'
    ]
    ohlcv_cols_for_agg = ['open', 'high', 'low', 'close', 'volume']

    for window_val_mins, period_label_str_config in sorted(rolling_windows_to_calculate.items()):
        if window_val_mins <= 0: 
            logger.warning(f"{log_prefix_main} Ignorer le calcul de la fenêtre glissante pour fenêtre non positive: {window_val_mins} ({period_label_str_config})")
            continue
        logger.info(f"{log_prefix_main} Calcul des klines glissantes {period_label_str_config} (taille fenêtre: {window_val_mins})...")
        
        df_processed[f"Kline_{period_label_str_config}_open"] = df_1m_raw['open'].rolling(window=window_val_mins, min_periods=window_val_mins).apply(lambda x: x[0] if len(x) >= window_val_mins and len(x) > 0 else np.nan, raw=True)
        df_processed[f"Kline_{period_label_str_config}_high"] = df_1m_raw['high'].rolling(window=window_val_mins, min_periods=window_val_mins).max()
        df_processed[f"Kline_{period_label_str_config}_low"] = df_1m_raw['low'].rolling(window=window_val_mins, min_periods=window_val_mins).min()
        df_processed[f"Kline_{period_label_str_config}_close"] = df_1m_raw['close'].rolling(window=window_val_mins, min_periods=window_val_mins).apply(lambda x: x[-1] if len(x) >= window_val_mins and len(x) > 0 else np.nan, raw=True)
        df_processed[f"Kline_{period_label_str_config}_volume"] = df_1m_raw['volume'].rolling(window=window_val_mins, min_periods=window_val_mins).sum()

        for taker_col in taker_vol_cols_to_agg:
            if taker_col in df_1m_raw.columns:
                df_processed[f"Kline_{period_label_str_config}_{taker_col}"] = df_1m_raw[taker_col].rolling(window=window_val_mins, min_periods=window_val_mins).sum()
            else:
                df_processed[f"Kline_{period_label_str_config}_{taker_col}"] = np.nan
        
    logger.debug(f"{log_prefix_main} Colonnes df_processed après agrégation Klines: {df_processed.columns.tolist()}")
    
    atr_period_sl_tp = strategy_params.get('atr_period_sl_tp', strategy_params.get('atr_period'))
    atr_freq_sl_tp_raw = strategy_params.get('atr_base_frequency_sl_tp', strategy_params.get('atr_base_frequency'))
    atr_freq_for_col_name = atr_freq_sl_tp_raw # This should be correct as '5min', '15min' etc.

    if atr_period_sl_tp and atr_freq_for_col_name:
        # CORRECTED: Use "Kline_" (singular) prefix
        atr_high_col = f"Kline_{atr_freq_for_col_name}_high"
        atr_low_col = f"Kline_{atr_freq_for_col_name}_low"
        atr_close_col = f"Kline_{atr_freq_for_col_name}_close"
        logger.debug(f"{log_prefix_main} Recherche ATR_strat avec colonnes sources: H={atr_high_col}, L={atr_low_col}, C={atr_close_col}")
        if all(c in df_processed.columns for c in [atr_high_col, atr_low_col, atr_close_col]):
            df_processed["ATR_strat"] = _calculate_atr_rolling(
                df_processed[atr_high_col], df_processed[atr_low_col], df_processed[atr_close_col],
                period=int(atr_period_sl_tp), series_name_debug=f"ATR_SL_TP_source({atr_freq_for_col_name})"
            )
            logger.info(f"{log_prefix_main} ATR_strat (pour SL/TP) calculé sur Klines {atr_freq_for_col_name}. NaNs: {df_processed['ATR_strat'].isnull().sum()}/{len(df_processed)}")
        else:
            logger.warning(f"{log_prefix_main} Colonnes sources pour ATR_strat ({atr_high_col}, {atr_low_col}, {atr_close_col}) non trouvées. ATR_strat sera NaN.")
            df_processed["ATR_strat"] = np.nan
    else:
        df_processed["ATR_strat"] = np.nan

    current_strategy_name_lower = strategy_name.lower()

    def get_effective_kline_freq_label(raw_freq_from_params: Optional[str]) -> str:
        if not raw_freq_from_params: return "1min" 
        prefix = data_utils.get_kline_prefix_effective(raw_freq_from_params) 
        if not prefix: 
            return "1min" 
        freq_part = prefix.replace("Klines_", "") # Assuming data_utils.get_kline_prefix_effective returns "Klines_..."
        return freq_part if freq_part else "1min"

    if current_strategy_name_lower == "emamacdatrstrategy":
        ema_s_p = strategy_params.get('ema_short_period'); ema_l_p = strategy_params.get('ema_long_period')
        ema_freq_label = get_effective_kline_freq_label(strategy_params.get('indicateur_frequence_ema'))
        # CORRECTED: Use "Kline_" (singular) prefix
        ema_close_col = f"Kline_{ema_freq_label}_close" if ema_freq_label != "1min" else "close"
        logger.debug(f"{log_prefix_main} EMA: src_col='{ema_close_col}', short_p={ema_s_p}, long_p={ema_l_p}")
        if ema_s_p and ema_l_p and ema_close_col in df_processed and df_processed[ema_close_col].notna().any():
            df_processed["EMA_short_strat"] = _calculate_ema_rolling(df_processed[ema_close_col], period=int(ema_s_p))
            df_processed["EMA_long_strat"] = _calculate_ema_rolling(df_processed[ema_close_col], period=int(ema_l_p))
        else: df_processed["EMA_short_strat"] = np.nan; df_processed["EMA_long_strat"] = np.nan
        if "EMA_short_strat" in df_processed: logger.debug(f"{log_prefix_main} EMA_short_strat NaNs: {df_processed['EMA_short_strat'].isnull().sum()}/{len(df_processed)}")
        
        macd_f = strategy_params.get('macd_fast_period'); macd_s = strategy_params.get('macd_slow_period'); macd_sig = strategy_params.get('macd_signal_period')
        macd_freq_label = get_effective_kline_freq_label(strategy_params.get('indicateur_frequence_macd'))
        # CORRECTED: Use "Kline_" (singular) prefix
        macd_close_col = f"Kline_{macd_freq_label}_close" if macd_freq_label != "1min" else "close"
        logger.debug(f"{log_prefix_main} MACD: src_col='{macd_close_col}', f={macd_f}, s={macd_s}, sig={macd_sig}")
        if macd_f and macd_s and macd_sig and macd_close_col in df_processed and df_processed[macd_close_col].notna().any():
            macd_df = ta.macd(df_processed[macd_close_col].dropna(), fast=int(macd_f), slow=int(macd_s), signal=int(macd_sig), append=False)
            if macd_df is not None:
                df_processed["MACD_line_strat"] = macd_df.iloc[:,0]; df_processed["MACD_hist_strat"] = macd_df.iloc[:,1]; df_processed["MACD_signal_strat"] = macd_df.iloc[:,2]
        else: df_processed["MACD_line_strat"]=np.nan; df_processed["MACD_hist_strat"]=np.nan; df_processed["MACD_signal_strat"]=np.nan
        if "MACD_line_strat" in df_processed: logger.debug(f"{log_prefix_main} MACD_line_strat NaNs: {df_processed['MACD_line_strat'].isnull().sum()}/{len(df_processed)}")

    elif current_strategy_name_lower == "bbandsvolumersistrategy":
        bb_p = strategy_params.get('bbands_period'); bb_std = strategy_params.get('bbands_std_dev')
        bb_freq_label = get_effective_kline_freq_label(strategy_params.get('indicateur_frequence_bbands'))
        # CORRECTED: Use "Kline_" (singular) prefix
        bb_close_col = f"Kline_{bb_freq_label}_close" if bb_freq_label != "1min" else "close"
        logger.debug(f"{log_prefix_main} BBands: src_col='{bb_close_col}', p={bb_p}, std={bb_std}")
        if bb_p and bb_std and bb_close_col in df_processed.columns and df_processed[bb_close_col].notna().any(): # check .columns explicitly
            logger.debug(f"BBands input series '{bb_close_col}' non-NaN count: {df_processed[bb_close_col].notna().sum()}")
            bb_input_series = df_processed[bb_close_col].dropna()
            if not bb_input_series.empty and len(bb_input_series) >= bb_p:
                 bb_df = ta.bbands(bb_input_series, length=int(bb_p), std=float(bb_std), append=False)
                 if bb_df is not None:
                    df_processed["BB_LOWER_strat"]=bb_df.iloc[:,0]; df_processed["BB_MIDDLE_strat"]=bb_df.iloc[:,1]; df_processed["BB_UPPER_strat"]=bb_df.iloc[:,2]; df_processed["BB_BANDWIDTH_strat"]=bb_df.iloc[:,3]
                 else:
                    logger.warning(f"{log_prefix_main} ta.bbands returned None for {bb_close_col}")
                    df_processed["BB_LOWER_strat"]=np.nan; df_processed["BB_MIDDLE_strat"]=np.nan; df_processed["BB_UPPER_strat"]=np.nan; df_processed["BB_BANDWIDTH_strat"]=np.nan
            else:
                logger.warning(f"{log_prefix_main} BBands input series for {bb_close_col} is empty or too short after dropna. Length: {len(bb_input_series)}, Required: {bb_p}")
                df_processed["BB_LOWER_strat"]=np.nan; df_processed["BB_MIDDLE_strat"]=np.nan; df_processed["BB_UPPER_strat"]=np.nan; df_processed["BB_BANDWIDTH_strat"]=np.nan
        else: 
            logger.warning(f"{log_prefix_main} Condition failed for BBands calculation. bb_close_col='{bb_close_col}'. Exists: {bb_close_col in df_processed.columns}. Has non-NaN: {df_processed[bb_close_col].notna().any() if bb_close_col in df_processed.columns else 'N/A'}")
            df_processed["BB_LOWER_strat"]=np.nan; df_processed["BB_MIDDLE_strat"]=np.nan; df_processed["BB_UPPER_strat"]=np.nan; df_processed["BB_BANDWIDTH_strat"]=np.nan
        if "BB_UPPER_strat" in df_processed: logger.debug(f"{log_prefix_main} BB_UPPER_strat NaNs: {df_processed['BB_UPPER_strat'].isnull().sum()}/{len(df_processed)}")

        vol_ma_p = strategy_params.get('volume_ma_period')
        vol_freq_label = get_effective_kline_freq_label(strategy_params.get('indicateur_frequence_volume'))
        # CORRECTED: Use "Kline_" (singular) prefix
        vol_kline_col = f"Kline_{vol_freq_label}_volume" if vol_freq_label != "1min" else "volume"
        logger.debug(f"{log_prefix_main} Volume MA: src_col='{vol_kline_col}', p={vol_ma_p}")
        if vol_ma_p and vol_kline_col in df_processed.columns and df_processed[vol_kline_col].notna().any():
            df_processed["Volume_MA_strat"] = _calculate_ema_rolling(df_processed[vol_kline_col], period=int(vol_ma_p))
        else: 
            logger.warning(f"{log_prefix_main} Condition failed for Volume MA calculation. vol_kline_col='{vol_kline_col}'. Exists: {vol_kline_col in df_processed.columns}. Has non-NaN: {df_processed[vol_kline_col].notna().any() if vol_kline_col in df_processed.columns else 'N/A'}")
            df_processed["Volume_MA_strat"] = np.nan
        if "Volume_MA_strat" in df_processed: logger.debug(f"{log_prefix_main} Volume_MA_strat NaNs: {df_processed['Volume_MA_strat'].isnull().sum()}/{len(df_processed)}")
        # REMOVED/COMMENTED: The line that might create a typo'd column
        # if vol_kline_col not in df_processed.columns: df_processed[vol_kline_col] = np.nan

        rsi_p = strategy_params.get('rsi_period')
        rsi_freq_label = get_effective_kline_freq_label(strategy_params.get('indicateur_frequence_rsi'))
        # CORRECTED: Use "Kline_" (singular) prefix
        rsi_close_col = f"Kline_{rsi_freq_label}_close" if rsi_freq_label != "1min" else "close"
        logger.debug(f"{log_prefix_main} RSI: src_col='{rsi_close_col}', p={rsi_p}")
        if rsi_p and rsi_close_col in df_processed.columns and df_processed[rsi_close_col].notna().any():
            rsi_input_series = df_processed[rsi_close_col].dropna()
            if not rsi_input_series.empty and len(rsi_input_series) >= rsi_p :
                 df_processed["RSI_strat"] = ta.rsi(rsi_input_series, length=int(rsi_p), append=False)
            else:
                logger.warning(f"{log_prefix_main} RSI input series for {rsi_close_col} is empty or too short after dropna. Length: {len(rsi_input_series)}, Required: {rsi_p}")
                df_processed["RSI_strat"] = np.nan
        else: 
            logger.warning(f"{log_prefix_main} Condition failed for RSI calculation. rsi_close_col='{rsi_close_col}'. Exists: {rsi_close_col in df_processed.columns}. Has non-NaN: {df_processed[rsi_close_col].notna().any() if rsi_close_col in df_processed.columns else 'N/A'}")
            df_processed["RSI_strat"] = np.nan
        if "RSI_strat" in df_processed: logger.debug(f"{log_prefix_main} RSI_strat NaNs: {df_processed['RSI_strat'].isnull().sum()}/{len(df_processed)}")

    elif current_strategy_name_lower == "kamaadxstochstrategy":
        kama_p = strategy_params.get('kama_period'); kama_f = strategy_params.get('kama_fast_ema'); kama_s = strategy_params.get('kama_slow_ema')
        kama_freq_label = get_effective_kline_freq_label(strategy_params.get('indicateur_frequence_kama'))
        # CORRECTED: Use "Kline_" (singular) prefix
        kama_close_col = f"Kline_{kama_freq_label}_close" if kama_freq_label != "1min" else "close"
        logger.debug(f"{log_prefix_main} KAMA: src_col='{kama_close_col}', p={kama_p}, f={kama_f}, s={kama_s}")
        if kama_p and kama_f and kama_s and kama_close_col in df_processed.columns and df_processed[kama_close_col].notna().any():
            df_processed["KAMA_strat"] = ta.kama(df_processed[kama_close_col].dropna(), length=int(kama_p), fast=int(kama_f), slow=int(kama_s), append=False)
        else: df_processed["KAMA_strat"] = np.nan
        if "KAMA_strat" in df_processed: logger.debug(f"{log_prefix_main} KAMA_strat NaNs: {df_processed['KAMA_strat'].isnull().sum()}/{len(df_processed)}")

        adx_p = strategy_params.get('adx_period')
        adx_freq_label = get_effective_kline_freq_label(strategy_params.get('indicateur_frequence_adx'))
        # CORRECTED: Use "Kline_" (singular) prefix
        adx_h_col = f"Kline_{adx_freq_label}_high" if adx_freq_label != "1min" else "high"
        adx_l_col = f"Kline_{adx_freq_label}_low" if adx_freq_label != "1min" else "low"
        adx_c_col = f"Kline_{adx_freq_label}_close" if adx_freq_label != "1min" else "close"
        logger.debug(f"{log_prefix_main} ADX: src_cols H='{adx_h_col}', L='{adx_l_col}', C='{adx_c_col}', p={adx_p}")
        if adx_p and all(c in df_processed.columns and df_processed[c].notna().any() for c in [adx_h_col, adx_l_col, adx_c_col]):
            adx_df = ta.adx(df_processed[adx_h_col].dropna(), df_processed[adx_l_col].dropna(), df_processed[adx_c_col].dropna(), length=int(adx_p), append=False)
            if adx_df is not None:
                df_processed["ADX_strat"]=adx_df.iloc[:,0]; df_processed["ADX_DMP_strat"]=adx_df.iloc[:,1]; df_processed["ADX_DMN_strat"]=adx_df.iloc[:,2]
        else: df_processed["ADX_strat"]=np.nan; df_processed["ADX_DMP_strat"]=np.nan; df_processed["ADX_DMN_strat"]=np.nan
        if "ADX_strat" in df_processed: logger.debug(f"{log_prefix_main} ADX_strat NaNs: {df_processed['ADX_strat'].isnull().sum()}/{len(df_processed)}")
            
        st_k=strategy_params.get('stoch_k_period'); st_d=strategy_params.get('stoch_d_period'); st_slow=strategy_params.get('stoch_slowing')
        st_freq_label = get_effective_kline_freq_label(strategy_params.get('indicateur_frequence_stoch'))
        # CORRECTED: Use "Kline_" (singular) prefix
        st_h_col = f"Kline_{st_freq_label}_high" if st_freq_label != "1min" else "high"
        st_l_col = f"Kline_{st_freq_label}_low" if st_freq_label != "1min" else "low"
        st_c_col = f"Kline_{st_freq_label}_close" if st_freq_label != "1min" else "close"
        logger.debug(f"{log_prefix_main} Stoch: src_cols H='{st_h_col}', L='{st_l_col}', C='{st_c_col}', k={st_k}, d={st_d}, slow={st_slow}")
        if st_k and st_d and st_slow and all(c in df_processed.columns and df_processed[c].notna().any() for c in [st_h_col, st_l_col, st_c_col]):
            st_df = ta.stoch(df_processed[st_h_col].dropna(), df_processed[st_l_col].dropna(), df_processed[st_c_col].dropna(), k=int(st_k), d=int(st_d), smooth_k=int(st_slow), append=False)
            if st_df is not None:
                df_processed["STOCH_K_strat"]=st_df.iloc[:,0]; df_processed["STOCH_D_strat"]=st_df.iloc[:,1]
        else: df_processed["STOCH_K_strat"]=np.nan; df_processed["STOCH_D_strat"]=np.nan
        if "STOCH_K_strat" in df_processed: logger.debug(f"{log_prefix_main} STOCH_K_strat NaNs: {df_processed['STOCH_K_strat'].isnull().sum()}/{len(df_processed)}")

    taker_p = strategy_params.get('taker_pressure_indicator_period')
    taker_freq_raw = strategy_params.get('indicateur_frequence_taker_pressure')
    if taker_p and taker_freq_raw:
        taker_freq_label = get_effective_kline_freq_label(taker_freq_raw)
        # CORRECTED: Use "Kline_" (singular) prefix
        buy_col_src = f"Kline_{taker_freq_label}_taker_buy_base_asset_volume" if taker_freq_label != "1min" else "taker_buy_base_asset_volume"
        sell_col_src = f"Kline_{taker_freq_label}_taker_sell_base_asset_volume" if taker_freq_label != "1min" else "taker_sell_base_asset_volume"
        logger.debug(f"{log_prefix_main} Taker Pressure: src_buy='{buy_col_src}', src_sell='{sell_col_src}', period_ma={taker_p}")
        if buy_col_src in df_processed.columns and sell_col_src in df_processed.columns and df_processed[buy_col_src].notna().any() and df_processed[sell_col_src].notna().any():
            df_processed = data_utils.calculate_taker_pressure_ratio(df_processed, buy_col_src, sell_col_src, "TakerPressureRatio_temp")
            if "TakerPressureRatio_temp" in df_processed.columns and df_processed["TakerPressureRatio_temp"].notna().any():
                df_processed["TakerPressureRatio_MA_strat"] = _calculate_ema_rolling(df_processed["TakerPressureRatio_temp"], period=int(taker_p))
                df_processed.drop(columns=["TakerPressureRatio_temp"], inplace=True, errors='ignore')
                if "TakerPressureRatio_MA_strat" in df_processed: logger.info(f"{log_prefix_main} TakerPressureRatio_MA_strat NaNs: {df_processed['TakerPressureRatio_MA_strat'].isnull().sum()}/{len(df_processed)}")
            else: 
                logger.warning(f"{log_prefix_main} TakerPressureRatio_temp not calculated or all NaN.")
                df_processed["TakerPressureRatio_MA_strat"] = np.nan
        else: 
            logger.warning(f"{log_prefix_main} Colonnes sources Taker ('{buy_col_src}', '{sell_col_src}') non trouvées ou entièrement NaN. TakerPressureRatio_MA_strat sera NaN.")
            df_processed["TakerPressureRatio_MA_strat"] = np.nan
            
    indicator_cols = [col for col in df_processed.columns if "_strat" in col.lower()] 
    if indicator_cols:
        df_processed[indicator_cols] = df_processed[indicator_cols].ffill()
        logger.info(f"{log_prefix_main} Colonnes d'indicateurs propagées (ffill): {indicator_cols}")

    df_final_to_save = df_processed.reset_index()

    expected_cols = ['timestamp'] + ohlcv_cols_for_agg 
    for _, period_lbl_cfg in rolling_windows_to_calculate.items(): 
        for ohlc_part in ohlcv_cols_for_agg: 
            expected_cols.append(f"Kline_{period_lbl_cfg}_{ohlc_part}") # This should use the correct period_lbl_cfg from rolling_windows_to_calculate
        for taker_part in taker_vol_cols_to_agg: 
            expected_cols.append(f"Kline_{period_lbl_cfg}_{taker_part}")
            
    # Add _strat columns to expected_cols if they exist, to ensure they are included and ordered
    for col in df_final_to_save.columns:
        if "_strat" in col.lower() and col not in expected_cols:
            expected_cols.append(col)
            
    current_cols = df_final_to_save.columns.tolist()
    final_ordered_cols = []
    # Add expected columns first, in their defined order
    for col in expected_cols:
        if col in current_cols and col not in final_ordered_cols:
            final_ordered_cols.append(col)
    # Add any other columns that might have been created but weren't in expected_cols (should be rare with good logic)
    for col in current_cols: 
        if col not in final_ordered_cols:
            final_ordered_cols.append(col)
            logger.warning(f"{log_prefix_main} Colonne '{col}' présente dans df_final_to_save mais pas dans expected_cols (ou déjà ajoutée). Ajoutée à la fin si manquante.")
            
    df_final_to_save = df_final_to_save.reindex(columns=final_ordered_cols)

    logger.info(f"{log_prefix_main} Colonnes dans le CSV final avant sauvegarde: {df_final_to_save.columns.tolist()}")

    try:
        df_final_to_save.to_csv(processed_output_path, index=False, float_format='%.8f')
        processing_time = time.time() - start_time
        logger.info(f"{log_prefix_main} Données prétraitées sauvegardées vers: {processed_output_path} ({len(df_final_to_save.columns)} cols, {df_final_to_save.shape[0]} lignes, {processing_time:.3f}s)")
    except Exception as e:
        logger.error(f"{log_prefix_main} Erreur lors de la sauvegarde des données live prétraitées vers {processed_output_path}: {e}", exc_info=True)
        return None
        
    if not df_final_to_save.empty:
        if 'timestamp' in df_final_to_save.columns:
            return df_final_to_save.set_index('timestamp').iloc[-1:] 
        else: 
            logger.error(f"{log_prefix_main} Colonne 'timestamp' manquante dans df_final_to_save avant de retourner la dernière ligne.")
            return None
    else:
        logger.warning(f"{log_prefix_main} DataFrame prétraité pour la stratégie live est vide, retourne None.")
        return None