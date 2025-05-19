import logging
from typing import Optional, Dict, Any, Union, List
import re 

import pandas as pd
import numpy as np
import pandas_ta as ta

logger = logging.getLogger(__name__)

def parse_frequency_to_pandas_offset(freq_str: str) -> Optional[str]:
    """
    Convertit une chaîne de fréquence personnalisée (ex: "5min", "1h") en un offset pandas.
    """
    if not freq_str:
        return None
    
    freq_str_lower = freq_str.lower().strip()
    
    match = re.fullmatch(r"(\d+)(min|m|h|d)", freq_str_lower)
    if match:
        num, unit_abbr = match.groups()
        if unit_abbr in ['min', 'm']:
            return f"{num}T"
        elif unit_abbr == 'h':
            return f"{num}H"
        elif unit_abbr == 'd':
            return f"{num}D"
            
    logger.error(f"Format de chaîne de fréquence non supporté pour l'offset pandas: '{freq_str}'")
    return None

def get_kline_prefix_effective(freq_param_value: Optional[str]) -> str:
    """
    Retourne le préfixe de colonne Klines effectif, gérant l'alias '1h' -> '60min' et None/empty.
    Exemple: "5min" -> "Klines_5min", "1h" -> "Klines_60min", "1min" -> "", None -> ""
    """
    if not freq_param_value: 
        logger.debug(f"get_kline_prefix_effective: Fréquence non valide ou manquante reçue: '{freq_param_value}'. Utilisation du préfixe vide (données 1min).")
        return "" # Pour les données 1min brutes ou si la fréquence est invalide/None
    
    freq_lower = freq_param_value.lower().strip()
    if freq_lower == "1h":
        return "Klines_60min"
    elif freq_lower == "1min":
        return "" # Pas de préfixe pour les données 1min brutes
    
    match = re.fullmatch(r"(\d+)(min|m|h|d)", freq_lower) 
    if match:
        num_str, unit_abbr = match.groups()
        
        standard_unit = unit_abbr
        if unit_abbr == 'm': 
            standard_unit = 'min'
        
        standard_freq_label = f"{num_str}{standard_unit}" 
        
        return f"Klines_{standard_freq_label}"
    else:
        logger.warning(f"get_kline_prefix_effective: Format de fréquence non reconnu '{freq_param_value}'. Utilisation du préfixe vide.")
        return ""

def aggregate_klines_rolling_for_current_timestamp(
    df_1min_slice: pd.DataFrame,
    window_size_minutes: int,
    aggregation_rules: Optional[Dict[str, Any]] = None
) -> Optional[pd.Series]:
    """
    Agrège les N dernières klines de 1 minute pour obtenir la kline agrégée actuelle.
    """
    if df_1min_slice.empty:
        logger.warning("aggregate_klines_rolling: DataFrame d'entrée vide.")
        return None
    if len(df_1min_slice) < window_size_minutes:
        logger.debug(f"aggregate_klines_rolling: Pas assez de données ({len(df_1min_slice)}) pour la fenêtre de {window_size_minutes} minutes.")
        return None

    if aggregation_rules is None:
        aggregation_rules = {
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'taker_buy_base_asset_volume': 'sum', 
            'taker_sell_base_asset_volume': 'sum',
            'taker_buy_quote_asset_volume': 'sum',
            'taker_sell_quote_asset_volume': 'sum',
            'number_of_trades': 'sum'
        }
    
    try:
        window_data = df_1min_slice.iloc[-window_size_minutes:]
        
        aggregated_values = {}
        for col, rule in aggregation_rules.items():
            if col in window_data.columns:
                if rule == 'first':
                    aggregated_values[col] = window_data[col].iloc[0]
                elif rule == 'last':
                    aggregated_values[col] = window_data[col].iloc[-1]
                elif rule == 'max':
                    aggregated_values[col] = window_data[col].max()
                elif rule == 'min':
                    aggregated_values[col] = window_data[col].min()
                elif rule == 'sum':
                    aggregated_values[col] = window_data[col].sum()
            else:
                pass 
        
        if not aggregated_values:
            logger.warning("aggregate_klines_rolling: Aucune valeur agrégée produite.")
            return None

        aggregated_series = pd.Series(aggregated_values)
        aggregated_series.name = window_data.index[-1] 
        return aggregated_series
        
    except Exception as e:
        logger.error(f"Erreur lors de l'agrégation glissante pour la fenêtre {window_size_minutes}min: {e}", exc_info=True)
        return None

def aggregate_klines_to_dataframe(
    df_1min: pd.DataFrame, 
    timeframe_minutes: int,
    extra_agg_rules: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """
    Agrège un DataFrame de klines 1-minute à un timeframe supérieur.
    """
    if not isinstance(df_1min.index, pd.DatetimeIndex):
        logger.error("df_1min doit avoir un DatetimeIndex.")
        raise ValueError("df_1min doit avoir un DatetimeIndex.")

    if df_1min.empty:
        logger.warning(f"DataFrame 1-minute vide fourni pour l'agrégation à {timeframe_minutes}min.")
        return pd.DataFrame()

    resample_period = f'{timeframe_minutes}T' 
    
    agg_rules = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }
    
    if extra_agg_rules:
        agg_rules.update(extra_agg_rules)

    final_agg_rules = {col: rule for col, rule in agg_rules.items() if col in df_1min.columns}
    
    if not final_agg_rules:
        logger.warning(f"Aucune règle d'agrégation applicable pour les colonnes présentes dans df_1min pour {timeframe_minutes}min.")
        return pd.DataFrame(index=df_1min.resample(resample_period, label='right', closed='right').first().index)

    df_aggregated = df_1min.resample(resample_period, label='right', closed='right').agg(final_agg_rules)
    
    ohlc_cols = [col for col in ['open', 'high', 'low', 'close'] if col in df_aggregated.columns]
    if ohlc_cols:
        df_aggregated = df_aggregated.dropna(subset=ohlc_cols, how='all')
    
    return df_aggregated

def calculate_atr_for_dataframe(
    df: pd.DataFrame, 
    atr_low: int = 10, 
    atr_high: int = 21, 
    atr_step: int = 1
) -> pd.DataFrame:
    """
    Calcule l'ATR pour différentes périodes sur un DataFrame.
    """
    if df.empty:
        logger.warning("DataFrame vide fourni à calculate_atr_for_dataframe.")
        return df
    
    required_cols = ['high', 'low', 'close']
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        logger.error(f"DataFrame doit contenir les colonnes 'high', 'low', 'close' pour le calcul de l'ATR. Manquant: {missing}")
        return df 

    df_with_atr = df.copy()
    
    for col in required_cols:
        df_with_atr[col] = pd.to_numeric(df_with_atr[col], errors='coerce')
    df_with_atr.replace([np.inf, -np.inf], np.nan, inplace=True) 

    for period in range(atr_low, atr_high + atr_step, atr_step):
        if period <= 0:
            logger.warning(f"Période ATR invalide ({period}). Ignorée.")
            continue
        try:
            atr_series = df_with_atr.ta.atr(length=period, append=False) 
            
            if atr_series is not None and isinstance(atr_series, pd.Series):
                df_with_atr[f'ATR_{period}'] = atr_series 
            else:
                logger.warning(f"Le calcul de l'ATR pour la période {period} n'a pas retourné de Series valide.")
                df_with_atr[f'ATR_{period}'] = np.nan
        except Exception as e:
            logger.error(f"Erreur lors du calcul de l'ATR pour la période {period}: {e}", exc_info=True)
            df_with_atr[f'ATR_{period}'] = np.nan
            
    return df_with_atr

def calculate_taker_pressure_ratio(
    df: pd.DataFrame, 
    taker_buy_volume_col: str, 
    taker_sell_volume_col: str, 
    output_col_name: str
) -> pd.DataFrame:
    """
    Calcule le ratio de pression Taker (achats Taker / ventes Taker).
    """
    df_copy = df.copy()
    if taker_buy_volume_col not in df_copy.columns or taker_sell_volume_col not in df_copy.columns:
        logger.warning(f"Les colonnes de volume Taker '{taker_buy_volume_col}' ou '{taker_sell_volume_col}' sont manquantes.")
        df_copy[output_col_name] = np.nan
        return df_copy

    buy_vol = pd.to_numeric(df_copy[taker_buy_volume_col], errors='coerce')
    sell_vol = pd.to_numeric(df_copy[taker_sell_volume_col], errors='coerce')
    
    conditions = [
        (sell_vol == 0) & (buy_vol > 0),
        (sell_vol == 0) & (buy_vol == 0),
        (sell_vol > 0)                  
    ]
    choices = [
        np.nan, 
        1.0,    
        buy_vol / sell_vol
    ]
    
    df_copy[output_col_name] = np.select(conditions, choices, default=np.nan)
    
    logger.debug(f"Colonne de ratio de pression Taker '{output_col_name}' calculée.")
    return df_copy

def calculate_taker_pressure_delta(
    df: pd.DataFrame, 
    taker_buy_volume_col: str, 
    taker_sell_volume_col: str, 
    output_col_name: str
) -> pd.DataFrame:
    """
    Calcule le delta de pression Taker (achats Taker - ventes Taker).
    """
    df_copy = df.copy()
    if taker_buy_volume_col not in df_copy.columns or taker_sell_volume_col not in df_copy.columns:
        logger.warning(f"Les colonnes de volume Taker '{taker_buy_volume_col}' ou '{taker_sell_volume_col}' sont manquantes.")
        df_copy[output_col_name] = np.nan
        return df_copy

    buy_vol = pd.to_numeric(df_copy[taker_buy_volume_col], errors='coerce')
    sell_vol = pd.to_numeric(df_copy[taker_sell_volume_col], errors='coerce')
    
    df_copy[output_col_name] = buy_vol - sell_vol
    
    logger.debug(f"Colonne de delta de pression Taker '{output_col_name}' calculée.")
    return df_copy
