import logging
import time
import importlib
from typing import Any, Dict, Optional, Tuple, List, Type, Union
from datetime import timezone 
import uuid 

import numpy as np
import pandas as pd
import pandas_ta as ta
import optuna

from src.backtesting.simulator import BacktestSimulator
from src.data import data_utils
from src.strategies.base import BaseStrategy
from src.config.definitions import ParamDetail 
# Importation de la fonction publique depuis data_utils
from src.data.data_utils import get_kline_prefix_effective, calculate_taker_pressure_ratio, calculate_taker_pressure_delta 

logger = logging.getLogger(__name__)

class ObjectiveEvaluator:
    def __init__(self,
                 strategy_name: str,
                 strategy_config_dict: Dict[str, Any], 
                 data_1min_cleaned_slice: pd.DataFrame, 
                 simulation_settings: Dict[str, Any],
                 optuna_objectives_config: Dict[str, Any],
                 pair_symbol: str,
                 symbol_info_data: Dict[str, Any],
                 is_oos_eval: bool = False,
                 app_config: Optional[Any] = None, 
                 is_trial_number_for_oos_log: Optional[int] = None): 

        self.strategy_name = strategy_name
        self.strategy_config_dict = strategy_config_dict 
        self.df_enriched_slice = data_1min_cleaned_slice.copy() 
        self.simulation_settings_global_defaults = simulation_settings
        self.optuna_objectives_config = optuna_objectives_config
        self.pair_symbol = pair_symbol
        self.symbol_info_data = symbol_info_data
        self.is_oos_eval = is_oos_eval
        self.app_config = app_config 
        self.is_trial_number_for_oos_log = is_trial_number_for_oos_log 
        
        self.strategy_script_ref = self.strategy_config_dict.get('script_reference')
        self.strategy_class_name = self.strategy_config_dict.get('class_name')

        if not self.strategy_script_ref or not self.strategy_class_name:
            raise ValueError("ObjectiveEvaluator: script_reference ou class_name manquant dans strategy_config_dict")

        self.params_space_details: Dict[str, ParamDetail] = {}
        raw_params_space = self.strategy_config_dict.get('params_space', {})

        if isinstance(raw_params_space, dict):
            for param_key, param_value_obj in raw_params_space.items():
                if isinstance(param_value_obj, ParamDetail): 
                    self.params_space_details[param_key] = param_value_obj
                elif isinstance(param_value_obj, dict): 
                    try:
                        self.params_space_details[param_key] = ParamDetail(**param_value_obj)
                    except Exception as e_pd_init:
                        logger.error(f"Erreur lors de la création de ParamDetail pour {param_key} à partir du dict dans ObjectiveEvaluator.__init__: {e_pd_init}")
                else:
                    logger.warning(f"L'élément '{param_key}' dans params_space pour la stratégie '{self.strategy_name}' n'est ni un objet ParamDetail ni un dict. Ignoré. Reçu: {type(param_value_obj)}")
        else:
            logger.error(f"params_space pour la stratégie '{self.strategy_name}' n'est pas un dictionnaire comme attendu. Type: {type(raw_params_space)}")

        if not self.params_space_details:
            logger.error(f"CRITIQUE: self.params_space_details est VIDE pour la stratégie {self.strategy_name}. Cela mènera à un élagage. params_space brut était: {raw_params_space}")
        else:
            logger.info(f"params_space_details peuplé avec succès pour {self.strategy_name} avec {len(self.params_space_details)} paramètres.")
        
        if not isinstance(self.df_enriched_slice.index, pd.DatetimeIndex):
            if 'timestamp' in self.df_enriched_slice.columns:
                self.df_enriched_slice['timestamp'] = pd.to_datetime(self.df_enriched_slice['timestamp'], utc=True)
                self.df_enriched_slice = self.df_enriched_slice.set_index('timestamp') 
            else:
                raise ValueError("ObjectiveEvaluator: Les données d'entrée doivent avoir un DatetimeIndex ou une colonne 'timestamp'.")
        
        if self.df_enriched_slice.index.tz is None:
            self.df_enriched_slice.index = self.df_enriched_slice.index.tz_localize('UTC')
        elif self.df_enriched_slice.index.tz.utcoffset(self.df_enriched_slice.index[0]) != timezone.utc.utcoffset(self.df_enriched_slice.index[0]):
             self.df_enriched_slice.index = self.df_enriched_slice.index.tz_convert('UTC')

    # La méthode _get_kline_prefix_effective a été supprimée d'ici et est maintenant dans data_utils.py

    def _calculate_indicator_on_selected_klines(self, 
                                                df_source_enriched: pd.DataFrame, 
                                                indicator_type: str, 
                                                indicator_params: Dict[str, Any],
                                                kline_ohlc_prefix: str 
                                                ) -> Optional[Union[pd.Series, pd.DataFrame]]:
        if df_source_enriched.empty:
            logger.warning(f"df_source_enriched est vide pour {indicator_type} avec préfixe {kline_ohlc_prefix}.")
            return None
        
        open_col = f"{kline_ohlc_prefix}_open" if kline_ohlc_prefix else "open"
        high_col = f"{kline_ohlc_prefix}_high" if kline_ohlc_prefix else "high"
        low_col = f"{kline_ohlc_prefix}_low" if kline_ohlc_prefix else "low"
        close_col = f"{kline_ohlc_prefix}_close" if kline_ohlc_prefix else "close"
        volume_col = f"{kline_ohlc_prefix}_volume" if kline_ohlc_prefix else "volume"

        required_ta_cols_map = {'close': close_col} 
        indicator_type_lower = indicator_type.lower()

        if indicator_type_lower in ['psar', 'adx', 'atr', 'cci', 'donchian', 'ichimoku', 'supertrend', 'kama', 'bbands', 'kc', 'macd', 'rsi', 'stoch', 'roc', 'mom', 'ao', 'apo', 'aroon', 'chop', 'coppock', 'dm', 'fisher', 'kst', 'massi', 'natr', 'ppo', 'qstick', 'stc', 'trix', 'tsi', 'uo', 'vhf', 'vortex', 'willr']:
            required_ta_cols_map['high'] = high_col
            required_ta_cols_map['low'] = low_col
        if indicator_type_lower in ['psar', 'adx', 'cci', 'donchian', 'ichimoku', 'kama', 'bbands', 'kc', 'ao', 'apo', 'aroon', 'chop', 'coppock', 'dm', 'fisher', 'kst', 'massi', 'natr', 'ppo', 'qstick', 'stc', 'trix', 'tsi', 'uo', 'vhf', 'vortex', 'willr']:
            if open_col in df_source_enriched.columns : 
                 required_ta_cols_map['open'] = open_col
        if indicator_type_lower in ['obv', 'vwap', 'ad', 'adosc', 'cmf', 'efi', 'mfi', 'nvi', 'pvi', 'pvol', 'pvr', 'pvt']:
            required_ta_cols_map['volume'] = volume_col

        ta_inputs: Dict[str, pd.Series] = {}
        all_required_cols_present = True
        for ta_key, source_col_name in required_ta_cols_map.items():
            if source_col_name not in df_source_enriched.columns:
                logger.warning(f"Colonne source requise '{source_col_name}' pour l'indicateur '{indicator_type}' (clé TA: '{ta_key}') non trouvée dans df_source_enriched. Préfixe: {kline_ohlc_prefix}. Colonnes disponibles: {df_source_enriched.columns.tolist()}")
                all_required_cols_present = False
                break 
            
            series_for_ta = df_source_enriched[source_col_name] 
            if series_for_ta.isnull().all():
                 logger.warning(f"Colonne '{source_col_name}' pour l'indicateur '{indicator_type}' (clé TA: '{ta_key}') est entièrement NaN. Préfixe: {kline_ohlc_prefix}.")
                 if ta_key == 'close': 
                    all_required_cols_present = False; break
            ta_inputs[ta_key] = series_for_ta
        
        if not all_required_cols_present:
            return pd.Series(np.nan, index=df_source_enriched.index) if indicator_type_lower == 'close' or 'close' not in ta_inputs else None

        if not ta_inputs.get('close', pd.Series(dtype=float)).notna().any():
            logger.warning(f"La série 'close' ({close_col}) pour {indicator_type} n'a aucune valeur non-NaN. Impossible de calculer l'indicateur.")
            return pd.Series(np.nan, index=df_source_enriched.index)

        try:
            indicator_function = getattr(ta, indicator_type_lower, None)
            if indicator_function is None: 
                for category in [ta.trend, ta.momentum, ta.overlap, ta.volume, ta.volatility, ta.cycles, ta.statistics, ta.transform, ta.utils]:
                    if hasattr(category, indicator_type_lower):
                        indicator_function = getattr(category, indicator_type_lower)
                        break
            if indicator_function is None:
                logger.error(f"Fonction d'indicateur '{indicator_type}' non trouvée dans pandas_ta.")
                return None
    
            result = indicator_function(**ta_inputs, **indicator_params, append=False)
            return result 
        except Exception as e:
            logger.error(f"Erreur lors du calcul de {indicator_type} avec params {indicator_params} utilisant le préfixe {kline_ohlc_prefix}: {e}", exc_info=True)
            return None

    def _prepare_data_with_dynamic_indicators(self, trial_params: Dict[str, Any], trial_number_for_log: Optional[Union[int, str]] = None) -> pd.DataFrame:
        df_for_simulation = self.df_enriched_slice[['open', 'high', 'low', 'close', 'volume']].copy()
        
        current_trial_num_str = str(trial_number_for_log) if trial_number_for_log is not None else "N/A_IS_Prep" # Changed placeholder
        if self.is_oos_eval and self.is_trial_number_for_oos_log is not None:
            current_trial_num_str = f"OOS_for_IS_{self.is_trial_number_for_oos_log}"
        
        log_prefix_prepare = f"[{self.strategy_name}/{self.pair_symbol}/Trial-{current_trial_num_str}]"

        logger.debug(f"{log_prefix_prepare} Début de _prepare_data_with_dynamic_indicators.")
        logger.debug(f"{log_prefix_prepare} Colonnes disponibles dans self.df_enriched_slice: {self.df_enriched_slice.columns.tolist()[:20]}... (total {len(self.df_enriched_slice.columns)})")
        if not self.df_enriched_slice.empty:
            logger.debug(f"{log_prefix_prepare} Plage de dates de self.df_enriched_slice: {self.df_enriched_slice.index.min()} à {self.df_enriched_slice.index.max()}")
        logger.debug(f"{log_prefix_prepare} Nombre total de lignes dans self.df_enriched_slice: {len(self.df_enriched_slice)}")

        atr_period_sl_tp_key = 'atr_period_sl_tp' if 'atr_period_sl_tp' in trial_params else 'atr_period'
        atr_base_freq_key = 'atr_base_frequency_sl_tp' if 'atr_base_frequency_sl_tp' in trial_params else 'atr_base_frequency'

        atr_period_sl_tp = trial_params.get(atr_period_sl_tp_key)
        atr_freq_sl_tp_raw = trial_params.get(atr_base_freq_key)
        
        # Utiliser data_utils.get_kline_prefix_effective pour obtenir le nom de la partie fréquence (ex: 60min, 5min)
        # La fonction retourne le préfixe complet (ex: Klines_60min), on extrait la partie fréquence.
        temp_prefix_for_atr_freq = data_utils.get_kline_prefix_effective(atr_freq_sl_tp_raw)
        atr_freq_sl_tp_effective = temp_prefix_for_atr_freq.replace("Klines_", "") if temp_prefix_for_atr_freq else "1min"

        logger.debug(f"{log_prefix_prepare} Paramètres ATR pour SL/TP: Période={atr_period_sl_tp}, Fréquence Orig='{atr_freq_sl_tp_raw}', Fréquence Effective pour nom de colonne='{atr_freq_sl_tp_effective}'")

        if atr_period_sl_tp is not None and atr_freq_sl_tp_effective is not None:
            atr_period_sl_tp = int(atr_period_sl_tp) 
            atr_col_name_enriched = f"Klines_{atr_freq_sl_tp_effective}_ATR_{atr_period_sl_tp}"
            
            logger.debug(f"{log_prefix_prepare} Recherche de la colonne ATR pré-calculée: '{atr_col_name_enriched}'")

            if atr_col_name_enriched in self.df_enriched_slice.columns:
                df_for_simulation['ATR_strat'] = self.df_enriched_slice[atr_col_name_enriched].reindex(df_for_simulation.index, method='ffill')
                logger.info(f"{log_prefix_prepare} ATR_strat chargé depuis '{atr_col_name_enriched}'. NaNs: {df_for_simulation['ATR_strat'].isnull().sum()}/{len(df_for_simulation)}")
            elif atr_freq_sl_tp_effective == "1min":
                 logger.info(f"{log_prefix_prepare} Calcul dynamique de ATR_strat pour 1min (Période: {atr_period_sl_tp}).")
                 atr_1min_series = self._calculate_indicator_on_selected_klines(
                     self.df_enriched_slice, 'atr', {'length': atr_period_sl_tp}, "" 
                 )
                 df_for_simulation['ATR_strat'] = atr_1min_series.reindex(df_for_simulation.index, method='ffill') if isinstance(atr_1min_series, pd.Series) else np.nan
                 if isinstance(df_for_simulation.get('ATR_strat'), pd.Series):
                    logger.info(f"{log_prefix_prepare} ATR_strat (1min dyn.) NaNs: {df_for_simulation['ATR_strat'].isnull().sum()}/{len(df_for_simulation)}")
                 else:
                    logger.warning(f"{log_prefix_prepare} ATR_strat (1min dyn.) est entièrement NaN après calcul.")
            else:
                logger.warning(f"{log_prefix_prepare} Colonne ATR pré-calculée '{atr_col_name_enriched}' non trouvée. ATR_strat sera NaN. Colonnes Klines disponibles (échantillon): {[col for col in self.df_enriched_slice.columns if 'Klines_' in col and f'_{atr_freq_sl_tp_effective}_' in col][:10]}")
                df_for_simulation['ATR_strat'] = np.nan
        else:
            logger.warning(f"{log_prefix_prepare} Paramètres ATR (période ou fréquence) manquants. ATR_strat sera NaN.")
            df_for_simulation['ATR_strat'] = np.nan

        if self.strategy_name == "ma_crossover_strategy":
            ma_type = trial_params['ma_type']
            freq_fast_raw = trial_params['indicateur_frequence_ma_rapide'] 
            kline_prefix_fast = data_utils.get_kline_prefix_effective(freq_fast_raw) # UTILISATION CORRIGÉE
            period_fast = int(trial_params['fast_ma_period'])
            
            ma_fast_result = self._calculate_indicator_on_selected_klines(self.df_enriched_slice, ma_type, {'length': period_fast}, kline_prefix_fast)
            df_for_simulation['MA_FAST_strat'] = ma_fast_result.reindex(df_for_simulation.index, method='ffill') if isinstance(ma_fast_result, pd.Series) else np.nan
            if isinstance(df_for_simulation.get('MA_FAST_strat'), pd.Series): logger.debug(f"{log_prefix_prepare} MA_FAST_strat (src: {kline_prefix_fast}_close) NaNs: {df_for_simulation['MA_FAST_strat'].isnull().sum()}/{len(df_for_simulation)}")

            freq_slow_raw = trial_params['indicateur_frequence_ma_lente'] 
            kline_prefix_slow = data_utils.get_kline_prefix_effective(freq_slow_raw) # UTILISATION CORRIGÉE
            period_slow = int(trial_params['slow_ma_period'])
            
            ma_slow_result = self._calculate_indicator_on_selected_klines(self.df_enriched_slice, ma_type, {'length': period_slow}, kline_prefix_slow)
            df_for_simulation['MA_SLOW_strat'] = ma_slow_result.reindex(df_for_simulation.index, method='ffill') if isinstance(ma_slow_result, pd.Series) else np.nan
            if isinstance(df_for_simulation.get('MA_SLOW_strat'), pd.Series): logger.debug(f"{log_prefix_prepare} MA_SLOW_strat (src: {kline_prefix_slow}_close) NaNs: {df_for_simulation['MA_SLOW_strat'].isnull().sum()}/{len(df_for_simulation)}")
    
        elif self.strategy_name == "kama_crossover_otoco":
            freq_kama_raw = trial_params['indicateur_frequence_kama']
            kline_prefix_kama = data_utils.get_kline_prefix_effective(freq_kama_raw) # UTILISATION CORRIGÉE
            kama_params = {
                'length': int(trial_params['kama_period']),
                'fast': int(trial_params['kama_fast_ema']),
                'slow': int(trial_params['kama_slow_ema'])
            }
            kama_result = self._calculate_indicator_on_selected_klines(self.df_enriched_slice, 'kama', kama_params, kline_prefix_kama)
            df_for_simulation['KAMA_strat'] = kama_result.reindex(df_for_simulation.index, method='ffill') if isinstance(kama_result, pd.Series) else np.nan
            if isinstance(df_for_simulation.get('KAMA_strat'), pd.Series): logger.debug(f"{log_prefix_prepare} KAMA_strat (src: {kline_prefix_kama}_close) NaNs: {df_for_simulation['KAMA_strat'].isnull().sum()}/{len(df_for_simulation)}")

        elif self.strategy_name == "psar_reversal_otoco":
            freq_psar_raw = trial_params['indicateur_frequence_psar']
            kline_prefix_psar = data_utils.get_kline_prefix_effective(freq_psar_raw) # UTILISATION CORRIGÉE
            psar_params = {
                'initial_af': float(trial_params['psar_step']), 
                'af': float(trial_params['psar_step']), 
                'max_af': float(trial_params['psar_max_step'])
            }
            psar_df_result = self._calculate_indicator_on_selected_klines(self.df_enriched_slice, 'psar', psar_params, kline_prefix_psar)
            if psar_df_result is not None and isinstance(psar_df_result, pd.DataFrame) and not psar_df_result.empty:
                psarl_col_name = next((col for col in psar_df_result.columns if col.lower().startswith("psarl")), None)
                psars_col_name = next((col for col in psar_df_result.columns if col.lower().startswith("psars")), None)
                
                df_for_simulation['PSARl_strat'] = psar_df_result[psarl_col_name].reindex(df_for_simulation.index, method='ffill') if psarl_col_name else np.nan
                df_for_simulation['PSARs_strat'] = psar_df_result[psars_col_name].reindex(df_for_simulation.index, method='ffill') if psars_col_name else np.nan
            else:
                df_for_simulation['PSARl_strat'] = np.nan; df_for_simulation['PSARs_strat'] = np.nan
            if isinstance(df_for_simulation.get('PSARl_strat'), pd.Series): logger.debug(f"{log_prefix_prepare} PSARl_strat NaNs: {df_for_simulation['PSARl_strat'].isnull().sum()}/{len(df_for_simulation)}")

        elif self.strategy_name == "adx_direction_otoco":
            freq_adx_raw = trial_params['indicateur_frequence_adx']
            kline_prefix_adx = data_utils.get_kline_prefix_effective(freq_adx_raw) # UTILISATION CORRIGÉE
            adx_params = {'length': int(trial_params['adx_period'])} 
            adx_df_result = self._calculate_indicator_on_selected_klines(self.df_enriched_slice, 'adx', adx_params, kline_prefix_adx)
            if adx_df_result is not None and isinstance(adx_df_result, pd.DataFrame) and not adx_df_result.empty:
                df_for_simulation['ADX_strat'] = adx_df_result.iloc[:,0].reindex(df_for_simulation.index, method='ffill')
                df_for_simulation['DMP_strat'] = adx_df_result.iloc[:,1].reindex(df_for_simulation.index, method='ffill') 
                df_for_simulation['DMN_strat'] = adx_df_result.iloc[:,2].reindex(df_for_simulation.index, method='ffill')
            else:
                df_for_simulation['ADX_strat'] = np.nan; df_for_simulation['DMP_strat'] = np.nan; df_for_simulation['DMN_strat'] = np.nan
            if isinstance(df_for_simulation.get('ADX_strat'), pd.Series): logger.debug(f"{log_prefix_prepare} ADX_strat NaNs: {df_for_simulation['ADX_strat'].isnull().sum()}/{len(df_for_simulation)}")

        elif self.strategy_name == "EmaMacdAtrStrategy":
            ema_short_p = int(trial_params['ema_short_period'])
            ema_long_p = int(trial_params['ema_long_period'])
            ema_freq_raw = trial_params['indicateur_frequence_ema']
            kline_prefix_ema = data_utils.get_kline_prefix_effective(ema_freq_raw) # UTILISATION CORRIGÉE
            
            ema_short_series = self._calculate_indicator_on_selected_klines(self.df_enriched_slice, 'ema', {'length': ema_short_p}, kline_prefix_ema)
            df_for_simulation['EMA_short_strat'] = ema_short_series.reindex(df_for_simulation.index, method='ffill') if isinstance(ema_short_series, pd.Series) else np.nan
            if isinstance(df_for_simulation.get('EMA_short_strat'), pd.Series): logger.debug(f"{log_prefix_prepare} EMA_short_strat NaNs: {df_for_simulation['EMA_short_strat'].isnull().sum()}/{len(df_for_simulation)}")

            ema_long_series = self._calculate_indicator_on_selected_klines(self.df_enriched_slice, 'ema', {'length': ema_long_p}, kline_prefix_ema)
            df_for_simulation['EMA_long_strat'] = ema_long_series.reindex(df_for_simulation.index, method='ffill') if isinstance(ema_long_series, pd.Series) else np.nan
            if isinstance(df_for_simulation.get('EMA_long_strat'), pd.Series): logger.debug(f"{log_prefix_prepare} EMA_long_strat NaNs: {df_for_simulation['EMA_long_strat'].isnull().sum()}/{len(df_for_simulation)}")

            macd_fast = int(trial_params['macd_fast_period'])
            macd_slow = int(trial_params['macd_slow_period'])
            macd_sig = int(trial_params['macd_signal_period'])
            macd_freq_raw = trial_params['indicateur_frequence_macd']
            kline_prefix_macd = data_utils.get_kline_prefix_effective(macd_freq_raw) # UTILISATION CORRIGÉE
            
            macd_df = self._calculate_indicator_on_selected_klines(self.df_enriched_slice, 'macd', {'fast': macd_fast, 'slow': macd_slow, 'signal': macd_sig}, kline_prefix_macd)
            if macd_df is not None and isinstance(macd_df, pd.DataFrame) and not macd_df.empty:
                df_for_simulation['MACD_line_strat'] = macd_df.iloc[:,0].reindex(df_for_simulation.index, method='ffill')
                df_for_simulation['MACD_hist_strat'] = macd_df.iloc[:,1].reindex(df_for_simulation.index, method='ffill')
                df_for_simulation['MACD_signal_strat'] = macd_df.iloc[:,2].reindex(df_for_simulation.index, method='ffill')
            else:
                df_for_simulation['MACD_line_strat'] = np.nan; df_for_simulation['MACD_hist_strat'] = np.nan; df_for_simulation['MACD_signal_strat'] = np.nan
            if isinstance(df_for_simulation.get('MACD_line_strat'), pd.Series): logger.debug(f"{log_prefix_prepare} MACD_line_strat NaNs: {df_for_simulation['MACD_line_strat'].isnull().sum()}/{len(df_for_simulation)}")

            atr_vol_p = trial_params.get('atr_volatility_filter_period')
            atr_vol_freq_raw = trial_params.get('indicateur_frequence_atr_volatility')
            if atr_vol_p is not None and atr_vol_freq_raw is not None:
                kline_prefix_atr_vol = data_utils.get_kline_prefix_effective(atr_vol_freq_raw) # UTILISATION CORRIGÉE
                atr_vol_series = self._calculate_indicator_on_selected_klines(self.df_enriched_slice, 'atr', {'length': int(atr_vol_p)}, kline_prefix_atr_vol)
                df_for_simulation['ATR_volatility_filter_strat'] = atr_vol_series.reindex(df_for_simulation.index, method='ffill') if isinstance(atr_vol_series, pd.Series) else np.nan
                if isinstance(df_for_simulation.get('ATR_volatility_filter_strat'), pd.Series): logger.debug(f"{log_prefix_prepare} ATR_volatility_filter_strat NaNs: {df_for_simulation['ATR_volatility_filter_strat'].isnull().sum()}/{len(df_for_simulation)}")


        elif self.strategy_name == "BbandsVolumeRsiStrategy":
            bb_p = int(trial_params['bbands_period'])
            bb_std = float(trial_params['bbands_std_dev'])
            bb_freq_raw = trial_params['indicateur_frequence_bbands']
            kline_prefix_bb = data_utils.get_kline_prefix_effective(bb_freq_raw) # UTILISATION CORRIGÉE

            bb_df = self._calculate_indicator_on_selected_klines(self.df_enriched_slice, 'bbands', {'length': bb_p, 'std': bb_std}, kline_prefix_bb)
            if bb_df is not None and isinstance(bb_df, pd.DataFrame) and not bb_df.empty:
                df_for_simulation['BB_LOWER_strat'] = bb_df.iloc[:,0].reindex(df_for_simulation.index, method='ffill')
                df_for_simulation['BB_MIDDLE_strat'] = bb_df.iloc[:,1].reindex(df_for_simulation.index, method='ffill')
                df_for_simulation['BB_UPPER_strat'] = bb_df.iloc[:,2].reindex(df_for_simulation.index, method='ffill')
                df_for_simulation['BB_BANDWIDTH_strat'] = bb_df.iloc[:,3].reindex(df_for_simulation.index, method='ffill')
            else:
                for col in ["BB_LOWER_strat", "BB_MIDDLE_strat", "BB_UPPER_strat", "BB_BANDWIDTH_strat"]: df_for_simulation[col] = np.nan
            if isinstance(df_for_simulation.get('BB_UPPER_strat'), pd.Series): logger.debug(f"{log_prefix_prepare} BB_UPPER_strat NaNs: {df_for_simulation['BB_UPPER_strat'].isnull().sum()}/{len(df_for_simulation)}")

            vol_ma_p = int(trial_params['volume_ma_period'])
            vol_freq_raw = trial_params['indicateur_frequence_volume']
            kline_prefix_vol = data_utils.get_kline_prefix_effective(vol_freq_raw) # UTILISATION CORRIGÉE
            source_vol_col = f"{kline_prefix_vol}_volume" if kline_prefix_vol else "volume"

            if source_vol_col in self.df_enriched_slice.columns:
                 volume_series_source = self.df_enriched_slice[source_vol_col]
                 if volume_series_source.notna().any():
                     df_for_simulation['Volume_MA_strat'] = ta.sma(volume_series_source, length=vol_ma_p, append=False).reindex(df_for_simulation.index, method='ffill')
                 else:
                     df_for_simulation['Volume_MA_strat'] = np.nan
            else:
                df_for_simulation['Volume_MA_strat'] = np.nan
            if isinstance(df_for_simulation.get('Volume_MA_strat'), pd.Series): logger.debug(f"{log_prefix_prepare} Volume_MA_strat (src: {source_vol_col}) NaNs: {df_for_simulation['Volume_MA_strat'].isnull().sum()}/{len(df_for_simulation)}")
            
            if source_vol_col not in df_for_simulation.columns : 
                 df_for_simulation[source_vol_col] = self.df_enriched_slice[source_vol_col].reindex(df_for_simulation.index, method='ffill') if source_vol_col in self.df_enriched_slice else np.nan

            rsi_p = int(trial_params['rsi_period'])
            rsi_freq_raw = trial_params['indicateur_frequence_rsi']
            kline_prefix_rsi = data_utils.get_kline_prefix_effective(rsi_freq_raw) # UTILISATION CORRIGÉE
            rsi_series = self._calculate_indicator_on_selected_klines(self.df_enriched_slice, 'rsi', {'length': rsi_p}, kline_prefix_rsi)
            df_for_simulation['RSI_strat'] = rsi_series.reindex(df_for_simulation.index, method='ffill') if isinstance(rsi_series, pd.Series) else np.nan
            if isinstance(df_for_simulation.get('RSI_strat'), pd.Series): logger.debug(f"{log_prefix_prepare} RSI_strat NaNs: {df_for_simulation['RSI_strat'].isnull().sum()}/{len(df_for_simulation)}")

        elif self.strategy_name == "KamaAdxStochStrategy":
            kama_p = int(trial_params['kama_period'])
            kama_fast = int(trial_params['kama_fast_ema'])
            kama_slow = int(trial_params['kama_slow_ema'])
            kama_freq_raw = trial_params['indicateur_frequence_kama']
            kline_prefix_kama = data_utils.get_kline_prefix_effective(kama_freq_raw) # UTILISATION CORRIGÉE
            kama_series = self._calculate_indicator_on_selected_klines(self.df_enriched_slice, 'kama', {'length': kama_p, 'fast': kama_fast, 'slow': kama_slow}, kline_prefix_kama)
            df_for_simulation['KAMA_strat'] = kama_series.reindex(df_for_simulation.index, method='ffill') if isinstance(kama_series, pd.Series) else np.nan
            if isinstance(df_for_simulation.get('KAMA_strat'), pd.Series): logger.debug(f"{log_prefix_prepare} KAMA_strat NaNs: {df_for_simulation['KAMA_strat'].isnull().sum()}/{len(df_for_simulation)}")

            adx_p = int(trial_params['adx_period'])
            adx_freq_raw = trial_params['indicateur_frequence_adx']
            kline_prefix_adx = data_utils.get_kline_prefix_effective(adx_freq_raw) # UTILISATION CORRIGÉE
            adx_df = self._calculate_indicator_on_selected_klines(self.df_enriched_slice, 'adx', {'length': adx_p}, kline_prefix_adx)
            if adx_df is not None and isinstance(adx_df, pd.DataFrame) and not adx_df.empty:
                df_for_simulation['ADX_strat'] = adx_df.iloc[:,0].reindex(df_for_simulation.index, method='ffill')
                df_for_simulation['ADX_DMP_strat'] = adx_df.iloc[:,1].reindex(df_for_simulation.index, method='ffill') 
                df_for_simulation['ADX_DMN_strat'] = adx_df.iloc[:,2].reindex(df_for_simulation.index, method='ffill') 
            else:
                df_for_simulation['ADX_strat'] = np.nan; df_for_simulation['ADX_DMP_strat'] = np.nan; df_for_simulation['ADX_DMN_strat'] = np.nan
            if isinstance(df_for_simulation.get('ADX_strat'), pd.Series): logger.debug(f"{log_prefix_prepare} ADX_strat NaNs: {df_for_simulation['ADX_strat'].isnull().sum()}/{len(df_for_simulation)}")
            
            stoch_k = int(trial_params['stoch_k_period'])
            stoch_d = int(trial_params['stoch_d_period'])
            stoch_slowing = int(trial_params['stoch_slowing'])
            stoch_freq_raw = trial_params['indicateur_frequence_stoch']
            kline_prefix_stoch = data_utils.get_kline_prefix_effective(stoch_freq_raw) # UTILISATION CORRIGÉE
            stoch_df = self._calculate_indicator_on_selected_klines(self.df_enriched_slice, 'stoch', {'k': stoch_k, 'd': stoch_d, 'smooth_k': stoch_slowing}, kline_prefix_stoch)
            if stoch_df is not None and isinstance(stoch_df, pd.DataFrame) and not stoch_df.empty:
                df_for_simulation['STOCH_K_strat'] = stoch_df.iloc[:,0].reindex(df_for_simulation.index, method='ffill')
                df_for_simulation['STOCH_D_strat'] = stoch_df.iloc[:,1].reindex(df_for_simulation.index, method='ffill')
            else:
                df_for_simulation['STOCH_K_strat'] = np.nan; df_for_simulation['STOCH_D_strat'] = np.nan
            if isinstance(df_for_simulation.get('STOCH_K_strat'), pd.Series): logger.debug(f"{log_prefix_prepare} STOCH_K_strat NaNs: {df_for_simulation['STOCH_K_strat'].isnull().sum()}/{len(df_for_simulation)}")
        
        taker_pressure_period = trial_params.get('taker_pressure_indicator_period')
        taker_pressure_freq_raw = trial_params.get('indicateur_frequence_taker_pressure')

        if taker_pressure_period is not None and taker_pressure_freq_raw is not None:
            taker_pressure_period = int(taker_pressure_period)
            kline_prefix_taker = data_utils.get_kline_prefix_effective(taker_pressure_freq_raw) # UTILISATION CORRIGÉE
            
            taker_buy_vol_col_source = f"{kline_prefix_taker}_taker_buy_base_asset_volume" if kline_prefix_taker else "taker_buy_base_asset_volume"
            taker_sell_vol_col_source = f"{kline_prefix_taker}_taker_sell_base_asset_volume" if kline_prefix_taker else "taker_sell_base_asset_volume"
            
            logger.debug(f"{log_prefix_prepare} Recherche des colonnes Taker: Buy='{taker_buy_vol_col_source}', Sell='{taker_sell_vol_col_source}'")

            if taker_buy_vol_col_source in self.df_enriched_slice.columns and \
               taker_sell_vol_col_source in self.df_enriched_slice.columns:
                
                df_temp_taker_calc = self.df_enriched_slice[[taker_buy_vol_col_source, taker_sell_vol_col_source]].copy()
                
                df_temp_taker_calc = calculate_taker_pressure_ratio(
                    df_temp_taker_calc, 
                    taker_buy_volume_col=taker_buy_vol_col_source, 
                    taker_sell_volume_col=taker_sell_vol_col_source,
                    output_col_name="TakerPressureRatio_raw"
                )
                
                if "TakerPressureRatio_raw" in df_temp_taker_calc.columns and df_temp_taker_calc["TakerPressureRatio_raw"].notna().any():
                    taker_ratio_ma_series = ta.ema(df_temp_taker_calc["TakerPressureRatio_raw"], length=taker_pressure_period, append=False)
                    df_for_simulation['TakerPressureRatio_MA_strat'] = taker_ratio_ma_series.reindex(df_for_simulation.index, method='ffill')
                    if isinstance(df_for_simulation.get('TakerPressureRatio_MA_strat'), pd.Series) :
                        logger.info(f"{log_prefix_prepare} TakerPressureRatio_MA_strat calculé (Période MA: {taker_pressure_period} sur Klines {taker_pressure_freq_raw}). NaNs: {df_for_simulation['TakerPressureRatio_MA_strat'].isnull().sum()}/{len(df_for_simulation)}")
                else:
                    df_for_simulation['TakerPressureRatio_MA_strat'] = np.nan
                    logger.warning(f"{log_prefix_prepare} TakerPressureRatio_raw est vide ou NaN. TakerPressureRatio_MA_strat sera NaN.")
            else:
                logger.warning(f"{log_prefix_prepare} Colonnes sources pour l'indicateur Taker Pressure non trouvées ('{taker_buy_vol_col_source}', '{taker_sell_vol_col_source}'). L'indicateur sera NaN.")
                df_for_simulation['TakerPressureRatio_MA_strat'] = np.nan
        else:
            if 'TakerPressureRatio_MA_strat' not in df_for_simulation.columns:
                 df_for_simulation['TakerPressureRatio_MA_strat'] = np.nan
            if 'TakerPressureDelta_MA_strat' not in df_for_simulation.columns: 
                 df_for_simulation['TakerPressureDelta_MA_strat'] = np.nan


        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col not in df_for_simulation.columns: 
                 logger.error(f"{log_prefix_prepare} Colonne de base {col} manquante dans df_for_simulation. C'est inattendu.")
                 df_for_simulation[col] = np.nan 
        
        if 'timestamp' not in df_for_simulation.columns:
             df_for_simulation['timestamp'] = df_for_simulation.index
        
        logger.debug(f"{log_prefix_prepare} Colonnes dans df_for_simulation avant retour: {df_for_simulation.columns.tolist()}")
        return df_for_simulation
    
    
    def _suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        params_for_trial = {}
        if not self.params_space_details: 
            logger.error(f"params_space_details non initialisé pour la stratégie '{self.strategy_name}'. Impossible de suggérer des paramètres.")
            return {}

        for param_name, param_detail_obj in self.params_space_details.items():
            param_type = param_detail_obj.type
            low = param_detail_obj.low
            high = param_detail_obj.high
            step = param_detail_obj.step
            choices = param_detail_obj.choices

            try:
                if param_type == 'int':
                    if low is None or high is None: raise ValueError(f"low/high manquant pour le paramètre int '{param_name}'")
                    params_for_trial[param_name] = trial.suggest_int(param_name, int(low), int(high), step=int(step if step is not None else 1))
                elif param_type == 'float':
                    if low is None or high is None: raise ValueError(f"low/high manquant pour le paramètre float '{param_name}'")
                    params_for_trial[param_name] = trial.suggest_float(param_name, float(low), float(high), step=step) 
                elif param_type == 'categorical':
                    if choices is None or not isinstance(choices, list) or not choices: raise ValueError(f"choices manquant/invalide pour le paramètre catégoriel '{param_name}'")
                    params_for_trial[param_name] = trial.suggest_categorical(param_name, choices)
            except Exception as e_suggest: 
                    logger.error(f"Erreur lors de la suggestion du paramètre '{param_name}' de type '{param_type}': {e_suggest}", exc_info=True)
                    if param_detail_obj.choices and param_detail_obj.choices: 
                        params_for_trial[param_name] = param_detail_obj.choices[0]
                    elif param_detail_obj.low is not None: 
                        params_for_trial[param_name] = param_detail_obj.low
                    else: 
                        logger.warning(f"Impossible de définir une valeur de fallback pour le paramètre '{param_name}'.")
        return params_for_trial

    def __call__(self, trial: optuna.Trial) -> Tuple[float, ...]:
        start_time_trial = time.time()
        
        trial_id_for_log: str 
        trial_number_for_prepare_log: Optional[int] = None

        if self.is_oos_eval:
            is_trial_num_oos = self.is_trial_number_for_oos_log 
            if is_trial_num_oos is not None:
                trial_id_for_log = f"oos_pour_is_trial_{is_trial_num_oos}"
                trial_number_for_prepare_log = is_trial_num_oos 
            else: 
                trial_id_for_log = f"oos_trial_inconnu_{trial.number if hasattr(trial, 'number') else uuid.uuid4().hex[:6]}"
                logger.warning(f"is_trial_number_for_oos_log non défini pour l'évaluation OOS du trial {trial_id_for_log}.")
                if hasattr(trial, 'number'): trial_number_for_prepare_log = trial.number
        else: 
            trial_id_for_log = str(trial.number) if hasattr(trial, 'number') and trial.number is not None else f"is_unknown_trial_{uuid.uuid4().hex[:6]}"
            if hasattr(trial, 'number'): trial_number_for_prepare_log = trial.number


        log_prefix = f"[Stratégie: {self.strategy_name}][Paire: {self.pair_symbol}][Trial {trial_id_for_log}]"
        
        StrategyClass_local: Optional[Type[BaseStrategy]] = None
        try:
            module_path = self.strategy_script_ref.replace('.py', '').replace('/', '.')
            module = importlib.import_module(module_path)
            StrategyClass_local = getattr(module, self.strategy_class_name)
            if not StrategyClass_local or not issubclass(StrategyClass_local, BaseStrategy):
                raise ImportError(f"Impossible de charger une sous-classe BaseStrategy valide: {self.strategy_class_name}")
        except Exception as e_load_strat:
            logger.error(f"{log_prefix} Échec du chargement de la classe de stratégie {self.strategy_class_name} dans le worker: {e_load_strat}", exc_info=True)
            raise optuna.exceptions.TrialPruned(f"Échec du chargement de la classe de stratégie dans le worker: {e_load_strat}")

        current_trial_params: Dict[str, Any]
        if self.is_oos_eval:
            current_trial_params = trial.params.copy() 
            current_trial_params.pop("_trial_id_for_oos", None) 
        else:
            current_trial_params = self._suggest_params(trial)
        
        if not current_trial_params: 
            logger.error(f"{log_prefix} Aucun paramètre suggéré/fourni. Élagage du trial.")
            raise optuna.exceptions.TrialPruned("Aucun paramètre à optimiser pour ce trial.")
        
        logger.info(f"{log_prefix} Évaluation avec les paramètres: {current_trial_params}")
        
        try:
            data_for_simulation = self._prepare_data_with_dynamic_indicators(current_trial_params, trial_number_for_log=trial_number_for_prepare_log)
            
            if data_for_simulation.empty or data_for_simulation[['open', 'high', 'low', 'close']].isnull().all().all():
                logger.error(f"{log_prefix} Données pour simulation vides ou OHLC entièrement NaN après préparation. Élagage.")
                raise optuna.exceptions.TrialPruned("Données pour simulation vides ou OHLC entièrement NaN après préparation.")
        except Exception as e_prepare:
            logger.error(f"{log_prefix} Erreur lors de la préparation des données avec indicateurs dynamiques: {e_prepare}", exc_info=True)
            return tuple([-float('inf') if direction == "maximize" else float('inf')
                            for direction in self.optuna_objectives_config['objectives_directions']])

        strategy_instance = StrategyClass_local(params=current_trial_params)
        
        sim_settings_for_trial = self.simulation_settings_global_defaults.copy()
        sim_settings_for_trial['symbol'] = self.pair_symbol
        sim_settings_for_trial['symbol_info'] = self.symbol_info_data
        
        simulator = BacktestSimulator(
            historical_data_with_indicators=data_for_simulation, 
            strategy_instance=strategy_instance,
            simulation_settings=sim_settings_for_trial,
            output_dir=None 
        )
        
        backtest_results = simulator.run_simulation()
        metrics = backtest_results.get("metrics", {})
        
        objective_values = []
        for i, metric_name in enumerate(self.optuna_objectives_config['objectives_names']):
            value = metrics.get(metric_name)
            direction = self.optuna_objectives_config['objectives_directions'][i]
            
            if value is None or not isinstance(value, (int, float)) or not np.isfinite(value):
                logger.warning(f"{log_prefix} Objectif '{metric_name}' est None, NaN ou Inf ({value}). Assignation de la pire valeur par défaut.")
                value = -float('inf') if direction == "maximize" else float('inf')
            objective_values.append(float(value)) 
        
        end_time_trial = time.time()
        logger.info(f"{log_prefix} Évaluation terminée en {end_time_trial - start_time_trial:.2f}s. Objectifs: {objective_values}")
        
        return tuple(objective_values)
