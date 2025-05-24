import logging
import time
import importlib
from typing import Any, Dict, Optional, Tuple, List, Type, Union
from datetime import timezone # Déjà présent, ok
from pathlib import Path # <<< IMPORT AJOUTÉ ICI
import uuid

import numpy as np
import pandas as pd
import pandas_ta as ta # type: ignore
import optuna

try:
    from src.backtesting.simulator import BacktestSimulator
    from src.data import data_utils # Pour get_kline_prefix_effective
    from src.strategies.base import BaseStrategy
    from src.config.definitions import ParamDetail, AppConfig # Importer AppConfig
except ImportError as e:
    # Fallback logger if main logging isn't set up yet
    logging.basicConfig(level=logging.WARNING)
    logging.getLogger(__name__).critical(f"ObjectiveEvaluator: Critical import error: {e}. Ensure PYTHONPATH is correct.", exc_info=True)
    raise

logger = logging.getLogger(__name__)

class ObjectiveEvaluator:
    def __init__(self,
                 strategy_name: str,
                 strategy_config_dict: Dict[str, Any],
                 df_enriched_slice: pd.DataFrame, # Renamed from data_1min_cleaned_slice
                 simulation_settings: Dict[str, Any],
                 optuna_objectives_config: Dict[str, Any],
                 pair_symbol: str,
                 symbol_info_data: Dict[str, Any],
                 app_config: AppConfig, # Added app_config
                 is_oos_eval: bool = False,
                 is_trial_number_for_oos_log: Optional[int] = None):

        self.strategy_name = strategy_name
        self.strategy_config_dict = strategy_config_dict
        self.df_enriched_slice = df_enriched_slice.copy() # Source unique de données
        self.simulation_settings_global_defaults = simulation_settings
        self.optuna_objectives_config = optuna_objectives_config
        self.pair_symbol = pair_symbol
        self.symbol_info_data = symbol_info_data
        self.is_oos_eval = is_oos_eval
        self.app_config = app_config # Store app_config
        self.is_trial_number_for_oos_log = is_trial_number_for_oos_log

        self.strategy_script_ref = self.strategy_config_dict.get('script_reference')
        self.strategy_class_name = self.strategy_config_dict.get('class_name')

        if not self.strategy_script_ref or not self.strategy_class_name:
            raise ValueError("ObjectiveEvaluator: script_reference or class_name missing in strategy_config_dict")

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
                        logger.error(f"Error creating ParamDetail for {param_key} from dict: {e_pd_init}")
                else:
                    logger.warning(f"Item '{param_key}' in params_space for strategy '{self.strategy_name}' is not ParamDetail or dict. Type: {type(param_value_obj)}")
        else:
            logger.error(f"params_space for strategy '{self.strategy_name}' is not a dict. Type: {type(raw_params_space)}")

        if not self.params_space_details:
            logger.critical(f"CRITICAL: self.params_space_details is EMPTY for strategy {self.strategy_name}. This will likely lead to pruning. Raw params_space: {raw_params_space}")


        if not isinstance(self.df_enriched_slice.index, pd.DatetimeIndex):
            if 'timestamp' in self.df_enriched_slice.columns:
                self.df_enriched_slice['timestamp'] = pd.to_datetime(self.df_enriched_slice['timestamp'], utc=True, errors='coerce')
                self.df_enriched_slice.dropna(subset=['timestamp'], inplace=True)
                self.df_enriched_slice = self.df_enriched_slice.set_index('timestamp')
            else:
                raise ValueError("ObjectiveEvaluator: df_enriched_slice must have a DatetimeIndex or a 'timestamp' column.")

        if self.df_enriched_slice.index.tz is None:
            self.df_enriched_slice.index = self.df_enriched_slice.index.tz_localize('UTC')
        elif self.df_enriched_slice.index.tz.utcoffset(self.df_enriched_slice.index[0]) != timezone.utc.utcoffset(self.df_enriched_slice.index[0]):
            self.df_enriched_slice.index = self.df_enriched_slice.index.tz_convert('UTC')
        
        if not self.df_enriched_slice.index.is_monotonic_increasing:
            self.df_enriched_slice.sort_index(inplace=True)
        if not self.df_enriched_slice.index.is_unique:
            self.df_enriched_slice = self.df_enriched_slice[~self.df_enriched_slice.index.duplicated(keep='first')]

        logger.debug(f"ObjectiveEvaluator for {self.strategy_name} on {self.pair_symbol} initialized. Enriched data shape: {self.df_enriched_slice.shape}")

    def _calculate_indicator_on_selected_klines(self,
                                                df_source_enriched: pd.DataFrame,
                                                indicator_type: str,
                                                indicator_params: Dict[str, Any],
                                                kline_ohlc_prefix: str
                                                ) -> Optional[Union[pd.Series, pd.DataFrame]]:
        if df_source_enriched.empty:
            logger.warning(f"df_source_enriched is empty for {indicator_type} with prefix '{kline_ohlc_prefix}'.")
            return None

        open_col = f"{kline_ohlc_prefix}_open" if kline_ohlc_prefix else "open"
        high_col = f"{kline_ohlc_prefix}_high" if kline_ohlc_prefix else "high"
        low_col = f"{kline_ohlc_prefix}_low" if kline_ohlc_prefix else "low"
        close_col = f"{kline_ohlc_prefix}_close" if kline_ohlc_prefix else "close"
        volume_col = f"{kline_ohlc_prefix}_volume" if kline_ohlc_prefix else "volume"

        required_ta_cols_map: Dict[str, str] = {}
        indicator_type_lower = indicator_type.lower()
        required_ta_cols_map['close'] = close_col
        if indicator_type_lower in ['psar', 'adx', 'atr', 'cci', 'donchian', 'ichimoku', 'supertrend', 'kama', 'bbands', 'kc', 'stoch', 'roc', 'mom', 'ao', 'apo', 'aroon', 'chop', 'coppock', 'dm', 'fisher', 'kst', 'massi', 'natr', 'ppo', 'qstick', 'stc', 'trix', 'tsi', 'uo', 'vhf', 'vortex', 'willr', 'alma', 'dema', 'ema', 'fwma', 'hma', 'linreg', 'midpoint', 'midprice', 'rma', 'sinwma', 'sma', 'smma', 'ssf', 'tema', 'trima', 'vidya', 'vwma', 'wcp', 'wma', 'zlma']:
            required_ta_cols_map['high'] = high_col
            required_ta_cols_map['low'] = low_col
        if indicator_type_lower in ['ichimoku', 'ao', 'ha', 'ohlc4']:
             required_ta_cols_map['open'] = open_col
        if indicator_type_lower in ['obv', 'vwap', 'ad', 'adosc', 'cmf', 'efi', 'mfi', 'nvi', 'pvi', 'pvol', 'pvr', 'pvt', 'vwma', 'adx']:
            required_ta_cols_map['volume'] = volume_col

        ta_inputs: Dict[str, pd.Series] = {}
        all_required_cols_present = True
        for ta_key, source_col_name in required_ta_cols_map.items():
            if source_col_name not in df_source_enriched.columns:
                logger.warning(f"Source column '{source_col_name}' (for TA key '{ta_key}') for indicator '{indicator_type}' not found in df_source_enriched (prefix: '{kline_ohlc_prefix}').")
                if ta_key == 'close':
                    all_required_cols_present = False; break
                continue

            series_for_ta = df_source_enriched[source_col_name]
            if series_for_ta.isnull().all():
                logger.warning(f"Source column '{source_col_name}' (for TA key '{ta_key}') for indicator '{indicator_type}' is entirely NaN (prefix: '{kline_ohlc_prefix}').")
                if ta_key == 'close':
                     all_required_cols_present = False; break
            ta_inputs[ta_key] = series_for_ta.astype(float)

        if not all_required_cols_present or not ta_inputs.get('close', pd.Series(dtype=float)).notna().any():
            logger.error(f"Critical source data (e.g., close) missing or all NaN for indicator '{indicator_type}' with prefix '{kline_ohlc_prefix}'. Cannot calculate.")
            return pd.Series(np.nan, index=df_source_enriched.index) if 'close' not in ta_inputs or ta_inputs['close'].empty else None

        try:
            indicator_function = getattr(ta, indicator_type_lower, None)
            if indicator_function is None:
                for category in [ta.trend, ta.momentum, ta.overlap, ta.volume, ta.volatility, ta.cycles, ta.statistics, ta.transform, ta.utils]:
                    if hasattr(category, indicator_type_lower):
                        indicator_function = getattr(category, indicator_type_lower)
                        break
            if indicator_function is None:
                logger.error(f"Indicator function '{indicator_type_lower}' not found in pandas_ta or its submodules.")
                return None
            
            logger.debug(f"Calculating {indicator_type_lower} with params: {indicator_params} on columns with prefix '{kline_ohlc_prefix}'. Input keys for TA: {list(ta_inputs.keys())}")
            result = indicator_function(**ta_inputs, **indicator_params, append=False)
            
            if isinstance(result, pd.DataFrame) or isinstance(result, pd.Series):
                return result
            else:
                logger.warning(f"Indicator {indicator_type_lower} did not return a Series or DataFrame. Got: {type(result)}")
                return None
        except Exception as e:
            logger.error(f"Error calculating {indicator_type} with params {indicator_params} using prefix '{kline_ohlc_prefix}': {e}", exc_info=True)
            return None

    def _prepare_data_with_dynamic_indicators(self, trial_params: Dict[str, Any], trial_number_for_log: Optional[Union[int, str]] = None) -> pd.DataFrame:
        df_for_simulation = self.df_enriched_slice[['open', 'high', 'low', 'close', 'volume']].copy()
        current_trial_num_str = str(trial_number_for_log) if trial_number_for_log is not None else "N/A_IS_Prep"
        if self.is_oos_eval and self.is_trial_number_for_oos_log is not None:
            current_trial_num_str = f"OOS_for_IS_{self.is_trial_number_for_oos_log}"
        log_prefix = f"[{self.strategy_name}/{self.pair_symbol}/Trial-{current_trial_num_str}]"

        logger.info(f"{log_prefix} Preparing data with dynamic indicators. Enriched slice shape: {self.df_enriched_slice.shape}")
        logger.debug(f"{log_prefix} Trial params: {trial_params}")

        atr_period_key = 'atr_period_sl_tp' if 'atr_period_sl_tp' in trial_params else 'atr_period'
        atr_freq_key = 'atr_base_frequency_sl_tp' if 'atr_base_frequency_sl_tp' in trial_params else 'atr_base_frequency'
        atr_period_param = trial_params.get(atr_period_key)
        atr_freq_param_raw = trial_params.get(atr_freq_key)

        if atr_period_param is not None and atr_freq_param_raw is not None:
            atr_period_val = int(atr_period_param)
            kline_prefix_atr_source = data_utils.get_kline_prefix_effective(str(atr_freq_param_raw))
            atr_source_col_name = f"{kline_prefix_atr_source}_ATR_{atr_period_val}" if kline_prefix_atr_source else f"ATR_{atr_period_val}"
            logger.debug(f"{log_prefix} ATR_strat: Trying to use pre-calculated column '{atr_source_col_name}' from enriched data.")

            if atr_source_col_name in self.df_enriched_slice.columns:
                df_for_simulation['ATR_strat'] = self.df_enriched_slice[atr_source_col_name].reindex(df_for_simulation.index, method='ffill')
                logger.info(f"{log_prefix} ATR_strat loaded from pre-calculated '{atr_source_col_name}'. NaNs: {df_for_simulation['ATR_strat'].isnull().sum()}/{len(df_for_simulation)}")
            elif kline_prefix_atr_source == "":
                logger.info(f"{log_prefix} ATR_strat: Pre-calculated ATR for 1-min not found ('{atr_source_col_name}'). Calculating ATR({atr_period_val}) on 1-min base data.")
                atr_series_1min = self._calculate_indicator_on_selected_klines(
                    self.df_enriched_slice, 'atr', {'length': atr_period_val}, ""
                )
                df_for_simulation['ATR_strat'] = atr_series_1min.reindex(df_for_simulation.index, method='ffill') if isinstance(atr_series_1min, pd.Series) else np.nan
                if isinstance(df_for_simulation.get('ATR_strat'), pd.Series):
                     logger.info(f"{log_prefix} ATR_strat (1-min dynamically calculated) NaNs: {df_for_simulation['ATR_strat'].isnull().sum()}/{len(df_for_simulation)}")
            else:
                logger.warning(f"{log_prefix} ATR_strat: Pre-calculated ATR column '{atr_source_col_name}' NOT FOUND in enriched data for frequency '{atr_freq_param_raw}'. ATR_strat will be NaN.")
                df_for_simulation['ATR_strat'] = np.nan
        else:
            logger.warning(f"{log_prefix} ATR_strat: Parameters '{atr_period_key}' or '{atr_freq_key}' missing in trial_params. ATR_strat will be NaN.")
            df_for_simulation['ATR_strat'] = np.nan

        # Dynamically calculate other indicators based on strategy type and params
        if self.strategy_name == "MaCrossoverStrategy":
            ma_fast_p = int(trial_params['fast_ma_period'])
            ma_slow_p = int(trial_params['slow_ma_period'])
            ma_type = str(trial_params.get('ma_type', 'sma'))
            freq_fast_raw = str(trial_params['indicateur_frequence_ma_rapide'])
            kline_prefix_fast = data_utils.get_kline_prefix_effective(freq_fast_raw)
            freq_slow_raw = str(trial_params['indicateur_frequence_ma_lente'])
            kline_prefix_slow = data_utils.get_kline_prefix_effective(freq_slow_raw)

            ma_fast_series = self._calculate_indicator_on_selected_klines(self.df_enriched_slice, ma_type, {'length': ma_fast_p}, kline_prefix_fast)
            df_for_simulation['MA_FAST_strat'] = ma_fast_series.reindex(df_for_simulation.index, method='ffill') if isinstance(ma_fast_series, pd.Series) else np.nan
            
            ma_slow_series = self._calculate_indicator_on_selected_klines(self.df_enriched_slice, ma_type, {'length': ma_slow_p}, kline_prefix_slow)
            df_for_simulation['MA_SLOW_strat'] = ma_slow_series.reindex(df_for_simulation.index, method='ffill') if isinstance(ma_slow_series, pd.Series) else np.nan

        elif self.strategy_name == "PsarReversalOtocoStrategy":
            psar_params = {
                'step': float(trial_params['psar_step']),
                'max_step': float(trial_params['psar_max_step'])
            }
            psar_freq_raw = str(trial_params['indicateur_frequence_psar'])
            kline_prefix_psar = data_utils.get_kline_prefix_effective(psar_freq_raw)
            psar_df_result = self._calculate_indicator_on_selected_klines(self.df_enriched_slice, 'psar', psar_params, kline_prefix_psar)
            if psar_df_result is not None and isinstance(psar_df_result, pd.DataFrame) and not psar_df_result.empty:
                df_for_simulation['PSARl_strat'] = psar_df_result.iloc[:,0].reindex(df_for_simulation.index, method='ffill')
                df_for_simulation['PSARs_strat'] = psar_df_result.iloc[:,1].reindex(df_for_simulation.index, method='ffill')
            else:
                df_for_simulation['PSARl_strat'] = np.nan
                df_for_simulation['PSARs_strat'] = np.nan

        elif self.strategy_name == "BbandsVolumeRsiStrategy":
            bb_period = int(trial_params['bbands_period'])
            bb_std = float(trial_params['bbands_std_dev'])
            bb_freq_raw = str(trial_params['indicateur_frequence_bbands'])
            kline_prefix_bb = data_utils.get_kline_prefix_effective(bb_freq_raw)
            bb_df_result = self._calculate_indicator_on_selected_klines(self.df_enriched_slice, 'bbands', {'length': bb_period, 'std': bb_std}, kline_prefix_bb)
            if bb_df_result is not None and isinstance(bb_df_result, pd.DataFrame) and not bb_df_result.empty:
                df_for_simulation['BB_LOWER_strat'] = bb_df_result.iloc[:,0].reindex(df_for_simulation.index, method='ffill')
                df_for_simulation['BB_MIDDLE_strat'] = bb_df_result.iloc[:,1].reindex(df_for_simulation.index, method='ffill')
                df_for_simulation['BB_UPPER_strat'] = bb_df_result.iloc[:,2].reindex(df_for_simulation.index, method='ffill')
                df_for_simulation['BB_BANDWIDTH_strat'] = bb_df_result.iloc[:,3].reindex(df_for_simulation.index, method='ffill')
            else:
                for col in ['BB_LOWER_strat', 'BB_MIDDLE_strat', 'BB_UPPER_strat', 'BB_BANDWIDTH_strat']: df_for_simulation[col] = np.nan

            vol_ma_p = int(trial_params['volume_ma_period'])
            vol_freq_raw = str(trial_params['indicateur_frequence_volume'])
            kline_prefix_vol = data_utils.get_kline_prefix_effective(vol_freq_raw)
            vol_col_name_source = f"{kline_prefix_vol}_volume" if kline_prefix_vol else "volume"
            if vol_col_name_source in self.df_enriched_slice.columns:
                # Note: pandas_ta.sma needs 'close' as input name by default.
                # We need to pass the correct volume series with the name 'close' for ta.sma, or use a different MA function.
                # For simplicity, let's assume self._calculate_indicator_on_selected_klines can handle 'sma' on a generic series.
                # If _calculate_indicator_on_selected_klines uses getattr(ta, 'sma'), it will expect a 'close' kwarg.
                # A more robust way is to ensure the 'volume' series is passed correctly.
                # Let's assume _calculate_indicator_on_selected_klines is adapted or we use direct ewm for sma.
                # For now, this might fail if 'sma' in pandas_ta strictly requires a 'close' named series.
                # We will pass the volume series as the primary series to a generic MA calculation.
                # Let's make _calculate_indicator_on_selected_klines more flexible or use a direct pandas ewm/rolling.
                # For now, we'll assume it works by passing the 'volume' column as the main series.
                # This requires _calculate_indicator_on_selected_klines to be flexible, e.g., if indicator_type is 'sma', it uses the 'close' key from ta_inputs.
                # We need to ensure the volume column is correctly identified and passed.
                # The current _calculate_indicator_on_selected_klines maps 'close' to the close_col.
                # This needs adjustment for volume MA.
                # Quick fix: use pandas rolling mean for SMA directly here.
                if self.df_enriched_slice[vol_col_name_source].notna().any():
                    df_for_simulation['Volume_MA_strat'] = self.df_enriched_slice[vol_col_name_source].rolling(window=vol_ma_p, min_periods=vol_ma_p).mean().reindex(df_for_simulation.index, method='ffill')
                else:
                    df_for_simulation['Volume_MA_strat'] = np.nan
                df_for_simulation['Kline_Volume_Source_strat'] = self.df_enriched_slice[vol_col_name_source].reindex(df_for_simulation.index, method='ffill')
            else:
                df_for_simulation['Volume_MA_strat'] = np.nan
                df_for_simulation['Kline_Volume_Source_strat'] = np.nan

            rsi_p = int(trial_params['rsi_period'])
            rsi_freq_raw = str(trial_params['indicateur_frequence_rsi'])
            kline_prefix_rsi = data_utils.get_kline_prefix_effective(rsi_freq_raw)
            rsi_series = self._calculate_indicator_on_selected_klines(self.df_enriched_slice, 'rsi', {'length': rsi_p}, kline_prefix_rsi)
            df_for_simulation['RSI_strat'] = rsi_series.reindex(df_for_simulation.index, method='ffill') if isinstance(rsi_series, pd.Series) else np.nan

        elif self.strategy_name == "TripleMAAnticipationStrategy":
            ma_s_p = int(trial_params['ma_short_period'])
            ma_m_p = int(trial_params['ma_medium_period'])
            ma_l_p = int(trial_params['ma_long_period'])
            
            freq_s_raw = str(trial_params['indicateur_frequence_mms'])
            kline_prefix_s = data_utils.get_kline_prefix_effective(freq_s_raw)
            ma_s_series = self._calculate_indicator_on_selected_klines(self.df_enriched_slice, 'sma', {'length': ma_s_p}, kline_prefix_s)
            df_for_simulation['MA_SHORT_strat'] = ma_s_series.reindex(df_for_simulation.index, method='ffill') if isinstance(ma_s_series, pd.Series) else np.nan

            freq_m_raw = str(trial_params['indicateur_frequence_mmm'])
            kline_prefix_m = data_utils.get_kline_prefix_effective(freq_m_raw)
            ma_m_series = self._calculate_indicator_on_selected_klines(self.df_enriched_slice, 'sma', {'length': ma_m_p}, kline_prefix_m)
            df_for_simulation['MA_MEDIUM_strat'] = ma_m_series.reindex(df_for_simulation.index, method='ffill') if isinstance(ma_m_series, pd.Series) else np.nan

            freq_l_raw = str(trial_params['indicateur_frequence_mml'])
            kline_prefix_l = data_utils.get_kline_prefix_effective(freq_l_raw)
            ma_l_series = self._calculate_indicator_on_selected_klines(self.df_enriched_slice, 'sma', {'length': ma_l_p}, kline_prefix_l)
            df_for_simulation['MA_LONG_strat'] = ma_l_series.reindex(df_for_simulation.index, method='ffill') if isinstance(ma_l_series, pd.Series) else np.nan

            if trial_params.get('anticipate_crossovers', False):
                slope_period = int(trial_params['anticipation_slope_period'])
                # Calculate slopes based on the MAs just calculated and added to df_for_simulation
                if df_for_simulation['MA_SHORT_strat'].notna().any():
                    df_for_simulation['SLOPE_MA_SHORT_strat'] = self._calculate_slope(df_for_simulation['MA_SHORT_strat'], slope_period)
                else: df_for_simulation['SLOPE_MA_SHORT_strat'] = np.nan
                
                if df_for_simulation['MA_MEDIUM_strat'].notna().any():
                    df_for_simulation['SLOPE_MA_MEDIUM_strat'] = self._calculate_slope(df_for_simulation['MA_MEDIUM_strat'], slope_period)
                else: df_for_simulation['SLOPE_MA_MEDIUM_strat'] = np.nan


        indicator_strat_cols = [col for col in df_for_simulation.columns if col.endswith('_strat')]
        if indicator_strat_cols:
            df_for_simulation[indicator_strat_cols] = df_for_simulation[indicator_strat_cols].ffill()
            logger.debug(f"{log_prefix} Applied ffill to _strat columns: {indicator_strat_cols}")
        
        essential_ohlcv = ['open', 'high', 'low', 'close']
        if df_for_simulation[essential_ohlcv].isnull().all().any():
            logger.error(f"{log_prefix} One or more essential OHLCV columns are entirely NaN in df_for_simulation.")

        logger.info(f"{log_prefix} Data preparation complete. df_for_simulation shape: {df_for_simulation.shape}. Columns: {df_for_simulation.columns.tolist()}")
        return df_for_simulation


    def _suggest_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        params_for_trial: Dict[str, Any] = {}
        if not self.params_space_details:
            logger.error(f"params_space_details not initialized for strategy '{self.strategy_name}'. Cannot suggest parameters.")
            raise optuna.exceptions.TrialPruned("Parameter space not defined for the strategy.")

        for param_name, p_detail in self.params_space_details.items():
            if param_name in params_for_trial: continue
            try:
                if p_detail.type == 'int':
                    params_for_trial[param_name] = trial.suggest_int(param_name, int(p_detail.low), int(p_detail.high), step=int(p_detail.step or 1))
                elif p_detail.type == 'float':
                    params_for_trial[param_name] = trial.suggest_float(param_name, float(p_detail.low), float(p_detail.high), step=p_detail.step)
                elif p_detail.type == 'categorical' and p_detail.choices:
                    params_for_trial[param_name] = trial.suggest_categorical(param_name, p_detail.choices)
            except Exception as e_suggest:
                logger.error(f"Error suggesting param '{param_name}' (type: {p_detail.type}): {e_suggest}", exc_info=True)
                if p_detail.choices: params_for_trial[param_name] = p_detail.choices[0]
                elif p_detail.low is not None: params_for_trial[param_name] = p_detail.low
                else:
                    logger.warning(f"Cannot set fallback for param '{param_name}'. Trial might be unstable.")
                    raise optuna.exceptions.TrialPruned(f"Failed to suggest or fallback for param {param_name}")
        return params_for_trial

    def __call__(self, trial: optuna.Trial) -> Tuple[float, ...]:
        start_time_trial = time.time()
        trial_id_for_log: str
        trial_number_for_prepare_log: Optional[int] = None

        if self.is_oos_eval:
            is_trial_num_oos = self.is_trial_number_for_oos_log
            if is_trial_num_oos is not None:
                trial_id_for_log = f"OOS_for_IS_Trial_{is_trial_num_oos}"
                trial_number_for_prepare_log = is_trial_num_oos
            else:
                trial_id_for_log = f"OOS_Trial_UnknownOrigin_{trial.number if hasattr(trial, 'number') else uuid.uuid4().hex[:6]}"
                if hasattr(trial, 'number'): trial_number_for_prepare_log = trial.number
        else:
            trial_id_for_log = str(trial.number) if hasattr(trial, 'number') and trial.number is not None else f"IS_Trial_Unknown_{uuid.uuid4().hex[:6]}"
            if hasattr(trial, 'number'): trial_number_for_prepare_log = trial.number

        log_prefix = f"[{self.strategy_name}/{self.pair_symbol}/Trial-{trial_id_for_log}]"
        logger.info(f"{log_prefix} Starting evaluation...")

        StrategyClass_local: Optional[Type[BaseStrategy]] = None
        try:
            module_path = self.strategy_script_ref.replace('.py', '').replace('/', '.')
            module = importlib.import_module(module_path)
            StrategyClass_local = getattr(module, self.strategy_class_name)
            if not StrategyClass_local or not issubclass(StrategyClass_local, BaseStrategy):
                raise ImportError(f"Could not load a valid BaseStrategy subclass: {self.strategy_class_name}")
        except Exception as e_load_strat:
            logger.error(f"{log_prefix} Failed to load strategy class {self.strategy_class_name}: {e_load_strat}", exc_info=True)
            raise optuna.exceptions.TrialPruned(f"Strategy class load failure: {e_load_strat}")

        current_trial_params: Dict[str, Any]
        if self.is_oos_eval:
            current_trial_params = trial.params.copy()
            current_trial_params.pop("_trial_id_for_oos", None)
            logger.info(f"{log_prefix} OOS evaluation using fixed params: {current_trial_params}")
        else:
            try:
                current_trial_params = self._suggest_params(trial)
                if not current_trial_params:
                    logger.error(f"{log_prefix} No parameters were suggested. Pruning trial.")
                    raise optuna.exceptions.TrialPruned("No parameters suggested.")
                logger.info(f"{log_prefix} IS evaluation with suggested params: {current_trial_params}")
            except optuna.exceptions.TrialPruned:
                raise
            except Exception as e_suggest:
                logger.error(f"{log_prefix} Error during parameter suggestion: {e_suggest}", exc_info=True)
                raise optuna.exceptions.TrialPruned(f"Parameter suggestion failed: {e_suggest}")

        try:
            data_for_simulation = self._prepare_data_with_dynamic_indicators(current_trial_params, trial_number_for_log=trial_number_for_prepare_log)
            if data_for_simulation.empty or data_for_simulation[['open', 'high', 'low', 'close']].isnull().all().all():
                logger.error(f"{log_prefix} Data for simulation is empty or OHLC fully NaN after preparation. Pruning.")
                raise optuna.exceptions.TrialPruned("Data preparation resulted in unusable data.")
        except optuna.exceptions.TrialPruned:
            raise
        except Exception as e_prepare:
            logger.error(f"{log_prefix} Error preparing data with dynamic indicators: {e_prepare}", exc_info=True)
            return tuple([-float('inf') if direction == "maximize" else float('inf')
                          for direction in self.optuna_objectives_config['objectives_directions']])

        strategy_instance = StrategyClass_local(params=current_trial_params)
        sim_settings_for_trial = self.app_config.global_config.simulation_defaults.__dict__.copy()
        sim_settings_for_trial['symbol'] = self.pair_symbol
        sim_settings_for_trial['symbol_info'] = self.symbol_info_data
        sim_settings_for_trial['capital_allocation_pct'] = current_trial_params.get('capital_allocation_pct', sim_settings_for_trial.get('capital_allocation_pct'))
        sim_settings_for_trial['margin_leverage'] = current_trial_params.get('margin_leverage', sim_settings_for_trial.get('margin_leverage'))
        
        simulator = BacktestSimulator(
            historical_data_with_indicators=data_for_simulation,
            strategy_instance=strategy_instance,
            simulation_settings=sim_settings_for_trial,
            output_dir=None 
        )

        try:
            backtest_results = simulator.run_simulation()
        except Exception as e_sim:
            logger.error(f"{log_prefix} Error during BacktestSimulator execution: {e_sim}", exc_info=True)
            return tuple([-float('inf') if direction == "maximize" else float('inf')
                          for direction in self.optuna_objectives_config['objectives_directions']])


        metrics = backtest_results.get("metrics", {})
        if not metrics or metrics.get("Status") == "EARLY_STOPPED_EQUITY_ZERO_OR_NEGATIVE" or "error" in metrics :
            logger.warning(f"{log_prefix} Simulation metrics missing, indicated critical failure, or contained error. Assigning worst objective values. Metrics: {metrics}")
            return tuple([-float('inf') if direction == "maximize" else float('inf')
                          for direction in self.optuna_objectives_config['objectives_directions']])


        objective_values: List[float] = []
        for i, metric_name in enumerate(self.optuna_objectives_config['objectives_names']):
            value = metrics.get(metric_name)
            direction = self.optuna_objectives_config['objectives_directions'][i]

            if value is None or not isinstance(value, (int, float)) or not np.isfinite(value):
                logger.warning(f"{log_prefix} Objective '{metric_name}' is None, NaN, or Inf (value: {value}). Assigning default worst value for this objective.")
                value = -float('inf') if direction == "maximize" else float('inf')
            objective_values.append(float(value))
        
        logger.debug(f"{log_prefix} All metrics for trial: {metrics}")
        end_time_trial = time.time()
        logger.info(f"{log_prefix} Evaluation finished in {end_time_trial - start_time_trial:.2f}s. Objectives: {objective_values}")

        return tuple(objective_values)
