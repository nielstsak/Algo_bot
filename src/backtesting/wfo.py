import json
import logging
import importlib
import time
import traceback
import sys
import argparse
import math
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    from src.backtesting.optimizer.optimization_orchestrator import run_optimization_for_fold
    from src.backtesting.optimizer.objective_evaluator import ObjectiveEvaluator
    from src.backtesting.optimizer.study_manager import StudyManager
    from src.backtesting.optimizer.results_analyzer import ResultsAnalyzer
    from src.strategies.base import BaseStrategy
    from src.config.loader import load_config, AppConfig, load_all_configs, GlobalConfig
    from src.config.definitions import WfoSettings
    from src.live.execution import OrderExecutionClient
except ImportError as e:
    logging.critical(f"WFO: Could not import required modules: {e}. Check PYTHONPATH.", exc_info=True)
    raise

logger = logging.getLogger(__name__)

def _get_expanding_folds(
    df_enriched: pd.DataFrame,
    n_splits: int,
    oos_percent: float
) -> Generator[Tuple[pd.DataFrame, pd.DataFrame, int, pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp], None, None]:
    if df_enriched.empty:
        logger.error("Input DataFrame df_enriched is empty. Cannot generate folds.")
        return
    if not isinstance(df_enriched.index, pd.DatetimeIndex):
        logger.error("df_enriched must have a DatetimeIndex.")
        return
    if df_enriched.index.tz is None or df_enriched.index.tz.utcoffset(df_enriched.index[0]) != timezone.utc.utcoffset(df_enriched.index[0]):
        logger.error("df_enriched index must be timezone-aware and UTC.")
        return
    if not df_enriched.index.is_monotonic_increasing:
        df_enriched = df_enriched.sort_index()

    n_total_points = len(df_enriched)
    if n_total_points < 2:
        logger.error(f"Not enough data ({n_total_points} points) in df_enriched to create folds.")
        return
    if not (0 < oos_percent < 100):
        logger.error("oos_percent must be between 0 and 100 (exclusive).")
        return
    if n_splits < 1:
        logger.error("n_splits must be at least 1.")
        return

    a_timestamp = df_enriched.index.min()
    e_timestamp = df_enriched.index.max()
    total_duration_seconds = (e_timestamp - a_timestamp).total_seconds()

    if total_duration_seconds <= 0:
        logger.error("Total duration of the dataset is not positive. Cannot split.")
        return

    oos_duration_seconds = total_duration_seconds * (oos_percent / 100.0)
    d_approx_split_point = e_timestamp - timedelta(seconds=oos_duration_seconds)
    
    possible_d_timestamps = df_enriched.index[df_enriched.index <= d_approx_split_point]
    if possible_d_timestamps.empty:
        logger.error(f"OOS percentage ({oos_percent}%) is too high or dataset too short. No data points available for the In-Sample period before the OOS cutoff ({d_approx_split_point}).")
        return
    d_timestamp = possible_d_timestamps.max()

    df_is_total_enriched = df_enriched.loc[:d_timestamp]
    
    oos_start_candidates = df_enriched.index[df_enriched.index > d_timestamp]
    if oos_start_candidates.empty:
        df_oos_fixed_enriched = pd.DataFrame(columns=df_enriched.columns, index=pd.DatetimeIndex([], tz='UTC'))
    else:
        actual_start_oos_ts = oos_start_candidates.min()
        df_oos_fixed_enriched = df_enriched.loc[actual_start_oos_ts:]
    
    start_oos_fixed_ts = df_oos_fixed_enriched.index.min() if not df_oos_fixed_enriched.empty else pd.NaT
    end_oos_fixed_ts = df_oos_fixed_enriched.index.max() if not df_oos_fixed_enriched.empty else pd.NaT

    if df_is_total_enriched.empty:
        logger.error("Total In-Sample period is empty after OOS split. Cannot generate IS folds.")
        return
    
    is_total_start_ts = df_is_total_enriched.index.min()
    is_total_duration_td = d_timestamp - is_total_start_ts

    if is_total_duration_td.total_seconds() <= 0 and n_splits > 0:
        logger.error(f"Total IS duration is not positive ({is_total_duration_td}). Cannot create {n_splits} IS segments.")
        return
        
    segment_duration_td = is_total_duration_td / n_splits if n_splits > 0 else is_total_duration_td
                                        
    for k_segments_in_fold in range(1, n_splits + 1):
        fold_idx_wfo_convention = k_segments_in_fold -1 

        start_is_k_approx_ts = d_timestamp - (k_segments_in_fold * segment_duration_td)
        
        if start_is_k_approx_ts < is_total_start_ts:
            actual_start_is_fold_ts = is_total_start_ts
        else:
            idx_pos = df_is_total_enriched.index.searchsorted(start_is_k_approx_ts, side='left')
            if idx_pos < len(df_is_total_enriched.index):
                actual_start_is_fold_ts = df_is_total_enriched.index[idx_pos]
            else: 
                actual_start_is_fold_ts = d_timestamp 
        
        actual_end_is_fold_ts = d_timestamp

        if actual_start_is_fold_ts > actual_end_is_fold_ts :
            continue

        df_is_enriched_fold = df_is_total_enriched.loc[actual_start_is_fold_ts : actual_end_is_fold_ts]

        if df_is_enriched_fold.empty:
            continue
        
        yield (df_is_enriched_fold, 
               df_oos_fixed_enriched.copy() if not df_oos_fixed_enriched.empty else pd.DataFrame(columns=df_enriched.columns, index=pd.DatetimeIndex([], tz='UTC')), 
               fold_idx_wfo_convention, 
               actual_start_is_fold_ts, 
               actual_end_is_fold_ts, 
               start_oos_fixed_ts, 
               end_oos_fixed_ts)


class WalkForwardOptimizer:
    def __init__(self, app_config: AppConfig):
        self.app_config: AppConfig = app_config
        self.global_config_obj: GlobalConfig = self.app_config.global_config
        self.wfo_settings: WfoSettings = self.global_config_obj.wfo_settings
        
        self.strategies_config_dict: Dict[str, Dict[str, Any]] = {}
        if hasattr(self.app_config.strategies_config, 'strategies') and \
           isinstance(self.app_config.strategies_config.strategies, dict):
            for name, strategy_params_obj in self.app_config.strategies_config.strategies.items():
                if hasattr(strategy_params_obj, '__dict__'): 
                     self.strategies_config_dict[name] = strategy_params_obj.__dict__
                elif isinstance(strategy_params_obj, dict): 
                     self.strategies_config_dict[name] = strategy_params_obj
        else:
            logger.error("AppConfig.strategies_config.strategies is not a dictionary.")

        self.paths_config: Dict[str, Any] = self.global_config_obj.paths.__dict__
        self.simulation_defaults: Dict[str, Any] = self.global_config_obj.simulation_defaults.__dict__

        if not (0 < self.wfo_settings.oos_percent < 100):
            raise ValueError("'oos_percent' must be between 0 and 100 (exclusive).")

        run_timestamp_str = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        logs_opt_path_str = self.paths_config.get('logs_backtest_optimization', 'logs/backtest_optimization')
        self.run_output_dir = Path(logs_opt_path_str) / run_timestamp_str
        self.run_output_dir.mkdir(parents=True, exist_ok=True)

        live_settings = self.app_config.live_config.global_live_settings
        self.execution_client = OrderExecutionClient(
            api_key=self.app_config.api_keys.binance_api_key,
            api_secret=self.app_config.api_keys.binance_secret_key,
            account_type=getattr(live_settings, 'account_type', "MARGIN"), 
            is_testnet=getattr(live_settings, 'is_testnet', False)
        )
        if not self.execution_client.test_connection():
            logger.warning("Failed Binance REST API connection during WFO init. Symbol info might fail.")


    def run(self, pairs: List[str], context_labels: List[str]): 
        all_wfo_run_results = {}

        for pair_symbol in pairs:
            current_context_label = context_labels[0] if context_labels else "default_context"
            
            symbol_info_data = self.execution_client.get_symbol_info(pair_symbol)
            if not symbol_info_data:
                continue
            
            enriched_data_dir_path_str = self.app_config.global_config.paths.data_historical_processed_enriched
            if not enriched_data_dir_path_str: 
                continue
            enriched_data_dir_path = Path(enriched_data_dir_path_str)
            enriched_filename = f"{pair_symbol}_enriched.parquet"
            enriched_filepath = enriched_data_dir_path / enriched_filename

            if not enriched_filepath.exists():
                continue
            
            try:
                data_enriched_full = pd.read_parquet(enriched_filepath)
                if 'timestamp' not in data_enriched_full.columns:
                    raise ValueError("Loaded enriched data is missing 'timestamp' column.")
                
                data_enriched_full['timestamp'] = pd.to_datetime(data_enriched_full['timestamp'], utc=True)
                data_enriched_full = data_enriched_full.set_index('timestamp')
                
                if data_enriched_full.index.tz is None:
                    data_enriched_full.index = data_enriched_full.index.tz_localize('UTC')
                elif data_enriched_full.index.tz.utcoffset(data_enriched_full.index[0]) != timezone.utc.utcoffset(data_enriched_full.index[0]):
                     data_enriched_full.index = data_enriched_full.index.tz_convert('UTC')
                
                data_enriched_full.sort_index(inplace=True)
                if not data_enriched_full.index.is_unique:
                    data_enriched_full = data_enriched_full[~data_enriched_full.index.duplicated(keep='first')]

            except Exception as e_load:
                logger.error(f"Failed to load or process enriched file {enriched_filepath}: {e_load}", exc_info=True)
                continue
            
            active_strategies = {
                name: cfg_dict for name, cfg_dict in self.strategies_config_dict.items()
                if isinstance(cfg_dict, dict) and cfg_dict.get('active_for_optimization', False)
            }

            if not active_strategies:
                continue

            for strat_name, strat_config_dict_for_opt in active_strategies.items():
                strategy_pair_output_dir = self.run_output_dir / strat_name / pair_symbol / current_context_label
                strategy_pair_output_dir.mkdir(parents=True, exist_ok=True)
                
                fold_summaries = []
                
                try:
                    folds_generator = _get_expanding_folds(
                        df_enriched=data_enriched_full, 
                        n_splits=self.wfo_settings.n_splits,
                        oos_percent=self.wfo_settings.oos_percent
                    )
                    for df_is_enriched, df_oos_enriched, fold_idx, start_is, end_is, start_oos, end_oos in folds_generator:
                        
                        fold_output_path = strategy_pair_output_dir / f"fold_{fold_idx}"
                        fold_output_path.mkdir(parents=True, exist_ok=True)
                        
                        final_params_for_fold, representative_oos_metrics_fold = run_optimization_for_fold(
                            strategy_name=strat_name,
                            strategy_config_dict=strat_config_dict_for_opt,
                            data_1min_cleaned_is_slice=df_is_enriched, 
                            data_1min_cleaned_oos_slice=df_oos_enriched, 
                            app_config=self.app_config,
                            output_dir_fold=fold_output_path,
                            pair_symbol=pair_symbol, 
                            symbol_info_data=symbol_info_data, 
                            objective_evaluator_class=ObjectiveEvaluator,
                            study_manager_class=StudyManager,
                            results_analyzer_class=ResultsAnalyzer
                        )
                        
                        fold_status = "COMPLETED"
                        if final_params_for_fold is None or representative_oos_metrics_fold is None:
                            fold_status = "OPTIMIZATION_FAILED"

                        fold_summary_entry = {
                            "fold_index": fold_idx, "status": fold_status,
                            "is_period_start": start_is.isoformat() if pd.notna(start_is) else None,
                            "is_period_end": end_is.isoformat() if pd.notna(end_is) else None,
                            "oos_period_start": start_oos.isoformat() if pd.notna(start_oos) else None,
                            "oos_period_end": end_oos.isoformat() if pd.notna(end_oos) else None,
                            "selected_params_for_fold": final_params_for_fold,
                            "representative_oos_metrics": representative_oos_metrics_fold
                        }
                        fold_summaries.append(fold_summary_entry)
                
                except Exception as e_fold_loop:
                    logger.error(f"WFO loop failed for {strat_name} on {pair_symbol}: {e_fold_loop}", exc_info=True)
                
                wfo_summary_for_strategy_pair = {
                    "strategy_name": strat_name, "pair_symbol": pair_symbol,
                    "context_label": current_context_label,
                    "wfo_run_timestamp": self.run_output_dir.name,
                    "folds_data": fold_summaries
                }
                
                summary_file_path = strategy_pair_output_dir / "wfo_strategy_pair_summary.json"
                try:
                    with open(summary_file_path, 'w', encoding='utf-8') as f_sum:
                        json.dump(wfo_summary_for_strategy_pair, f_sum, indent=4, default=str) 
                except Exception as e_save_final:
                    logger.error(f"Failed to save final WFO summary for {strat_name}/{pair_symbol}: {e_save_final}", exc_info=True)
                
                all_wfo_run_results[f"{strat_name}_{pair_symbol}_{current_context_label}"] = wfo_summary_for_strategy_pair
        
        return all_wfo_run_results
