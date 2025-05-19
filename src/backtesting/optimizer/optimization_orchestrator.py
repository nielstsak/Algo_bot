import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type, TYPE_CHECKING, List

import pandas as pd
import optuna

if TYPE_CHECKING:
    from src.config.loader import AppConfig
    from src.backtesting.optimizer.objective_evaluator import ObjectiveEvaluator
    from src.backtesting.optimizer.study_manager import StudyManager
    from src.backtesting.optimizer.results_analyzer import ResultsAnalyzer

logger = logging.getLogger(__name__)

def run_optimization_for_fold(
    strategy_name: str,
    strategy_config_dict: Dict[str, Any], 
    data_1min_cleaned_is_slice: pd.DataFrame, 
    data_1min_cleaned_oos_slice: Optional[pd.DataFrame], 
    app_config: 'AppConfig',
    output_dir_fold: Path,
    pair_symbol: str, 
    symbol_info_data: Dict[str, Any], 
    objective_evaluator_class: Type['ObjectiveEvaluator'],
    study_manager_class: Type['StudyManager'],
    results_analyzer_class: Type['ResultsAnalyzer']
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:

    fold_log_prefix = f"[{strategy_name}][{pair_symbol}/{output_dir_fold.name}]"
    logger.info(f"{fold_log_prefix} Starting optimization orchestrator for fold.")

    if data_1min_cleaned_is_slice.empty:
        logger.error(f"{fold_log_prefix} IS data slice is empty. Aborting optimization for this fold.")
        return None, None

    logger.info(f"{fold_log_prefix} Initializing StudyManager for IS optimization.")
    study_manager_instance = study_manager_class(
        app_config=app_config,
        strategy_name=strategy_name,
        strategy_config_dict=strategy_config_dict,
        study_output_dir=output_dir_fold,
        pair_symbol=pair_symbol, 
        symbol_info_data=symbol_info_data 
    )

    is_study: Optional[optuna.Study] = None
    try:
        is_study = study_manager_instance.run_study(
            data_1min_cleaned_is_slice=data_1min_cleaned_is_slice, 
            objective_evaluator_class=objective_evaluator_class
        )
        if is_study:
            logger.info(f"{fold_log_prefix} IS study completed. Study name: {is_study.study_name}")
        else: 
            logger.error(f"{fold_log_prefix} IS study did not return a valid study object.")
            return None, None
    except Exception as e_is_study:
        logger.error(f"{fold_log_prefix} Error during IS study execution: {e_is_study}", exc_info=True)
        return None, None

    logger.info(f"{fold_log_prefix} Initializing ResultsAnalyzer.")
    results_analyzer_instance = results_analyzer_class(
        app_config=app_config,
        study=is_study, 
        objective_evaluator_class=objective_evaluator_class,
        strategy_name=strategy_name,
        strategy_config_dict=strategy_config_dict, 
        output_dir_fold=output_dir_fold,
        pair_symbol=pair_symbol, 
        symbol_info_data=symbol_info_data 
    )
    
    num_top_trials_to_validate = app_config.global_config.optuna_settings.n_best_for_oos

    selected_is_trials = results_analyzer_instance._select_n_best_trials_from_pareto(n=num_top_trials_to_validate)

    if not selected_is_trials:
        logger.warning(f"{fold_log_prefix} No IS trials selected for OOS validation.")
        if is_study and is_study.best_trials: 
             best_is_trial_overall = is_study.best_trials[0] 
             logger.info(f"{fold_log_prefix} Returning best IS trial ({best_is_trial_overall.number}) due to no trials selected for OOS.")
             return best_is_trial_overall.params, {"IS_ONLY_METRICS": best_is_trial_overall.values, "message": "Used best IS trial as no OOS selection was made."}
        return None, None

    oos_validation_results: List[Dict[str, Any]] = []
    if data_1min_cleaned_oos_slice is not None and not data_1min_cleaned_oos_slice.empty:
        oos_validation_results = results_analyzer_instance.run_oos_validation_for_trials(
            selected_is_trials=selected_is_trials,
            data_1min_cleaned_oos_slice=data_1min_cleaned_oos_slice 
        )
        results_analyzer_instance.save_oos_validation_results(oos_validation_results)
    else:
        logger.warning(f"{fold_log_prefix} OOS data is empty or not provided. Skipping OOS validation.")
    
    final_selected_params = results_analyzer_instance.select_final_parameters_for_live(oos_validation_results)
    
    representative_oos_metrics: Optional[Dict[str, Any]] = None
    if oos_validation_results: 
        found_final_metrics = False
        if final_selected_params: 
            for res in oos_validation_results:
                if res.get("is_trial_params") == final_selected_params:
                    representative_oos_metrics = res.get("oos_metrics")
                    found_final_metrics = True
                    break
        
        if not found_final_metrics and oos_validation_results: 
            best_pnl_oos = -float('inf')
            temp_metrics = None
            for res in oos_validation_results:
                pnl = res.get("oos_metrics", {}).get("Total Net PnL USDC", -float('inf'))
                if isinstance(pnl, (int, float)) and np.isfinite(pnl) and pnl > best_pnl_oos:
                    best_pnl_oos = pnl
                    temp_metrics = res.get("oos_metrics")
            if temp_metrics:
                representative_oos_metrics = temp_metrics
    
    if not final_selected_params and is_study and is_study.best_trials:
        best_is_trial_overall = is_study.best_trials[0] 
        final_selected_params = best_is_trial_overall.params
        if representative_oos_metrics is None: 
            representative_oos_metrics = {"IS_ONLY_METRICS": best_is_trial_overall.values, "message": "Used best IS trial as no OOS selection or OOS metrics available."}
        logger.warning(f"{fold_log_prefix} No final parameters selected from OOS. Falling back to best IS trial: {best_is_trial_overall.number}")
    
    if final_selected_params:
        logger.info(f"{fold_log_prefix} Orchestration complete. Final selected params: {final_selected_params}")
    else:
        logger.error(f"{fold_log_prefix} Orchestration complete, but NO final parameters could be selected.")

    return final_selected_params, representative_oos_metrics
