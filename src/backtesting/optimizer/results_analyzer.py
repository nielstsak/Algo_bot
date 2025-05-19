import logging
import json
from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple, TYPE_CHECKING, Union

import optuna
import pandas as pd
import numpy as np
import math 

if TYPE_CHECKING:
    from src.config.loader import AppConfig, OptunaSettings 
    from src.backtesting.optimizer.objective_evaluator import ObjectiveEvaluator

logger = logging.getLogger(__name__)

class ResultsAnalyzer:
    def __init__(self,
                 app_config: 'AppConfig',
                 study: optuna.Study, 
                 objective_evaluator_class: type['ObjectiveEvaluator'],
                 strategy_name: str,
                 strategy_config_dict: Dict[str, Any],
                 output_dir_fold: Path,
                 pair_symbol: str, 
                 symbol_info_data: Dict[str, Any] 
                 ):
        self.app_config = app_config
        self.optuna_settings_obj = self.app_config.global_config.optuna_settings
        self.study = study
        self.objective_evaluator_class = objective_evaluator_class
        self.strategy_name = strategy_name
        self.strategy_config_dict = strategy_config_dict
        self.output_dir_fold = output_dir_fold
        self.pair_symbol = pair_symbol 
        self.symbol_info_data = symbol_info_data 

    def _select_n_best_trials_from_pareto(self, n: int = 10) -> List[optuna.trial.FrozenTrial]:
        """
        Selects the n best trials from the Pareto front of a multi-objective study.
        Handles different selection strategies based on Optuna settings.
        Filters trials with non-finite or missing objective values.
        Applies a PNL threshold if configured.
        Sorts trials based on a composite score or a primary metric (e.g., PNL).

        Args:
            n (int): The number of best trials to select.

        Returns:
            List[optuna.trial.FrozenTrial]: A list of the selected best trials.
                                            Returns an empty list if no valid trials are found.
        """
        if not self.study.best_trials: 
            logger.warning(f"[{self.strategy_name}][{self.pair_symbol}] No best_trials (Pareto front) found in the study. Cannot select trials.")
            return []

        pareto_trials = self.study.best_trials
        
        objectives_names = self.optuna_settings_obj.objectives_names
        objectives_directions = self.optuna_settings_obj.objectives_directions
        selection_strategy = self.optuna_settings_obj.pareto_selection_strategy
        selection_weights = self.optuna_settings_obj.pareto_selection_weights
        pnl_threshold = self.optuna_settings_obj.pareto_selection_pnl_threshold

        # Filter out trials with None values or non-finite values (inf, -inf, nan)
        valid_trials = []
        for t in pareto_trials:
            if t.values is not None and all(isinstance(v, (float, int)) and np.isfinite(v) for v in t.values):
                valid_trials.append(t)
            else:
                logger.debug(f"[{self.strategy_name}][{self.pair_symbol}] Trial {t.number} excluded from Pareto selection due to invalid objective values: {t.values}")
        
        if not valid_trials:
            logger.warning(f"[{self.strategy_name}][{self.pair_symbol}] No trials with valid (finite, non-null) objective values found in Pareto front.")
            return []

        # Apply PNL threshold if configured
        if pnl_threshold is not None:
            try:
                # Assuming "Total Net PnL USDC" is one of the objectives
                pnl_metric_index = objectives_names.index("Total Net PnL USDC")
                trials_above_threshold = [
                    t for t in valid_trials 
                    if t.values and len(t.values) > pnl_metric_index and t.values[pnl_metric_index] >= pnl_threshold
                ]
                if trials_above_threshold:
                    logger.info(f"[{self.strategy_name}][{self.pair_symbol}] Applied PNL threshold of {pnl_threshold}. {len(trials_above_threshold)} trials remaining from {len(valid_trials)} valid Pareto trials.")
                    valid_trials = trials_above_threshold
                else:
                    logger.warning(f"[{self.strategy_name}][{self.pair_symbol}] No trials met PNL threshold of {pnl_threshold}. Proceeding with all {len(valid_trials)} valid Pareto trials.")
            except (ValueError, IndexError):
                logger.warning(f"[{self.strategy_name}][{self.pair_symbol}] 'Total Net PnL USDC' not found in objectives_names or index issue. PNL threshold not applied.")
                pass # PNL metric not found, skip thresholding


        # Sort based on selection strategy
        if selection_strategy == "SCORE_COMPOSITE" and selection_weights and objectives_names:
            scored_trials = []
            for trial in valid_trials:
                score = 0.0
                is_valid_for_scoring = True
                for i, obj_name in enumerate(objectives_names):
                    weight = selection_weights.get(obj_name, 0.0)
                    if weight != 0.0: # Only consider objectives with non-zero weights
                        if trial.values and i < len(trial.values) and isinstance(trial.values[i], (float, int)) and np.isfinite(trial.values[i]):
                            score += trial.values[i] * weight
                        else:
                            # This trial cannot be scored if a weighted objective value is missing/invalid
                            is_valid_for_scoring = False
                            logger.debug(f"Trial {trial.number} invalid for composite scoring: objective {obj_name} value missing or invalid.")
                            break 
                if is_valid_for_scoring:
                    scored_trials.append({'trial': trial, 'score': score})
            
            if scored_trials:
                scored_trials.sort(key=lambda x: x['score'], reverse=True) # Higher score is better
                valid_trials = [st['trial'] for st in scored_trials]
                logger.info(f"[{self.strategy_name}][{self.pair_symbol}] Sorted {len(valid_trials)} trials by SCORE_COMPOSITE.")
            else: 
                # Fallback if no trials could be scored (e.g., all had missing values for weighted objectives)
                logger.warning(f"[{self.strategy_name}][{self.pair_symbol}] No trials could be scored with SCORE_COMPOSITE. Falling back to primary objective sort.")
                # Fallback: sort by the first objective's direction
                if objectives_directions:
                     sort_descending_fallback = objectives_directions[0] == "maximize"
                     valid_trials.sort(key=lambda t: t.values[0] if t.values else (-float('inf') if sort_descending_fallback else float('inf')), 
                                       reverse=sort_descending_fallback)
                else: # Should not happen if objectives_names is present
                     valid_trials.sort(key=lambda t: t.values[0] if t.values else -float('inf'), reverse=True)
        
        else: # Default or PNL_MAX or other simple strategies
            # Default to sorting by "Total Net PnL USDC" if available, otherwise first objective
            try:
                metric_index_for_sort = objectives_names.index("Total Net PnL USDC")
                direction_for_sort = objectives_directions[metric_index_for_sort]
            except (ValueError, IndexError): # Fallback to the first objective
                logger.debug(f"[{self.strategy_name}][{self.pair_symbol}] 'Total Net PnL USDC' not found for default sort. Using first objective.")
                metric_index_for_sort = 0
                direction_for_sort = objectives_directions[0] if objectives_directions else "maximize" # Default direction if not specified
            
            sort_descending = direction_for_sort == "maximize"
            valid_trials.sort(key=lambda t: t.values[metric_index_for_sort] if t.values and len(t.values) > metric_index_for_sort else (-float('inf') if sort_descending else float('inf')), 
                              reverse=sort_descending)
            logger.info(f"[{self.strategy_name}][{self.pair_symbol}] Sorted {len(valid_trials)} trials by objective '{objectives_names[metric_index_for_sort]}'.")

        selected_trials = valid_trials[:n]
        logger.info(f"[{self.strategy_name}][{self.pair_symbol}] Selected {len(selected_trials)} best trials for OOS validation from Pareto front.")
        return selected_trials

    def run_oos_validation_for_trials(self,
                                      selected_is_trials: List[optuna.trial.FrozenTrial],
                                      data_1min_cleaned_oos_slice: pd.DataFrame 
                                      ) -> List[Dict[str, Any]]:
        """
        Runs Out-of-Sample (OOS) validation for a list of selected In-Sample (IS) trials.

        Args:
            selected_is_trials: A list of FrozenTrial objects from IS optimization.
            data_1min_cleaned_oos_slice: DataFrame containing the OOS period data.

        Returns:
            A list of dictionaries, each containing IS trial info and its OOS metrics.
        """
        oos_results_list = []
        if not selected_is_trials:
            logger.warning(f"[{self.strategy_name}][{self.pair_symbol}] No In-Sample trials provided for OOS validation.")
            return oos_results_list
        if data_1min_cleaned_oos_slice.empty:
            logger.warning(f"[{self.strategy_name}][{self.pair_symbol}] OOS data slice is empty. Skipping OOS validation.")
            return oos_results_list

        optuna_objectives_config_for_evaluator = {
            'objectives_names': self.optuna_settings_obj.objectives_names,
            'objectives_directions': self.optuna_settings_obj.objectives_directions
        }
        
        num_objectives = len(self.optuna_settings_obj.objectives_directions)

        for i, is_trial in enumerate(selected_is_trials):
            logger.info(f"[{self.strategy_name}][{self.pair_symbol}] Running OOS validation for IS Trial {is_trial.number} ({i+1}/{len(selected_is_trials)}). Params: {is_trial.params}")
            
            # Instantiate ObjectiveEvaluator for OOS run
            # MODIFICATION: Pass is_trial.number for logging purposes
            oos_objective_instance = self.objective_evaluator_class(
                strategy_name=self.strategy_name,
                strategy_config_dict=self.strategy_config_dict,
                data_1min_cleaned_slice=data_1min_cleaned_oos_slice, # Use OOS data
                simulation_settings=self.app_config.global_config.simulation_defaults.__dict__,
                optuna_objectives_config=optuna_objectives_config_for_evaluator,
                pair_symbol=self.pair_symbol,
                symbol_info_data=self.symbol_info_data,
                is_oos_eval=True, # Indicate this is an OOS evaluation
                app_config=self.app_config, # Pass the full app_config
                is_trial_number_for_oos_log=is_trial.number # NOUVEAU: Passer le numéro du trial IS
            )
            
            # MODIFICATION: params for create_trial should be is_trial.params directly
            # This ensures params.keys() matches distributions.keys()
            # The _trial_id_for_oos was causing the "Inconsistent parameters" error.
            # The ObjectiveEvaluator now gets the IS trial number via is_trial_number_for_oos_log.
            
            safe_placeholder_values = [0.0] * num_objectives
            
            dummy_oos_trial = optuna.trial.create_trial(
                params=is_trial.params,  # UTILISER is_trial.params directement
                distributions=is_trial.distributions, 
                values=safe_placeholder_values 
            )
            
            try:
                oos_metrics_tuple = oos_objective_instance(dummy_oos_trial) # Call the evaluator
                logger.debug(f"[{self.strategy_name}][{self.pair_symbol}] OOS metrics for IS Trial {is_trial.number}: {oos_metrics_tuple}")
            except optuna.exceptions.TrialPruned as e_pruned_oos:
                logger.warning(f"[{self.strategy_name}][{self.pair_symbol}] OOS evaluation pruned for IS Trial {is_trial.number}: {e_pruned_oos}")
                oos_metrics_tuple = tuple([-float('inf') if direction == "maximize" else float('inf')
                                           for direction in self.optuna_settings_obj.objectives_directions])
            except Exception as e_oos_eval:
                logger.error(f"[{self.strategy_name}][{self.pair_symbol}] Error during OOS evaluation for IS Trial {is_trial.number}: {e_oos_eval}", exc_info=True)
                oos_metrics_tuple = tuple([-float('inf') if direction == "maximize" else float('inf')
                                           for direction in self.optuna_settings_obj.objectives_directions])

            oos_metrics_dict = {}
            for name, value in zip(self.optuna_settings_obj.objectives_names, oos_metrics_tuple):
                oos_metrics_dict[name] = value
            
            oos_results_list.append({
                "is_trial_number": is_trial.number,
                "is_trial_params": is_trial.params,
                "is_trial_values": is_trial.values, 
                "oos_metrics": oos_metrics_dict    
            })
            
        logger.info(f"[{self.strategy_name}][{self.pair_symbol}] Completed OOS validation for {len(selected_is_trials)} trials.")
        return oos_results_list

    def save_oos_validation_results(self, oos_results: List[Dict[str, Any]]):
        """Saves the OOS validation summary to a JSON file."""
        if not oos_results:
            logger.info(f"[{self.strategy_name}][{self.pair_symbol}] No OOS results to save.")
            return

        file_path = self.output_dir_fold / "oos_validation_summary_TOP_N_TRIALS.json"
        try:
            serializable_results = []
            for res_entry in oos_results:
                entry_copy = res_entry.copy()
                if 'is_trial_values' in entry_copy and entry_copy['is_trial_values'] is not None:
                    entry_copy['is_trial_values'] = [
                        (float(v) if isinstance(v, (int, float)) and np.isfinite(v) else None) 
                        for v in entry_copy['is_trial_values']
                    ]
                
                if 'oos_metrics' in entry_copy and isinstance(entry_copy['oos_metrics'], dict):
                    entry_copy['oos_metrics'] = {
                        k: (float(v) if isinstance(v, (int, float)) and np.isfinite(v) else str(v) if not pd.isna(v) else None) 
                        for k, v in entry_copy['oos_metrics'].items()
                    }
                serializable_results.append(entry_copy)

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(serializable_results, f, indent=4, default=str) 
            logger.info(f"[{self.strategy_name}][{self.pair_symbol}] OOS validation summary saved to: {file_path}")
        except Exception as e:
            logger.error(f"[{self.strategy_name}][{self.pair_symbol}] Failed to save OOS validation summary: {e}", exc_info=True)

    def select_final_parameters_for_live(self, oos_results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Selects the final set of parameters for live trading based on OOS results.
        Currently, selects based on the best "Total Net PnL USDC" in OOS.
        """
        if not oos_results:
            logger.warning(f"[{self.strategy_name}][{self.pair_symbol}] No OOS results provided to select final parameters.")
            return None

        best_oos_trial_info = None
        best_oos_pnl = -float('inf') 

        for result_entry in oos_results:
            oos_metrics = result_entry.get("oos_metrics", {})
            current_pnl = oos_metrics.get("Total Net PnL USDC", -float('inf')) 
            
            if isinstance(current_pnl, (int,float)) and np.isfinite(current_pnl):
                if current_pnl > best_oos_pnl:
                    best_oos_pnl = current_pnl
                    best_oos_trial_info = result_entry
            else:
                logger.debug(f"[{self.strategy_name}][{self.pair_symbol}] Skipping OOS result for IS trial {result_entry.get('is_trial_number')} due to invalid PNL: {current_pnl}")
        
        if best_oos_trial_info:
            final_params = best_oos_trial_info.get("is_trial_params")
            logger.info(f"[{self.strategy_name}][{self.pair_symbol}] Final parameters selected from IS Trial {best_oos_trial_info.get('is_trial_number')} with OOS PNL {best_oos_pnl:.2f}: {final_params}")
            return final_params
        else:
            logger.warning(f"[{self.strategy_name}][{self.pair_symbol}] Could not select final parameters. No OOS trial had valid and positive PNL, or OOS results were empty/invalid.")
            return None
