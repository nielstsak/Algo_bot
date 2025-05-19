import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type, TYPE_CHECKING

import optuna
import pandas as pd

if TYPE_CHECKING:
    from src.config.loader import AppConfig, OptunaSettings # OptunaSettings is already part of AppConfig.global_config
    from src.backtesting.optimizer.objective_evaluator import ObjectiveEvaluator

logger = logging.getLogger(__name__)

class StudyManager:
    def __init__(self,
                 app_config: 'AppConfig',
                 strategy_name: str,
                 strategy_config_dict: Dict[str, Any], 
                 study_output_dir: Path,
                 pair_symbol: str, 
                 symbol_info_data: Dict[str, Any]
                 ):
        self.app_config = app_config
        # OptunaSettings is accessed via app_config.global_config.optuna_settings
        self.optuna_settings_obj = self.app_config.global_config.optuna_settings
        self.strategy_name = strategy_name
        self.strategy_config_dict = strategy_config_dict # This contains params_space
        self.study_output_dir = study_output_dir
        self.pair_symbol = pair_symbol 
        self.symbol_info_data = symbol_info_data
        
        self.study_output_dir.mkdir(parents=True, exist_ok=True)
        # Ensure the path for the database is correctly resolved and directory exists
        db_file_name = f"is_opt_{self.strategy_name}_{self.pair_symbol}.db" # More specific DB name
        self.db_path_str = str((self.study_output_dir / db_file_name).resolve())
        self.storage_url = f"sqlite:///{self.db_path_str}"
        logger.info(f"Optuna study storage URL for {strategy_name}/{pair_symbol}: {self.storage_url}")


    def _create_sampler(self) -> optuna.samplers.BaseSampler:
        sampler_name = self.optuna_settings_obj.sampler.lower()
        # Add more samplers as needed
        if sampler_name == 'nsgaiisampler':
            return optuna.samplers.NSGAIISampler()
        elif sampler_name == 'tpesampler':
            return optuna.samplers.TPESampler()
        elif sampler_name == 'cmaessampler':
            return optuna.samplers.CmaEsSampler()
        # Add other samplers like BoTorchSampler if BoTorch is installed and configured
        # elif sampler_name == 'botorchsampler' and optuna.integration.BoTorchSampler is not None:
        #     return optuna.integration.BoTorchSampler()
        else:
            logger.warning(f"Unknown sampler '{sampler_name}'. Defaulting to NSGAIISampler.")
            return optuna.samplers.NSGAIISampler()

    def _create_pruner(self) -> optuna.pruners.BasePruner:
        pruner_name = self.optuna_settings_obj.pruner.lower()
        # Add more pruners as needed
        if pruner_name == 'medianpruner':
            return optuna.pruners.MedianPruner()
        elif pruner_name == 'hyperbandpruner':
            # HyperbandPruner might require specific setup for multi-objective
            return optuna.pruners.HyperbandPruner()
        elif pruner_name == 'nopruner':
            return optuna.pruners.NopPruner()
        else:
            logger.warning(f"Unknown pruner '{pruner_name}'. Defaulting to MedianPruner.")
            return optuna.pruners.MedianPruner()

    def run_study(self,
                  data_1min_cleaned_is_slice: pd.DataFrame, # This is now the enriched IS slice
                  objective_evaluator_class: Type['ObjectiveEvaluator'] 
                  ) -> optuna.Study:

        # Construct a unique study name
        study_name_parts = [
            self.strategy_name,
            self.pair_symbol, 
            self.study_output_dir.name, # Typically includes a WFO fold identifier
            "is_opt" # Indicates In-Sample optimization
        ]
        study_name = "_".join(filter(None, study_name_parts))
        
        objectives_names = self.optuna_settings_obj.objectives_names
        objectives_directions = self.optuna_settings_obj.objectives_directions

        if len(objectives_names) != len(objectives_directions):
            logger.error("Mismatch between objectives_names and objectives_directions in Optuna settings. Defaulting to single PNL objective.")
            # Fallback to a sensible default if configuration is inconsistent
            objectives_names = ["Total Net PnL USDC"]
            objectives_directions = ["maximize"]
        
        try:
            study = optuna.create_study(
                study_name=study_name,
                storage=self.storage_url,
                sampler=self._create_sampler(),
                pruner=self._create_pruner(),
                directions=objectives_directions, # Pass list of directions for multi-objective
                load_if_exists=True # Allows resuming studies
            )
        except Exception as e_create:
            logger.critical(f"Failed to create or load Optuna study '{study_name}' at '{self.storage_url}': {e_create}", exc_info=True)
            raise # Re-raise the exception as this is critical

        optuna_objectives_config_for_evaluator = {
            'objectives_names': objectives_names,
            'objectives_directions': objectives_directions
        }

        # Instantiate ObjectiveEvaluator, now passing app_config
        objective_instance = objective_evaluator_class(
            strategy_name=self.strategy_name,
            strategy_config_dict=self.strategy_config_dict,
            data_1min_cleaned_slice=data_1min_cleaned_is_slice, # Pass the enriched IS data
            simulation_settings=self.app_config.global_config.simulation_defaults.__dict__,
            optuna_objectives_config=optuna_objectives_config_for_evaluator,
            pair_symbol=self.pair_symbol,
            symbol_info_data=self.symbol_info_data,
            is_oos_eval=False, # This is for In-Sample optimization
            app_config=self.app_config # Pass the full app_config
        )

        n_trials = self.optuna_settings_obj.n_trials
        n_jobs = self.optuna_settings_obj.n_jobs
        if n_jobs == 0 or n_jobs < -1: # Optuna interprets 0 as invalid, -1 means use all CPUs
            logger.warning(f"Invalid n_jobs ({n_jobs}) in Optuna settings. Defaulting to 1.")
            n_jobs = 1 
        
        # Check completed trials to support resuming
        completed_trials_count = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        remaining_trials = n_trials - completed_trials_count

        if remaining_trials <= 0 and completed_trials_count >= n_trials :
            logger.info(f"Study '{study_name}' already has {completed_trials_count} completed trials (target: {n_trials}). No new trials will be run for this fold.")
        else:
            logger.info(f"Starting Optuna IS optimization for {study_name} ({remaining_trials} new trials out of {n_trials} total, using {n_jobs} job(s)).")
            try:
                study.optimize(
                    objective_instance, # The callable ObjectiveEvaluator instance
                    n_trials=remaining_trials, # Run only the remaining number of trials
                    n_jobs=n_jobs,
                    gc_after_trial=True, # Helps manage memory
                    callbacks=[] # Add any Optuna callbacks if needed, e.g., for logging
                )
            except Exception as e_optimize:
                logger.error(f"Error during Optuna study.optimize for {study_name}: {e_optimize}", exc_info=True)
                # Depending on the error, you might want to handle it or re-raise
                # For now, we log and let the study object be returned as is.

        final_completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        logger.info(f"Optuna IS optimization finished for {study_name}. Total completed trials: {final_completed_trials}.")
        return study
