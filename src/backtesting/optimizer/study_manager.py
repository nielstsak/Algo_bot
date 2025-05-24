import logging
from pathlib import Path
from typing import Any, Dict, Optional, Type, TYPE_CHECKING, List

import optuna # type: ignore
import pandas as pd

if TYPE_CHECKING:
    from src.config.definitions import AppConfig, OptunaSettings
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
        self.optuna_settings: 'OptunaSettings' = self.app_config.global_config.optuna_settings
        self.strategy_name = strategy_name
        self.strategy_config_dict = strategy_config_dict
        self.study_output_dir = study_output_dir
        self.pair_symbol = pair_symbol
        self.symbol_info_data = symbol_info_data
        
        self.log_prefix = f"[{self.strategy_name}][{self.pair_symbol}][Fold: {self.study_output_dir.name}]"
        logger.info(f"{self.log_prefix} StudyManager initialized.")

        self.study_output_dir.mkdir(parents=True, exist_ok=True)
        
        db_file_name = f"optuna_is_study_{self.strategy_name}_{self.pair_symbol}_{self.study_output_dir.name}.db"
        self.db_path_str = str((self.study_output_dir / db_file_name).resolve())
        self.storage_url = f"sqlite:///{self.db_path_str}"
        logger.info(f"{self.log_prefix} Optuna study storage URL: {self.storage_url}")


    def _create_sampler(self) -> optuna.samplers.BaseSampler:
        # Assurez-vous que self.optuna_settings.sampler_name existe
        if not hasattr(self.optuna_settings, 'sampler_name') or not self.optuna_settings.sampler_name:
            logger.error(f"{self.log_prefix} 'sampler_name' manquant dans OptunaSettings. Utilisation de TPESampler par défaut.")
            return optuna.samplers.TPESampler()
            
        sampler_name = self.optuna_settings.sampler_name.lower()
        sampler_params = self.optuna_settings.sampler_params or {}
        logger.info(f"{self.log_prefix} Creating sampler: {sampler_name} with params: {sampler_params}")
        if sampler_name == 'tpesampler':
            return optuna.samplers.TPESampler(**sampler_params)
        elif sampler_name == 'nsgaiisampler':
            return optuna.samplers.NSGAIISampler(**sampler_params)
        elif sampler_name == 'cmaessampler':
            return optuna.samplers.CmaEsSampler(**sampler_params)
        elif sampler_name == 'randomsampler':
            return optuna.samplers.RandomSampler(**sampler_params)
        else:
            logger.warning(f"{self.log_prefix} Unknown sampler '{sampler_name}'. Defaulting to TPESampler.")
            return optuna.samplers.TPESampler()

    def _create_pruner(self) -> optuna.pruners.BasePruner:
        if not hasattr(self.optuna_settings, 'pruner_name') or not self.optuna_settings.pruner_name:
            logger.error(f"{self.log_prefix} 'pruner_name' manquant dans OptunaSettings. Utilisation de MedianPruner par défaut.")
            return optuna.pruners.MedianPruner()

        pruner_name = self.optuna_settings.pruner_name.lower()
        pruner_params = self.optuna_settings.pruner_params or {}
        logger.info(f"{self.log_prefix} Creating pruner: {pruner_name} with params: {pruner_params}")
        if pruner_name == 'medianpruner':
            return optuna.pruners.MedianPruner(**pruner_params)
        elif pruner_name == 'hyperbandpruner':
            return optuna.pruners.HyperbandPruner(**pruner_params)
        elif pruner_name == 'nopruner':
            return optuna.pruners.NopPruner()
        elif pruner_name == 'patientpruner':
            try:
                return optuna.pruners.PatientPruner(**pruner_params)
            except TypeError as e:
                logger.warning(f"{self.log_prefix} PatientPruner creation failed with params {pruner_params} (may require a trial): {e}. Falling back to MedianPruner.")
                return optuna.pruners.MedianPruner()
        elif pruner_name == 'successivehalvingpruner':
            return optuna.pruners.SuccessiveHalvingPruner(**pruner_params)
        else:
            logger.warning(f"{self.log_prefix} Unknown pruner '{pruner_name}'. Defaulting to MedianPruner.")
            return optuna.pruners.MedianPruner()

    def run_study(self,
                  data_1min_cleaned_is_slice: pd.DataFrame, 
                  objective_evaluator_class: Type['ObjectiveEvaluator']
                  ) -> optuna.Study:
        
        study_name = f"{self.strategy_name}_{self.pair_symbol}_{self.study_output_dir.name}_is_opt"
        logger.info(f"{self.log_prefix} Preparing to run/load Optuna study: '{study_name}'")

        objectives_names = self.optuna_settings.objectives_names
        objectives_directions = self.optuna_settings.objectives_directions

        if len(objectives_names) != len(objectives_directions):
            logger.error(f"{self.log_prefix} Mismatch between objectives_names ({len(objectives_names)}) and objectives_directions ({len(objectives_directions)}). Defaulting to single PNL objective.")
            objectives_names = ["Total Net PnL USDC"]
            objectives_directions = ["maximize"]
        
        for direction in objectives_directions:
            if direction not in ["maximize", "minimize"]:
                logger.error(f"{self.log_prefix} Invalid objective direction '{direction}'. Defaulting to 'maximize' for all.")
                objectives_directions = ["maximize"] * len(objectives_names)
                break
        
        study: optuna.Study
        try:
            study = optuna.create_study(
                study_name=study_name,
                storage=self.storage_url,
                sampler=self._create_sampler(),
                pruner=self._create_pruner(),
                directions=objectives_directions,
                load_if_exists=True
            )
            logger.info(f"{self.log_prefix} Optuna study '{study.study_name}' created/loaded successfully from {self.storage_url}.")
        except Exception as e_create:
            logger.critical(f"{self.log_prefix} Failed to create or load Optuna study '{study_name}' at '{self.storage_url}': {e_create}", exc_info=True)
            raise

        optuna_objectives_config_for_evaluator = {
            'objectives_names': objectives_names,
            'objectives_directions': objectives_directions
        }

        objective_instance = objective_evaluator_class(
            strategy_name=self.strategy_name,
            strategy_config_dict=self.strategy_config_dict,
            df_enriched_slice=data_1min_cleaned_is_slice, # Nom d'argument corrigé
            simulation_settings=self.app_config.global_config.simulation_defaults.__dict__,
            optuna_objectives_config=optuna_objectives_config_for_evaluator,
            pair_symbol=self.pair_symbol,
            symbol_info_data=self.symbol_info_data,
            app_config=self.app_config,
            is_oos_eval=False,
            is_trial_number_for_oos_log=None
        )

        n_trials_config = self.optuna_settings.n_trials
        n_jobs_config = self.optuna_settings.n_jobs
        
        if n_jobs_config == 0:
            logger.warning(f"{self.log_prefix} Optuna n_jobs is 0, which is invalid. Defaulting to 1 (sequential).")
            n_jobs_config = 1

        completed_trials_count = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        remaining_trials = n_trials_config - completed_trials_count

        if remaining_trials <= 0 and completed_trials_count >= n_trials_config:
            logger.info(f"{self.log_prefix} Study '{study.study_name}' already has {completed_trials_count} completed trials (target: {n_trials_config}). No new trials will be run for this fold.")
        else:
            if remaining_trials < 0 :
                logger.warning(f"{self.log_prefix} remaining_trials is negative ({remaining_trials}). This is unexpected. Setting to 0.")
                remaining_trials = 0
            
            logger.info(f"{self.log_prefix} Starting Optuna IS optimization for '{study.study_name}'. "
                        f"Target total trials: {n_trials_config}. Completed: {completed_trials_count}. Remaining: {remaining_trials}. "
                        f"Using {n_jobs_config} job(s).")
            try:
                study.optimize(
                    objective_instance,
                    n_trials=remaining_trials,
                    n_jobs=n_jobs_config,
                    gc_after_trial=True,
                    callbacks=[]
                )
            except Exception as e_optimize:
                logger.error(f"{self.log_prefix} Error during Optuna study.optimize for '{study.study_name}': {e_optimize}", exc_info=True)
        
        final_completed_trials = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        logger.info(f"{self.log_prefix} Optuna IS optimization finished for '{study.study_name}'. Total completed trials in study: {final_completed_trials}.")
        return study
