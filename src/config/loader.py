import json
import logging
import os
import sys
import dataclasses
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Dict, Optional, Any, List, Union, Type, get_origin, get_args
import typing # Ajout de l'importation explicite de typing
from pathlib import Path
from dotenv import load_dotenv

try:
    from ..utils.logging_setup import setup_logging
except ImportError:
    try:
        from src.utils.logging_setup import setup_logging
    except ImportError:
        def setup_logging(*args, **kwargs):
            logging.basicConfig(level=logging.WARNING, format='%(asctime)s - FALLBACK_SETUP - %(levelname)s - %(message)s')
            logging.getLogger(__name__).warning("Logging setup skipped due to import error in loader.py.")
            pass

logger = logging.getLogger(__name__)

try:
    from .definitions import (
        PathsConfig, LoggingConfig, SimulationDefaults, WfoSettings, OptunaSettings,
        GlobalConfig, SourceDetails, AssetsAndTimeframes, HistoricalPeriod,
        FetchingOptions, DataConfig, ParamDetail, StrategyParams, StrategiesConfig,
        GlobalLiveSettings, OverrideRiskSettings, StrategyDeployment,
        LiveLoggingConfig, LiveFetchConfig, LiveConfig, ApiKeys, AppConfig
    )
except ImportError:
    logger.error("Fallback: Could not import dataclass definitions from .definitions. Using local fallback definitions.", exc_info=True)
    # Fallback definitions (comme dans votre version originale)
    @dataclass
    class PathsConfig:
        data_historical_raw: str
        data_historical_processed_cleaned: str 
        logs_backtest_optimization: str
        logs_live: str
        results: str
        data_live_raw: str
        data_live_processed: str
        live_state: str

    @dataclass
    class LoggingConfig:
        level: str
        format: str
        log_to_file: bool
        log_filename_global: str
        log_filename_live: Optional[str] = None

    @dataclass
    class SimulationDefaults:
        initial_capital: float
        margin_leverage: int
        transaction_fee_pct: float
        slippage_pct: float

    @dataclass
    class WfoSettings:
        n_splits: int
        oos_percent: int
        metric_to_optimize: str
        optimization_direction: str

    @dataclass
    class OptunaSettings:
        n_trials: int
        sampler: str
        pruner: str
        storage: Optional[str] = None
        n_jobs: int = 1
        objectives_names: List[str] = field(default_factory=lambda: ["Total Net PnL USDC", "Sharpe Ratio"])
        objectives_directions: List[str] = field(default_factory=lambda: ["maximize", "maximize"])
        pareto_selection_strategy: Optional[str] = "PNL_MAX"
        pareto_selection_weights: Optional[Dict[str, float]] = None
        pareto_selection_pnl_threshold: Optional[float] = None
        n_best_for_oos: int = 10

    @dataclass
    class GlobalConfig:
        project_name: str
        paths: PathsConfig
        logging: LoggingConfig
        simulation_defaults: SimulationDefaults
        wfo_settings: WfoSettings
        optuna_settings: OptunaSettings

    @dataclass
    class SourceDetails:
        exchange: str
        asset_type: str

    @dataclass
    class AssetsAndTimeframes:
        pairs: List[str]
        timeframes: List[str]

    @dataclass
    class HistoricalPeriod:
        start_date: str
        end_date: Optional[str] = None

    @dataclass
    class FetchingOptions:
        max_workers: int
        batch_size: int

    @dataclass
    class DataConfig:
        source_details: SourceDetails
        assets_and_timeframes: AssetsAndTimeframes
        historical_period: HistoricalPeriod
        fetching_options: FetchingOptions

    @dataclass
    class ParamDetail:
        type: str
        low: Optional[Union[float, int]] = None
        high: Optional[Union[float, int]] = None
        step: Optional[Union[float, int]] = None
        choices: Optional[List[Any]] = None

    @dataclass
    class StrategyParams:
        active_for_optimization: bool
        script_reference: str
        class_name: str
        params_space: Dict[str, ParamDetail]

    @dataclass
    class StrategiesConfig:
        strategies: Dict[str, StrategyParams]
    
    @dataclass
    class GlobalLiveSettings:
        run_live_trading: bool
        account_type: str
        max_concurrent_strategies: int
        default_position_sizing_pct_capital: float
        global_risk_limit_pct_capital: float
        is_testnet: bool = False

    @dataclass
    class OverrideRiskSettings:
        position_sizing_pct_capital: Optional[float] = None
        max_loss_per_trade_pct: Optional[float] = None

    @dataclass
    class StrategyDeployment:
        active: bool
        strategy_id: str
        results_config_path: str
        override_risk_settings: Optional[OverrideRiskSettings] = None

    @dataclass
    class LiveLoggingConfig:
        level: str = "INFO"
        log_to_file: bool = True
        log_filename_live: str = "live_trading_specific.log"

    @dataclass
    class LiveFetchConfig:
        crypto_pairs: List[str]
        intervals: List[str]
        limit_init_history: int
        limit_per_fetch: int
        max_retries: int
        retry_backoff: float

    @dataclass
    class LiveConfig:
        global_live_settings: GlobalLiveSettings
        strategy_deployments: List[StrategyDeployment]
        live_fetch: LiveFetchConfig
        live_logging: LiveLoggingConfig

    @dataclass
    class ApiKeys:
        binance_api_key: Optional[str] = None
        binance_secret_key: Optional[str] = None

    @dataclass
    class AppConfig:
        global_config: GlobalConfig
        data_config: DataConfig
        strategies_config: StrategiesConfig
        live_config: LiveConfig
        api_keys: ApiKeys
        project_root: str = field(init=False)


def _load_json(file_path: str) -> Dict[str, Any]:
    if not os.path.exists(file_path):
        logger.error(f"Configuration file not found: {file_path}")
        raise FileNotFoundError(f"Configuration file not found: {file_path}")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from file: {file_path} - {e}")
        raise
    except Exception as e:
        logger.error(f"An unexpected error occurred while reading {file_path}: {e}")
        raise

def _is_optional_union(field_type: Any) -> bool:
    return get_origin(field_type) is Union and type(None) in get_args(field_type)

def _get_non_none_type_from_optional_union(field_type: Any) -> Any:
    if _is_optional_union(field_type):
        args = get_args(field_type)
        return next(arg for arg in args if arg is not type(None))
    return field_type

def _create_dataclass_from_dict(dataclass_type: Type[Any], data: Dict[str, Any]) -> Any:
    if not is_dataclass(dataclass_type):
        return data

    field_info = {f.name: f for f in fields(dataclass_type)}
    init_kwargs: Dict[str, Any] = {}

    for name_from_json, value_from_json in data.items():
        if name_from_json not in field_info:
            logger.debug(f"Key '{name_from_json}' in JSON data is not a field in dataclass '{dataclass_type.__name__}'. Skipping.")
            continue

        field_obj = field_info[name_from_json]
        actual_field_type = _get_non_none_type_from_optional_union(field_obj.type)
        origin_type = get_origin(actual_field_type)
        type_args = get_args(actual_field_type)
        
        is_generic_alias = isinstance(actual_field_type, typing._GenericAlias) if hasattr(typing, '_GenericAlias') else False 
        if not is_generic_alias and hasattr(actual_field_type, "__origin__"): 
             is_generic_alias = True


        if value_from_json is None and _is_optional_union(field_obj.type):
            init_kwargs[name_from_json] = None # Correction: utiliser name_from_json
        elif is_dataclass(actual_field_type) and isinstance(value_from_json, dict):
            init_kwargs[name_from_json] = _create_dataclass_from_dict(actual_field_type, value_from_json)
        elif origin_type is list and type_args and len(type_args) == 1 and is_dataclass(type_args[0]) and isinstance(value_from_json, list):
            list_item_dataclass_type = type_args[0]
            init_kwargs[name_from_json] = [_create_dataclass_from_dict(list_item_dataclass_type, item) for item in value_from_json if isinstance(item, dict)]
        elif origin_type is dict and type_args and len(type_args) == 2 and is_dataclass(type_args[1]) and isinstance(value_from_json, dict):
            dict_value_dataclass_type = type_args[1]
            processed_dict = {}
            for k, v_item in value_from_json.items():
                if isinstance(v_item, dict):
                    try:
                        item_instance = _create_dataclass_from_dict(dict_value_dataclass_type, v_item)
                        if item_instance is not None: processed_dict[k] = item_instance
                    except Exception as inner_e:
                        logger.error(f"Error creating inner dataclass item for key '{k}' type {dict_value_dataclass_type.__name__}: {inner_e}", exc_info=False)
                else:
                    logger.warning(f"Value for key '{k}' in dict field '{name_from_json}' is not a dict. Skipping.")
            init_kwargs[name_from_json] = processed_dict
        else:
            try:
                expected_base_type = origin_type if is_generic_alias else actual_field_type
                
                if isinstance(value_from_json, expected_base_type):
                    init_kwargs[name_from_json] = value_from_json
                elif expected_base_type == int and not isinstance(value_from_json, bool):
                    init_kwargs[name_from_json] = int(value_from_json)
                elif expected_base_type == float:
                    init_kwargs[name_from_json] = float(value_from_json)
                elif expected_base_type == str:
                    init_kwargs[name_from_json] = str(value_from_json)
                elif expected_base_type == bool and isinstance(value_from_json, (int, float)):
                    init_kwargs[name_from_json] = bool(value_from_json)
                else:
                    init_kwargs[name_from_json] = value_from_json
            except (ValueError, TypeError) as conv_err:
                logger.warning(f"Could not convert value '{value_from_json}' to type {actual_field_type} for field '{name_from_json}'. Using original value. Error: {conv_err}")
                init_kwargs[name_from_json] = value_from_json
    
    for field_name_req, field_obj_req in field_info.items():
        if field_name_req not in init_kwargs: 
            if field_obj_req.default is not dataclasses.MISSING:
                init_kwargs[field_name_req] = field_obj_req.default
            elif field_obj_req.default_factory is not dataclasses.MISSING:
                init_kwargs[field_name_req] = field_obj_req.default_factory()

    try:
        instance = dataclass_type(**init_kwargs)
        return instance
    except Exception as e:
        logger.error(f"Error instantiating dataclass {dataclass_type.__name__} with kwargs {list(init_kwargs.keys())}: {e}", exc_info=True)
        raise

def load_config(config_file_path: str) -> Dict[str, Any]:
    return _load_json(config_file_path)

def load_all_configs(project_root: Optional[str] = None) -> AppConfig:
    if project_root is None:
        try:
            project_root = str(Path(__file__).resolve().parent.parent.parent)
        except NameError: 
            project_root = str(Path('.').resolve())
            logger.info(f"__file__ not defined, project root detected as current directory: {project_root}")
    else:
        project_root = os.path.abspath(project_root)

    env_path = Path(project_root) / '.env'
    load_dotenv(dotenv_path=env_path)

    api_keys = ApiKeys(
        binance_api_key=os.getenv('BINANCE_API_KEY'),
        binance_secret_key=os.getenv('BINANCE_SECRET_KEY')
    )

    config_dir = Path(project_root) / 'config'
    global_config_path = config_dir / 'config_global.json'
    data_config_path = config_dir / 'config_data.json'
    strategies_config_path = config_dir / 'config_strategies.json'
    live_config_path = config_dir / 'config_live.json'

    global_config_dict = load_config(str(global_config_path))
    data_config_dict = load_config(str(data_config_path))
    strategies_config_dict_raw = load_config(str(strategies_config_path))
    live_config_dict = load_config(str(live_config_path))

    global_cfg = _create_dataclass_from_dict(GlobalConfig, global_config_dict)
    data_cfg = _create_dataclass_from_dict(DataConfig, data_config_dict)
    strategies_cfg = _create_dataclass_from_dict(StrategiesConfig, {"strategies": strategies_config_dict_raw})
    live_cfg = _create_dataclass_from_dict(LiveConfig, live_config_dict)

    if hasattr(global_cfg, 'paths') and global_cfg.paths:
        for path_field in fields(global_cfg.paths):
            field_name = path_field.name
            relative_path = getattr(global_cfg.paths, field_name)
            if isinstance(relative_path, str):
                absolute_path = (Path(project_root) / relative_path).resolve()
                setattr(global_cfg.paths, field_name, str(absolute_path))
                if any(k in field_name for k in ["log", "data", "results", "state"]):
                    try:
                        absolute_path.mkdir(parents=True, exist_ok=True)
                    except OSError as e:
                        logger.error(f"Failed dir creation {absolute_path}: {e}")
            else:
                logger.warning(f"Path value for '{field_name}' non-string: {relative_path}. Skipping resolution.")
    else:
        logger.warning("GlobalConfig.paths missing or empty. Path resolution skipped.")

    log_conf_obj = None
    root_log_level = logging.INFO
    log_dir_final_str = str(Path(project_root) / 'logs' / 'general_loader_logs')
    log_filename_final = 'loader_default.log'

    if hasattr(global_cfg, 'logging') and global_cfg.logging:
        log_conf_obj = global_cfg.logging
        log_dir_path_str_from_config = getattr(global_cfg.paths, 'logs_backtest_optimization', None) 
        if log_dir_path_str_from_config and isinstance(log_dir_path_str_from_config, str): 
            log_dir_final_str = log_dir_path_str_from_config
        
        log_filename_final = getattr(log_conf_obj, 'log_filename_global', 'global_run.log')
        root_log_level_str = getattr(log_conf_obj, 'level', 'INFO').upper()
        root_log_level = getattr(logging, root_log_level_str, logging.INFO)

    if log_conf_obj:
        try:
            Path(log_dir_final_str).mkdir(parents=True, exist_ok=True)
            setup_logging(log_conf_obj, log_dir_final_str, log_filename_final, root_log_level)
        except Exception as e_log_setup:
            logging.basicConfig(level=logging.WARNING, format='%(asctime)s - LOADER_LOG_SETUP_FAIL - %(levelname)s - %(message)s')
            logger.error(f"Failed during logging setup in loader: {e_log_setup}. Basic logging active for loader.", exc_info=True)
    else:
        logging.basicConfig(level=root_log_level, format='%(asctime)s - LOADER_BASIC_CFG - %(levelname)s - %(message)s')
        logger.warning("Logging config section missing in GlobalConfig. Using basicConfig for loader.")

    app_config = AppConfig(
        global_config=global_cfg,
        data_config=data_cfg,
        strategies_config=strategies_cfg,
        live_config=live_cfg,
        api_keys=api_keys
    )
    app_config.project_root = project_root
    
    logger.info(f"All configurations loaded successfully for project: {getattr(app_config.global_config, 'project_name', 'N/A')}")
    return app_config
