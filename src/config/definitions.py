import dataclasses
from dataclasses import dataclass, field
from typing import Dict, Optional, Any, List, Union

@dataclass
class PathsConfig:
    data_historical_raw: str
    data_historical_processed_cleaned: str
    data_historical_processed_enriched: str # Added this line
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
    # params_space_to_process: Optional[Dict[str, ParamDetail]] = field(default_factory=dict) # This was commented out or removed in original, keeping it as is

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
    log_filename_live: str = "live_trading_specific_run.log"

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
