import importlib
import logging
import threading
import time
import json
import math
import dataclasses
from pathlib import Path
from typing import Dict, List, Optional, Type, Any, Tuple
from datetime import datetime, timezone, timedelta
import uuid
import re 

import pandas as pd

from src.config.loader import AppConfig
from src.data import acquisition_live
from src.data import preprocessing_live
from src.strategies.base import BaseStrategy
from src.live.state import LiveTradingState, STATUT_1_NO_TRADE_NO_OCO, STATUT_2_ENTRY_FILLED_OCO_PENDING, STATUT_3_OCO_ACTIVE
from src.live.execution import OrderExecutionClient
from src.utils.exchange_utils import get_precision_from_filter, adjust_precision

logger = logging.getLogger(__name__)

USDC_ASSET = "USDC"
MIN_USDC_BALANCE_FOR_TRADE = 10.0
SIGNIFICANT_LOAN_THRESHOLD_USDC_EQUIVALENT = 5.0
API_CALL_DELAY_S = 0.25
MAIN_LOOP_SLEEP_S = 5
KLINE_FETCH_LIMIT_LIVE = 5
ORDER_STATUS_CHECK_DELAY_S = 2
MAX_OCO_CONFIRMATION_ATTEMPTS = 10
FULL_STATE_SYNC_INTERVAL_MINUTES = 5
MIN_EXECUTED_QTY_THRESHOLD = 1e-9


class LiveTradingManager:
    def __init__(self, app_config: AppConfig, pair_to_trade: str, context_label: str):
        self.app_config = app_config
        self.pair_symbol = pair_to_trade.upper()
        self.context_label = context_label 
        # self.strategy_operational_timeframe est supprimé

        self.shutdown_event = threading.Event()
        self.strategy: Optional[BaseStrategy] = None
        self.state_manager: Optional[LiveTradingState] = None
        self.execution_client: Optional[OrderExecutionClient] = None
        
        self.project_root = Path(self.app_config.project_root)
        self.raw_data_dir = self.project_root / self.app_config.global_config.paths.data_live_raw
        self.processed_data_dir = self.project_root / self.app_config.global_config.paths.data_live_processed
        self.state_dir = self.project_root / self.app_config.global_config.paths.live_state
        self.trades_log_dir = self.project_root / self.app_config.global_config.paths.logs_live

        self.raw_1min_data_file_path = self.raw_data_dir / f"{self.pair_symbol}_1min.csv"
        
        cleaned_context_for_file = re.sub(r'[^\w\-_\.]', '_', self.context_label).strip('_')
        if not cleaned_context_for_file: cleaned_context_for_file = "default_context"
        self.processed_data_file_path = self.processed_data_dir / f"{self.pair_symbol}_{cleaned_context_for_file}_processed_live.csv"
        
        self.last_1m_kline_open_timestamp: Optional[pd.Timestamp] = None
        
        self.is_isolated_margin_trading = self.app_config.live_config.global_live_settings.account_type == "ISOLATED_MARGIN"
        self.base_asset = ""
        self.quote_asset = ""
        self.oco_confirmation_attempts = 0
        self.last_full_state_sync_time: Optional[datetime] = None
        self.current_trade_cycle_id: Optional[str] = None

        self._initialize_components()

    def _initialize_components(self):
        logger.info(f"[{self.pair_symbol}][Ctx:{self.context_label}] Initializing components...")
        
        self._load_strategy_config_and_params() 
        if not self.strategy: 
            raise RuntimeError(f"[{self.pair_symbol}][Ctx:{self.context_label}] Critical: Strategy could not be loaded.")

        logger.info(f"[{self.pair_symbol}][Ctx:{self.context_label}] Path for processed live data: {self.processed_data_file_path}")

        cleaned_context_label = re.sub(r'[^\w\-_\.]', '_', self.context_label).strip('_')
        if not cleaned_context_label: cleaned_context_label = "defaultcontext"
            
        state_file_name = f"{self.pair_symbol}_{cleaned_context_label}_state.json" # Nom de fichier d'état sans op_timeframe
        state_file_path = self.state_dir / state_file_name
        
        self.state_manager = LiveTradingState(self.pair_symbol, state_file_path)
        self.current_trade_cycle_id = self.state_manager.get_state_snapshot().get("current_trade_cycle_id") 

        live_settings = self.app_config.live_config.global_live_settings
        self.execution_client = OrderExecutionClient(
            api_key=self.app_config.api_keys.binance_api_key,
            api_secret=self.app_config.api_keys.binance_secret_key,
            account_type=live_settings.account_type,
            is_testnet=getattr(live_settings, 'is_testnet', False)
        )
        if not self.execution_client.test_connection():
            raise ConnectionError(f"[{self.pair_symbol}][Ctx:{self.context_label}] Failed Binance REST API connection.")

        symbol_info = self.execution_client.get_symbol_info(self.pair_symbol)
        if not symbol_info or not symbol_info.get('baseAsset') or not symbol_info.get('quoteAsset'):
            raise ValueError(f"[{self.pair_symbol}][Ctx:{self.context_label}] Could not retrieve valid symbol info (base/quote asset).")
        self.base_asset = symbol_info['baseAsset']
        self.quote_asset = symbol_info['quoteAsset']
        
        self._fetch_initial_1min_klines()
        self._run_initial_preprocessing() 
        
        self._determine_initial_status()
        self.last_full_state_sync_time = datetime.now(timezone.utc)
        logger.info(f"[{self.pair_symbol}][Ctx:{self.context_label}] Component initialization complete.")

    def _load_strategy_config_and_params(self):
        live_config = self.app_config.live_config
        strategy_deployment_config = None
        
        candidate_deployments = []
        for dep_cfg_obj in live_config.strategy_deployments:
            strategy_id_val = getattr(dep_cfg_obj, 'strategy_id', '')
            is_active = getattr(dep_cfg_obj, 'active', False)
            results_path_str = getattr(dep_cfg_obj, 'results_config_path', '')

            if is_active and self.pair_symbol in strategy_id_val:
                candidate_deployments.append({
                    "deployment_obj": dep_cfg_obj,
                    "strategy_id": strategy_id_val,
                    "results_path": results_path_str,
                    "context_match_score": 1 if self.context_label in results_path_str else 0
                })
        
        if not candidate_deployments:
            raise ValueError(f"[{self.pair_symbol}][Ctx:{self.context_label}] No active deployments found for pair '{self.pair_symbol}' in config_live.json.")

        candidate_deployments.sort(key=lambda x: x["context_match_score"], reverse=True)
        
        if not candidate_deployments[0]["context_match_score"] == 1 and len(candidate_deployments) > 0:
            logger.warning(f"[{self.pair_symbol}][Ctx:{self.context_label}] No deployment has context '{self.context_label}' in its results_config_path. Selecting the first available for the pair: {candidate_deployments[0]['strategy_id']}")

        selected_candidate = candidate_deployments[0]
        strategy_deployment_config = selected_candidate["deployment_obj"]
        strategy_id_val = selected_candidate["strategy_id"] # Cet ID est celui du déploiement, peut inclure le contexte.
        
        logger.info(f"[{self.pair_symbol}][Ctx:{self.context_label}] Selected deployment: ID '{strategy_id_val}' (Context match score: {selected_candidate['context_match_score']})")
        
        results_config_path_str = getattr(strategy_deployment_config, 'results_config_path')
        optimized_params_file = self.project_root / results_config_path_str
        if not optimized_params_file.exists():
            raise FileNotFoundError(f"[{self.pair_symbol}][Ctx:{self.context_label}] Optimized parameters file (live_config.json for context) not found: {optimized_params_file} (Deployment ID '{strategy_id_val}')")
        
        with open(optimized_params_file, 'r', encoding='utf-8') as f:
            live_params_config_from_file = json.load(f)
        
        # MODIFICATION: Lire 'strategy_name_base' depuis le fichier live_config.json chargé
        strategy_name_from_id = live_params_config_from_file.get("strategy_name_base")
        if not strategy_name_from_id:
            raise ValueError(f"[{self.pair_symbol}][Ctx:{self.context_label}] 'strategy_name_base' not found in loaded live_config file: {optimized_params_file}")
        
        logger.info(f"[{self.pair_symbol}][Ctx:{self.context_label}] Using strategy base name: {strategy_name_from_id} from {optimized_params_file}")

        strategy_definition_from_config_strategies = self.app_config.strategies_config.strategies.get(strategy_name_from_id)
        if not strategy_definition_from_config_strategies :
            raise ValueError(f"[{self.pair_symbol}][Ctx:{self.context_label}] Strategy base name '{strategy_name_from_id}' not found in config_strategies.json.")

        module_path_str = getattr(strategy_definition_from_config_strategies, 'script_reference')
        class_name_str = getattr(strategy_definition_from_config_strategies, 'class_name')
        
        try:
            module_path = module_path_str.replace('.py', '').replace('/', '.')
            StrategyClass = getattr(importlib.import_module(module_path), class_name_str)
        except Exception as e:
            logger.critical(f"[{self.pair_symbol}][Ctx:{self.context_label}] Error importing strategy {class_name_str} from {module_path_str}: {e}", exc_info=True)
            raise
        
        strategy_params_from_file = live_params_config_from_file.get("parameters", {})
        
        final_params = self.app_config.global_config.simulation_defaults.__dict__.copy()
        final_params.update(strategy_params_from_file) 
        
        override_settings = getattr(strategy_deployment_config, 'override_risk_settings', None)
        if override_settings and dataclasses.is_dataclass(override_settings):
            override_dict = {f.name: getattr(override_settings, f.name) for f in dataclasses.fields(override_settings) if getattr(override_settings, f.name) is not None}
            final_params.update(override_dict)

        self.strategy = StrategyClass(params=final_params)
        logger.info(f"[{self.pair_symbol}][Ctx:{self.context_label}] Strategy '{class_name_str}' loaded with params: {final_params}")

    def _fetch_initial_1min_klines(self):
        limit = self.app_config.live_config.live_fetch.limit_init_history
        account_type = self.app_config.live_config.global_live_settings.account_type
        
        acquisition_live.initialize_pair_data(
            pair=self.pair_symbol,
            config_interval_context="1min",  # Corrigé ici
            raw_path=self.raw_1min_data_file_path, 
            total_klines_to_fetch=limit,      # Corrigé ici
            account_type=account_type
        )
        if self.raw_1min_data_file_path.exists() and self.raw_1min_data_file_path.stat().st_size > 0:
            try:
                df_raw_check = pd.read_csv(self.raw_1min_data_file_path, usecols=['timestamp'])
                if not df_raw_check.empty:
                    self.last_1m_kline_open_timestamp = pd.to_datetime(df_raw_check['timestamp'].iloc[-1], errors='coerce', utc=True)
            except Exception as e:
                logger.error(f"[{self.pair_symbol}][Ctx:{self.context_label}] Error reading timestamp after 1-min data initialization: {e}")

    def _run_initial_preprocessing(self):
        if not self.strategy or not self.processed_data_file_path: 
            logger.warning(f"[{self.pair_symbol}][Ctx:{self.context_label}] Strategy or processed_data_file_path ('{self.processed_data_file_path}') not set. Initial preprocessing skipped.")
            return
            
        if not self.raw_1min_data_file_path.exists() or self.raw_1min_data_file_path.stat().st_size == 0:
            logger.warning(f"[{self.pair_symbol}][Ctx:{self.context_label}] Raw 1-minute data file {self.raw_1min_data_file_path} not found/empty. Preprocessing skipped.")
            return
            
        strategy_params_for_indicators = self.strategy.get_params()
        preprocessing_live.preprocess_live_data_for_strategy(
            raw_data_path=self.raw_1min_data_file_path,
            processed_output_path=self.processed_data_file_path,
            strategy_params=strategy_params_for_indicators,
            strategy_name=self.strategy.__class__.__name__
        )

    def _run_current_preprocessing_cycle(self):
        if not self.strategy or not self.processed_data_file_path:
            logger.warning(f"[{self.pair_symbol}][Ctx:{self.context_label}] Strategy or processed_data_file_path not set. Current preprocessing cycle skipped.")
            return
            
        strategy_params_for_indicators = self.strategy.get_params()
        preprocessing_live.preprocess_live_data_for_strategy(
            raw_data_path=self.raw_1min_data_file_path,
            processed_output_path=self.processed_data_file_path,
            strategy_params=strategy_params_for_indicators,
            strategy_name=self.strategy.__class__.__name__
        )

    def _determine_initial_status(self, is_periodic_sync: bool = False):
        if not self.execution_client or not self.state_manager: return
        log_prefix = f"[{self.pair_symbol}][Ctx:{self.context_label}]" + ("[SYNC] " if is_periodic_sync else "[INIT] ")
        
        logger.info(f"{log_prefix}Determining initial/sync status...")
        usdc_balance = self.execution_client.get_margin_usdc_balance(
            symbol_pair_for_isolated=self.pair_symbol if self.is_isolated_margin_trading else None
        )
        usdc_balance = usdc_balance if usdc_balance is not None else 0.0
        self.state_manager.update_specific_fields({"available_capital_at_last_check": usdc_balance})

        base_asset_balance_val = self.execution_client.get_margin_asset_balance(
            self.base_asset, 
            symbol_pair_for_isolated=self.pair_symbol if self.is_isolated_margin_trading else None
        )
        base_asset_balance_val = base_asset_balance_val if base_asset_balance_val is not None else 0.0
        
        active_loans = self.execution_client.get_active_margin_loans(
            asset=None, 
            isolated_symbol_pair=self.pair_symbol if self.is_isolated_margin_trading else None
        )
        usdc_loan_amount = sum(float(l.get('borrowed', 0.0)) for l in active_loans if l.get('asset', '').upper() == USDC_ASSET)
        base_asset_loan_amount = sum(float(l.get('borrowed', 0.0)) for l in active_loans if l.get('asset', '').upper() == self.base_asset)
        
        open_orders = self.execution_client.get_all_open_margin_orders(
            symbol=self.pair_symbol, 
            is_isolated=self.is_isolated_margin_trading
        )
        num_open_orders = len(open_orders)
        logger.info(f"{log_prefix}USDC Bal: {usdc_balance:.2f}, {self.base_asset} Bal: {base_asset_balance_val:.6f}, USDC Loan: {usdc_loan_amount:.2f}, {self.base_asset} Loan: {base_asset_loan_amount:.6f}, Open Orders: {num_open_orders}")
        
        latest_price = self._get_latest_price_from_agg_data()
        if latest_price == 0.0 and num_open_orders == 0 and (base_asset_loan_amount > MIN_EXECUTED_QTY_THRESHOLD or usdc_loan_amount > SIGNIFICANT_LOAN_THRESHOLD_USDC_EQUIVALENT):
            logger.warning(f"{log_prefix}Latest price is 0.0, cannot reliably determine loan significance if any. Proceeding with caution.")

        has_significant_base_loan = (base_asset_loan_amount * latest_price) > SIGNIFICANT_LOAN_THRESHOLD_USDC_EQUIVALENT if latest_price > 0 else base_asset_loan_amount > MIN_EXECUTED_QTY_THRESHOLD 
        has_significant_quote_loan = usdc_loan_amount > SIGNIFICANT_LOAN_THRESHOLD_USDC_EQUIVALENT
        has_any_significant_loan = has_significant_base_loan or has_significant_quote_loan
        logger.info(f"{log_prefix}Latest Price: {latest_price}, Sig. Base Loan: {has_significant_base_loan}, Sig. Quote Loan: {has_significant_quote_loan}, Any Sig. Loan: {has_any_significant_loan}")

        current_state_snapshot = self.state_manager.get_state_snapshot()
        current_bot_status = self.state_manager.get_current_status()

        if num_open_orders == 0 and not has_any_significant_loan:
            if current_bot_status != STATUT_1_NO_TRADE_NO_OCO or is_periodic_sync:
                logger.info(f"{log_prefix}Condition: No open orders, no significant loans. Transitioning to STATUT_1.")
                self.state_manager.transition_to_status_1("INIT_OR_SYNC_NO_POS_NO_OCO")
                self.current_trade_cycle_id = None
        
        elif num_open_orders == 1 and not has_any_significant_loan:
            pending_entry_id_state = current_state_snapshot.get("pending_entry_order_id")
            open_order_api = open_orders[0]
            if pending_entry_id_state and str(open_order_api.get("orderId")) == str(pending_entry_id_state):
                logger.info(f"{log_prefix}Condition: One open order (matches pending entry {pending_entry_id_state}), no loans. Ensuring STATUT_1 if not already.")
                if current_bot_status != STATUT_1_NO_TRADE_NO_OCO:
                     self.state_manager.update_specific_fields({"current_status": STATUT_1_NO_TRADE_NO_OCO}) 
            else:
                logger.warning(f"{log_prefix}Condition: One open order (ID: {open_order_api.get('orderId')}), no loans, but doesn't match pending entry in state ({pending_entry_id_state}). Cancelling order and resetting.")
                self._cancel_all_orders_for_pair(log_prefix + "[STALE_ORDER_NO_LOAN]")
                self.state_manager.transition_to_status_1("INIT_OR_SYNC_STALE_ORDER_NO_LOAN")
                self.current_trade_cycle_id = None

        elif num_open_orders > 1 and has_any_significant_loan: 
            if current_bot_status != STATUT_3_OCO_ACTIVE or is_periodic_sync:
                logger.info(f"{log_prefix}Condition: {num_open_orders} open orders, significant loan detected. Attempting to sync to STATUT_3 (OCO Active).")
                sl_order = next((o for o in open_orders if o.get('type') in ['STOP_LOSS_LIMIT', 'STOP_LOSS', 'STOP_MARKET']), None)
                tp_order = next((o for o in open_orders if o.get('type') in ['LIMIT', 'TAKE_PROFIT_LIMIT', 'TAKE_PROFIT', 'LIMIT_MAKER']), None)
                
                order_list_id_from_api = sl_order.get("orderListId", -1) if sl_order else -1
                if order_list_id_from_api == -1 and tp_order: order_list_id_from_api = tp_order.get("orderListId", -1)
                
                if order_list_id_from_api == -1 : order_list_id_from_api = current_state_snapshot.get("active_oco_order_list_id")

                oco_details_from_api = {
                    "orderListId": str(order_list_id_from_api) if order_list_id_from_api != -1 else None,
                    "listClientOrderId": current_state_snapshot.get("active_oco_list_client_order_id"), 
                    "orders": open_orders, 
                    "transactionTime": int(time.time() * 1000) 
                }
                self.state_manager.transition_to_status_3(oco_details_from_api)
            else:
                logger.info(f"{log_prefix}Already in STATUT_3_OCO_ACTIVE with open orders and loan. No state change.")

        elif num_open_orders == 1 and has_any_significant_loan:
            if current_bot_status != STATUT_2_ENTRY_FILLED_OCO_PENDING or is_periodic_sync:
                 logger.warning(f"{log_prefix}Condition: One open order and significant loan. This state is ambiguous. Forcing STATUT_2 and will re-evaluate OCO placement.")
                 self._deduce_position_from_loan_and_set_status2(log_prefix, base_asset_loan_amount, usdc_loan_amount, latest_price, {"current_status": STATUT_2_ENTRY_FILLED_OCO_PENDING})

        elif num_open_orders == 0 and has_any_significant_loan:
            if current_bot_status != STATUT_2_ENTRY_FILLED_OCO_PENDING or is_periodic_sync:
                 logger.info(f"{log_prefix}Condition: No open orders, but significant loan exists. Assuming position active, OCO needs placement/check. Syncing to STATUT_2.")
                 self._deduce_position_from_loan_and_set_status2(log_prefix, base_asset_loan_amount, usdc_loan_amount, latest_price, {"current_status": STATUT_2_ENTRY_FILLED_OCO_PENDING})
        
        elif num_open_orders > 0 and not has_any_significant_loan:
            logger.warning(f"{log_prefix}Condition: {num_open_orders} open orders exist, but NO significant loans. Likely stale orders. Cancelling and resetting to STATUT_1.")
            self._cancel_all_orders_for_pair(log_prefix + "[STALE_ORDERS_NO_LOAN]")
            self.state_manager.transition_to_status_1("INIT_OR_SYNC_STALE_ORDERS_NO_LOAN")
            self.current_trade_cycle_id = None
            
        else: 
            logger.warning(f"{log_prefix}Unhandled complex state during sync: {num_open_orders} orders, Loan: {has_any_significant_loan}. Forcing STATUT_1.")
            self._cancel_all_orders_for_pair(log_prefix + "[COMPLEX_STATE_FALLBACK]")
            if has_any_significant_loan:
                self._handle_loan_without_clear_position(log_prefix, base_asset_loan_amount, usdc_loan_amount, "COMPLEX_STATE_FALLBACK_REPAY")
            self.state_manager.transition_to_status_1("INIT_OR_SYNC_FALLBACK_COMPLEX")
            self.current_trade_cycle_id = None
        logger.info(f"{log_prefix}Status after sync: {self.state_manager.get_current_status_name()}")

    def _deduce_position_from_loan_and_set_status2(self, log_prefix:str, base_loan:float, quote_loan:float, current_price:float, updates_dict:dict):
        if not self.state_manager: return
        loan_asset_for_pos: Optional[str] = None
        loan_amount_for_pos = 0.0
        qty_deduced_from_loan = 0.0
        pos_side_deduced: Optional[str] = None
        entry_price_deduced = current_price if current_price > 0 else self.state_manager.get_state_snapshot().get("position_entry_price", 0.0)
        
        if entry_price_deduced == 0.0: 
            if self.execution_client:
                try:
                    client_instance: Client = self.execution_client.client # type: ignore
                    ticker = client_instance.get_symbol_ticker(symbol=self.pair_symbol)
                    if ticker and 'price' in ticker: entry_price_deduced = float(ticker['price'])
                except Exception as e_ticker:
                     logger.error(f"{log_prefix}Error fetching ticker price for loan deduction: {e_ticker}")
        
        if entry_price_deduced == 0.0:
            logger.error(f"{log_prefix}Cannot deduce position entry price for loan calculation. Aborting status 2 transition based on loan.")
            self._handle_loan_without_clear_position(log_prefix, base_loan, quote_loan, "PRICE_UNAVAILABLE_FOR_LOAN_DEDUCE")
            return

        if base_loan > MIN_EXECUTED_QTY_THRESHOLD and ( (base_loan * entry_price_deduced > quote_loan * 0.9) or quote_loan < SIGNIFICANT_LOAN_THRESHOLD_USDC_EQUIVALENT * 0.5 ):
            pos_side_deduced = "SELL" 
            qty_deduced_from_loan = base_loan
            loan_asset_for_pos = self.base_asset
            loan_amount_for_pos = base_loan
            logger.info(f"{log_prefix}Deducing SHORT position from base asset loan {self.base_asset}: {base_loan}")
        elif quote_loan > SIGNIFICANT_LOAN_THRESHOLD_USDC_EQUIVALENT:
            pos_side_deduced = "BUY"
            if entry_price_deduced > 0: 
                qty_deduced_from_loan = quote_loan / entry_price_deduced
            loan_asset_for_pos = self.quote_asset
            loan_amount_for_pos = quote_loan
            logger.info(f"{log_prefix}Deducing LONG position from quote asset loan {self.quote_asset}: {quote_loan}, Qty: {qty_deduced_from_loan}")
        
        if pos_side_deduced and qty_deduced_from_loan > MIN_EXECUTED_QTY_THRESHOLD:
            state_updates = updates_dict.copy()
            state_updates.update({
                "position_side": pos_side_deduced,
                "position_quantity": qty_deduced_from_loan,
                "position_entry_price": entry_price_deduced,
                "position_entry_timestamp": int(time.time() * 1000), 
                "loan_details": {"asset": loan_asset_for_pos, "amount": loan_amount_for_pos, "timestamp": int(time.time()*1000)},
                "entry_order_details": { 
                    "status": "FILLED_BY_LOAN_DEDUCTION", 
                    "side": pos_side_deduced.upper(), 
                    "executedQty": qty_deduced_from_loan, 
                    "cummulativeQuoteQty": qty_deduced_from_loan * entry_price_deduced if entry_price_deduced > 0 else 0.0,
                    "updateTime": int(time.time()*1000)
                },
                "pending_sl_tp_raw": self.state_manager.get_state_snapshot().get("pending_sl_tp_raw", {}) 
            })
            if not self.current_trade_cycle_id and not self.state_manager.get_state_snapshot().get("current_trade_cycle_id"):
                self.current_trade_cycle_id = f"recovered_trade_{self.pair_symbol}_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}"
                state_updates["current_trade_cycle_id"] = self.current_trade_cycle_id
                logger.info(f"{log_prefix}Generated new trade_cycle_id for recovered position: {self.current_trade_cycle_id}")
            elif self.current_trade_cycle_id:
                 state_updates["current_trade_cycle_id"] = self.current_trade_cycle_id
            
            self.state_manager.update_specific_fields(state_updates)
            logger.info(f"{log_prefix}Transitioned to STATUT_2 based on loan deduction. Side: {pos_side_deduced}, Qty: {qty_deduced_from_loan:.6f}")
        else:
            logger.warning(f"{log_prefix}Could not deduce clear position from loans (Base Loan: {base_loan}, Quote Loan: {quote_loan}, Price: {entry_price_deduced}). Attempting cleanup.")
            self._handle_loan_without_clear_position(log_prefix, base_loan, quote_loan, "AMBIGUOUS_LOAN_DURING_STATUS2_DEDUCE")

    def _handle_loan_without_clear_position(self, log_prefix: str, base_loan: float, quote_loan: float, reason_suffix: str):
        if not self.execution_client or not self.state_manager: return
        logger.warning(f"{log_prefix}Handling loans without clear open position. Base: {base_loan}, Quote: {quote_loan}. Reason: {reason_suffix}")
        cleaned_something = False
        if base_loan > MIN_EXECUTED_QTY_THRESHOLD :
            logger.info(f"{log_prefix}Attempting to repay base asset loan: {base_loan} {self.base_asset}")
            repay_res_base = self.execution_client.repay_margin_loan(asset=self.base_asset, amount=str(base_loan), isolated_symbol_pair=self.pair_symbol if self.is_isolated_margin_trading else None)
            self._log_trade_event(f"CLEANUP_REPAY_BASE_ASSET_{reason_suffix}", {"amount": base_loan, "api_response": repay_res_base})
            if repay_res_base and (repay_res_base.get("status") == "SUCCESS" or repay_res_base.get("tranId")):
                cleaned_something = True
        
        if quote_loan > SIGNIFICANT_LOAN_THRESHOLD_USDC_EQUIVALENT : 
            logger.info(f"{log_prefix}Attempting to repay quote asset loan: {quote_loan} {self.quote_asset}")
            repay_res_quote = self.execution_client.repay_margin_loan(asset=self.quote_asset, amount=str(quote_loan), isolated_symbol_pair=self.pair_symbol if self.is_isolated_margin_trading else None)
            self._log_trade_event(f"CLEANUP_REPAY_QUOTE_ASSET_{reason_suffix}", {"amount": quote_loan, "api_response": repay_res_quote})
            if repay_res_quote and (repay_res_quote.get("status") == "SUCCESS" or repay_res_quote.get("tranId")):
                cleaned_something = True
        
        if cleaned_something or self.state_manager.get_current_status() != STATUT_1_NO_TRADE_NO_OCO :
            self.state_manager.transition_to_status_1(f"CLEANUP_LOAN_{reason_suffix}")
            self.current_trade_cycle_id = None

    def _cancel_all_orders_for_pair(self, log_prefix: str):
        if not self.execution_client: return
        logger.info(f"{log_prefix} Attempting to cancel all open margin orders for {self.pair_symbol} (Isolated: {self.is_isolated_margin_trading})")
        open_orders_to_cancel = self.execution_client.get_all_open_margin_orders(self.pair_symbol, self.is_isolated_margin_trading)
        cancelled_ids_or_lists = []
        if not open_orders_to_cancel:
            logger.info(f"{log_prefix} No open orders found for {self.pair_symbol} to cancel.")
            return

        processed_order_list_ids = set() 

        for order in open_orders_to_cancel:
            order_id_to_cancel = order.get('orderId')
            client_order_id_to_cancel = order.get('clientOrderId')
            order_list_id = order.get('orderListId', -1) 

            if order_list_id != -1 and order_list_id not in processed_order_list_ids:
                logger.info(f"{log_prefix} Found OCO order (OrderListId: {order_list_id}). Attempting to cancel list.")
                cancel_res = self.execution_client.cancel_margin_oco_order(
                    symbol=self.pair_symbol, 
                    order_list_id=order_list_id,
                    list_client_order_id=order.get('listClientOrderId'),
                    is_isolated=self.is_isolated_margin_trading
                )
                event_type = "CANCEL_OCO_ORDER_LIST_ATTEMPT"
                processed_order_list_ids.add(order_list_id)
                id_logged = order_list_id
            elif order_list_id == -1 and order_id_to_cancel : 
                logger.info(f"{log_prefix} Found single order (ID: {order_id_to_cancel}). Attempting to cancel.")
                cancel_res = self.execution_client.cancel_margin_order(
                    symbol=self.pair_symbol, 
                    order_id=order_id_to_cancel, 
                    orig_client_order_id=client_order_id_to_cancel,
                    is_isolated=self.is_isolated_margin_trading
                )
                event_type = "CANCEL_SINGLE_ORDER_ATTEMPT"
                id_logged = order_id_to_cancel
            else: 
                continue 
            
            self._log_trade_event(event_type, {"order_to_cancel": order, "api_response": cancel_res})
            if cancel_res and ( (cancel_res.get("data") and (cancel_res.get("data").get("orderListId") or cancel_res.get("data").get("orderId"))) or cancel_res.get("status") == "SUCCESS" ): 
                cancelled_ids_or_lists.append(id_logged)
                logger.info(f"{log_prefix} Successfully sent cancel request for order/list {id_logged}.")
            else:
                logger.error(f"{log_prefix} Failed to cancel order/list {id_logged}. Response: {cancel_res}")
            time.sleep(API_CALL_DELAY_S) 
        logger.info(f"{log_prefix} Cancellation process finished. Cancelled IDs/Lists: {cancelled_ids_or_lists}")


    def _get_latest_price_from_agg_data(self) -> float:
        if not self.processed_data_file_path: 
             logger.warning(f"[{self.pair_symbol}][Ctx:{self.context_label}] Processed aggregated data file path not set. Cannot get latest price from it.")
             if self.execution_client and hasattr(self.execution_client, 'client'):
                try:
                    client_instance: Client = self.execution_client.client # type: ignore
                    ticker = client_instance.get_symbol_ticker(symbol=self.pair_symbol)
                    if ticker and 'price' in ticker: return float(ticker['price'])
                except Exception as e_ticker: 
                    logger.error(f"[{self.pair_symbol}][Ctx:{self.context_label}] Error fetching ticker price as fallback: {e_ticker}")
             return 0.0

        data = self._get_latest_processed_agg_data()
        if data is not None and not data.empty and 'close' in data.columns and pd.notna(data['close'].iloc[-1]):
            return float(data['close'].iloc[-1])
        
        if self.execution_client and hasattr(self.execution_client, 'client'):
            try:
                client_instance: Client = self.execution_client.client # type: ignore
                ticker = client_instance.get_symbol_ticker(symbol=self.pair_symbol)
                if ticker and 'price' in ticker: return float(ticker['price'])
            except Exception as e_ticker_fb: 
                logger.error(f"[{self.pair_symbol}][Ctx:{self.context_label}] Error fetching ticker price as fallback (after file check): {e_ticker_fb}")
        return 0.0

    def _get_latest_processed_agg_data(self) -> Optional[pd.DataFrame]:
        if not self.processed_data_file_path: 
            logger.warning(f"[{self.pair_symbol}][Ctx:{self.context_label}] Processed aggregated data file path is not set. Cannot load.")
            return None
            
        if not self.processed_data_file_path.exists() or self.processed_data_file_path.stat().st_size == 0:
            logger.info(f"[{self.pair_symbol}][Ctx:{self.context_label}] Processed data file {self.processed_data_file_path} missing or empty. Attempting to run preprocessing.")
            if self.raw_1min_data_file_path.exists() and self.raw_1min_data_file_path.stat().st_size > 0:
                self._run_current_preprocessing_cycle() 
                if not self.processed_data_file_path.exists() or self.processed_data_file_path.stat().st_size == 0:
                    logger.warning(f"[{self.pair_symbol}][Ctx:{self.context_label}] Preprocessing did not create/populate {self.processed_data_file_path}.")
                    return None
            else:
                logger.warning(f"[{self.pair_symbol}][Ctx:{self.context_label}] Raw 1-min data file {self.raw_1min_data_file_path} also missing or empty. Cannot preprocess.")
                return None
        try:
            df = pd.read_csv(self.processed_data_file_path)
            if df.empty: 
                logger.warning(f"[{self.pair_symbol}][Ctx:{self.context_label}] Loaded processed aggregated data from {self.processed_data_file_path} is empty.")
                return None
            if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce', utc=True)
            df = df.set_index('timestamp').sort_index()
            return df
        except Exception as e:
            logger.error(f"[{self.pair_symbol}][Ctx:{self.context_label}] Error loading processed aggregated data from {self.processed_data_file_path}: {e}", exc_info=True)
            return None

    def _check_new_1min_kline_and_trigger_preprocessing(self) -> bool:
        if not self.execution_client or not hasattr(self.execution_client, 'client'): 
            logger.debug(f"[{self.pair_symbol}][Ctx:{self.context_label}] Skipping kline check: client not ready.")
            return False
        try:
            log_context_for_fetch = self.context_label 
            new_1m_klines_df_raw = acquisition_live.get_binance_klines(
                            symbol=self.pair_symbol, 
                            config_interval_context=log_context_for_fetch, # Corrigé ici
                            limit=KLINE_FETCH_LIMIT_LIVE, 
                            account_type=self.app_config.live_config.global_live_settings.account_type
                        )
            if new_1m_klines_df_raw is None or new_1m_klines_df_raw.empty: 
                logger.debug(f"[{self.pair_symbol}][Ctx:{self.context_label}] No new 1m klines fetched.")
                return False

            if 'kline_close_time' not in new_1m_klines_df_raw.columns or 'timestamp' not in new_1m_klines_df_raw.columns:
                logger.warning(f"[{self.pair_symbol}][Ctx:{self.context_label}] Fetched 1m klines missing 'kline_close_time' or 'timestamp'.")
                return False
            
            new_1m_klines_df_raw['timestamp'] = pd.to_datetime(new_1m_klines_df_raw['timestamp'], errors='coerce', utc=True)
            new_1m_klines_df_raw['kline_close_time'] = pd.to_datetime(new_1m_klines_df_raw['kline_close_time'], errors='coerce', utc=True)
            new_1m_klines_df_raw.dropna(subset=['timestamp', 'kline_close_time'], inplace=True)
            if new_1m_klines_df_raw.empty: 
                logger.debug(f"[{self.pair_symbol}][Ctx:{self.context_label}] 1m klines empty after NaN drop on timestamps.")
                return False
            
            new_1m_klines_df_raw = new_1m_klines_df_raw.sort_values(by='timestamp')
            
            client_instance: Client = self.execution_client.client # type: ignore
            server_time_response = client_instance.get_server_time() 
            current_server_time = pd.to_datetime(server_time_response['serverTime'], unit='ms', utc=True)
            
            closed_1m_klines_df = new_1m_klines_df_raw[new_1m_klines_df_raw['kline_close_time'] < current_server_time].copy()
            if closed_1m_klines_df.empty: 
                logger.debug(f"[{self.pair_symbol}][Ctx:{self.context_label}] No *closed* 1m klines among fetched. Server time: {current_server_time}")
                return False
            
            latest_closed_1m_kline_open_ts = closed_1m_klines_df['timestamp'].iloc[-1]

            if self.last_1m_kline_open_timestamp is None or latest_closed_1m_kline_open_ts > self.last_1m_kline_open_timestamp:
                logger.info(f"[{self.pair_symbol}][Ctx:{self.context_label}] New closed 1m kline detected. Open: {latest_closed_1m_kline_open_ts} (Prev known: {self.last_1m_kline_open_timestamp}). Appending to raw 1m file.")
                df_existing_1m_raw = pd.DataFrame(columns=acquisition_live.BASE_COLUMNS_RAW)
                if self.raw_1min_data_file_path.exists() and self.raw_1min_data_file_path.stat().st_size > 0:
                    try:
                        df_existing_1m_raw = pd.read_csv(self.raw_1min_data_file_path)
                        if 'timestamp' in df_existing_1m_raw.columns:
                             df_existing_1m_raw['timestamp'] = pd.to_datetime(df_existing_1m_raw['timestamp'], errors='coerce', utc=True)
                    except Exception as e_read_raw: 
                        logger.error(f"[{self.pair_symbol}][Ctx:{self.context_label}] Error reading existing raw 1m data: {e_read_raw}. Starting with empty DataFrame.")
                        df_existing_1m_raw = pd.DataFrame(columns=acquisition_live.BASE_COLUMNS_RAW)
                
                df_to_add = closed_1m_klines_df[acquisition_live.BASE_COLUMNS_RAW] 
                df_combined_1m = pd.concat([df_existing_1m_raw, df_to_add], ignore_index=True)
                if 'timestamp' in df_combined_1m.columns:
                    df_combined_1m.dropna(subset=['timestamp'], inplace=True)
                    df_combined_1m['timestamp'] = pd.to_datetime(df_combined_1m['timestamp'], errors='coerce', utc=True) 
                    df_combined_1m.dropna(subset=['timestamp'], inplace=True) 
                    df_combined_1m = df_combined_1m.sort_values(by='timestamp').drop_duplicates(subset=['timestamp'], keep='last')
                
                df_combined_1m[acquisition_live.BASE_COLUMNS_RAW].to_csv(self.raw_1min_data_file_path, index=False)
                logger.info(f"[{self.pair_symbol}][Ctx:{self.context_label}] Appended {len(df_to_add)} new 1m klines. Total in raw file: {len(df_combined_1m)}")
                
                self._run_current_preprocessing_cycle() 
                self.last_1m_kline_open_timestamp = latest_closed_1m_kline_open_ts
                return True
            else:
                logger.debug(f"[{self.pair_symbol}][Ctx:{self.context_label}] No new closed 1m kline since last check ({self.last_1m_kline_open_timestamp}). Latest server closed: {latest_closed_1m_kline_open_ts}")
            return False
        except Exception as e:
            logger.error(f"[{self.pair_symbol}][Ctx:{self.context_label}] Error in _check_new_1min_kline: {e}", exc_info=True)
            return False
            
    def _log_trade_event(self, event_type: str, details: Dict[str, Any]):
        if not self.strategy or not self.state_manager : return
        
        cycle_id_to_log = self.current_trade_cycle_id or self.state_manager.get_state_snapshot().get("current_trade_cycle_id")
        if not cycle_id_to_log: 
            if not event_type.startswith("ENTRY_SIGNAL_GENERATED"):
                cycle_id_to_log = f"orphan_{self.pair_symbol}_{int(time.time()*1000)}_{uuid.uuid4().hex[:4]}"
                logger.warning(f"[{self.pair_symbol}][Ctx:{self.context_label}] No current_trade_cycle_id, using generated orphan ID: {cycle_id_to_log} for event {event_type}")

        log_entry = {
            "log_timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "event_id": f"{event_type.lower().replace(' ', '_')}_{int(time.time()*1000)}_{uuid.uuid4().hex[:4]}",
            "trade_cycle_id": cycle_id_to_log, 
            "event_type": event_type, 
            "pair_symbol": self.pair_symbol, 
            "context_label": self.context_label,
            "strategy_name": self.strategy.__class__.__name__, 
            "current_bot_status": self.state_manager.get_current_status_name(), 
            "details": details
        }
        today_date_str = datetime.now(timezone.utc).strftime("%Y%m%d")
        cleaned_context_label_for_file = re.sub(r'[^\w\-_\.]', '_', self.context_label).strip('_')
        if not cleaned_context_label_for_file: cleaned_context_label_for_file = "default"

        log_file_name = f"trades_{self.pair_symbol}_{cleaned_context_label_for_file}_{today_date_str}.jsonl" # Nom de fichier sans op_timeframe
        log_file_path = self.trades_log_dir / log_file_name
        try:
            self.trades_log_dir.mkdir(parents=True, exist_ok=True)
            with open(log_file_path, 'a', encoding='utf-8') as f:
                json.dump(log_entry, f, default=str); f.write('\n')
        except Exception as e:
            logger.error(f"[{self.pair_symbol}][Ctx:{self.context_label}] Error logging trade event to {log_file_path}: {e}", exc_info=True)

    def _check_and_process_orders_via_rest(self):
        if not self.execution_client or not self.state_manager : return
        current_status = self.state_manager.get_current_status()
        state = self.state_manager.get_state_snapshot()
        
        active_cycle_id_for_log = self.current_trade_cycle_id or state.get("current_trade_cycle_id", f"NO_CYCLE_ORDERS_CHECK_{int(time.time())}")
        log_prefix = f"[{self.pair_symbol}][Ctx:{self.context_label}][Cycle:{active_cycle_id_for_log}]"

        pending_entry_id_server = state.get("pending_entry_order_id")
        pending_entry_id_client = state.get("pending_entry_client_order_id")

        if pending_entry_id_server and current_status == STATUT_1_NO_TRADE_NO_OCO:
            logger.debug(f"{log_prefix} Checking pending entry order ID: {pending_entry_id_server}")
            order_details = self.execution_client.get_margin_order_status(
                symbol=self.pair_symbol, order_id=pending_entry_id_server,
                orig_client_order_id=pending_entry_id_client,
                is_isolated=self.is_isolated_margin_trading
            )
            if order_details and order_details.get("status") == "FILLED":
                executed_qty_api = float(order_details.get("executedQty", 0.0))
                if executed_qty_api <= MIN_EXECUTED_QTY_THRESHOLD: 
                    logger.error(f"{log_prefix} Entry order {pending_entry_id_server} FILLED with zero/sub-threshold quantity: {executed_qty_api}.")
                    self._log_trade_event("ENTRY_ORDER_FILLED_ZERO_QTY_ERROR", {"order_api_response": order_details, "pending_entry_params": state.get("pending_entry_params")})
                    self.state_manager.set_last_error(f"Entry FILLED zero qty: {executed_qty_api}")
                    self.state_manager.transition_to_status_1("ENTRY_FILLED_ZERO_QTY")
                    self.current_trade_cycle_id = None; return

                logger.info(f"{log_prefix} Entry order {pending_entry_id_server} FILLED successfully.")
                self._log_trade_event("ENTRY_ORDER_FILLED", {"order_api_response": order_details, "pending_entry_params": state.get("pending_entry_params")})
                
                loan_asset = self.quote_asset if order_details.get("side") == "BUY" else self.base_asset 
                loan_amount = float(order_details.get("cummulativeQuoteQty")) if order_details.get("side") == "BUY" else executed_qty_api
                
                loan_info = {"asset": loan_asset, "amount": loan_amount, "timestamp": order_details.get("updateTime")}
                self.state_manager.transition_to_status_2(order_details, loan_info)
                self.oco_confirmation_attempts = 0 
            elif order_details and order_details.get("status") in ["CANCELED", "EXPIRED", "REJECTED", "PENDING_CANCEL", "NOT_FOUND"]:
                logger.warning(f"{log_prefix} Entry order {pending_entry_id_server} status: {order_details.get('status')}. Transitioning to STATUT_1.")
                self._log_trade_event(f"ENTRY_ORDER_{order_details.get('status', 'FAILED').upper()}", {"order_api_response": order_details, "pending_entry_params": state.get("pending_entry_params")})
                self.state_manager.transition_to_status_1(f"ENTRY_{order_details.get('status', 'FAILED')}")
                self.current_trade_cycle_id = None 
            elif order_details: 
                logger.debug(f"{log_prefix} Entry order {pending_entry_id_server} status still {order_details.get('status')}. No action.")
            else: 
                logger.error(f"{log_prefix} Could not get status for pending entry order {pending_entry_id_server}.")
                self.state_manager.set_last_error(f"Failed to get status for entry {pending_entry_id_server}")

        elif current_status == STATUT_2_ENTRY_FILLED_OCO_PENDING and state.get("pending_oco_list_client_order_id"):
            pending_list_client_id = state.get("pending_oco_list_client_order_id")
            logger.debug(f"{log_prefix} Checking pending OCO with listClientOrderId: {pending_list_client_id}")
            open_ocos = self.execution_client.get_open_margin_oco_orders(self.pair_symbol, self.is_isolated_margin_trading)
            found_oco_in_api = next((oco for oco in open_ocos if oco.get("listClientOrderId") == pending_list_client_id), None)
            
            if found_oco_in_api:
                logger.info(f"{log_prefix} OCO order (ListClientOrderID: {pending_list_client_id}, OrderListID: {found_oco_in_api.get('orderListId')}) confirmed ACTIVE on exchange.")
                self._log_trade_event("OCO_ORDER_CONFIRMED_ACTIVE", {"oco_api_response": found_oco_in_api, "oco_params_sent": state.get("oco_params_to_place")})
                self.state_manager.transition_to_status_3(found_oco_in_api)
                self.oco_confirmation_attempts = 0
            else:
                self.oco_confirmation_attempts += 1
                logger.warning(f"{log_prefix} Pending OCO (ListClientOrderID: {pending_list_client_id}) not found in open OCOs (Attempt {self.oco_confirmation_attempts}/{MAX_OCO_CONFIRMATION_ATTEMPTS}).")
                if self.oco_confirmation_attempts >= MAX_OCO_CONFIRMATION_ATTEMPTS:
                    logger.error(f"{log_prefix} Max attempts reached for OCO confirmation (ListClientOrderID: {pending_list_client_id}). Assuming OCO placement failed or was filled/cancelled externally.")
                    self._log_trade_event("OCO_CONFIRMATION_FAILED_MAX_ATTEMPTS", {"pending_oco_list_client_order_id": pending_list_client_id, "oco_params_sent": state.get("oco_params_to_place")})
                    self._handle_trade_closure_and_loan_repayment(state, "OCO_CONFIRMATION_FAILURE_CLOSING_POS", None)
                    self.oco_confirmation_attempts = 0 

        elif current_status == STATUT_3_OCO_ACTIVE and (state.get("active_oco_order_list_id") or state.get("active_oco_list_client_order_id")):
            oco_id_server = state.get("active_oco_order_list_id")
            oco_client_id = state.get("active_oco_list_client_order_id")
            logger.debug(f"{log_prefix} Checking active OCO (ServerID: {oco_id_server}, ClientID: {oco_client_id})")
            
            open_orders_for_pair = self.execution_client.get_all_open_margin_orders(self.pair_symbol, self.is_isolated_margin_trading)
            num_open_orders = len(open_orders_for_pair)
            
            is_oco_still_present_as_list = False
            if oco_id_server:
                 is_oco_still_present_as_list = any(str(o.get("orderListId")) == str(oco_id_server) for o in open_orders_for_pair)
            elif oco_client_id: 
                 is_oco_still_present_as_list = any(o.get("listClientOrderId") == oco_client_id for o in open_orders_for_pair if o.get("listClientOrderId"))

            if not is_oco_still_present_as_list and num_open_orders == 0 : 
                logger.info(f"{log_prefix} Active OCO (ServerID: {oco_id_server}, ClientID: {oco_client_id}) no longer found, and no open orders for the pair. Assuming one leg filled.")
                sl_order_id, tp_order_id = state.get("active_sl_order_id"), state.get("active_tp_order_id")
                closed_order_data, exit_reason_code = None, "OCO_DISAPPEARED_UNKNOWN_FILL"

                if sl_order_id:
                    sl_status = self.execution_client.get_margin_order_status(self.pair_symbol, sl_order_id, is_isolated=self.is_isolated_margin_trading)
                    if sl_status and sl_status.get("status") == "FILLED": 
                        closed_order_data, exit_reason_code = sl_status, "SL_FILLED"
                        logger.info(f"{log_prefix} SL order {sl_order_id} confirmed FILLED.")
                
                if not closed_order_data and tp_order_id:
                    tp_status = self.execution_client.get_margin_order_status(self.pair_symbol, tp_order_id, is_isolated=self.is_isolated_margin_trading)
                    if tp_status and tp_status.get("status") == "FILLED": 
                        closed_order_data, exit_reason_code = tp_status, "TP_FILLED"
                        logger.info(f"{log_prefix} TP order {tp_order_id} confirmed FILLED.")
                
                if not closed_order_data:
                    logger.warning(f"{log_prefix} OCO disappeared, but could not confirm fill status for SL ({sl_order_id}) or TP ({tp_order_id}). Using placeholder exit reason.")
                
                self._log_trade_event(exit_reason_code, {"closed_order_details": closed_order_data, "active_oco_details_before_closure": state.get("active_oco_details")})
                self._handle_trade_closure_and_loan_repayment(state, exit_reason_code, closed_order_data)
            elif num_open_orders > 0 and not is_oco_still_present_as_list:
                 logger.warning(f"{log_prefix} OCO list (ServerID: {oco_id_server}) not found, but {num_open_orders} individual orders still open. This is unexpected state post-OCO. Cancelling remaining orders.")
                 self._cancel_all_orders_for_pair(log_prefix + "[DANGLING_ORDERS_POST_OCO]")
                 self._handle_trade_closure_and_loan_repayment(state, "DANGLING_ORDERS_POST_OCO_CLOSING_POS", None)


    def _handle_status_1_no_trade(self):
        if not self.strategy or not self.execution_client or not self.state_manager or not self.processed_data_file_path: return
        
        current_state = self.state_manager.get_state_snapshot()
        if current_state.get("pending_entry_order_id"): 
            logger.debug(f"[{self.pair_symbol}][Ctx:{self.context_label}] Status 1: Entry order {current_state.get('pending_entry_order_id')} already pending. Skipping new signal check.")
            return
        
        current_usdc_balance = self.execution_client.get_margin_usdc_balance(
            symbol_pair_for_isolated=self.pair_symbol if self.is_isolated_margin_trading else None
        )
        current_usdc_balance = current_usdc_balance if current_usdc_balance is not None else 0.0
        self.state_manager.update_specific_fields({"available_capital_at_last_check": current_usdc_balance})

        if current_usdc_balance < MIN_USDC_BALANCE_FOR_TRADE:
            logger.warning(f"[{self.pair_symbol}][Ctx:{self.context_label}] Status 1: Insufficient {USDC_ASSET} balance ({current_usdc_balance:.2f}). Min required: {MIN_USDC_BALANCE_FOR_TRADE}. No new trade.")
            self.state_manager.set_last_error(f"Insufficient {USDC_ASSET} balance: {current_usdc_balance:.2f}")
            return

        latest_agg_data_df = self._get_latest_processed_agg_data()
        if latest_agg_data_df is None or latest_agg_data_df.empty: 
            logger.warning(f"[{self.pair_symbol}][Ctx:{self.context_label}] Status 1: No aggregated data available. Cannot generate signal.")
            return
        
        symbol_info = self.execution_client.get_symbol_info(self.pair_symbol)
        if not symbol_info: 
            logger.error(f"[{self.pair_symbol}][Ctx:{self.context_label}] Status 1: Failed to get symbol info. Cannot generate order request.")
            return
        
        order_request_tuple = self.strategy.generate_order_request(
            data=latest_agg_data_df, symbol=self.pair_symbol, current_position=0, 
            available_capital=current_usdc_balance, symbol_info=symbol_info
        )
        
        if order_request_tuple:
            entry_params, sl_tp_raw = order_request_tuple
            
            self.current_trade_cycle_id = f"trade_{self.pair_symbol}_{self.context_label.replace(' ','_')}_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}"
            logger.info(f"[{self.pair_symbol}][Ctx:{self.context_label}][Cycle:{self.current_trade_cycle_id}] Entry signal generated.")
            self._log_trade_event("ENTRY_SIGNAL_GENERATED", {"entry_params_suggested": entry_params, "sl_tp_raw": sl_tp_raw})
            
            entry_params_api = entry_params.copy()
            entry_params_api['isIsolated'] = "TRUE" if self.is_isolated_margin_trading else "FALSE"
            
            response = self.execution_client.place_margin_order(**entry_params_api)
            log_detail_entry_sent = {"params_sent_to_api": entry_params_api, "api_response": response, "sl_tp_intended": sl_tp_raw}
            self._log_trade_event("ENTRY_ORDER_SENT", log_detail_entry_sent)
            
            if response and response.get("status") == "SUCCESS" and response.get("data"):
                api_data = response["data"]
                order_id, client_order_id = api_data.get("orderId"), api_data.get("clientOrderId")
                if order_id and client_order_id:
                    logger.info(f"[{self.pair_symbol}][Ctx:{self.context_label}][Cycle:{self.current_trade_cycle_id}] Entry order submitted. OrderID: {order_id}, ClientOrderID: {client_order_id}")
                    self.state_manager.prepare_for_entry_order(entry_params, sl_tp_raw, self.current_trade_cycle_id) 
                    self.state_manager.record_placed_entry_order(order_id, client_order_id)
                else: 
                    logger.error(f"[{self.pair_symbol}][Ctx:{self.context_label}][Cycle:{self.current_trade_cycle_id}] Entry API success, but missing OrderID or ClientOrderID in response: {api_data}")
                    self.state_manager.set_last_error(f"Entry API success, missing IDs: {api_data}"); 
                    self.current_trade_cycle_id = None 
            else: 
                error_msg = response.get("message", 'No response data') if response and isinstance(response, dict) else 'Placement failed, no response'
                logger.error(f"[{self.pair_symbol}][Ctx:{self.context_label}][Cycle:{self.current_trade_cycle_id}] Entry order placement failed: {error_msg}")
                self.state_manager.set_last_error(f"Entry placement failed: {error_msg}"); 
                self.current_trade_cycle_id = None 


    def _handle_status_2_oco_pending(self):
        if not self.strategy or not self.execution_client or not self.state_manager: return
        state = self.state_manager.get_state_snapshot()
        active_cycle_id_for_log = self.current_trade_cycle_id or state.get("current_trade_cycle_id", f"NO_CYCLE_OCO_PENDING_{int(time.time())}")
        log_prefix = f"[{self.pair_symbol}][Ctx:{self.context_label}][Cycle:{active_cycle_id_for_log}]"

        if state.get("pending_oco_list_client_order_id"): 
            logger.debug(f"{log_prefix} Status 2: OCO order (ListClientOrderID: {state.get('pending_oco_list_client_order_id')}) already pending. Skipping new OCO placement attempt.")
            return

        pos_side = state.get("position_side")
        pos_qty_float = state.get("position_quantity") 
        sl_tp_raw = state.get("pending_sl_tp_raw", {})

        if not (pos_side and isinstance(pos_qty_float, float) and pos_qty_float > MIN_EXECUTED_QTY_THRESHOLD and 
                sl_tp_raw.get('sl_price') is not None and sl_tp_raw.get('tp_price') is not None):
            error_msg = f"STATUT_2: Missing critical data for OCO placement. Side:{pos_side}, Qty:{pos_qty_float}, SL/TP Raw:{sl_tp_raw}"
            logger.error(f"{log_prefix} {error_msg}")
            self.state_manager.set_last_error(error_msg)
            if pos_qty_float is not None and pos_qty_float <= MIN_EXECUTED_QTY_THRESHOLD:
                logger.error(f"{log_prefix} Position quantity ({pos_qty_float}) is zero or sub-threshold in STATUT_2. Cannot place OCO. Attempting to close trade.")
                self._handle_trade_closure_and_loan_repayment(state, "ZERO_POSITION_QTY_IN_STATUS_2", state.get("entry_order_details"))
            return
        
        symbol_info = self.execution_client.get_symbol_info(self.pair_symbol)
        if not symbol_info: 
            logger.error(f"{log_prefix} Status 2: Failed to get symbol info for OCO. Cannot proceed.")
            self.state_manager.set_last_error("OCO: No symbol info."); return
        
        price_prec_val = get_precision_from_filter(symbol_info, 'PRICE_FILTER', 'tickSize')
        qty_prec_val = get_precision_from_filter(symbol_info, 'LOT_SIZE', 'stepSize')
        if price_prec_val is None or qty_prec_val is None: 
            logger.error(f"{log_prefix} Status 2: Failed to get price/qty precision for OCO.")
            self.state_manager.set_last_error("OCO: No precision."); return
        
        oco_params = self.strategy._build_oco_params(
            symbol=self.pair_symbol, position_side=str(pos_side), executed_qty=float(pos_qty_float), 
            sl_price_raw=float(sl_tp_raw['sl_price']), tp_price_raw=float(sl_tp_raw['tp_price']), 
            price_precision=price_prec_val, qty_precision=qty_prec_val, symbol_info=symbol_info
        )
        if not oco_params: 
            logger.error(f"{log_prefix} Status 2: Failed to build OCO parameters from strategy.")
            self.state_manager.set_last_error("Failed to build OCO params."); return
        
        oco_params_api = oco_params.copy()
        oco_params_api['isIsolated'] = "TRUE" if self.is_isolated_margin_trading else "FALSE"
        if "listClientOrderId" not in oco_params_api or not oco_params_api["listClientOrderId"]:
            base_oco_client_id_part = active_cycle_id_for_log.split('_')[-1] if active_cycle_id_for_log and '_' in active_cycle_id_for_log else uuid.uuid4().hex[:4]
            oco_params_api["listClientOrderId"] = f"oco_{self.pair_symbol[:3]}_{base_oco_client_id_part}_{uuid.uuid4().hex[:4]}"
        
        logger.info(f"{log_prefix} Status 2: Attempting to place OCO order. Params: {oco_params_api}")
        response = self.execution_client.place_margin_oco_order(**oco_params_api)
        log_detail_oco_sent = {"params_sent_to_api": oco_params_api, "api_response": response, "entry_order_details": state.get("entry_order_details")}
        self._log_trade_event("OCO_ORDER_SENT", log_detail_oco_sent)
        
        if response and response.get("status") == "SUCCESS" and response.get("data"):
            api_data = response["data"]
            list_client_id = api_data.get("listClientOrderId", oco_params_api["listClientOrderId"]) 
            order_list_id_api = api_data.get("orderListId") 
            logger.info(f"{log_prefix} Status 2: OCO order submitted. ListClientOrderID: {list_client_id}, OrderListID API: {order_list_id_api}")
            self.state_manager.prepare_for_oco_order(oco_params, list_client_id, str(order_list_id_api) if order_list_id_api else None)
            self.oco_confirmation_attempts = 0 
        else: 
            err_msg = response.get("message", "OCO placement failed, no response data") if response and isinstance(response,dict) else "OCO placement failed, no response"
            logger.error(f"{log_prefix} Status 2: OCO order placement failed: {err_msg}")
            self.state_manager.set_last_error(f"OCO placement failed: {err_msg}")

    def _handle_status_3_oco_active(self):
        if not self.execution_client or not self.state_manager: return
        state = self.state_manager.get_state_snapshot()
        active_cycle_id_for_log = self.current_trade_cycle_id or state.get("current_trade_cycle_id", f"NO_CYCLE_OCO_ACTIVE_{int(time.time())}")
        log_prefix = f"[{self.pair_symbol}][Ctx:{self.context_label}][Cycle:{active_cycle_id_for_log}]"

        logger.debug(f"{log_prefix} Status 3: OCO is active. Monitoring.")
        pass

    def _handle_trade_closure_and_loan_repayment(self, closed_trade_state_snapshot: Dict[str, Any], exit_reason: str, closed_order_details: Optional[Dict[str,Any]]):
        if not self.execution_client or not self.state_manager: return
        
        trade_cycle_id_of_closed_trade = closed_trade_state_snapshot.get("current_trade_cycle_id", f"CLOSURE_UNKNOWN_CYCLE_{int(time.time())}")
        log_prefix = f"[{self.pair_symbol}][Ctx:{self.context_label}][Cycle:{trade_cycle_id_of_closed_trade}]"

        logger.info(f"{log_prefix} Handling trade closure. Reason: {exit_reason}.")
        
        asset_to_repay: Optional[str] = closed_trade_state_snapshot.get("loan_details", {}).get("asset")
        if not asset_to_repay:
            position_side_closed = closed_trade_state_snapshot.get("position_side")
            if position_side_closed == "BUY": 
                asset_to_repay = self.quote_asset
            elif position_side_closed == "SELL": 
                asset_to_repay = self.base_asset
            else: 
                logger.error(f"{log_prefix} Cannot determine asset to repay: position_side missing or invalid in state snapshot.")
                asset_to_repay = None
        
        if asset_to_repay:
            current_loans_for_asset = self.execution_client.get_active_margin_loans(
                asset=asset_to_repay, 
                isolated_symbol_pair=self.pair_symbol if self.is_isolated_margin_trading else None
            )
            actual_loan_to_repay = sum(float(l.get("borrowed",0.0)) for l in current_loans_for_asset)
            
            min_repay_threshold = SIGNIFICANT_LOAN_THRESHOLD_USDC_EQUIVALENT if asset_to_repay == self.quote_asset else MIN_EXECUTED_QTY_THRESHOLD * 0.9 
            
            if actual_loan_to_repay > min_repay_threshold:
                logger.info(f"{log_prefix} Attempting to repay loan of {actual_loan_to_repay:.8f} {asset_to_repay}.")
                repay_res = self.execution_client.repay_margin_loan(
                    asset=asset_to_repay, 
                    amount=f"{actual_loan_to_repay:.8f}", 
                    isolated_symbol_pair=self.pair_symbol if self.is_isolated_margin_trading else None
                )
                self._log_trade_event("LOAN_REPAY_ATTEMPT", {"asset": asset_to_repay, "amount_attempted": actual_loan_to_repay, "api_response": repay_res})
                if not (repay_res and (repay_res.get("status") == "SUCCESS" or repay_res.get("tranId"))): 
                    logger.error(f"{log_prefix} Loan repayment for {asset_to_repay} might have failed or status unknown. Response: {repay_res}")
                    self.state_manager.set_last_error(f"Repayment for {asset_to_repay} failed/unknown: {repay_res.get('message','No response') if repay_res else 'No response'}")
                else:
                    logger.info(f"{log_prefix} Loan repayment for {actual_loan_to_repay:.8f} {asset_to_repay} seems successful. TranID: {repay_res.get('tranId')}")
            else:
                logger.info(f"{log_prefix} No significant loan ({actual_loan_to_repay:.8f} {asset_to_repay}) found to repay for asset {asset_to_repay} or below threshold {min_repay_threshold}.")
        else:
            logger.warning(f"{log_prefix} Could not determine asset to repay for loan.")

        
        exit_px_val, pnl_est = None, None
        if closed_order_details and closed_order_details.get("status") == "FILLED":
             exec_q = float(closed_order_details.get("executedQty",0.0))
             cum_q = float(closed_order_details.get("cummulativeQuoteQty",0.0))
             if exec_q > 0: exit_px_val = cum_q / exec_q
        
        entry_px = closed_trade_state_snapshot.get("position_entry_price")
        qty = closed_trade_state_snapshot.get("position_quantity")
        side = closed_trade_state_snapshot.get("position_side")
        if exit_px_val and entry_px and qty and side:
            pnl_est = (exit_px_val - float(entry_px)) * float(qty) if side == "BUY" else (float(entry_px) - exit_px_val) * float(qty)
            logger.info(f"{log_prefix} Estimated PnL for closed trade: {pnl_est:.2f} USDC (Entry: {entry_px}, Exit: {exit_px_val}, Qty: {qty}, Side: {side})")

        self.state_manager.record_closed_trade(exit_reason, exit_px_val, pnl_est, closed_order_details)
        self.state_manager.transition_to_status_1(exit_reason, closed_order_details) 
        self.current_trade_cycle_id = None 
        self.oco_confirmation_attempts = 0
        logger.info(f"{log_prefix} Trade cycle {trade_cycle_id_of_closed_trade} closed. Manager reset to STATUT_1.")

    def _periodic_full_state_sync(self):
        logger.info(f"[{self.pair_symbol}][Ctx:{self.context_label}] Performing periodic full state sync...")
        self._determine_initial_status(is_periodic_sync=True)
        self.last_full_state_sync_time = datetime.now(timezone.utc)
        logger.info(f"[{self.pair_symbol}][Ctx:{self.context_label}] Periodic full state sync completed.")


    def run(self):
        logger.info(f"[{self.pair_symbol}][Ctx:{self.context_label}] Starting LiveTradingManager main loop...")
        try:
            while not self.shutdown_event.is_set():
                current_time_utc = datetime.now(timezone.utc)
                if self.last_full_state_sync_time is None or \
                   (current_time_utc - self.last_full_state_sync_time) >= timedelta(minutes=FULL_STATE_SYNC_INTERVAL_MINUTES):
                    self._periodic_full_state_sync()
                
                new_agg_candle_generated = self._check_new_1min_kline_and_trigger_preprocessing()
                
                self._check_and_process_orders_via_rest()
                
                if not self.state_manager: self.shutdown_event.set(); break 
                current_status = self.state_manager.get_current_status()
                state_snapshot = self.state_manager.get_state_snapshot()
                
                self.current_trade_cycle_id = state_snapshot.get("current_trade_cycle_id") 

                if new_agg_candle_generated: 
                    if current_status == STATUT_1_NO_TRADE_NO_OCO:
                        if not state_snapshot.get("pending_entry_order_id"):
                            self._handle_status_1_no_trade()
                    elif current_status == STATUT_2_ENTRY_FILLED_OCO_PENDING:
                        if not state_snapshot.get("pending_oco_list_client_order_id"):
                            self._handle_status_2_oco_pending()
                    elif current_status == STATUT_3_OCO_ACTIVE: 
                        self._handle_status_3_oco_active() 
                    else: 
                        logger.error(f"[{self.pair_symbol}][Ctx:{self.context_label}][Cycle:{self.current_trade_cycle_id}] Reached unknown status '{current_status}'. Resetting to STATUT_1.")
                        self._cancel_all_orders_for_pair(f"[UNKNOWN_STATUS_RESET_Ctx:{self.context_label}]")
                        self.state_manager.transition_to_status_1("UNKNOWN_STATUS_RESET")
                        self.current_trade_cycle_id = None
                else:
                    log_msg_no_candle = f"[{self.pair_symbol}][Ctx:{self.context_label}] No new aggregated candle."
                    if current_status == STATUT_1_NO_TRADE_NO_OCO and state_snapshot.get("pending_entry_order_id"):
                         log_msg_no_candle += f" Waiting for entry {state_snapshot.get('pending_entry_order_id')}."
                    elif current_status == STATUT_2_ENTRY_FILLED_OCO_PENDING and state_snapshot.get("pending_oco_list_client_order_id"):
                         log_msg_no_candle += f" Waiting for OCO confirmation {state_snapshot.get('pending_oco_list_client_order_id')}."
                    elif current_status == STATUT_3_OCO_ACTIVE:
                         log_msg_no_candle += " OCO active, monitoring."
                    logger.debug(log_msg_no_candle + f" Current status: {current_status}")

                
                if self.shutdown_event.wait(timeout=MAIN_LOOP_SLEEP_S): break
        except Exception as e:
            logger.critical(f"[{self.pair_symbol}][Ctx:{self.context_label}] CRITICAL ERROR in manager loop: {e}", exc_info=True)
            if self.state_manager: self.state_manager.set_last_error(f"Critical loop error: {str(e)[:250]}")
        finally:
            self.stop_trading()

    def stop_trading(self):
        if self.shutdown_event.is_set(): return 
        self.shutdown_event.set()
        ctx_log = self.context_label or "UNKNOWN_CTX"
        logger.info(f"[{self.pair_symbol}][Ctx:{ctx_log}] LiveTradingManager signaled to stop. Cleaning up...")
        if self.execution_client and hasattr(self.execution_client, 'close'): 
            try:
                self.execution_client.close()
            except Exception as e_close:
                logger.error(f"[{self.pair_symbol}][Ctx:{ctx_log}] Error during execution_client close: {e_close}")
        logger.info(f"[{self.pair_symbol}][Ctx:{ctx_log}] LiveTradingManager stopped.")
