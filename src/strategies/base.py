import logging
import math
import time
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

try:
    from src.utils.exchange_utils import (adjust_precision,
                                          get_filter_value,
                                          get_precision_from_filter)
except ImportError:
    def adjust_precision(value, precision, method=None): return value # type: ignore
    def get_precision_from_filter(info, ftype, key): return 8 # type: ignore
    def get_filter_value(info, ftype, key): return None # type: ignore
    logging.getLogger(__name__).error("Failed to import exchange_utils. Using dummy functions.")


logger = logging.getLogger(__name__)

class BaseStrategy(ABC):
    def __init__(self, params: dict):
        if not isinstance(params, dict):
            raise TypeError("Strategy parameters must be provided as a dictionary.")
        self.params = params
        self._signals: Optional[pd.DataFrame] = None
        logger.debug(f"Initializing strategy {self.__class__.__name__} with params: {params}")

    @abstractmethod
    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Processes the input data, primarily verifying the presence of pre-calculated indicators.

        In the refactored backtesting flow, 'data' is a 1-minute DataFrame where 
        strategy-specific indicator columns (e.g., 'KAMA_strat', 'ATR_strat', 'MA_FAST_strat')
        have already been prepared by the ObjectiveEvaluator. These indicators are
        derived from aggregated K-lines and pre-calculated ATRs based on Optuna parameters 
        (frequency, period).

        This method in concrete strategy classes should:
        1. Verify the presence of the expected final indicator columns (e.g., self.kama_col_strat)
           that were prepared by ObjectiveEvaluator.
        2. Perform any minor post-processing or derive secondary signals if necessary,
           using these already prepared columns. For example, calculating a difference
           between two pre-calculated MAs.
        3. It MUST NOT recalculate primary indicators like MAs, KAMA, PSAR, ADX, or the 
           main ATR for stop loss/take profit from raw OHLCV data, as this is now handled 
           upstream by ObjectiveEvaluator using the enriched data source.
        
        For live trading, this method's behavior should align. The LiveTradingManager 
        should aim to provide data in a similar format, with indicators pre-calculated 
        based on the strategy's deployed parameters.

        Args:
            data (pd.DataFrame): DataFrame with 1-minute resolution, expected to contain
                                 pre-calculated and named indicator columns (e.g., 'MA_FAST_strat')
                                 and the original 1-minute OHLCV columns.

        Returns:
            pd.DataFrame: The input DataFrame, typically unchanged or with very minor additions
                          if secondary signals/transformations were performed.
        """
        raise NotImplementedError("Subclasses must implement _calculate_indicators.")

    @abstractmethod
    def generate_signals(self, data: pd.DataFrame):
        """
        Generates trading signals based on the input data.

        The input 'data' DataFrame is expected to have a 1-minute resolution and
        contain all necessary final indicator columns (e.g., 'MA_FAST_strat', 'ATR_strat')
        prepared by an upstream process like ObjectiveEvaluator (for backtesting)
        or LiveTradingManager (for live trading). It also contains the original 1-minute
        OHLCV data.

        This method should use these pre-calculated final indicator columns and the 
        1-minute 'close' (or other 1-min OHLCV) for crossover conditions or reference.
        The SL/TP levels should be calculated using the pre-calculated 'ATR_strat' column.
        The method populates self._signals with entry/exit signals and SL/TP levels.

        Args:
            data (pd.DataFrame): DataFrame with 1-minute resolution, containing
                                 pre-calculated final indicator columns and 1-min OHLCV.
        """
        raise NotImplementedError("Subclasses must implement generate_signals.")

    @abstractmethod
    def generate_order_request(self,
                               data: pd.DataFrame,
                               symbol: str,
                               current_position: int,
                               available_capital: float,
                               symbol_info: dict
                               ) -> Optional[Tuple[Dict[str, Any], Dict[str, float]]]:
        """
        Generates an order request for live trading or simulation.

        The input 'data' DataFrame is expected to have a 1-minute resolution and
        contain all necessary final indicator columns (e.g., 'MA_FAST_strat', 'ATR_strat')
        prepared by an upstream process. The latest row of this DataFrame is typically used
        to make a trading decision.

        Args:
            data (pd.DataFrame): DataFrame with 1-minute resolution, containing
                                 pre-calculated final indicator columns and 1-min OHLCV. 
                                 The latest row is typically used.
            symbol (str): The trading symbol.
            current_position (int): Current position size (0 for none, >0 for long, <0 for short).
            available_capital (float): Available capital for trading.
            symbol_info (dict): Exchange-provided information about the symbol (filters, precision).

        Returns:
            Optional[Tuple[Dict[str, Any], Dict[str, float]]]:
                A tuple containing (entry_order_params, sl_tp_raw_prices_dict), or None if no order.
                - entry_order_params: Dictionary of parameters for placing the entry order.
                - sl_tp_raw_prices_dict: Dictionary with 'sl_price' and 'tp_price' (raw, unadjusted values).
        """
        raise NotImplementedError("Subclasses must implement generate_order_request.")

    def get_signals(self) -> pd.DataFrame:
        if self._signals is None:
            logger.error("generate_signals must be called before get_signals.")
            return pd.DataFrame(columns=['entry_long', 'exit_long', 'entry_short', 'exit_short', 'sl', 'tp'])

        bool_cols = ['entry_long', 'exit_long', 'entry_short', 'exit_short']
        for col in bool_cols:
            if col not in self._signals.columns:
                self._signals[col] = False 
            else:
                self._signals.loc[:, col] = self._signals[col].fillna(False).astype(bool)
        
        float_cols = ['sl', 'tp']
        for col in float_cols:
            if col not in self._signals.columns:
                self._signals[col] = np.nan 
            else:
                self._signals.loc[:, col] = pd.to_numeric(self._signals[col], errors='coerce')
        return self._signals

    def get_params(self) -> Dict[str, Any]:
        return self.params

    def get_param(self, key: str, default: Any = None) -> Any:
        return self.params.get(key, default)

    def _calculate_quantity(self,
                            entry_price: float,
                            available_capital: float,
                            qty_precision: Optional[int],
                            symbol_info: dict,
                            symbol: Optional[str] = None 
                           ) -> Optional[float]:
        symbol_log = symbol or symbol_info.get('symbol', 'N/A_SYMBOL')
        if available_capital <= 0 or entry_price <= 0:
            return None
        if qty_precision is None or not isinstance(qty_precision, int) or qty_precision < 0:
             qty_precision = 8

        leverage = float(self.get_param('margin_leverage', 1.0))
        capital_alloc_pct = float(self.get_param('capital_allocation_pct', 0.90)) 
        if not (0 < capital_alloc_pct <= 1.0): capital_alloc_pct = 0.90 
        if leverage <= 0: leverage = 1.0 

        notional_target = available_capital * capital_alloc_pct * leverage
        raw_quantity = notional_target / entry_price
        
        adjusted_quantity = adjust_precision(raw_quantity, qty_precision, math.floor)
        if adjusted_quantity is None or adjusted_quantity <= 1e-9: 
            return None

        min_qty_filter = get_filter_value(symbol_info, 'LOT_SIZE', 'minQty')
        max_qty_filter = get_filter_value(symbol_info, 'LOT_SIZE', 'maxQty')

        if min_qty_filter is not None and adjusted_quantity < min_qty_filter * (1 - 1e-9): 
            return None
        if max_qty_filter is not None and adjusted_quantity > max_qty_filter * (1 + 1e-9): 
            adjusted_quantity = adjust_precision(max_qty_filter, qty_precision, math.floor)
            if adjusted_quantity is None or (min_qty_filter is not None and adjusted_quantity < min_qty_filter * (1 - 1e-9)):
                return None
        
        price_prec_filter = get_precision_from_filter(symbol_info, 'PRICE_FILTER', 'tickSize')
        if price_prec_filter is None: price_prec_filter = 8 
        
        entry_price_adj_check = adjust_precision(entry_price, price_prec_filter, round) 
        if entry_price_adj_check is None or entry_price_adj_check <= 0:
            return None

        if not self._validate_notional(adjusted_quantity, entry_price_adj_check, symbol_info):
            return None 
            
        return adjusted_quantity

    def _validate_notional(self, quantity: float, price: float, symbol_info: dict) -> bool:
        min_notional_filter = get_filter_value(symbol_info, 'MIN_NOTIONAL', 'minNotional')
        if min_notional_filter is None: 
            min_notional_filter = get_filter_value(symbol_info, 'NOTIONAL', 'minNotional')

        if min_notional_filter is not None:
            notional_value = abs(quantity * price)
            if notional_value < min_notional_filter * (1 - 1e-9): 
                return False
        return True

    def _build_entry_params_formatted(self,
                                      symbol: str,
                                      side: str,
                                      quantity_str: str,
                                      entry_price_str: Optional[str] = None, 
                                      order_type: str = "LIMIT" 
                                     ) -> Dict[str, Any]:
        current_timestamp_ms = int(time.time() * 1000)
        client_order_id = f"entry_{symbol[:4].lower()}_{current_timestamp_ms % 1000000}_{uuid.uuid4().hex[:4]}"

        params: Dict[str, Any] = {
            "symbol": symbol.upper(),
            "side": side.upper(),
            "type": order_type.upper(),
            "quantity": quantity_str,
            "newClientOrderId": client_order_id,
            "newOrderRespType": "FULL" 
        }

        if order_type.upper() in ["LIMIT", "LIMIT_MAKER"]:
            if not entry_price_str:
                raise ValueError(f"Entry price required for {order_type} orders.")
            params["price"] = entry_price_str
            params["timeInForce"] = "GTC" 
        
        if side.upper() == 'BUY':
            params["sideEffectType"] = "MARGIN_BUY" 
        elif side.upper() == 'SELL':
            params["sideEffectType"] = "AUTO_BORROW_REPAY" 
        
        return params

    def _build_oco_params(self,
                          symbol: str,
                          position_side: str, 
                          executed_qty: float,
                          sl_price_raw: float,
                          tp_price_raw: float,
                          price_precision: int,
                          qty_precision: int,
                          symbol_info: dict 
                         ) -> Optional[Dict[str, Any]]:
        
        oco_side = 'SELL' if position_side.upper() == 'BUY' else 'BUY' 
        quantity_str = f"{executed_qty:.{qty_precision}f}"
        
        tick_size = 1 / (10**price_precision) if price_precision > 0 else 1.0

        if oco_side == 'SELL': 
            tp_price_adj = adjust_precision(tp_price_raw, price_precision, math.floor)
            sl_price_adj = adjust_precision(sl_price_raw, price_precision, math.ceil)
            
            if tp_price_adj is None or sl_price_adj is None or tp_price_adj <= sl_price_adj + tick_size * 0.5:
                return None
        elif oco_side == 'BUY': 
            tp_price_adj = adjust_precision(tp_price_raw, price_precision, math.ceil)
            sl_price_adj = adjust_precision(sl_price_raw, price_precision, math.floor)

            if tp_price_adj is None or sl_price_adj is None or tp_price_adj >= sl_price_adj - tick_size * 0.5:
                return None
        else:
            return None 

        sl_price_str = f"{sl_price_adj:.{price_precision}f}"
        tp_price_str = f"{tp_price_adj:.{price_precision}f}"

        current_timestamp_ms = int(time.time() * 1000)
        base_client_order_id = f"oco_{symbol[:4].lower()}_{current_timestamp_ms % 1000000}_{uuid.uuid4().hex[:4]}"
        
        oco_api_params: Dict[str, Any] = {
            "symbol": symbol.upper(),
            "side": oco_side, 
            "quantity": quantity_str,
            "price": tp_price_str, 
            "stopPrice": sl_price_str, 
            "stopLimitPrice": sl_price_str, 
            "stopLimitTimeInForce": "GTC",
            "listClientOrderId": base_client_order_id, 
            "sideEffectType": "AUTO_REPAY", 
            "newOrderRespType": "FULL"
        }
        return oco_api_params
