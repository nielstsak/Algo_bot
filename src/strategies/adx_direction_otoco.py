import logging
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pandas_ta as ta

try:
    from .base import BaseStrategy
    from src.utils.exchange_utils import (adjust_precision,
                                          get_filter_value,
                                          get_precision_from_filter)
except ImportError:
    logging.getLogger(__name__).error("Failed to import BaseStrategy or exchange_utils. Using dummy BaseStrategy.")
    from abc import ABC, abstractmethod
    class BaseStrategy(ABC): # type: ignore
        def __init__(self, params: dict): self.params = params
        @abstractmethod
        def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame: raise NotImplementedError
        @abstractmethod
        def generate_signals(self, data: pd.DataFrame): raise NotImplementedError
        def get_signals(self) -> pd.DataFrame: raise NotImplementedError("Signals not generated") 
        def get_params(self) -> Dict: return self.params
        def get_param(self, key: str, default: Any = None) -> Any: return self.params.get(key, default)
        @abstractmethod
        def generate_order_request(self, data: pd.DataFrame, symbol: str, current_position: int, available_capital: float, symbol_info: dict) -> Optional[Tuple[Dict[str, Any], Dict[str, float]]]: raise NotImplementedError
        def _calculate_quantity(self, *args, **kwargs) -> Optional[float]: return None
        def _build_entry_params_formatted(self, *args, **kwargs) -> Optional[Dict]: return None
        def _build_oco_params(self, *args, **kwargs) -> Optional[Dict]: return None

logger = logging.getLogger(__name__)

class AdxDirectionOtocoStrategy(BaseStrategy):
    def __init__(self, params: dict):
        super().__init__(params)
        required_params = ['adx_period', 'adx_threshold', 
                           'atr_period', 'sl_atr_mult', 'tp_atr_mult', 
                           'indicateur_frequence_adx', 'atr_base_frequency'] 
        missing = [key for key in required_params if self.get_param(key) is None]
        if missing:
            raise ValueError(f"Missing parameters for {self.__class__.__name__}: {missing}")

        self.adx_col_strat = "ADX_strat" 
        self.dmp_col_strat = "DMP_strat" 
        self.dmn_col_strat = "DMN_strat" 
        self.atr_col_strat = "ATR_strat" 
        self._signals: Optional[pd.DataFrame] = None
        self.strategy_name_log_prefix = f"[{self.__class__.__name__}]"

    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()

        if self.adx_col_strat not in df.columns:
            df[self.adx_col_strat] = np.nan
        if self.dmp_col_strat not in df.columns:
            df[self.dmp_col_strat] = np.nan
        if self.dmn_col_strat not in df.columns:
            df[self.dmn_col_strat] = np.nan
        if self.atr_col_strat not in df.columns:
            df[self.atr_col_strat] = np.nan

        required_ohlc = ['open', 'high', 'low', 'close', 'volume']
        missing_ohlc = [col for col in required_ohlc if col not in df.columns]
        if missing_ohlc:
            for col in missing_ohlc: df[col] = np.nan
            
        return df

    def generate_signals(self, data: pd.DataFrame):
        df_with_indicators = self._calculate_indicators(data)
        
        required_cols_for_signal = [self.adx_col_strat, self.dmp_col_strat, self.dmn_col_strat, self.atr_col_strat, 'close']
        if df_with_indicators.empty or \
           any(col not in df_with_indicators.columns for col in required_cols_for_signal) or \
           df_with_indicators[required_cols_for_signal].isnull().all().all() or \
           len(df_with_indicators) < 2: 
            self._signals = pd.DataFrame(index=data.index, columns=['entry_long', 'exit_long', 'entry_short', 'exit_short', 'sl', 'tp'])
            self._signals[['sl', 'tp']] = np.nan
            self._signals[self._signals.select_dtypes(include=bool).columns] = False
            return

        adx_threshold = self.get_param('adx_threshold')
        sl_atr_mult = self.get_param('sl_atr_mult')
        tp_atr_mult = self.get_param('tp_atr_mult')

        adx_curr = df_with_indicators[self.adx_col_strat]
        dmp_curr = df_with_indicators[self.dmp_col_strat]
        dmn_curr = df_with_indicators[self.dmn_col_strat]
        close_curr = df_with_indicators['close']
        atr_curr = df_with_indicators[self.atr_col_strat]

        trending_curr = adx_curr > adx_threshold
        bullish_curr = dmp_curr > dmn_curr
        bearish_curr = dmn_curr > dmp_curr
        
        long_cond_now = trending_curr & bullish_curr & adx_curr.notna() & dmp_curr.notna() & dmn_curr.notna()
        short_cond_now = trending_curr & bearish_curr & adx_curr.notna() & dmp_curr.notna() & dmn_curr.notna()

        adx_prev = df_with_indicators[self.adx_col_strat].shift(1)
        dmp_prev = df_with_indicators[self.dmp_col_strat].shift(1)
        dmn_prev = df_with_indicators[self.dmn_col_strat].shift(1)

        trending_prev = adx_prev > adx_threshold
        bullish_prev = dmp_prev > dmn_prev
        bearish_prev = dmn_prev > dmp_prev
        
        long_cond_prev = trending_prev & bullish_prev & adx_prev.notna() & dmp_prev.notna() & dmn_prev.notna()
        short_cond_prev = trending_prev & bearish_prev & adx_prev.notna() & dmp_prev.notna() & dmn_prev.notna()

        entry_long_trigger = long_cond_now & ~long_cond_prev.fillna(False) 
        entry_short_trigger = short_cond_now & ~short_cond_prev.fillna(False) 

        signals_df = pd.DataFrame(index=df_with_indicators.index)
        signals_df['entry_long'] = entry_long_trigger
        signals_df['entry_short'] = entry_short_trigger
        signals_df['exit_long'] = False
        signals_df['exit_short'] = False

        signals_df['sl'] = np.where(
            entry_long_trigger & atr_curr.notna(), close_curr - sl_atr_mult * atr_curr,
            np.where(entry_short_trigger & atr_curr.notna(), close_curr + sl_atr_mult * atr_curr, np.nan)
        )
        signals_df['tp'] = np.where(
            entry_long_trigger & atr_curr.notna(), close_curr + tp_atr_mult * atr_curr,
            np.where(entry_short_trigger & atr_curr.notna(), close_curr - tp_atr_mult * atr_curr, np.nan)
        )
        
        self._signals = signals_df.reindex(data.index)

    def generate_order_request(self,
                               data: pd.DataFrame,
                               symbol: str,
                               current_position: int,
                               available_capital: float,
                               symbol_info: dict
                               ) -> Optional[Tuple[Dict[str, Any], Dict[str, float]]]:
        if current_position != 0:
            return None
        if len(data) < 2: 
            return None

        df_verified_indicators = self._calculate_indicators(data.copy())

        latest_data = df_verified_indicators.iloc[-1]
        previous_data = df_verified_indicators.iloc[-2]
        
        required_cols = ['close', self.adx_col_strat, self.dmp_col_strat, self.dmn_col_strat, self.atr_col_strat]
        if latest_data[required_cols].isnull().any() or \
           previous_data[['close', self.adx_col_strat, self.dmp_col_strat, self.dmn_col_strat]].isnull().any():
            return None

        adx_curr = latest_data[self.adx_col_strat]
        dmp_curr = latest_data[self.dmp_col_strat]
        dmn_curr = latest_data[self.dmn_col_strat]
        adx_prev = previous_data[self.adx_col_strat]
        dmp_prev = previous_data[self.dmp_col_strat]
        dmn_prev = previous_data[self.dmn_col_strat]
        
        close_curr = latest_data['close']
        atr_value = latest_data[self.atr_col_strat]

        adx_threshold = self.get_param('adx_threshold')
        sl_atr_mult = self.get_param('sl_atr_mult')
        tp_atr_mult = self.get_param('tp_atr_mult')

        side: Optional[str] = None
        stop_loss_price_raw: Optional[float] = None
        take_profit_price_raw: Optional[float] = None

        trending_curr = adx_curr > adx_threshold
        bullish_curr = dmp_curr > dmn_curr
        bearish_curr = dmn_curr > dmp_curr
        long_cond_now = trending_curr and bullish_curr

        trending_prev = adx_prev > adx_threshold
        bullish_prev = dmp_prev > dmn_prev
        long_cond_prev = trending_prev and bullish_prev
        
        short_cond_now = trending_curr and bearish_curr
        bearish_prev = dmn_prev > dmp_prev # Recalculate for clarity
        short_cond_prev = trending_prev and bearish_prev

        if long_cond_now and not long_cond_prev:
            side = 'BUY'
            stop_loss_price_raw = close_curr - sl_atr_mult * atr_value
            take_profit_price_raw = close_curr + tp_atr_mult * atr_value
        elif short_cond_now and not short_cond_prev:
            side = 'SELL'
            stop_loss_price_raw = close_curr + sl_atr_mult * atr_value
            take_profit_price_raw = close_curr - tp_atr_mult * atr_value

        if side and stop_loss_price_raw is not None and take_profit_price_raw is not None:
            if atr_value is None or atr_value <= 1e-9 or np.isnan(atr_value):
                return None

            price_precision_val = get_precision_from_filter(symbol_info, 'PRICE_FILTER', 'tickSize')
            qty_precision_val = get_precision_from_filter(symbol_info, 'LOT_SIZE', 'stepSize')
            if price_precision_val is None or qty_precision_val is None:
                return None

            quantity = self._calculate_quantity(
                entry_price=close_curr, available_capital=available_capital,
                qty_precision=qty_precision_val, symbol_info=symbol_info, symbol=symbol
            )
            if quantity is None or quantity <= 0:
                return None

            entry_price_adjusted = adjust_precision(close_curr, price_precision_val, round)
            if entry_price_adjusted is None:
                return None
            
            entry_price_str = f"{entry_price_adjusted:.{price_precision_val}f}"
            quantity_str = f"{quantity:.{qty_precision_val}f}"

            entry_params = self._build_entry_params_formatted(
                symbol=symbol, side=side, quantity_str=quantity_str,
                entry_price_str=entry_price_str, order_type="LIMIT"
            )
            if not entry_params:
                return None

            sl_tp_prices = {'sl_price': stop_loss_price_raw, 'tp_price': take_profit_price_raw}
            return entry_params, sl_tp_prices
        
        return None
