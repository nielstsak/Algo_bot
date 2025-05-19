import json
import logging
import math
import os
import time
import uuid
import datetime 
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple , Union

import numpy as np
import pandas as pd

try:
    from src.strategies.base import BaseStrategy
    from src.utils.exchange_utils import adjust_precision, get_precision_from_filter, get_filter_value
except ImportError as e:
    logging.critical(f"Simulator: Failed to import BaseStrategy or exchange_utils: {e}", exc_info=True)
    from abc import ABC, abstractmethod
    class BaseStrategy(ABC): # type: ignore
        def __init__(self, params: dict): self.params = params if params is not None else {}
        @abstractmethod
        def generate_signals(self, data: pd.DataFrame): raise NotImplementedError
        def get_signals(self) -> pd.DataFrame: raise NotImplementedError("Signals not generated")
        def get_params(self) -> Dict: return getattr(self, 'params', {})
        @abstractmethod
        def generate_order_request(self, data: pd.DataFrame, symbol: str, current_position: int, available_capital: float, symbol_info: dict) -> Optional[Tuple[Dict[str, Any], Dict[str, float]]]: raise NotImplementedError
    def adjust_precision(value, precision, method=None): return value # type: ignore
    def get_precision_from_filter(info, ftype, key): return 8 # type: ignore
    def get_filter_value(info, ftype, key): return None # type: ignore

try:
    from src.backtesting.performance import calculate_performance_metrics
except ImportError:
    def calculate_performance_metrics(*args, **kwargs) -> Dict[str, Any]: # type: ignore
        return {"error": "Performance calculation module missing.", "Sharpe Ratio": np.nan, "Total Net PnL USDC": 0.0, "Win Rate Pct": np.nan}
    logging.warning("Simulator: Could not import calculate_performance_metrics. Using placeholder.")

logger = logging.getLogger(__name__)

# Seuil par défaut pour l'arrêt anticipé si l'équité descend en dessous de cette valeur.
# Peut être surchargé par simulation_settings.
DEFAULT_EARLY_STOP_EQUITY_THRESHOLD = 800.0 

class BacktestSimulator:
    def __init__(self,
                 historical_data_with_indicators: pd.DataFrame, 
                 strategy_instance: BaseStrategy,
                 simulation_settings: Dict[str, Any],
                 output_dir: Optional[Union[str, Path]]): 
        
        self.sim_log_prefix = f"[{self.__class__.__name__}][{simulation_settings.get('symbol', 'N/A')}]"
        logger.info(f"{self.sim_log_prefix} Initialisation du simulateur...")
        
        self.early_stop_equity_threshold = float(simulation_settings.get('early_stop_equity_threshold', DEFAULT_EARLY_STOP_EQUITY_THRESHOLD))
        logger.info(f"{self.sim_log_prefix} Seuil d'arrêt anticipé pour l'équité: {self.early_stop_equity_threshold:.2f} USDC")
        logger.debug(f"{self.sim_log_prefix} Paramètres de simulation: Capital Initial={simulation_settings.get('initial_capital')}, Frais={simulation_settings.get('transaction_fee_pct')*100}%, Slippage={simulation_settings.get('slippage_pct')*100}%, Levier={simulation_settings.get('margin_leverage')}")


        if historical_data_with_indicators is None or historical_data_with_indicators.empty:
            raise ValueError("Historical data (with indicators) cannot be None or empty.")
        if not isinstance(strategy_instance, BaseStrategy):
            raise TypeError("strategy_instance must be an instance inheriting from BaseStrategy.")
        if simulation_settings is None:
            raise ValueError("simulation_settings cannot be None.")

        required_settings = ['initial_capital', 'transaction_fee_pct', 'slippage_pct', 'margin_leverage', 'symbol']
        missing_settings = [k for k in required_settings if k not in simulation_settings]
        if missing_settings:
            raise ValueError(f"Missing required simulation settings: {missing_settings}")
        if simulation_settings['initial_capital'] <= 0:
            raise ValueError("initial_capital must be positive.")

        self.strategy = strategy_instance
        self.settings = simulation_settings
        self.output_dir = Path(output_dir) if output_dir else None
        self.strategy_name = strategy_instance.__class__.__name__
        self.symbol = simulation_settings['symbol']

        self.data_input = historical_data_with_indicators.copy() 
        if not isinstance(self.data_input.index, pd.DatetimeIndex):
            if 'timestamp' in self.data_input.columns:
                self.data_input['timestamp'] = pd.to_datetime(self.data_input['timestamp'], errors='coerce', utc=True)
                self.data_input = self.data_input.set_index('timestamp')
                if self.data_input.index.isnull().any(): 
                    raise ValueError("Failed to convert 'timestamp' column to valid DatetimeIndex.")
            else:
                raise TypeError("Input historical_data_with_indicators must have a DatetimeIndex or a 'timestamp' column.")

        if self.data_input.index.tz is None:
             self.data_input.index = self.data_input.index.tz_localize('UTC')
        elif self.data_input.index.tz is not datetime.timezone.utc: 
             self.data_input.index = self.data_input.index.tz_convert('UTC')

        if not self.data_input.index.is_monotonic_increasing:
            self.data_input.sort_index(inplace=True)
        if not self.data_input.index.is_unique:
            initial_len = len(self.data_input)
            self.data_input = self.data_input[~self.data_input.index.duplicated(keep='first')]
            logger.debug(f"{self.sim_log_prefix} Removed {initial_len - len(self.data_input)} duplicate index entries.")
        
        required_ohlcv_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_ohlcv_cols = [c for c in required_ohlcv_cols if c not in self.data_input.columns]
        if missing_ohlcv_cols:
             raise ValueError(f"Input data missing OHLCV columns: {missing_ohlcv_cols}")

        nan_ohlcv_before = self.data_input[required_ohlcv_cols].isnull().sum().sum()
        if nan_ohlcv_before > 0:
            logger.debug(f"{self.sim_log_prefix} Found {nan_ohlcv_before} NaNs in OHLCV. Applying ffill & bfill.")
            self.data_input[required_ohlcv_cols] = self.data_input[required_ohlcv_cols].ffill().bfill() 
            if self.data_input[required_ohlcv_cols].isnull().sum().sum() > 0: 
                logger.debug(f"{self.sim_log_prefix} NaNs still present after ffill/bfill. Dropping rows with any OHLCV NaN.")
                self.data_input.dropna(subset=required_ohlcv_cols, inplace=True) 
        
        if self.data_input.empty:
            raise ValueError("Input data became empty after NaN handling for OHLCV.")
        
        logger.debug(f"{self.sim_log_prefix} Données d'entrée après nettoyage initial: Shape={self.data_input.shape}, Dates: {self.data_input.index.min()} à {self.data_input.index.max()}")
        logger.debug(f"{self.sim_log_prefix} Colonnes dans data_input: {self.data_input.columns.tolist()}")

        self.initial_capital = float(self.settings['initial_capital'])
        self.equity = self.initial_capital
        self.equity_curve = pd.Series(dtype=float) 
        self.position = 0.0 
        self.entry_price = 0.0 
        self.entry_timestamp: Optional[pd.Timestamp] = None 
        self.current_sl_price: Optional[float] = None 
        self.current_tp_price: Optional[float] = None 
        self.order_intent: Optional[Dict[str, Any]] = None 
        self.trades_log_full: List[Dict[str, Any]] = [] 
        self.trades_log_simple: List[Dict[str, Any]] = [] 
        self.order_id_counter = 0 
        self.early_stopped_reason: Optional[str] = None # Pour l'arrêt anticipé

        if self.output_dir:
            try:
                self.output_dir.mkdir(parents=True, exist_ok=True)
            except OSError as e:
                logger.warning(f"{self.sim_log_prefix} Could not create output directory {self.output_dir}: {e}. No detailed artifacts will be saved.")
                self.output_dir = None 

    def _calculate_quantity(self, entry_price: float) -> Optional[float]:
        if self.equity <= 0 or entry_price <= 0:
            logger.debug(f"{self.sim_log_prefix} Calcul quantité: Équité ({self.equity}) ou prix d'entrée ({entry_price}) invalide.")
            return None

        symbol_info_data = self.settings.get('symbol_info', {})
        qty_precision = get_precision_from_filter(symbol_info_data, 'LOT_SIZE', 'stepSize')
        if qty_precision is None: qty_precision = 8 
        
        qty = self.strategy._calculate_quantity(
            entry_price=entry_price,
            available_capital=self.equity, 
            qty_precision=qty_precision,
            symbol_info=symbol_info_data,
            symbol=self.symbol
        )
        logger.debug(f"{self.sim_log_prefix} Calcul quantité: Entrée Px={entry_price}, Capital={self.equity:.2f}, Qty Prec={qty_precision} -> Quantité calculée={qty}")
        return qty

    def _apply_slippage(self, price: float, side: str) -> float:
        slippage = self.settings.get('slippage_pct', 0.0)
        factor = 1 + slippage if side.upper() == 'BUY' else 1 - slippage
        slipped_price = max(0.0, price * factor) 
        return slipped_price

    def _calculate_commission(self, position_value: float) -> float:
        fee = self.settings.get('transaction_fee_pct', 0.0)
        commission = abs(position_value) * fee
        return commission

    def _generate_otoco_intent(self, timestamp: pd.Timestamp, entry_price: float, quantity: float,
                               side: str, sl_price: Optional[float], tp_price: Optional[float]) -> Dict[str, Any]:
        return {
            "intent_timestamp_utc": timestamp.isoformat(),
            "entry_side": side,
            "entry_price_theoretical": entry_price, 
            "quantity": quantity,
            "intended_sl": sl_price,
            "intended_tp": tp_price,
        }

    def run_simulation(self) -> Dict[str, Any]:
        logger.info(f"{self.sim_log_prefix} Démarrage de la simulation pour la stratégie '{self.strategy_name}'.")
        try:
            self.strategy.generate_signals(self.data_input) 
            signals_df = self.strategy.get_signals()
            if signals_df.empty:
                logger.warning(f"{self.sim_log_prefix} Aucun signal généré par la stratégie.")
            else:
                logger.debug(f"{self.sim_log_prefix} Signaux générés. Entry_long: {signals_df['entry_long'].sum()}, Entry_short: {signals_df['entry_short'].sum()}.")

        except Exception as e:
            logger.error(f"{self.sim_log_prefix} Erreur lors de strategy.generate_signals: {e}", exc_info=True)
            return self._finalize_simulation(success=False, error_message=f"generate_signals failed: {e}")

        if not self.data_input.index.equals(signals_df.index):
            logger.warning(f"{self.sim_log_prefix} L'index des signaux ne correspond pas à l'index des données d'entrée. Réindexation des signaux.")
            signals_df = signals_df.reindex(self.data_input.index) 

        try:
            required_signal_cols = ['entry_long', 'exit_long', 'entry_short', 'exit_short', 'sl', 'tp']
            for col in required_signal_cols:
                if col not in signals_df.columns:
                    if col in ['sl', 'tp']:
                        signals_df[col] = np.nan
                    else:
                        signals_df[col] = False
            
            self.data_for_loop = pd.merge(self.data_input, signals_df, left_index=True, right_index=True, how='left')
            
            # Log des données d'entrée et des signaux (échantillon)
            log_cols_sample = ['close', 'entry_long', 'entry_short', 'sl', 'tp']
            # Essayer d'ajouter une colonne ATR si elle existe pour le contexte
            atr_col_present = next((col for col in self.data_for_loop.columns if 'ATR_strat' in col), None)
            if atr_col_present: log_cols_sample.append(atr_col_present)
            
            # logger.debug(f"{self.sim_log_prefix} Données fusionnées pour la boucle. Premières lignes:\n{self.data_for_loop[log_cols_sample].head().to_string()}")
            # logger.debug(f"{self.sim_log_prefix} Données fusionnées pour la boucle. Dernières lignes:\n{self.data_for_loop[log_cols_sample].tail().to_string()}")


        except Exception as e:
             logger.error(f"{self.sim_log_prefix} Erreur lors de la fusion des signaux avec les données d'entrée: {e}", exc_info=True)
             return self._finalize_simulation(success=False, error_message=f"Error during signal join: {e}")

        signal_cols_bool = ['entry_long', 'exit_long', 'entry_short', 'exit_short']
        signal_cols_float = ['sl', 'tp']
        for col in signal_cols_bool:
            if col not in self.data_for_loop.columns: self.data_for_loop[col] = False 
            self.data_for_loop.loc[:, col] = self.data_for_loop[col].fillna(False).astype(bool)
        for col in signal_cols_float:
            if col not in self.data_for_loop.columns: self.data_for_loop[col] = np.nan 
            self.data_for_loop.loc[:, col] = pd.to_numeric(self.data_for_loop[col], errors='coerce')

        if self.data_for_loop.empty:
            logger.error(f"{self.sim_log_prefix} Données vides après la fusion des signaux.")
            return self._finalize_simulation(success=False, error_message="Empty data after signal join")
        
        self.equity_curve = pd.Series(index=self.data_for_loop.index, dtype=float)
        if not self.data_for_loop.empty: 
            self.equity_curve.iloc[0] = self.initial_capital
        else: 
            logger.error(f"{self.sim_log_prefix} Données vides avant l'initialisation de la courbe d'équité.")
            return self._finalize_simulation(success=False, error_message="Data empty before equity init")

        last_equity_update_idx = 0 
        logger.info(f"{self.sim_log_prefix} Démarrage de la boucle de simulation sur {len(self.data_for_loop)} bougies.")

        for i in range(1, len(self.data_for_loop)):
            current_timestamp = self.data_for_loop.index[i]
            current_row = self.data_for_loop.iloc[i]

            high_price = current_row['high']
            low_price = current_row['low']
            close_price = current_row['close']
            open_price = current_row['open'] 

            if pd.isna(high_price) or pd.isna(low_price) or pd.isna(close_price) or pd.isna(open_price):
                 self.equity_curve.iloc[i] = self.equity_curve.iloc[last_equity_update_idx] 
                 continue

            self.equity_curve.iloc[i] = self.equity_curve.iloc[last_equity_update_idx] 
            current_bar_equity = self.equity 

            exit_reason: Optional[str] = None
            exit_price: Optional[float] = None
            exit_timestamp: Optional[pd.Timestamp] = None

            if self.position != 0: 
                if self.current_sl_price is not None and pd.notna(self.current_sl_price):
                    if (self.position > 0 and low_price <= self.current_sl_price) or \
                       (self.position < 0 and high_price >= self.current_sl_price):
                        exit_reason = 'SL_HIT'; exit_price = self.current_sl_price; exit_timestamp = current_timestamp
                        logger.debug(f"{self.sim_log_prefix} SL touché à {exit_price} pour position {self.position_side_log()}.")
                
                if exit_reason is None and self.current_tp_price is not None and pd.notna(self.current_tp_price):
                     if (self.position > 0 and high_price >= self.current_tp_price) or \
                        (self.position < 0 and low_price <= self.current_tp_price):
                         exit_reason = 'TP_HIT'; exit_price = self.current_tp_price; exit_timestamp = current_timestamp
                         logger.debug(f"{self.sim_log_prefix} TP touché à {exit_price} pour position {self.position_side_log()}.")
                
                if exit_reason is None: 
                     if (self.position > 0 and current_row.get('exit_long', False)) or \
                        (self.position < 0 and current_row.get('exit_short', False)):
                         exit_reason = 'EXIT_SIGNAL'; exit_price = open_price; exit_timestamp = current_timestamp 
                         logger.debug(f"{self.sim_log_prefix} Signal de sortie de stratégie pour position {self.position_side_log()}.")

                if exit_reason and exit_price is not None:
                    simulated_exit_price = self._apply_slippage(exit_price, 'SELL' if self.position > 0 else 'BUY')
                    position_value_at_exit = abs(self.position) * simulated_exit_price 
                    pnl_gross = (simulated_exit_price - self.entry_price) * self.position 
                    commission_exit = self._calculate_commission(position_value_at_exit)
                    pnl_net = pnl_gross - commission_exit

                    equity_before_exit = self.equity 
                    self.equity += pnl_net
                    self.equity_curve.iloc[i] = self.equity 
                    last_equity_update_idx = i
                    logger.info(f"{self.sim_log_prefix} SORTIE: {exit_reason} @ {simulated_exit_price:.4f}, Qty: {abs(self.position):.4f}, PnL Net: {pnl_net:.2f}, Équité: {self.equity:.2f}")

                    pnl_percent = (pnl_net / equity_before_exit) * 100 if equity_before_exit > 1e-9 else 0.0
                    
                    trade_closure_details = {
                        "entry_timestamp": self.entry_timestamp.isoformat() if self.entry_timestamp else None,
                        "exit_timestamp": exit_timestamp.isoformat() if exit_timestamp else current_timestamp.isoformat(), 
                        "entry_price": self.entry_price, "exit_price": simulated_exit_price, 
                        "pnl_usd": pnl_net, "pnl_percent": pnl_percent, "exit_reason": exit_reason,
                        "success": 'YES' if pnl_net > 0 else 'NO' 
                    }
                    full_trade_record = {"order_intent": self.order_intent, "closure_details": trade_closure_details}
                    self.trades_log_full.append(full_trade_record)
                    
                    simple_trade_record = {
                        "entry_timestamp": self.entry_timestamp, "exit_timestamp": exit_timestamp, "symbol": self.symbol,
                        "side": "LONG" if self.position > 0 else "SHORT", "entry_price": self.entry_price, "exit_price": simulated_exit_price,
                        "quantity": abs(self.position), 
                        "initial_sl_price": self.order_intent.get('intended_sl') if self.order_intent else None,
                        "initial_tp_price": self.order_intent.get('intended_tp') if self.order_intent else None,
                        "exit_reason": exit_reason, "pnl_gross_usd": pnl_gross, "commission_usd": commission_exit,
                        "pnl_net_usd": pnl_net, "pnl_net_pct": pnl_percent, "cumulative_equity_usd": self.equity
                    }
                    self.trades_log_simple.append(simple_trade_record)

                    self.position = 0.0; self.entry_price = 0.0; self.entry_timestamp = None
                    self.current_sl_price = None; self.current_tp_price = None; self.order_intent = None

                    # Vérification pour l'arrêt anticipé après la clôture d'un trade
                    if self.equity < self.early_stop_equity_threshold:
                        self.early_stopped_reason = "EQUITY_TOO_LOW"
                        logger.warning(f"{self.sim_log_prefix} ARRÊT ANTICIPÉ: Équité ({self.equity:.2f}) < Seuil ({self.early_stop_equity_threshold:.2f}) à {current_timestamp}.")
                        self.equity_curve.iloc[i:] = self.equity 
                        break 

            if self.position == 0: 
                entry_side: Optional[str] = None
                theoretical_entry_price: Optional[float] = None 
                sl_price_signal = current_row.get('sl') 
                tp_price_signal = current_row.get('tp')
                sl_price_signal = float(sl_price_signal) if pd.notna(sl_price_signal) else None
                tp_price_signal = float(tp_price_signal) if pd.notna(tp_price_signal) else None

                if current_row.get('entry_long', False):
                    entry_side = "LONG"; theoretical_entry_price = open_price 
                elif current_row.get('entry_short', False):
                    entry_side = "SHORT"; theoretical_entry_price = open_price 

                if entry_side and theoretical_entry_price is not None:
                    # Log des indicateurs et SL/TP du signal
                    # atr_val_log = current_row.get(atr_col_present, np.nan) if atr_col_present else np.nan
                    # logger.debug(f"{self.sim_log_prefix} Signal d'entrée {entry_side} à {current_timestamp}. Prix Théo: {theoretical_entry_price:.4f}, SL Sig: {sl_price_signal}, TP Sig: {tp_price_signal}, ATR_strat: {atr_val_log:.4f}")
                    
                    valid_sl_tp_from_signal = True
                    if sl_price_signal is None or tp_price_signal is None:
                        valid_sl_tp_from_signal = False
                    elif entry_side == "LONG": 
                         if sl_price_signal >= theoretical_entry_price: valid_sl_tp_from_signal = False
                         if tp_price_signal <= theoretical_entry_price: valid_sl_tp_from_signal = False
                    elif entry_side == "SHORT": 
                         if sl_price_signal <= theoretical_entry_price: valid_sl_tp_from_signal = False
                         if tp_price_signal >= theoretical_entry_price: valid_sl_tp_from_signal = False
                    
                    if not valid_sl_tp_from_signal:
                        logger.debug(f"{self.sim_log_prefix} SL/TP du signal invalide pour {entry_side} à {theoretical_entry_price:.4f} (SL: {sl_price_signal}, TP: {tp_price_signal}). Pas d'entrée.")
                    else: 
                        quantity = self._calculate_quantity(theoretical_entry_price) 
                        
                        if quantity is not None and quantity > 0:
                            self.current_sl_price = sl_price_signal
                            self.current_tp_price = tp_price_signal
                            self.order_intent = self._generate_otoco_intent(current_timestamp, theoretical_entry_price, quantity, entry_side, sl_price_signal, tp_price_signal)
                            
                            simulated_entry_price = self._apply_slippage(theoretical_entry_price, 'BUY' if entry_side == 'LONG' else 'SELL')
                            position_value = quantity * simulated_entry_price
                            commission_entry = self._calculate_commission(position_value)
                            
                            leverage = float(self.settings.get('margin_leverage', 1.0))
                            if leverage <= 0: leverage = 1.0
                            required_margin = position_value / leverage 

                            if current_bar_equity - commission_entry >= required_margin: 
                                self.equity = current_bar_equity - commission_entry 
                                self.equity_curve.iloc[i] = self.equity 
                                last_equity_update_idx = i

                                self.position = quantity if entry_side == 'LONG' else -quantity
                                self.entry_price = simulated_entry_price
                                self.entry_timestamp = current_timestamp 
                                logger.info(f"{self.sim_log_prefix} ENTRÉE: {entry_side} Qty: {quantity:.4f} @ {simulated_entry_price:.4f} (Théo: {theoretical_entry_price:.4f}). SL: {self.current_sl_price:.4f}, TP: {self.current_tp_price:.4f}. Équité: {self.equity:.2f}")
                                
                                # Vérification pour l'arrêt anticipé après l'entrée (si la commission fait chuter l'équité)
                                if self.equity < self.early_stop_equity_threshold:
                                    self.early_stopped_reason = "EQUITY_TOO_LOW_POST_ENTRY"
                                    logger.warning(f"{self.sim_log_prefix} ARRÊT ANTICIPÉ: Équité ({self.equity:.2f}) < Seuil ({self.early_stop_equity_threshold:.2f}) juste après l'entrée à {current_timestamp}.")
                                    self.equity_curve.iloc[i:] = self.equity
                                    # Clôturer immédiatement la position si l'équité est trop basse après l'entrée
                                    # Cela nécessite de simuler une clôture au prochain prix disponible (ex: close_price de la barre actuelle)
                                    # Pour simplifier, on sort de la boucle. _finalize_simulation gérera la clôture.
                                    break 
                            else:
                                logger.warning(f"{self.sim_log_prefix} Marge insuffisante pour l'entrée {entry_side}. Requis: {required_margin:.2f} + Comm: {commission_entry:.2f}, Dispo: {current_bar_equity:.2f}. Pas d'entrée.")
                                self.order_intent = None; self.current_sl_price = None; self.current_tp_price = None 
                        else:
                            logger.debug(f"{self.sim_log_prefix} Quantité calculée nulle ou invalide ({quantity}). Pas d'entrée pour signal {entry_side}.")
                            self.current_sl_price = None; self.current_tp_price = None 
            
            if self.equity <= 0 and not self.early_stopped_reason : # Si l'équité est nulle pour une autre raison
                self.early_stopped_reason = "EQUITY_ZERO_OR_NEGATIVE"
                logger.warning(f"{self.sim_log_prefix} ARRÊT ANTICIPÉ: Équité ({self.equity:.2f}) <= 0 à {current_timestamp}.")
                self.equity_curve.iloc[i:] = self.equity 
                break 
        
        logger.info(f"{self.sim_log_prefix} Fin de la boucle de simulation." + (f" Raison arrêt anticipé: {self.early_stopped_reason}" if self.early_stopped_reason else ""))
        return self._finalize_simulation()

    def _finalize_simulation(self, success: bool = True, error_message: Optional[str] = None) -> Dict[str, Any]:
        log_final_prefix = f"{self.sim_log_prefix}[Finalize]"
        logger.info(f"{log_final_prefix} Finalisation de la simulation. Succès initial: {success}" + (f", Erreur: {error_message}" if error_message else "") + (f", Arrêt anticipé: {self.early_stopped_reason}" if self.early_stopped_reason else ""))
        
        if self.position != 0 and not self.data_for_loop.empty : 
            last_timestamp = self.data_for_loop.index[-1]
            last_close_price = self.data_for_loop['close'].iloc[-1]
            if pd.isna(last_close_price): 
                 last_valid_closes = self.data_for_loop['close'].ffill()
                 if not last_valid_closes.empty and pd.notna(last_valid_closes.iloc[-1]):
                     last_close_price = last_valid_closes.iloc[-1]
                 else: 
                      last_close_price = None 
            
            if last_close_price is not None: 
                 logger.info(f"{log_final_prefix} Clôture de la position ouverte ({self.position_side_log()} {abs(self.position):.4f}) à la fin des données au prix {last_close_price:.4f}.")
                 exit_reason = self.early_stopped_reason if self.early_stopped_reason else 'END_OF_DATA'
                 exit_price = last_close_price; exit_timestamp = last_timestamp
                 simulated_exit_price = self._apply_slippage(exit_price, 'SELL' if self.position > 0 else 'BUY')
                 position_value_at_exit = abs(self.position) * simulated_exit_price
                 pnl_gross = (simulated_exit_price - self.entry_price) * self.position
                 commission_exit = self._calculate_commission(position_value_at_exit)
                 pnl_net = pnl_gross - commission_exit
                 equity_before_exit = self.equity 
                 self.equity += pnl_net

                 if not self.equity_curve.empty:
                     if last_timestamp not in self.equity_curve.index: 
                          new_index = self.equity_curve.index.union([last_timestamp])
                          self.equity_curve = self.equity_curve.reindex(new_index).ffill() 
                     self.equity_curve.loc[last_timestamp] = self.equity
                 
                 logger.info(f"{log_final_prefix} SORTIE ({exit_reason}): PnL Net: {pnl_net:.2f}, Équité Finale: {self.equity:.2f}")
                 pnl_percent = (pnl_net / equity_before_exit) * 100 if equity_before_exit > 1e-9 else 0.0
                 
                 trade_closure_details = {
                     "entry_timestamp": self.entry_timestamp.isoformat() if self.entry_timestamp else None,
                     "exit_timestamp": exit_timestamp.isoformat(), "entry_price": self.entry_price,
                     "exit_price": simulated_exit_price, "pnl_usd": pnl_net, "pnl_percent": pnl_percent,
                     "exit_reason": exit_reason, "success": 'YES' if pnl_net > 0 else 'NO' }
                 full_trade_record = {"order_intent": self.order_intent, "closure_details": trade_closure_details}
                 self.trades_log_full.append(full_trade_record)
                 simple_trade_record = {
                     "entry_timestamp": self.entry_timestamp, "exit_timestamp": exit_timestamp, "symbol": self.symbol,
                     "side": "LONG" if self.position > 0 else "SHORT", "entry_price": self.entry_price, "exit_price": simulated_exit_price,
                     "quantity": abs(self.position), 
                     "initial_sl_price": self.order_intent.get('intended_sl') if self.order_intent else None,
                     "initial_tp_price": self.order_intent.get('intended_tp') if self.order_intent else None,
                     "exit_reason": exit_reason, "pnl_gross_usd": pnl_gross, "commission_usd": commission_exit,
                     "pnl_net_usd": pnl_net, "pnl_net_pct": pnl_percent, "cumulative_equity_usd": self.equity }
                 self.trades_log_simple.append(simple_trade_record)
            else:
                logger.warning(f"{log_final_prefix} Impossible de clôturer la position ouverte en fin de données car le dernier prix de clôture est invalide.")
            self.position = 0.0 

        if not self.equity_curve.empty:
            self.equity_curve = self.equity_curve.ffill().fillna(self.initial_capital) 
        else: 
             if not self.data_for_loop.empty: self.equity_curve = pd.Series(self.initial_capital, index=self.data_for_loop.index)
             else: self.equity_curve = pd.Series(self.initial_capital, index=[pd.Timestamp.now(tz='UTC')]) 


        trades_df = pd.DataFrame(self.trades_log_simple)
        if not trades_df.empty:
            for col in ['entry_timestamp', 'exit_timestamp']:
                 if col in trades_df.columns:
                      trades_df[col] = pd.to_datetime(trades_df[col], errors='coerce', utc=True)
            num_cols_to_convert = ['entry_price', 'exit_price', 'quantity', 'initial_sl_price', 'initial_tp_price',
                                   'pnl_gross_usd', 'commission_usd', 'pnl_net_usd', 'pnl_net_pct', 'cumulative_equity_usd']
            for col in num_cols_to_convert:
                 if col in trades_df.columns:
                      trades_df[col] = pd.to_numeric(trades_df[col], errors='coerce')

        equity_series = self.equity_curve.rename('equity_usd') 
        if equity_series.index.name != 'timestamp': equity_series.index.name = 'timestamp' 

        metrics = {}
        if self.early_stopped_reason:
            logger.warning(f"{log_final_prefix} Simulation arrêtée prématurément: {self.early_stopped_reason}. Les métriques refléteront cela.")
            metrics = { 
                "Final Equity USDC": self.equity, 
                "Total Net PnL USDC": self.equity - self.initial_capital,
                "Total Trades": len(trades_df), 
                "Sharpe Ratio": np.nan, 
                "Max Drawdown Pct": calculate_performance_metrics(trades_df, equity_series, self.initial_capital).get("Max Drawdown Pct", np.nan), # Tenter de calculer MDD
                "Win Rate Pct": np.nan, 
                "Profit Factor": np.nan,
                "Status": f"EARLY_STOPPED_{self.early_stopped_reason.upper()}"
            }
            # S'assurer que les objectifs principaux sont très mauvais pour Optuna
            metrics["Total Net PnL USDC"] = min(metrics["Total Net PnL USDC"], -abs(self.initial_capital * 10)) # Pénalité sévère
            metrics["Win Rate Pct"] = 0.0 # Pire Win Rate
            metrics["Sharpe Ratio"] = -999 # Pire Sharpe
        elif not trades_df.empty and not equity_series.empty and equity_series.notna().any():
            try:
                metrics = calculate_performance_metrics(trades_df.dropna(subset=['pnl_net_usd']), equity_series.dropna(), self.initial_capital)
            except Exception as e:
                logger.error(f"{log_final_prefix} Erreur lors du calcul des métriques de performance: {e}", exc_info=True)
                metrics = {"error": f"Metric calculation failed: {e}", "Sharpe Ratio": np.nan, "Total Net PnL USDC": 0.0, "Win Rate Pct": np.nan}
        else: 
            logger.info(f"{log_final_prefix} Aucun trade exécuté ou courbe d'équité vide. Calcul des métriques par défaut.")
            metrics = { "Final Equity USDC": self.equity, "Total Net PnL USDC": self.equity - self.initial_capital,
                        "Total Trades": 0, "Sharpe Ratio": np.nan, "Max Drawdown Pct": 0.0,
                        "Win Rate Pct": np.nan, "Profit Factor": np.nan, "Status": "NO_TRADES" }
        
        if "Start Date" not in metrics : metrics["Start Date"] = equity_series.index.min().isoformat() if not equity_series.empty and equity_series.index.min() is not pd.NaT else None
        if "End Date" not in metrics: metrics["End Date"] = equity_series.index.max().isoformat() if not equity_series.empty and equity_series.index.max() is not pd.NaT else None


        logger.info(f"{log_final_prefix} Métriques de performance finales: PnL Total={metrics.get('Total Net PnL USDC', 'N/A')}, Win Rate={metrics.get('Win Rate Pct', 'N/A')}%, Status={metrics.get('Status', 'N/A')}")

        strat_params = self.strategy.get_params()
        serializable_settings = {k: v for k, v in self.settings.items() if isinstance(v, (str, int, float, bool, list, dict, type(None)))}
        if 'symbol_info' in serializable_settings and isinstance(serializable_settings['symbol_info'], dict) :
             serializable_settings['symbol_info'] = {
                 "symbol": serializable_settings['symbol_info'].get('symbol'),
                 "status": serializable_settings['symbol_info'].get('status'),
                 "filters_summary": f"{len(serializable_settings['symbol_info'].get('filters',[]))} filters present"
             }

        summary_data = { "simulation_settings": serializable_settings, "strategy_params": strat_params, "metrics": metrics }
        if not success and error_message: summary_data["error_message"] = error_message
        if self.early_stopped_reason: summary_data["early_stopped_reason"] = self.early_stopped_reason


        if self.output_dir:
            logger.info(f"{log_final_prefix} Sauvegarde des artefacts de simulation dans {self.output_dir}")
            try:
                parquet_engine = 'pyarrow' 
                trade_json_path = self.output_dir / f"trade_log_full_{self.strategy_name}_{self.symbol}.json"
                with open(trade_json_path, 'w', encoding='utf-8') as f: json.dump(self.trades_log_full, f, indent=2, default=str) 
                logger.debug(f"{log_final_prefix} Log de trade détaillé sauvegardé: {trade_json_path}")

                trades_path = self.output_dir / f"trades_{self.symbol}.parquet"
                if not trades_df.empty: trades_df.to_parquet(trades_path, index=False, engine=parquet_engine)
                logger.debug(f"{log_final_prefix} Log de trade simplifié sauvegardé: {trades_path}")
                
                equity_path = self.output_dir / f"equity_curve_{self.symbol}.parquet"
                equity_series.to_frame().to_parquet(equity_path, index=True, engine=parquet_engine) 
                logger.debug(f"{log_final_prefix} Courbe d'équité sauvegardée: {equity_path}")

                summary_json_path = self.output_dir / f"summary_{self.symbol}.json"
                serializable_summary_content = json.loads(json.dumps(summary_data, default=lambda x: x.item() if isinstance(x, np.generic) else str(x) if isinstance(x, (pd.Timestamp, datetime.datetime, datetime.timedelta)) else None if pd.isna(x) else x, allow_nan=True)) # type: ignore
                if 'metrics' in serializable_summary_content and isinstance(serializable_summary_content['metrics'], dict):
                    serializable_summary_content['metrics'] = {k: (None if isinstance(v, float) and (np.isnan(v) or np.isinf(v)) else v) for k, v in serializable_summary_content['metrics'].items()}

                with open(summary_json_path, 'w', encoding='utf-8') as f: json.dump(serializable_summary_content, f, indent=4)
                logger.info(f"{log_final_prefix} Résumé de la simulation sauvegardé: {summary_json_path}")
            except Exception as e:
                logger.error(f"{log_final_prefix} Erreur lors de la sauvegarde des artefacts de simulation: {e}", exc_info=True)
                summary_data["save_error"] = f"Failed to save artifacts: {e}"
        
        return {
            "trades": trades_df if not trades_df.empty else pd.DataFrame(), 
            "equity_curve": equity_series,
            "metrics": metrics,
            "summary_data": summary_data 
        }

    def position_side_log(self) -> str:
        if self.position > 0: return "LONG"
        if self.position < 0: return "SHORT"
        return "FLAT"

