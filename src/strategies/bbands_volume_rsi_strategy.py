# Fichier: src/strategies/bbands_volume_rsi_strategy.py
import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from .base import BaseStrategy
    from src.utils.exchange_utils import (adjust_precision,
                                          get_filter_value, # Supposons que cette fonction est importée
                                          get_precision_from_filter)
except ImportError:
    # Fallback pour les tests unitaires ou environnements isolés
    logging.getLogger(__name__).error("BbandsVolumeRsiStrategy: Failed to import BaseStrategy or exchange_utils. Using dummy BaseStrategy.")
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
        def _calculate_quantity(self, *args, **kwargs) -> Optional[float]: return None # type: ignore
        def _build_entry_params_formatted(self, *args, **kwargs) -> Optional[Dict]: return None # type: ignore
        def _build_oco_params(self, *args, **kwargs) -> Optional[Dict]: return None # type: ignore
    # Définition factice de get_filter_value si l'import échoue, pour la complétude du fallback
    def get_filter_value(symbol_info: dict, filter_type: str, filter_key: str) -> Optional[Any]: return None


logger = logging.getLogger(__name__)

class BbandsVolumeRsiStrategy(BaseStrategy):
    """
    Stratégie de trading basée sur la cassure de range des Bandes de Bollinger,
    confirmée par le volume et le RSI. L'ATR est utilisé pour SL/TP.
    """
    def __init__(self, params: dict):
        """
        Initialise la stratégie avec les paramètres fournis.
        """
        super().__init__(params)
        required_params = [
            'bbands_period', 'bbands_std_dev', 'indicateur_frequence_bbands',
            'volume_ma_period', 'indicateur_frequence_volume',
            'rsi_period', 'indicateur_frequence_rsi',
            'rsi_buy_breakout_threshold', 'rsi_sell_breakout_threshold',
            'atr_period_sl_tp', 'atr_base_frequency_sl_tp', 'sl_atr_mult', 'tp_atr_mult'
        ]
        missing = [key for key in required_params if self.get_param(key) is None]
        if missing:
            err_msg = f"Paramètres manquants pour {self.__class__.__name__}: {missing}"
            logger.error(err_msg)
            raise ValueError(err_msg)

        self.bb_upper_col_strat = "BB_UPPER_strat"
        self.bb_middle_col_strat = "BB_MIDDLE_strat"
        self.bb_lower_col_strat = "BB_LOWER_strat"
        self.bb_bandwidth_col_strat = "BB_BANDWIDTH_strat"
        
        vol_freq_param = self.get_param('indicateur_frequence_volume')
        if vol_freq_param and vol_freq_param.lower() != "1min":
            self.volume_kline_col_strat = f"Kline_{vol_freq_param}_volume"
        else:
            self.volume_kline_col_strat = "volume" 

        self.volume_ma_col_strat = "Volume_MA_strat"
        self.rsi_col_strat = "RSI_strat"
        self.atr_col_strat = "ATR_strat"

        self._signals: Optional[pd.DataFrame] = None
        self.strategy_name_log_prefix = f"[{self.__class__.__name__}]"
        logger.info(f"{self.strategy_name_log_prefix} Stratégie initialisée avec les paramètres: {self.params}")
        logger.info(f"{self.strategy_name_log_prefix} Colonne volume source attendue pour la stratégie: {self.volume_kline_col_strat}")


    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        logger.debug(f"{self.strategy_name_log_prefix} Entrée _calculate_indicators. Colonnes reçues: {df.columns.tolist()}")
        expected_indicator_cols = [
            self.bb_upper_col_strat, self.bb_middle_col_strat, self.bb_lower_col_strat, 
            self.bb_bandwidth_col_strat, self.volume_kline_col_strat, 
            self.volume_ma_col_strat, self.rsi_col_strat, self.atr_col_strat
        ]
        for col_name in expected_indicator_cols:
            if col_name not in df.columns:
                df[col_name] = np.nan
                logger.warning(f"{self.strategy_name_log_prefix} Colonne indicateur attendue '{col_name}' manquante dans les données fournies à la stratégie. Ajoutée avec NaN.")
        
        required_ohlc = ['open', 'high', 'low', 'close', 'volume']
        for col in required_ohlc:
            if col not in df.columns:
                df[col] = np.nan
                logger.warning(f"{self.strategy_name_log_prefix} Colonne OHLC de base '{col}' manquante. Ajoutée avec NaN.")
        
        logger.debug(f"{self.strategy_name_log_prefix} Sortie _calculate_indicators. Colonnes présentes: {df.columns.tolist()}")
        return df

    def generate_signals(self, data: pd.DataFrame):
        logger.debug(f"{self.strategy_name_log_prefix} Génération des signaux (pour backtesting)...")
        df_with_indicators = self._calculate_indicators(data)
        required_cols_for_signal = [
            self.bb_upper_col_strat, self.bb_lower_col_strat,
            self.volume_kline_col_strat, self.volume_ma_col_strat,
            self.rsi_col_strat, self.atr_col_strat, 'close', 'open'
        ]
        if df_with_indicators.empty or \
           any(col not in df_with_indicators.columns for col in required_cols_for_signal) or \
           df_with_indicators[required_cols_for_signal].isnull().all().all() or \
           len(df_with_indicators) < 2 :
            logger.warning(f"{self.strategy_name_log_prefix} Données insuffisantes ou colonnes essentielles entièrement NaN pour la génération de signaux (backtesting).")
            self._signals = pd.DataFrame(index=data.index, columns=['entry_long', 'exit_long', 'entry_short', 'exit_short', 'sl', 'tp'])
            self._signals[['sl', 'tp']] = np.nan
            self._signals[self._signals.select_dtypes(include=bool).columns] = False
            return

        sl_atr_mult = self.get_param('sl_atr_mult')
        tp_atr_mult = self.get_param('tp_atr_mult')
        rsi_buy_thresh = self.get_param('rsi_buy_breakout_threshold')
        rsi_sell_thresh = self.get_param('rsi_sell_breakout_threshold')

        close_curr = df_with_indicators['close'] 
        close_prev = close_curr.shift(1)
        bb_upper_curr = df_with_indicators[self.bb_upper_col_strat]
        bb_lower_curr = df_with_indicators[self.bb_lower_col_strat]
        volume_kline_curr = df_with_indicators[self.volume_kline_col_strat]
        volume_ma_curr = df_with_indicators[self.volume_ma_col_strat]
        rsi_curr = df_with_indicators[self.rsi_col_strat]

        long_breakout_cond_curr = close_curr > bb_upper_curr
        long_volume_conf_curr = volume_kline_curr > volume_ma_curr
        long_rsi_conf_curr = rsi_curr > rsi_buy_thresh
        all_long_cond_curr = long_breakout_cond_curr & long_volume_conf_curr & long_rsi_conf_curr

        long_breakout_cond_prev = close_prev > bb_upper_curr.shift(1) 
        long_volume_conf_prev = volume_kline_curr.shift(1) > volume_ma_curr.shift(1)
        long_rsi_conf_prev = rsi_curr.shift(1) > rsi_buy_thresh
        all_long_cond_prev = long_breakout_cond_prev & long_volume_conf_prev & long_rsi_conf_prev
        entry_long_trigger = all_long_cond_curr & ~all_long_cond_prev.fillna(False)

        short_breakout_cond_curr = close_curr < bb_lower_curr
        short_volume_conf_curr = volume_kline_curr > volume_ma_curr 
        short_rsi_conf_curr = rsi_curr < rsi_sell_thresh
        all_short_cond_curr = short_breakout_cond_curr & short_volume_conf_curr & short_rsi_conf_curr

        short_breakout_cond_prev = close_prev < bb_lower_curr.shift(1)
        short_volume_conf_prev = volume_kline_curr.shift(1) > volume_ma_curr.shift(1)
        short_rsi_conf_prev = rsi_curr.shift(1) < rsi_sell_thresh
        all_short_cond_prev = short_breakout_cond_prev & short_volume_conf_prev & short_rsi_conf_prev
        entry_short_trigger = all_short_cond_curr & ~all_short_cond_prev.fillna(False)

        signals_df = pd.DataFrame(index=df_with_indicators.index)
        signals_df['entry_long'] = entry_long_trigger
        signals_df['entry_short'] = entry_short_trigger
        signals_df['exit_long'] = False 
        signals_df['exit_short'] = False 
        atr_curr = df_with_indicators[self.atr_col_strat]
        entry_price_series = df_with_indicators['close'] 
        signals_df['sl'] = np.where(
            entry_long_trigger & atr_curr.notna(), entry_price_series - sl_atr_mult * atr_curr,
            np.where(entry_short_trigger & atr_curr.notna(), entry_price_series + sl_atr_mult * atr_curr, np.nan)
        )
        signals_df['tp'] = np.where(
            entry_long_trigger & atr_curr.notna(), entry_price_series + tp_atr_mult * atr_curr,
            np.where(entry_short_trigger & atr_curr.notna(), entry_price_series - tp_atr_mult * atr_curr, np.nan)
        )
        self._signals = signals_df.reindex(data.index)
        logger.debug(f"{self.strategy_name_log_prefix} Signaux générés (backtesting). Longs: {signals_df['entry_long'].sum()}, Shorts: {signals_df['entry_short'].sum()}")

    def generate_order_request(self,
                               data: pd.DataFrame,
                               symbol: str,
                               current_position: int,
                               available_capital: float,
                               symbol_info: dict
                               ) -> Optional[Tuple[Dict[str, Any], Dict[str, float]]]:
        log_prefix_live = f"{self.strategy_name_log_prefix}[LiveOrder][{symbol}]"
        logger.info(f"{log_prefix_live} Appel de generate_order_request. Position actuelle: {current_position}, Capital dispo: {available_capital:.2f}")

        if current_position != 0:
            logger.info(f"{log_prefix_live} Position existante ({current_position}). Pas de nouvelle requête d'ordre.")
            return None
        
        if data.empty or len(data) < 2:
            logger.warning(f"{log_prefix_live} Données d'entrée vides ou insuffisantes (lignes: {len(data)}). Pas de nouvelle requête d'ordre.")
            return None

        df_verified_indicators = self._calculate_indicators(data.copy())
        
        if len(df_verified_indicators) < 2:
             logger.warning(f"{log_prefix_live} Pas assez de lignes après _calculate_indicators (lignes: {len(df_verified_indicators)}) pour évaluer les conditions. Pas de requête d'ordre.")
             return None
             
        latest_data = df_verified_indicators.iloc[-1]
        previous_data = df_verified_indicators.iloc[-2]

        required_cols_check = [
            'close', 'open', self.bb_upper_col_strat, self.bb_lower_col_strat,
            self.volume_kline_col_strat, self.volume_ma_col_strat,
            self.rsi_col_strat, self.atr_col_strat
        ]
        
        log_msg_indicators = f"{log_prefix_live} Valeurs indicateurs (dernière ligne avant vérif NaN): "
        for col in required_cols_check:
            val = latest_data.get(col)
            log_msg_indicators += f"{col}=" + (f"{val:.5f}" if isinstance(val, float) else str(val)) + "; "
        logger.info(log_msg_indicators)

        if latest_data[required_cols_check].isnull().any():
            nan_cols_latest = latest_data[required_cols_check].index[latest_data[required_cols_check].isnull()].tolist()
            logger.warning(f"{log_prefix_live} Indicateurs essentiels manquants (NaN) sur la dernière donnée: {nan_cols_latest}. Pas de requête d'ordre.")
            return None

        required_cols_previous_check = [col for col in required_cols_check if col != self.atr_col_strat]
        if previous_data[required_cols_previous_check].isnull().any():
            nan_cols_previous = previous_data[required_cols_previous_check].index[previous_data[required_cols_previous_check].isnull()].tolist()
            logger.warning(f"{log_prefix_live} Indicateurs essentiels manquants (NaN) sur l'avant-dernière donnée: {nan_cols_previous}. Pas de requête d'ordre.")
            return None

        close_curr = latest_data['close']
        bb_upper_curr = latest_data[self.bb_upper_col_strat]
        bb_lower_curr = latest_data[self.bb_lower_col_strat]
        volume_kline_curr = latest_data[self.volume_kline_col_strat]
        volume_ma_curr = latest_data[self.volume_ma_col_strat]
        rsi_curr = latest_data[self.rsi_col_strat]
        atr_value = latest_data[self.atr_col_strat]
        
        close_prev = previous_data['close']
        bb_upper_prev = previous_data[self.bb_upper_col_strat]
        bb_lower_prev = previous_data[self.bb_lower_col_strat]
        volume_kline_prev = previous_data[self.volume_kline_col_strat]
        volume_ma_prev = previous_data[self.volume_ma_col_strat]
        rsi_prev = previous_data[self.rsi_col_strat]
        
        rsi_buy_thresh = self.get_param('rsi_buy_breakout_threshold')
        rsi_sell_thresh = self.get_param('rsi_sell_breakout_threshold')
        logger.info(f"{log_prefix_live} Seuils RSI: Achat > {rsi_buy_thresh}, Vente < {rsi_sell_thresh}")

        long_breakout_cond_curr = close_curr > bb_upper_curr
        long_volume_conf_curr = volume_kline_curr > volume_ma_curr
        long_rsi_conf_curr = rsi_curr > rsi_buy_thresh
        all_long_cond_curr = long_breakout_cond_curr and long_volume_conf_curr and long_rsi_conf_curr
        logger.info(f"{log_prefix_live} Cond. Achat Actuelles: BB_Breakout({close_curr:.5f} > {bb_upper_curr:.5f})={long_breakout_cond_curr}, Vol_Conf({volume_kline_curr:.2f} > {volume_ma_curr:.2f})={long_volume_conf_curr}, RSI_Conf({rsi_curr:.2f} > {rsi_buy_thresh})={long_rsi_conf_curr} -> Total={all_long_cond_curr}")

        long_breakout_cond_prev = close_prev > bb_upper_prev
        long_volume_conf_prev = volume_kline_prev > volume_ma_prev
        long_rsi_conf_prev = rsi_prev > rsi_buy_thresh
        all_long_cond_prev = long_breakout_cond_prev and long_volume_conf_prev and long_rsi_conf_prev
        
        short_breakout_cond_curr = close_curr < bb_lower_curr
        short_volume_conf_curr = volume_kline_curr > volume_ma_curr
        short_rsi_conf_curr = rsi_curr < rsi_sell_thresh
        all_short_cond_curr = short_breakout_cond_curr and short_volume_conf_curr and short_rsi_conf_curr
        logger.info(f"{log_prefix_live} Cond. Vente Actuelles: BB_Breakout({close_curr:.5f} < {bb_lower_curr:.5f})={short_breakout_cond_curr}, Vol_Conf({volume_kline_curr:.2f} > {volume_ma_curr:.2f})={short_volume_conf_curr}, RSI_Conf({rsi_curr:.2f} < {rsi_sell_thresh})={short_rsi_conf_curr} -> Total={all_short_cond_curr}")

        short_breakout_cond_prev = close_prev < bb_lower_prev
        short_volume_conf_prev = volume_kline_prev > volume_ma_prev
        short_rsi_conf_prev = rsi_prev < rsi_sell_thresh
        all_short_cond_prev = short_breakout_cond_prev and short_volume_conf_prev and short_rsi_conf_prev

        side: Optional[str] = None
        stop_loss_price_raw: Optional[float] = None
        take_profit_price_raw: Optional[float] = None
        
        entry_price_for_order = latest_data['open'] 
        sl_atr_mult = self.get_param('sl_atr_mult')
        tp_atr_mult = self.get_param('tp_atr_mult')

        if all_long_cond_curr and not all_long_cond_prev:
            side = 'BUY'
            if pd.notna(atr_value) and pd.notna(sl_atr_mult) and pd.notna(tp_atr_mult) and pd.notna(entry_price_for_order):
                stop_loss_price_raw = entry_price_for_order - sl_atr_mult * atr_value
                take_profit_price_raw = entry_price_for_order + tp_atr_mult * atr_value
                logger.info(f"{log_prefix_live} Signal d'ACHAT généré. Entrée: {entry_price_for_order:.5f}, ATR: {atr_value:.5f}, SL brut: {stop_loss_price_raw:.5f}, TP brut: {take_profit_price_raw:.5f}")
            else:
                logger.warning(f"{log_prefix_live} Signal d'ACHAT mais ATR/SL/TP/Prix d'entrée invalide. ATR:{atr_value}, SL_mult:{sl_atr_mult}, TP_mult:{tp_atr_mult}, EntryPx:{entry_price_for_order}")
                side = None
        elif all_short_cond_curr and not all_short_cond_prev:
            side = 'SELL'
            if pd.notna(atr_value) and pd.notna(sl_atr_mult) and pd.notna(tp_atr_mult) and pd.notna(entry_price_for_order):
                stop_loss_price_raw = entry_price_for_order + sl_atr_mult * atr_value
                take_profit_price_raw = entry_price_for_order - tp_atr_mult * atr_value
                logger.info(f"{log_prefix_live} Signal de VENTE généré. Entrée: {entry_price_for_order:.5f}, ATR: {atr_value:.5f}, SL brut: {stop_loss_price_raw:.5f}, TP brut: {take_profit_price_raw:.5f}")
            else:
                logger.warning(f"{log_prefix_live} Signal de VENTE mais ATR/SL/TP/Prix d'entrée invalide. ATR:{atr_value}, SL_mult:{sl_atr_mult}, TP_mult:{tp_atr_mult}, EntryPx:{entry_price_for_order}")
                side = None

        if side and stop_loss_price_raw is not None and take_profit_price_raw is not None:
            if atr_value is None or atr_value <= 1e-9 or np.isnan(atr_value):
                logger.warning(f"{log_prefix_live} Valeur ATR invalide ({atr_value}) après génération de signal. Pas d'ordre.")
                return None

            price_precision_val = get_precision_from_filter(symbol_info, 'PRICE_FILTER', 'tickSize')
            qty_precision_val = get_precision_from_filter(symbol_info, 'LOT_SIZE', 'stepSize')
            if price_precision_val is None or qty_precision_val is None:
                logger.error(f"{log_prefix_live} Précision de prix ou de quantité non trouvée pour {symbol}.")
                return None

            quantity = self._calculate_quantity(
                entry_price=entry_price_for_order, available_capital=available_capital,
                qty_precision=qty_precision_val, symbol_info=symbol_info, symbol=symbol
            )
            if quantity is None or quantity <= 0:
                logger.warning(f"{log_prefix_live} Quantité calculée invalide ({quantity}). Pas d'ordre.")
                return None

            entry_price_adjusted = adjust_precision(entry_price_for_order, price_precision_val, round)
            if entry_price_adjusted is None: 
                logger.error(f"{log_prefix_live} Échec de l'ajustement du prix d'entrée (valeur None).")
                return None
            
            # CORRECTION: Appel à get_filter_value avec 3 arguments, gestion de la valeur par défaut séparément
            tick_size_str = get_filter_value(symbol_info, 'PRICE_FILTER', 'tickSize')
            if tick_size_str is None:
                logger.warning(f"{log_prefix_live} 'tickSize' non trouvé dans PRICE_FILTER pour {symbol}. Utilisation d'une valeur par défaut très petite.")
                tick_size = 0.00000001 # Valeur par défaut si non trouvé
            else:
                try:
                    tick_size = float(tick_size_str)
                except ValueError:
                    logger.error(f"{log_prefix_live} Impossible de convertir tickSize '{tick_size_str}' en float. Utilisation d'une valeur par défaut.")
                    tick_size = 0.00000001
            
            if side == 'BUY':
                if stop_loss_price_raw >= entry_price_adjusted:
                    logger.warning(f"{log_prefix_live} SL ({stop_loss_price_raw:.5f}) >= Prix d'entrée ({entry_price_adjusted:.5f}) pour un BUY. Ajustement du SL.")
                    stop_loss_price_raw = entry_price_adjusted - tick_size 
                if take_profit_price_raw <= entry_price_adjusted:
                    logger.warning(f"{log_prefix_live} TP ({take_profit_price_raw:.5f}) <= Prix d'entrée ({entry_price_adjusted:.5f}) pour un BUY. Ajustement du TP.")
                    take_profit_price_raw = entry_price_adjusted + tick_size
            elif side == 'SELL':
                if stop_loss_price_raw <= entry_price_adjusted:
                    logger.warning(f"{log_prefix_live} SL ({stop_loss_price_raw:.5f}) <= Prix d'entrée ({entry_price_adjusted:.5f}) pour un SELL. Ajustement du SL.")
                    stop_loss_price_raw = entry_price_adjusted + tick_size
                if take_profit_price_raw >= entry_price_adjusted:
                    logger.warning(f"{log_prefix_live} TP ({take_profit_price_raw:.5f}) >= Prix d'entrée ({entry_price_adjusted:.5f}) pour un SELL. Ajustement du TP.")
                    take_profit_price_raw = entry_price_adjusted - tick_size

            entry_price_str = f"{entry_price_adjusted:.{price_precision_val}f}"
            quantity_str = f"{quantity:.{qty_precision_val}f}"

            entry_params = self._build_entry_params_formatted(
                symbol=symbol, side=side, quantity_str=quantity_str,
                entry_price_str=entry_price_str, order_type="LIMIT"
            )
            if not entry_params:
                logger.error(f"{log_prefix_live} Échec de la construction des paramètres d'entrée formatés.")
                return None

            sl_tp_prices = {'sl_price': stop_loss_price_raw, 'tp_price': take_profit_price_raw}
            logger.info(f"{log_prefix_live} Requête d'ordre générée: {entry_params} avec SL/TP bruts: {sl_tp_prices}")
            return entry_params, sl_tp_prices
        
        logger.info(f"{log_prefix_live} Aucune condition d'entrée remplie pour un nouvel ordre (après vérification des conditions actuelles vs précédentes).")
        return None

