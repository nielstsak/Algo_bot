import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from .base import BaseStrategy
    from src.utils.exchange_utils import (adjust_precision,
                                          get_filter_value,
                                          get_precision_from_filter)
except ImportError:
    # Fallback pour les tests unitaires ou environnements isolés
    logging.getLogger(__name__).error("EmaMacdAtrStrategy: Failed to import BaseStrategy or exchange_utils. Using dummy BaseStrategy.")
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

logger = logging.getLogger(__name__)

class EmaMacdAtrStrategy(BaseStrategy):
    """
    Stratégie de trading basée sur un croisement d'EMA pour la tendance,
    confirmé par le MACD pour le momentum. L'ATR est utilisé pour
    définir les niveaux de Stop Loss et Take Profit.
    """
    def __init__(self, params: dict):
        """
        Initialise la stratégie avec les paramètres fournis.

        Args:
            params (dict): Dictionnaire des paramètres de la stratégie. Doit contenir :
                - ema_short_period (int): Période de l'EMA courte.
                - ema_long_period (int): Période de l'EMA longue.
                - indicateur_frequence_ema (str): Fréquence des K-lines pour le calcul des EMA.
                - macd_fast_period (int): Période rapide du MACD.
                - macd_slow_period (int): Période lente du MACD.
                - macd_signal_period (int): Période du signal MACD.
                - indicateur_frequence_macd (str): Fréquence des K-lines pour le calcul du MACD.
                - atr_period_sl_tp (int): Période de l'ATR pour SL/TP.
                - atr_base_frequency_sl_tp (str): Fréquence des K-lines pour l'ATR (SL/TP).
                - sl_atr_mult (float): Multiplicateur ATR pour le Stop Loss.
                - tp_atr_mult (float): Multiplicateur ATR pour le Take Profit.
                - (Optionnel) atr_volatility_filter_period, indicateur_frequence_atr_volatility,
                  atr_volatility_threshold_mult pour un filtre de volatilité.
        """
        super().__init__(params)
        required_params = [
            'ema_short_period', 'ema_long_period', 'indicateur_frequence_ema',
            'macd_fast_period', 'macd_slow_period', 'macd_signal_period', 'indicateur_frequence_macd',
            'atr_period_sl_tp', 'atr_base_frequency_sl_tp', 'sl_atr_mult', 'tp_atr_mult'
        ]
        missing = [key for key in required_params if self.get_param(key) is None]
        if missing:
            err_msg = f"Paramètres manquants pour {self.__class__.__name__}: {missing}"
            logger.error(err_msg)
            raise ValueError(err_msg)

        self.ema_short_col_strat = "EMA_short_strat"
        self.ema_long_col_strat = "EMA_long_strat"
        self.macd_line_col_strat = "MACD_line_strat"
        self.macd_signal_col_strat = "MACD_signal_strat"
        self.macd_hist_col_strat = "MACD_hist_strat"
        self.atr_col_strat = "ATR_strat" # Utilisé pour SL/TP

        # Colonnes pour le filtre de volatilité optionnel (non implémenté dans la logique de signaux pour l'instant)
        self.atr_volatility_filter_col_strat = "ATR_volatility_filter_strat"

        self._signals: Optional[pd.DataFrame] = None
        self.strategy_name_log_prefix = f"[{self.__class__.__name__}]"
        logger.info(f"{self.strategy_name_log_prefix} Stratégie initialisée avec les paramètres: {self.params}")

    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Vérifie la présence des colonnes d'indicateurs pré-calculées.
        Ne recalcule pas les indicateurs ici, ils doivent être fournis.
        """
        df = data.copy()
        logger.debug(f"{self.strategy_name_log_prefix} Entrée _calculate_indicators. Colonnes reçues: {df.columns.tolist()}")

        # Vérification des colonnes EMA
        if self.ema_short_col_strat not in df.columns:
            df[self.ema_short_col_strat] = np.nan
            logger.debug(f"{self.strategy_name_log_prefix} Colonne {self.ema_short_col_strat} ajoutée (NaN).")
        if self.ema_long_col_strat not in df.columns:
            df[self.ema_long_col_strat] = np.nan
            logger.debug(f"{self.strategy_name_log_prefix} Colonne {self.ema_long_col_strat} ajoutée (NaN).")

        # Vérification des colonnes MACD
        if self.macd_line_col_strat not in df.columns:
            df[self.macd_line_col_strat] = np.nan
            logger.debug(f"{self.strategy_name_log_prefix} Colonne {self.macd_line_col_strat} ajoutée (NaN).")
        if self.macd_signal_col_strat not in df.columns:
            df[self.macd_signal_col_strat] = np.nan
            logger.debug(f"{self.strategy_name_log_prefix} Colonne {self.macd_signal_col_strat} ajoutée (NaN).")
        if self.macd_hist_col_strat not in df.columns:
            df[self.macd_hist_col_strat] = np.nan
            logger.debug(f"{self.strategy_name_log_prefix} Colonne {self.macd_hist_col_strat} ajoutée (NaN).")

        # Vérification de la colonne ATR (pour SL/TP)
        if self.atr_col_strat not in df.columns:
            df[self.atr_col_strat] = np.nan
            logger.debug(f"{self.strategy_name_log_prefix} Colonne {self.atr_col_strat} ajoutée (NaN).")

        # Vérification optionnelle pour filtre ATR de volatilité
        if self.get_param('atr_volatility_filter_period') is not None:
            if self.atr_volatility_filter_col_strat not in df.columns:
                df[self.atr_volatility_filter_col_strat] = np.nan
                logger.debug(f"{self.strategy_name_log_prefix} Colonne {self.atr_volatility_filter_col_strat} (optionnelle) ajoutée (NaN).")

        # Vérification des colonnes OHLCV de base (1-minute)
        required_ohlc = ['open', 'high', 'low', 'close', 'volume']
        missing_ohlc = [col for col in required_ohlc if col not in df.columns]
        if missing_ohlc:
            for col in missing_ohlc:
                df[col] = np.nan
                logger.debug(f"{self.strategy_name_log_prefix} Colonne OHLC '{col}' ajoutée (NaN).")
        
        logger.debug(f"{self.strategy_name_log_prefix} Sortie _calculate_indicators. Colonnes présentes: {df.columns.tolist()}")
        return df

    def generate_signals(self, data: pd.DataFrame):
        """
        Génère les signaux d'achat et de vente basés sur les indicateurs.
        """
        logger.debug(f"{self.strategy_name_log_prefix} Génération des signaux...")
        df_with_indicators = self._calculate_indicators(data)

        required_cols_for_signal = [
            self.ema_short_col_strat, self.ema_long_col_strat,
            self.macd_line_col_strat, self.macd_signal_col_strat, self.macd_hist_col_strat,
            self.atr_col_strat, 'close'
        ]

        # Vérification que le DataFrame et les colonnes nécessaires ne sont pas vides ou entièrement NaN
        if df_with_indicators.empty or \
           any(col not in df_with_indicators.columns for col in required_cols_for_signal) or \
           df_with_indicators[required_cols_for_signal].isnull().all().all() or \
           len(df_with_indicators) < 2 :
            logger.warning(f"{self.strategy_name_log_prefix} Données insuffisantes ou colonnes manquantes pour la génération de signaux. Colonnes requises: {required_cols_for_signal}. DataFrame vide: {df_with_indicators.empty}. Longueur: {len(df_with_indicators)}.")
            if not df_with_indicators.empty:
                 for col in required_cols_for_signal:
                     if col not in df_with_indicators.columns: logger.warning(f"Colonne manquante: {col}")
                     elif df_with_indicators[col].isnull().all(): logger.warning(f"Colonne {col} entièrement NaN.")

            self._signals = pd.DataFrame(index=data.index, columns=['entry_long', 'exit_long', 'entry_short', 'exit_short', 'sl', 'tp'])
            self._signals[['sl', 'tp']] = np.nan
            self._signals[self._signals.select_dtypes(include=bool).columns] = False
            return

        # Récupération des paramètres
        sl_atr_mult = self.get_param('sl_atr_mult')
        tp_atr_mult = self.get_param('tp_atr_mult')

        # Conditions d'entrée
        ema_short_curr = df_with_indicators[self.ema_short_col_strat]
        ema_long_curr = df_with_indicators[self.ema_long_col_strat]
        macd_line_curr = df_with_indicators[self.macd_line_col_strat]
        macd_signal_curr = df_with_indicators[self.macd_signal_col_strat]
        macd_hist_curr = df_with_indicators[self.macd_hist_col_strat]
        
        # Conditions pour la bougie précédente
        ema_short_prev = ema_short_curr.shift(1)
        ema_long_prev = ema_long_curr.shift(1)
        macd_line_prev = macd_line_curr.shift(1)
        macd_signal_prev = macd_signal_curr.shift(1)
        macd_hist_prev = macd_hist_curr.shift(1)

        # Conditions d'achat actuelles
        long_trend_cond_curr = ema_short_curr > ema_long_curr
        long_momentum_cond_curr = macd_line_curr > macd_signal_curr
        long_hist_cond_curr = macd_hist_curr > 0 # Histogramme positif
        all_long_cond_curr = long_trend_cond_curr & long_momentum_cond_curr & long_hist_cond_curr

        # Conditions d'achat précédentes (pour éviter les entrées répétées)
        long_trend_cond_prev = ema_short_prev > ema_long_prev
        long_momentum_cond_prev = macd_line_prev > macd_signal_prev
        long_hist_cond_prev = macd_hist_prev > 0
        all_long_cond_prev = long_trend_cond_prev & long_momentum_cond_prev & long_hist_cond_prev
        
        entry_long_trigger = all_long_cond_curr & ~all_long_cond_prev.fillna(False)

        # Conditions de vente actuelles
        short_trend_cond_curr = ema_short_curr < ema_long_curr
        short_momentum_cond_curr = macd_line_curr < macd_signal_curr
        short_hist_cond_curr = macd_hist_curr < 0 # Histogramme négatif
        all_short_cond_curr = short_trend_cond_curr & short_momentum_cond_curr & short_hist_cond_curr

        # Conditions de vente précédentes
        short_trend_cond_prev = ema_short_prev < ema_long_prev
        short_momentum_cond_prev = macd_line_prev < macd_signal_prev
        short_hist_cond_prev = macd_hist_prev < 0
        all_short_cond_prev = short_trend_cond_prev & short_momentum_cond_prev & short_hist_cond_prev

        entry_short_trigger = all_short_cond_curr & ~all_short_cond_prev.fillna(False)

        # Création du DataFrame des signaux
        signals_df = pd.DataFrame(index=df_with_indicators.index)
        signals_df['entry_long'] = entry_long_trigger
        signals_df['entry_short'] = entry_short_trigger
        signals_df['exit_long'] = False  # Les sorties sont gérées par SL/TP dans le simulateur
        signals_df['exit_short'] = False

        # Calcul SL/TP
        close_curr = df_with_indicators['close']
        atr_curr = df_with_indicators[self.atr_col_strat]

        signals_df['sl'] = np.where(
            entry_long_trigger & atr_curr.notna(), close_curr - sl_atr_mult * atr_curr,
            np.where(entry_short_trigger & atr_curr.notna(), close_curr + sl_atr_mult * atr_curr, np.nan)
        )
        signals_df['tp'] = np.where(
            entry_long_trigger & atr_curr.notna(), close_curr + tp_atr_mult * atr_curr,
            np.where(entry_short_trigger & atr_curr.notna(), close_curr - tp_atr_mult * atr_curr, np.nan)
        )
        
        self._signals = signals_df.reindex(data.index)
        logger.debug(f"{self.strategy_name_log_prefix} Signaux générés. Entrées Long: {signals_df['entry_long'].sum()}, Entrées Short: {signals_df['entry_short'].sum()}")

    def generate_order_request(self,
                               data: pd.DataFrame,
                               symbol: str,
                               current_position: int,
                               available_capital: float,
                               symbol_info: dict
                               ) -> Optional[Tuple[Dict[str, Any], Dict[str, float]]]:
        """
        Génère une requête d'ordre pour le trading en direct ou la simulation.
        """
        log_prefix_live = f"{self.strategy_name_log_prefix}[LiveOrder]"
        if current_position != 0:
            logger.debug(f"{log_prefix_live} Position existante ({current_position}). Pas de nouvelle requête d'ordre.")
            return None
        
        if len(data) < 2: # Nécessite au moins deux points pour le shift des conditions
            logger.debug(f"{log_prefix_live} Données insuffisantes (moins de 2 points). Pas de nouvelle requête d'ordre.")
            return None

        df_verified_indicators = self._calculate_indicators(data.copy())
        latest_data = df_verified_indicators.iloc[-1]
        previous_data = df_verified_indicators.iloc[-2]

        required_cols = [
            'close', self.ema_short_col_strat, self.ema_long_col_strat,
            self.macd_line_col_strat, self.macd_signal_col_strat, self.macd_hist_col_strat,
            self.atr_col_strat
        ]
        if latest_data[required_cols].isnull().any() or \
           previous_data[['close', self.ema_short_col_strat, self.ema_long_col_strat,
                           self.macd_line_col_strat, self.macd_signal_col_strat, self.macd_hist_col_strat]].isnull().any():
            logger.debug(f"{log_prefix_live} Indicateurs manquants sur les données récentes. Pas de requête d'ordre.")
            return None

        # Conditions d'achat actuelles
        ema_short_curr = latest_data[self.ema_short_col_strat]
        ema_long_curr = latest_data[self.ema_long_col_strat]
        macd_line_curr = latest_data[self.macd_line_col_strat]
        macd_signal_curr = latest_data[self.macd_signal_col_strat]
        macd_hist_curr = latest_data[self.macd_hist_col_strat]
        
        long_trend_cond_curr = ema_short_curr > ema_long_curr
        long_momentum_cond_curr = macd_line_curr > macd_signal_curr
        long_hist_cond_curr = macd_hist_curr > 0
        all_long_cond_curr = long_trend_cond_curr and long_momentum_cond_curr and long_hist_cond_curr

        # Conditions d'achat précédentes
        ema_short_prev = previous_data[self.ema_short_col_strat]
        ema_long_prev = previous_data[self.ema_long_col_strat]
        macd_line_prev = previous_data[self.macd_line_col_strat]
        macd_signal_prev = previous_data[self.macd_signal_col_strat]
        macd_hist_prev = previous_data[self.macd_hist_col_strat]

        long_trend_cond_prev = ema_short_prev > ema_long_prev
        long_momentum_cond_prev = macd_line_prev > macd_signal_prev
        long_hist_cond_prev = macd_hist_prev > 0
        all_long_cond_prev = long_trend_cond_prev and long_momentum_cond_prev and long_hist_cond_prev
        
        # Conditions de vente actuelles
        short_trend_cond_curr = ema_short_curr < ema_long_curr
        short_momentum_cond_curr = macd_line_curr < macd_signal_curr
        short_hist_cond_curr = macd_hist_curr < 0
        all_short_cond_curr = short_trend_cond_curr and short_momentum_cond_curr and short_hist_cond_curr

        # Conditions de vente précédentes
        short_trend_cond_prev = ema_short_prev < ema_long_prev
        short_momentum_cond_prev = macd_line_prev < macd_signal_prev
        short_hist_cond_prev = macd_hist_prev < 0
        all_short_cond_prev = short_trend_cond_prev and short_momentum_cond_prev and short_hist_cond_prev

        side: Optional[str] = None
        stop_loss_price_raw: Optional[float] = None
        take_profit_price_raw: Optional[float] = None
        
        close_curr = latest_data['close']
        atr_value = latest_data[self.atr_col_strat]
        sl_atr_mult = self.get_param('sl_atr_mult')
        tp_atr_mult = self.get_param('tp_atr_mult')

        if all_long_cond_curr and not all_long_cond_prev:
            side = 'BUY'
            stop_loss_price_raw = close_curr - sl_atr_mult * atr_value
            take_profit_price_raw = close_curr + tp_atr_mult * atr_value
            logger.info(f"{log_prefix_live} Signal d'ACHAT généré. Close: {close_curr}, ATR: {atr_value}")
        elif all_short_cond_curr and not all_short_cond_prev:
            side = 'SELL'
            stop_loss_price_raw = close_curr + sl_atr_mult * atr_value
            take_profit_price_raw = close_curr - tp_atr_mult * atr_value
            logger.info(f"{log_prefix_live} Signal de VENTE généré. Close: {close_curr}, ATR: {atr_value}")

        if side and stop_loss_price_raw is not None and take_profit_price_raw is not None:
            if atr_value is None or atr_value <= 1e-9 or np.isnan(atr_value):
                logger.warning(f"{log_prefix_live} Valeur ATR invalide ({atr_value}). Impossible de calculer SL/TP. Pas d'ordre.")
                return None

            price_precision_val = get_precision_from_filter(symbol_info, 'PRICE_FILTER', 'tickSize')
            qty_precision_val = get_precision_from_filter(symbol_info, 'LOT_SIZE', 'stepSize')
            if price_precision_val is None or qty_precision_val is None:
                logger.error(f"{log_prefix_live} Précision de prix ou de quantité non trouvée pour {symbol}.")
                return None

            quantity = self._calculate_quantity(
                entry_price=close_curr, available_capital=available_capital,
                qty_precision=qty_precision_val, symbol_info=symbol_info, symbol=symbol
            )
            if quantity is None or quantity <= 0:
                logger.warning(f"{log_prefix_live} Quantité calculée invalide ({quantity}). Pas d'ordre.")
                return None

            entry_price_adjusted = adjust_precision(close_curr, price_precision_val, round)
            if entry_price_adjusted is None:
                logger.error(f"{log_prefix_live} Échec de l'ajustement du prix d'entrée.")
                return None
            
            # Assurer que SL et TP sont différents du prix d'entrée après ajustement
            # (Surtout pour les tests avec ATR faible ou multiplicateurs très petits)
            tick_size = 1 / (10**price_precision_val) if price_precision_val > 0 else 1.0
            if side == 'BUY':
                if abs(stop_loss_price_raw - entry_price_adjusted) < tick_size:
                    stop_loss_price_raw = entry_price_adjusted - tick_size
                if abs(take_profit_price_raw - entry_price_adjusted) < tick_size:
                    take_profit_price_raw = entry_price_adjusted + tick_size
            elif side == 'SELL':
                if abs(stop_loss_price_raw - entry_price_adjusted) < tick_size:
                    stop_loss_price_raw = entry_price_adjusted + tick_size
                if abs(take_profit_price_raw - entry_price_adjusted) < tick_size:
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
            logger.info(f"{log_prefix_live} Requête d'ordre générée: {entry_params} avec SL/TP: {sl_tp_prices}")
            return entry_params, sl_tp_prices
        
        logger.debug(f"{log_prefix_live} Aucune condition d'entrée remplie pour un nouvel ordre.")
        return None