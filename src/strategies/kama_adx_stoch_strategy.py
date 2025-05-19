# Fichier: src/strategies/kama_adx_stoch_strategy.py
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
    logging.getLogger(__name__).error("KamaAdxStochStrategy: Failed to import BaseStrategy or exchange_utils. Using dummy BaseStrategy.")
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
        def _calculate_quantity(self, *args, **kwargs) -> Optional[float]: return None #type: ignore
        def _build_entry_params_formatted(self, *args, **kwargs) -> Optional[Dict]: return None #type: ignore
        def _build_oco_params(self, *args, **kwargs) -> Optional[Dict]: return None #type: ignore

logger = logging.getLogger(__name__)

class KamaAdxStochStrategy(BaseStrategy):
    """
    Stratégie de trading basée sur un rebond sur KAMA (support/résistance dynamique),
    filtré par l'ADX pour la force de la tendance et le Stochastique pour
    les conditions de surachat/survente. L'ATR est utilisé pour SL/TP.
    """
    def __init__(self, params: dict):
        """
        Initialise la stratégie avec les paramètres fournis.

        Args:
            params (dict): Dictionnaire des paramètres de la stratégie.
        """
        super().__init__(params)
        required_params = [
            'kama_period', 'kama_fast_ema', 'kama_slow_ema', 'indicateur_frequence_kama',
            'adx_period', 'adx_trend_threshold', 'indicateur_frequence_adx',
            'stoch_k_period', 'stoch_d_period', 'stoch_slowing', 'indicateur_frequence_stoch',
            'stoch_oversold_threshold', 'stoch_overbought_threshold',
            'atr_period_sl_tp', 'atr_base_frequency_sl_tp', 'sl_atr_mult', 'tp_atr_mult'
        ]
        missing = [key for key in required_params if self.get_param(key) is None]
        if missing:
            err_msg = f"Paramètres manquants pour {self.__class__.__name__}: {missing}"
            logger.error(err_msg)
            raise ValueError(err_msg)

        # Noms des colonnes pour les indicateurs (pré-calculés)
        self.kama_col_strat = "KAMA_strat"
        self.adx_col_strat = "ADX_strat"
        self.adx_dmp_col_strat = "ADX_DMP_strat"  # +DI
        self.adx_dmn_col_strat = "ADX_DMN_strat"  # -DI
        self.stoch_k_col_strat = "STOCH_K_strat"
        self.stoch_d_col_strat = "STOCH_D_strat"
        self.atr_col_strat = "ATR_strat"

        self._signals: Optional[pd.DataFrame] = None
        self.strategy_name_log_prefix = f"[{self.__class__.__name__}]"
        logger.info(f"{self.strategy_name_log_prefix} Stratégie initialisée avec les paramètres: {self.params}")

    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Vérifie la présence des colonnes d'indicateurs pré-calculées.
        """
        df = data.copy()
        logger.debug(f"{self.strategy_name_log_prefix} Entrée _calculate_indicators. Colonnes reçues: {df.columns.tolist()}")

        for col_name in [self.kama_col_strat, self.adx_col_strat, self.adx_dmp_col_strat,
                         self.adx_dmn_col_strat, self.stoch_k_col_strat, self.stoch_d_col_strat,
                         self.atr_col_strat]:
            if col_name not in df.columns:
                df[col_name] = np.nan
                logger.debug(f"{self.strategy_name_log_prefix} Colonne {col_name} ajoutée (NaN).")

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
            self.kama_col_strat, self.adx_col_strat, self.adx_dmp_col_strat, self.adx_dmn_col_strat,
            self.stoch_k_col_strat, self.stoch_d_col_strat, self.atr_col_strat,
            'close', 'open', 'high', 'low'
        ]

        if df_with_indicators.empty or \
           any(col not in df_with_indicators.columns for col in required_cols_for_signal) or \
           df_with_indicators[required_cols_for_signal].isnull().all().all() or \
           len(df_with_indicators) < 2:
            logger.warning(f"{self.strategy_name_log_prefix} Données insuffisantes ou colonnes manquantes pour la génération de signaux.")
            self._signals = pd.DataFrame(index=data.index, columns=['entry_long', 'exit_long', 'entry_short', 'exit_short', 'sl', 'tp'])
            self._signals[['sl', 'tp']] = np.nan
            self._signals[self._signals.select_dtypes(include=bool).columns] = False
            return

        # Récupération des paramètres
        sl_atr_mult = self.get_param('sl_atr_mult')
        tp_atr_mult = self.get_param('tp_atr_mult')
        adx_trend_thresh = self.get_param('adx_trend_threshold')
        stoch_oversold = self.get_param('stoch_oversold_threshold')
        stoch_overbought = self.get_param('stoch_overbought_threshold')

        # Indicateurs actuels et précédents
        close_curr = df_with_indicators['close']
        low_curr = df_with_indicators['low']
        high_curr = df_with_indicators['high']
        
        kama_curr = df_with_indicators[self.kama_col_strat]
        adx_curr = df_with_indicators[self.adx_col_strat]
        dmp_curr = df_with_indicators[self.adx_dmp_col_strat] # +DI
        dmn_curr = df_with_indicators[self.adx_dmn_col_strat] # -DI
        stoch_k_curr = df_with_indicators[self.stoch_k_col_strat]
        stoch_d_curr = df_with_indicators[self.stoch_d_col_strat]
        
        stoch_k_prev = stoch_k_curr.shift(1)
        stoch_d_prev = stoch_d_curr.shift(1)
        close_prev = close_curr.shift(1) # Utilisé pour la condition de croisement de KAMA

        # Conditions d'achat (Rebond Long) actuelles
        long_kama_cond_curr = (low_curr <= kama_curr) & (close_curr > kama_curr) # Rebond sur KAMA
        long_adx_filter_curr = (adx_curr > adx_trend_thresh) & (dmp_curr > dmn_curr)
        long_stoch_cond_curr = (stoch_k_curr > stoch_d_curr) & (stoch_k_prev <= stoch_d_prev) & (stoch_k_prev < stoch_oversold)
        all_long_cond_curr = long_kama_cond_curr & long_adx_filter_curr & long_stoch_cond_curr

        # Conditions d'achat (Rebond Long) précédentes (pour éviter les entrées répétées)
        # On peut simplifier en vérifiant juste si la condition globale n'était pas vraie avant
        all_long_cond_prev = all_long_cond_curr.shift(1).fillna(False)
        entry_long_trigger = all_long_cond_curr & ~all_long_cond_prev

        # Conditions de vente (Rejet Court) actuelles
        short_kama_cond_curr = (high_curr >= kama_curr) & (close_curr < kama_curr) # Rejet sous KAMA
        short_adx_filter_curr = (adx_curr > adx_trend_thresh) & (dmn_curr > dmp_curr)
        short_stoch_cond_curr = (stoch_k_curr < stoch_d_curr) & (stoch_k_prev >= stoch_d_prev) & (stoch_k_prev > stoch_overbought)
        all_short_cond_curr = short_kama_cond_curr & short_adx_filter_curr & short_stoch_cond_curr
        
        # Conditions de vente (Rejet Court) précédentes
        all_short_cond_prev = all_short_cond_curr.shift(1).fillna(False)
        entry_short_trigger = all_short_cond_curr & ~all_short_cond_prev
        
        # Création du DataFrame des signaux
        signals_df = pd.DataFrame(index=df_with_indicators.index)
        signals_df['entry_long'] = entry_long_trigger
        signals_df['entry_short'] = entry_short_trigger
        signals_df['exit_long'] = False
        signals_df['exit_short'] = False

        # Calcul SL/TP
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
        
        if len(data) < 2:
            logger.debug(f"{log_prefix_live} Données insuffisantes. Pas de nouvelle requête d'ordre.")
            return None

        df_verified_indicators = self._calculate_indicators(data.copy())
        latest_data = df_verified_indicators.iloc[-1]
        previous_data = df_verified_indicators.iloc[-2]

        required_cols = [
            'close', 'open', 'high', 'low', self.kama_col_strat, self.adx_col_strat,
            self.adx_dmp_col_strat, self.adx_dmn_col_strat, self.stoch_k_col_strat,
            self.stoch_d_col_strat, self.atr_col_strat
        ]
        if latest_data[required_cols].isnull().any() or \
           previous_data[required_cols[:-1]].isnull().any(): # Exclure ATR de la vérif pour previous_data
            logger.debug(f"{log_prefix_live} Indicateurs manquants sur les données récentes. Pas de requête d'ordre.")
            return None

        # Indicateurs actuels
        close_curr = latest_data['close']
        low_curr = latest_data['low']
        high_curr = latest_data['high']
        kama_curr = latest_data[self.kama_col_strat]
        adx_curr = latest_data[self.adx_col_strat]
        dmp_curr = latest_data[self.adx_dmp_col_strat]
        dmn_curr = latest_data[self.adx_dmn_col_strat]
        stoch_k_curr = latest_data[self.stoch_k_col_strat]
        stoch_d_curr = latest_data[self.stoch_d_col_strat]
        
        # Indicateurs précédents
        stoch_k_prev = previous_data[self.stoch_k_col_strat]
        stoch_d_prev = previous_data[self.stoch_d_col_strat]
        close_prev = previous_data['close'] # Utilisé pour la condition de croisement de KAMA

        adx_trend_thresh = self.get_param('adx_trend_threshold')
        stoch_oversold = self.get_param('stoch_oversold_threshold')
        stoch_overbought = self.get_param('stoch_overbought_threshold')

        # Conditions d'achat (Rebond Long) actuelles
        long_kama_cond_curr = (low_curr <= kama_curr) and (close_curr > kama_curr)
        long_adx_filter_curr = (adx_curr > adx_trend_thresh) and (dmp_curr > dmn_curr)
        long_stoch_cond_curr = (stoch_k_curr > stoch_d_curr) and (stoch_k_prev <= stoch_d_prev) and (stoch_k_prev < stoch_oversold)
        all_long_cond_curr = long_kama_cond_curr and long_adx_filter_curr and long_stoch_cond_curr

        # Conditions d'achat (Rebond Long) précédentes (pour s'assurer que le signal est nouveau)
        # Pour la requête d'ordre, on vérifie si la condition globale n'était PAS vraie à la bougie précédente
        # Cela nécessite de reconstruire all_long_cond_prev
        kama_prev_val = previous_data[self.kama_col_strat]
        adx_prev_val = previous_data[self.adx_col_strat]
        dmp_prev_val = previous_data[self.adx_dmp_col_strat]
        dmn_prev_val = previous_data[self.adx_dmn_col_strat]
        # stoch_k_prev et stoch_d_prev sont déjà définis
        low_prev_val = previous_data['low']

        long_kama_cond_prev_live = (low_prev_val <= kama_prev_val) and (close_prev > kama_prev_val)
        long_adx_filter_prev_live = (adx_prev_val > adx_trend_thresh) and (dmp_prev_val > dmn_prev_val)
        # Pour le stochastique, la condition de croisement est sur k_curr vs d_curr et k_prev vs d_prev.
        # Donc, pour all_long_cond_prev, on ne peut pas simplement shifter all_long_cond_curr.
        # On vérifie que la condition de signal n'était pas déjà active.
        # Une approximation simple est de vérifier si `all_long_cond_curr` est vrai et `all_long_cond_curr.shift(1)` était faux.
        # Pour generate_order_request, on ne peut pas utiliser .shift() directement sur latest_data.
        # On doit reconstruire la condition pour la bougie N-1.
        # Cependant, la logique de la BaseStrategy est souvent d'évaluer le signal sur la dernière bougie complète.
        # Pour simplifier, on va supposer que le signal est frais si les conditions sont remplies MAINTENANT.
        # La vérification "n'était pas vrai avant" est plus pour le backtesting sur une série.
        # Pour le live, on agit sur la dernière bougie fermée.
        # Si on veut être strict, il faudrait passer les données de N-2 aussi.
        # Pour l'instant, on va se baser sur le fait que si la condition est vraie maintenant, on prend.
        # La gestion de "déjà en position" empêche les ordres multiples.

        # Conditions de vente (Rejet Court) actuelles
        short_kama_cond_curr = (high_curr >= kama_curr) and (close_curr < kama_curr)
        short_adx_filter_curr = (adx_curr > adx_trend_thresh) and (dmn_curr > dmp_curr)
        short_stoch_cond_curr = (stoch_k_curr < stoch_d_curr) and (stoch_k_prev >= stoch_d_prev) and (stoch_k_prev > stoch_overbought)
        all_short_cond_curr = short_kama_cond_curr and short_adx_filter_curr and short_stoch_cond_curr

        side: Optional[str] = None
        stop_loss_price_raw: Optional[float] = None
        take_profit_price_raw: Optional[float] = None
        
        entry_price_for_order = latest_data['open'] # Ou 'close' selon la logique d'exécution
        atr_value = latest_data[self.atr_col_strat]
        sl_atr_mult = self.get_param('sl_atr_mult')
        tp_atr_mult = self.get_param('tp_atr_mult')

        # Pour éviter les entrées répétées, on vérifie que la condition n'était pas déjà active
        # Cette logique est un peu simplifiée pour generate_order_request par rapport à generate_signals
        # car on n'a pas facilement accès à `all_long_cond_curr.shift(1)` ici.
        # On se fie au fait que `current_position == 0`.
        # Si un signal est généré, on prend position. Si le même signal persiste, on est déjà en position.

        if all_long_cond_curr:
            side = 'BUY'
            stop_loss_price_raw = entry_price_for_order - sl_atr_mult * atr_value
            take_profit_price_raw = entry_price_for_order + tp_atr_mult * atr_value
            logger.info(f"{log_prefix_live} Signal d'ACHAT généré. Entrée: {entry_price_for_order}, ATR: {atr_value}")
        elif all_short_cond_curr:
            side = 'SELL'
            stop_loss_price_raw = entry_price_for_order + sl_atr_mult * atr_value
            take_profit_price_raw = entry_price_for_order - tp_atr_mult * atr_value
            logger.info(f"{log_prefix_live} Signal de VENTE généré. Entrée: {entry_price_for_order}, ATR: {atr_value}")

        if side and stop_loss_price_raw is not None and take_profit_price_raw is not None:
            if atr_value is None or atr_value <= 1e-9 or np.isnan(atr_value):
                logger.warning(f"{log_prefix_live} Valeur ATR invalide ({atr_value}). Pas d'ordre.")
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
                logger.error(f"{log_prefix_live} Échec de l'ajustement du prix d'entrée.")
                return None
            
            tick_size = 1 / (10**price_precision_val) if price_precision_val > 0 else 1.0
            if side == 'BUY':
                if abs(stop_loss_price_raw - entry_price_adjusted) < tick_size: stop_loss_price_raw = entry_price_adjusted - tick_size
                if abs(take_profit_price_raw - entry_price_adjusted) < tick_size: take_profit_price_raw = entry_price_adjusted + tick_size
            elif side == 'SELL':
                if abs(stop_loss_price_raw - entry_price_adjusted) < tick_size: stop_loss_price_raw = entry_price_adjusted + tick_size
                if abs(take_profit_price_raw - entry_price_adjusted) < tick_size: take_profit_price_raw = entry_price_adjusted - tick_size

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
