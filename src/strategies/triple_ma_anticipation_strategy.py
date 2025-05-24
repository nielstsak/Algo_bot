import logging
import math
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Tuple

try:
    import pandas_ta as ta
    PANDAS_TA_AVAILABLE = True
except ImportError:
    PANDAS_TA_AVAILABLE = False
    class DummyTaExtension:
        def __init__(self, df_instance): self.df = df_instance
        def sma(self, length, close, **kwargs): return pd.Series([np.nan]*len(self.df), index=self.df.index)
        def atr(self, length, high, low, close, **kwargs): return pd.Series([np.nan]*len(self.df), index=self.df.index)
    if not hasattr(pd.DataFrame, 'ta'):
        pd.DataFrame.ta = property(lambda df_instance: DummyTaExtension(df_instance))

try:
    from .base import BaseStrategy # Tente d'importer la BaseStrategy locale
except ImportError as e_local:
    logging.warning(f"[TripleMAAnticipationStrategy] Échec import local 'from .base import BaseStrategy': {e_local}. "
                    "Tentative d'import depuis src.strategies.base...")
    try:
        from src.strategies.base import BaseStrategy # Tentative d'import global si relatif échoue
    except ImportError as e_global:
        logging.critical(f"[TripleMAAnticipationStrategy] Erreur critique: Impossible d'importer BaseStrategy: {e_global}")
        raise

logger = logging.getLogger(__name__)

if not PANDAS_TA_AVAILABLE:
    logger.error("pandas_ta non disponible. TripleMAAnticipationStrategy ne fonctionnera pas correctement.")

class TripleMAAnticipationStrategy(BaseStrategy):
    def __init__(self, 
                 params: dict, 
                 pair_symbol: str = "DEFAULT_PAIR_OPTIM", # Valeur par défaut si non fourni
                 account_id: str = "OPTIMIZER_ACC_OPTIM", 
                 deployment_id: str = "OPTIMIZER_DEP_OPTIM", 
                 is_futures: bool = False):
        
        # Appel à super().__init__ en assumant que la BaseStrategy actuellement chargée
        # n'accepte que `params`. C'est pour correspondre à l'erreur :
        # "TypeError: BaseStrategy.__init__() takes 2 positional arguments but 6 were given"
        # Cela signifie que le BaseStrategy que vous avez posté au début (avec 6 args)
        # n'est PAS celui qui est utilisé par l'optimiseur pour les autres stratégies.
        try:
            super().__init__(params) # Appel compatible avec une BaseStrategy simple (self, params)
        except TypeError as te_super_simple:
            logger.warning(f"[TripleMAAnticipationStrategy] super().__init__(params) a échoué ({te_super_simple}). "
                           "Tentative avec la signature complète de BaseStrategy...")
            try:
                # Tentative d'appel avec la signature complète si la simple a échoué
                # Cela suppose que la BaseStrategy que vous avez fournie au début est maintenant active
                super().__init__(params, pair_symbol, account_id, deployment_id, is_futures)
            except TypeError as te_super_full:
                logger.critical(f"[TripleMAAnticipationStrategy] Les deux tentatives d'appel à super().__init__ ont échoué: "
                                f"Simple: {te_super_simple}, Complète: {te_super_full}. "
                                "Vérifiez la définition de BaseStrategy utilisée par votre environnement.")
                raise  # Impossible de continuer si l'initialisation de base échoue

        # Stocker les arguments supplémentaires manuellement car super() pourrait ne pas les avoir pris.
        # Ceux-ci seront utilisés par les méthodes de CETTE classe.
        # Le log_prefix de BaseStrategy pourrait utiliser des valeurs différentes si l'appel super() simple a été utilisé.
        self.pair_symbol = pair_symbol
        self.account_id = account_id
        self.deployment_id = deployment_id
        self.is_futures = is_futures

        # Construire un log_prefix spécifique à cette instance si celui de BaseStrategy n'est pas complet
        # ou si super().__init__(params) a été utilisé.
        if not hasattr(self, 'log_prefix') or "DEFAULT_PAIR" in self.log_prefix: # Heuristique
            self.log_prefix = f"[{self.account_id}][{self.deployment_id}][{self.pair_symbol}][{self.__class__.__name__}]"
            logger.info(f"{self.log_prefix} log_prefix reconstruit localement.")


        # --- Reste de l'initialisation de TripleMAAnticipationStrategy ---
        self.ma_short_period = int(self.get_param('ma_short_period', 7))
        self.ma_medium_period = int(self.get_param('ma_medium_period', 25))
        self.ma_long_period = int(self.get_param('ma_long_period', 99))
        self.atr_period = int(self.get_param('atr_period_sl_tp', 14))
        self.sl_atr_mult = float(self.get_param('sl_atr_mult', 1.5))
        self.tp_atr_mult = float(self.get_param('tp_atr_mult', 2.0))
        
        self.indicator_frequency = self.get_param('indicator_frequency', '1h')
        self.order_type_preference = self.get_param('order_type_preference', "MARKET")
        self.allow_shorting = bool(self.get_param('allow_shorting', False))

        self.anticipate_crossovers = bool(self.get_param('anticipate_crossovers', False))
        self.anticipation_slope_period = int(self.get_param('anticipation_slope_period', 3))
        if self.anticipation_slope_period < 2:
            logger.warning(f"{self.log_prefix} anticipation_slope_period ({self.anticipation_slope_period}) < 2. Mise à 2.")
            self.anticipation_slope_period = 2
        self.anticipation_convergence_threshold_pct = float(self.get_param('anticipation_convergence_threshold_pct', 0.005))

        self.active_sl_price: Optional[float] = None
        self.active_tp_price: Optional[float] = None
        
        logger.info(
            f"{self.log_prefix} {self.strategy_name if hasattr(self, 'strategy_name') else self.__class__.__name__} initialisée. " # strategy_name vient de BaseStrategy
            f"MAs: {self.ma_short_period}/{self.ma_medium_period}/{self.ma_long_period}, ATR: {self.atr_period}, "
            f"Shorting: {self.allow_shorting}, Anticipate: {self.anticipate_crossovers}"
        )
        if not PANDAS_TA_AVAILABLE:
            logger.error(f"{self.log_prefix} pandas_ta n'est pas disponible.")

    # Les méthodes _calculate_slope, _calculate_indicators, generate_signals, generate_order_request
    # restent les mêmes que dans la version précédente que je vous ai fournie.
    # Assurez-vous qu'elles utilisent self.log_prefix.

    def _calculate_slope(self, series: pd.Series, window: int) -> pd.Series:
        # ... (code de la méthode inchangé)
        if series.isnull().all() or len(series) < window:
            return pd.Series([np.nan] * len(series), index=series.index)
        def get_slope(y_values):
            y = y_values.dropna(); L = len(y)
            if L < 2: return np.nan
            x = np.arange(L)
            try: return np.polyfit(x,y,1)[0]
            except: return np.nan # type: ignore
        return series.rolling(window=window).apply(get_slope, raw=False)

    def _calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        # ... (code de la méthode inchangé, mais utilise self.log_prefix)
        df = data.copy()
        log_pref = self.log_prefix # Assigner à une variable locale pour concision

        if not PANDAS_TA_AVAILABLE:
            logger.error(f"{log_pref}/CalcIndic pandas_ta non disponible.")
            for col_name in ['MA_short', 'MA_medium', 'MA_long', 'ATR', 'slope_MA_short', 'slope_MA_medium']: df[col_name] = np.nan
            return df

        required_cols = ['high', 'low', 'close']
        if not all(col in df.columns for col in required_cols) or df.empty:
            logger.error(f"{log_pref}/CalcIndic Colonnes OHLC requises manquantes ou données vides.")
            for col_name in ['MA_short', 'MA_medium', 'MA_long', 'ATR', 'slope_MA_short', 'slope_MA_medium']: df[col_name] = np.nan
            return df
        
        for col in required_cols: df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df['MA_short'] = df.ta.sma(length=self.ma_short_period, close=df['close'], append=False)
        df['MA_medium'] = df.ta.sma(length=self.ma_medium_period, close=df['close'], append=False)
        df['MA_long'] = df.ta.sma(length=self.ma_long_period, close=df['close'], append=False)
        df['ATR'] = df.ta.atr(length=self.atr_period, high=df['high'], low=df['low'], close=df['close'], append=False)

        if self.anticipate_crossovers:
            df['slope_MA_short'] = self._calculate_slope(df['MA_short'], self.anticipation_slope_period)
            df['slope_MA_medium'] = self._calculate_slope(df['MA_medium'], self.anticipation_slope_period)
        else:
            df['slope_MA_short'] = np.nan
            df['slope_MA_medium'] = np.nan
        
        logger.debug(f"{log_pref}/CalcIndic Indicateurs calculés.")
        return df

    def generate_signals(self, data: pd.DataFrame):
        # ... (code de la méthode inchangé, mais utilise self.log_prefix)
        log_pref = self.log_prefix
        
        required_cols_for_signal = ['MA_short', 'MA_medium', 'MA_long', 'ATR', 'close']
        if self.anticipate_crossovers: required_cols_for_signal.extend(['slope_MA_short', 'slope_MA_medium'])
        
        df_with_indicators = data # Supposer que les indicateurs sont déjà dans 'data' suite à l'appel framework
        if not all(col in df_with_indicators.columns and df_with_indicators[col].notna().any() for col in ['MA_short', 'MA_medium', 'MA_long']):
            logger.info(f"{log_pref}/GenSig MAs non trouvées/NaN, recalcul via _calculate_indicators.")
            df_with_indicators = self._calculate_indicators(data) # Recalculer si manquant
        
        if df_with_indicators.empty or not all(col in df_with_indicators.columns for col in required_cols_for_signal) :
             logger.warning(f"{log_pref}/GenSig Colonnes/données insuffisantes après indicateurs. Skip.")
             self._signals = pd.DataFrame(index=data.index if not data.empty else None, columns=['entry_long', 'exit_long', 'entry_short', 'exit_short', 'sl', 'tp', 'ATR'])
             for col_bool in ['entry_long', 'exit_long', 'entry_short', 'exit_short']: self._signals[col_bool] = False
             for col_float in ['sl', 'tp', 'ATR']: self._signals[col_float] = np.nan
             return

        df = df_with_indicators
        # Reste de la logique de signaux
        actual_long_entry_cross = (df['MA_short'] > df['MA_medium']) & (df['MA_short'].shift(1) <= df['MA_medium'].shift(1)); actual_long_exit_cross = (df['MA_short'] < df['MA_medium']) & (df['MA_short'].shift(1) >= df['MA_medium'].shift(1))
        actual_short_entry_cross = pd.Series(False, index=df.index); actual_short_exit_cross = pd.Series(False, index=df.index)
        if self.allow_shorting: actual_short_entry_cross = (df['MA_short'] < df['MA_medium']) & (df['MA_short'].shift(1) >= df['MA_medium'].shift(1)); actual_short_exit_cross = (df['MA_short'] > df['MA_medium']) & (df['MA_short'].shift(1) <= df['MA_medium'].shift(1))
        anticipated_long_entry = pd.Series(False, index=df.index); anticipated_long_exit = pd.Series(False, index=df.index); anticipated_short_entry = pd.Series(False, index=df.index); anticipated_short_exit = pd.Series(False, index=df.index)
        if self.anticipate_crossovers and 'slope_MA_short' in df.columns and 'slope_MA_medium' in df.columns and df['slope_MA_short'].notna().any() and df['slope_MA_medium'].notna().any():
            convergence_dist = df['MA_medium'] * self.anticipation_convergence_threshold_pct; ma_diff_abs = abs(df['MA_short'] - df['MA_medium'])
            is_converging_up = df['slope_MA_short'] > df['slope_MA_medium']; is_below_and_close_long = (df['MA_short'] < df['MA_medium']) & (ma_diff_abs < convergence_dist); main_trend_bullish = df['MA_medium'] > df['MA_long']; anticipated_long_entry = is_converging_up & is_below_and_close_long & main_trend_bullish
            is_converging_down_for_exit = df['slope_MA_short'] < df['slope_MA_medium']; is_above_and_close_long_exit = (df['MA_short'] > df['MA_medium']) & (ma_diff_abs < convergence_dist); anticipated_long_exit = is_converging_down_for_exit & is_above_and_close_long_exit
            if self.allow_shorting:
                is_converging_down = df['slope_MA_short'] < df['slope_MA_medium']; is_above_and_close_short = (df['MA_short'] > df['MA_medium']) & (ma_diff_abs < convergence_dist); main_trend_bearish = df['MA_medium'] < df['MA_long']; anticipated_short_entry = is_converging_down & is_above_and_close_short & main_trend_bearish
                is_converging_up_for_exit = df['slope_MA_short'] > df['slope_MA_medium']; is_below_and_close_short_exit = (df['MA_short'] < df['MA_medium']) & (ma_diff_abs < convergence_dist); anticipated_short_exit = is_converging_up_for_exit & is_below_and_close_short_exit
        df['entry_long'] = actual_long_entry_cross | anticipated_long_entry; df['exit_long'] = actual_long_exit_cross | anticipated_long_exit; df['entry_short'] = actual_short_entry_cross | anticipated_short_entry; df['exit_short'] = actual_short_exit_cross | anticipated_short_exit
        df['sl'] = np.nan; df['tp'] = np.nan
        long_entry_indices = df[df['entry_long']].index
        if not long_entry_indices.empty: df.loc[long_entry_indices, 'sl'] = df.loc[long_entry_indices, 'close'] - (df.loc[long_entry_indices, 'ATR'] * self.sl_atr_mult); df.loc[long_entry_indices, 'tp'] = df.loc[long_entry_indices, 'close'] + (df.loc[long_entry_indices, 'ATR'] * self.tp_atr_mult)
        if self.allow_shorting: short_entry_indices = df[df['entry_short']].index; S = short_entry_indices; M = self
        if self.allow_shorting and not S.empty: df.loc[S,'sl']=df.loc[S,'close']+(df.loc[S,'ATR']*M.sl_atr_mult); df.loc[S,'tp']=df.loc[S,'close']-(df.loc[S,'ATR']*M.tp_atr_mult) # type: ignore
        self._signals = df[['entry_long', 'exit_long', 'entry_short', 'exit_short', 'sl', 'tp', 'ATR']].copy()
        for col in ['entry_long', 'exit_long', 'entry_short', 'exit_short']:
            if col in self._signals.columns: self._signals[col] = self._signals[col].fillna(False).astype(bool)
        logger.debug(f"{log_pref}/GenSig Signaux générés.")

    def generate_order_request(self,
                               data: pd.DataFrame,
                               current_position_size: float,
                               current_position_avg_price: float,
                               current_position_side: Optional[str],
                               current_strategy_capital: float,
                               position_sizing_pct_of_strategy_capital: float,
                               market_info: Dict[str, Any]
                               ) -> Optional[Tuple[str, float, Optional[float], Optional[str]]]:
        # ... (code de la méthode inchangé, mais utilise self.log_prefix)
        log_pref = self.log_prefix
        current_signals = self.get_signals()
        if current_signals is None or current_signals.empty: logger.warning(f"{log_pref}/GenOrder Aucun signal."); return None
        if len(current_signals) == 0: logger.warning(f"{log_pref}/GenOrder Signals DataFrame vide."); return None
        latest_signal_info = current_signals.iloc[-1]

        current_price = market_info.get('current_price')
        if current_price is None:
            if not data.empty and 'close' in data.columns and not data['close'].empty: current_price = data['close'].iloc[-1]; logger.warning(f"{log_pref}/GenOrder current_price non trouvé, utilisation dernière clôture: {current_price}")
            else: logger.error(f"{log_pref}/GenOrder current_price non dispo."); return None
        if not isinstance(current_price, (int, float)) or not np.isfinite(current_price): logger.error(f"{log_pref}/GenOrder current_price invalide: {current_price}."); return None

        atr_for_sl_tp_calc = latest_signal_info['ATR']
        order_to_place: Optional[Tuple[str, float, Optional[float], Optional[str]]] = None
        position_closed_this_cycle = False

        if current_position_side == "LONG": # ... (logique de sortie LONG) ...
            exit_reason = None
            if self.active_sl_price is not None and current_price <= self.active_sl_price: exit_reason = "Stop-Loss"
            elif self.active_tp_price is not None and current_price >= self.active_tp_price: exit_reason = "Take-Profit"
            elif latest_signal_info['exit_long']: exit_reason = "Exit Long Signal"
            if exit_reason: logger.info(f"{log_pref}/GenOrder Fermeture LONG: {exit_reason}."); order_price_limit = current_price if self.order_type_preference == "LIMIT" else None; order_to_place = ("SELL", current_position_size, order_price_limit, self.order_type_preference); self.active_sl_price, self.active_tp_price = None, None; position_closed_this_cycle = True
        elif current_position_side == "SHORT": # ... (logique de sortie SHORT) ...
            exit_reason = None
            if self.active_sl_price is not None and current_price >= self.active_sl_price: exit_reason = "Stop-Loss"
            elif self.active_tp_price is not None and current_price <= self.active_tp_price: exit_reason = "Take-Profit"
            elif latest_signal_info['exit_short']: exit_reason = "Exit Short Signal"
            if exit_reason: logger.info(f"{log_pref}/GenOrder Fermeture SHORT: {exit_reason}."); order_price_limit = current_price if self.order_type_preference == "LIMIT" else None; order_to_place = ("BUY", current_position_size, order_price_limit, self.order_type_preference); self.active_sl_price, self.active_tp_price = None, None; position_closed_this_cycle = True
        
        if not position_closed_this_cycle and current_position_side is None: # ... (logique d'entrée) ...
            entry_attempted = False
            if pd.isna(atr_for_sl_tp_calc) or atr_for_sl_tp_calc <= 0:
                if latest_signal_info['entry_long'] or (self.allow_shorting and latest_signal_info['entry_short']): logger.warning(f"{log_pref}/GenOrder ATR invalide ({atr_for_sl_tp_calc}). Pas d'entrée.")
            elif latest_signal_info['entry_long']:
                entry_attempted = True; logger.info(f"{log_pref}/GenOrder Signal Entrée Long."); capital_to_commit = current_strategy_capital * position_sizing_pct_of_strategy_capital; quantity = self._calculate_quantity(current_price, capital_to_commit, market_info)
                if quantity is not None and quantity > 0: self.active_sl_price = current_price-(atr_for_sl_tp_calc*self.sl_atr_mult); self.active_tp_price = current_price+(atr_for_sl_tp_calc*self.tp_atr_mult); logger.info(f"{log_pref}/GenOrder Nouveau LONG: SL={self.active_sl_price:.4f}, TP={self.active_tp_price:.4f}"); order_price_limit = current_price if self.order_type_preference=="LIMIT" else None; order_to_place = ("BUY", quantity, order_price_limit, self.order_type_preference)
                else: logger.warning(f"{log_pref}/GenOrder Qty calc échoué pour LONG.")
            elif self.allow_shorting and latest_signal_info['entry_short']:
                entry_attempted = True; logger.info(f"{log_pref}/GenOrder Signal Entrée Short."); capital_to_commit = current_strategy_capital * position_sizing_pct_of_strategy_capital; quantity = self._calculate_quantity(current_price, capital_to_commit, market_info)
                if quantity is not None and quantity > 0: self.active_sl_price = current_price+(atr_for_sl_tp_calc*self.sl_atr_mult); self.active_tp_price = current_price-(atr_for_sl_tp_calc*self.tp_atr_mult); logger.info(f"{log_pref}/GenOrder Nouveau SHORT: SL={self.active_sl_price:.4f}, TP={self.active_tp_price:.4f}"); order_price_limit = current_price if self.order_type_preference=="LIMIT" else None; order_to_place = ("SELL", quantity, order_price_limit, self.order_type_preference)
                else: logger.warning(f"{log_pref}/GenOrder Qty calc échoué pour SHORT.")
            if entry_attempted and order_to_place is None: self.active_sl_price, self.active_tp_price = None, None
        
        if order_to_place: logger.info(f"{log_pref}/GenOrder Requête: {order_to_place}")
        return order_to_place