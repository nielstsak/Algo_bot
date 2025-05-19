import json
import logging
import time
from pathlib import Path
from typing import Dict, Optional, Any, Union

logger = logging.getLogger(__name__)

STATUT_1_NO_TRADE_NO_OCO = "STATUT_1_NO_TRADE_NO_OCO"
STATUT_2_ENTRY_FILLED_OCO_PENDING = "STATUT_2_ENTRY_FILLED_OCO_PENDING"
STATUT_3_OCO_ACTIVE = "STATUT_3_OCO_ACTIVE"

USDC_ASSET = "USDC" # Définition de la constante manquante

class LiveTradingState:
    def __init__(self, pair_symbol: str, state_file_path: Union[str, Path]):
        self.pair_symbol = pair_symbol.upper()
        self.state_file_path = Path(state_file_path)
        self.state: Dict[str, Any] = self._load_state()

        if not self.state.get("current_status") or self.state.get("pair_symbol") != self.pair_symbol:
            logger.warning(f"[{self.pair_symbol}] État initial chargé invalide ou pour une paire différente. Forçage de la réinitialisation.")
            self.state = self._default_state()
            self._save_state()

        logger.info(f"[{self.pair_symbol}] LiveTradingState initialisé. Statut actuel : {self.get_current_status_name()}. Fichier : {self.state_file_path}")

    def _default_state(self) -> Dict[str, Any]:
        return {
            "pair_symbol": self.pair_symbol,
            "current_status": STATUT_1_NO_TRADE_NO_OCO,
            "current_trade_cycle_id": None,
            "last_status_update_timestamp": time.time() * 1000,
            "last_error": None,
            "available_capital_at_last_check": 0.0,
            "pending_entry_order_id": None,
            "pending_entry_client_order_id": None,
            "pending_entry_params": {},
            "pending_sl_tp_raw": {},
            "entry_order_details": {},
            "position_side": None,
            "position_quantity": 0.0,
            "position_entry_price": 0.0,
            "position_entry_timestamp": None,
            "position_total_commission_usdc_equivalent": 0.0,
            "loan_details": {
                "asset": None,
                "amount": 0.0,
                "timestamp": None
            },
            "oco_params_to_place": {},
            "pending_oco_list_client_order_id": None,
            "pending_oco_order_list_id_api": None, 
            "active_oco_details": {},
            "active_oco_list_client_order_id": None,
            "active_oco_order_list_id": None,
            "active_sl_order_id": None,
            "active_tp_order_id": None,
            "active_sl_price": None,
            "active_tp_price": None,
            "oco_active_timestamp": None,
            "last_closed_trade_info": {}
        }

    def _load_state(self) -> Dict[str, Any]:
        if self.state_file_path.exists():
            try:
                with open(self.state_file_path, 'r', encoding='utf-8') as f:
                    loaded_state = json.load(f)
                if "current_status" in loaded_state and loaded_state.get("pair_symbol") == self.pair_symbol:
                    logger.info(f"[{self.pair_symbol}] État chargé depuis {self.state_file_path}")
                    default_s = self._default_state()
                    for key, value in default_s.items():
                        if key not in loaded_state:
                            loaded_state[key] = value
                            logger.info(f"[{self.pair_symbol}] Ajout du champ manquant '{key}' avec la valeur par défaut '{value}' à l'état chargé.")
                    return loaded_state
                else:
                    logger.warning(f"[{self.pair_symbol}] Fichier d'état {self.state_file_path} invalide ou pour une paire différente. Réinitialisation.")
            except json.JSONDecodeError:
                logger.error(f"[{self.pair_symbol}] Erreur de décodage JSON du fichier d'état {self.state_file_path}. Réinitialisation.", exc_info=True)
            except Exception as e:
                logger.error(f"[{self.pair_symbol}] Erreur inattendue lors du chargement de l'état depuis {self.state_file_path} : {e}. Réinitialisation.", exc_info=True)
        else:
            logger.info(f"[{self.pair_symbol}] Fichier d'état non trouvé à {self.state_file_path}. Initialisation avec l'état par défaut.")
        return self._default_state()

    def _save_state(self):
        try:
            self.state_file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.state_file_path, 'w', encoding='utf-8') as f:
                json.dump(self.state, f, indent=4, default=str)
            logger.debug(f"[{self.pair_symbol}] État sauvegardé dans {self.state_file_path}")
        except Exception as e:
            logger.error(f"[{self.pair_symbol}] Erreur lors de la sauvegarde de l'état dans {self.state_file_path} : {e}", exc_info=True)

    def update_specific_fields(self, update_dict: Dict[str, Any]):
        for key, value in update_dict.items():
            self.state[key] = value
        self.state["last_status_update_timestamp"] = time.time() * 1000
        self._save_state()
        logger.info(f"[{self.pair_symbol}] Champs d'état mis à jour : {list(update_dict.keys())}. Statut actuel : {self.get_current_status_name()}")

    def get_current_status(self) -> str:
        return self.state.get("current_status", STATUT_1_NO_TRADE_NO_OCO)

    def get_current_status_name(self) -> str:
        return self.state.get("current_status", "UNKNOWN").replace("STATUT_", "")

    def get_state_snapshot(self) -> Dict[str, Any]:
        return self.state.copy()

    def set_last_error(self, error_message: Optional[str]):
        self.update_specific_fields({"last_error": error_message})
        if error_message:
            logger.error(f"[{self.pair_symbol}] Erreur enregistrée dans l'état : {error_message}")

    def clear_last_error(self):
        if self.state.get("last_error"):
            self.update_specific_fields({"last_error": None})
            logger.info(f"[{self.pair_symbol}] Erreur précédente effacée de l'état.")

    def transition_to_status_1(self, exit_reason: Optional[str] = "REINITIALIZATION",
                                 closed_trade_details: Optional[Dict[str, Any]] = None):
        logger.info(f"[{self.pair_symbol}] Transition vers STATUT_1_NO_TRADE_NO_OCO. Raison : {exit_reason}. Cycle ID actuel (avant réinitialisation) : {self.state.get('current_trade_cycle_id')}")
        
        current_available_capital = self.state.get("available_capital_at_last_check", 0.0)
        last_trade_info = self.state.get("last_closed_trade_info", {})
        if closed_trade_details: 
            last_trade_info = closed_trade_details

        new_state_fields = self._default_state() 
        new_state_fields["available_capital_at_last_check"] = current_available_capital
        new_state_fields["last_closed_trade_info"] = last_trade_info
        new_state_fields["current_trade_cycle_id"] = None 

        self.state.update(new_state_fields)
        self.state["last_status_update_timestamp"] = time.time() * 1000
        self.clear_last_error()
        self._save_state()

    def prepare_for_entry_order(self, entry_params: Dict[str, Any], sl_tp_raw: Dict[str, float], trade_cycle_id: str):
        logger.info(f"[{self.pair_symbol}][{trade_cycle_id}] Préparation pour l'ordre d'entrée. Paramètres : {entry_params}, SL/TP bruts : {sl_tp_raw}")
        self.update_specific_fields({
            "pending_entry_params": entry_params,
            "pending_sl_tp_raw": sl_tp_raw,
            "pending_entry_order_id": None,
            "pending_entry_client_order_id": None,
            "current_trade_cycle_id": trade_cycle_id
        })

    def record_placed_entry_order(self, order_id: Union[str, int], client_order_id: str):
        trade_cycle_id = self.state.get("current_trade_cycle_id", "UNKNOWN_CYCLE")
        logger.info(f"[{self.pair_symbol}][{trade_cycle_id}] Ordre d'entrée placé. OrderID : {order_id}, ClientOrderID : {client_order_id}")
        self.update_specific_fields({
            "pending_entry_order_id": str(order_id),
            "pending_entry_client_order_id": client_order_id
        })

    def transition_to_status_2(self, filled_entry_details: Dict[str, Any],
                                 loan_info: Dict[str, Any]):
        trade_cycle_id = self.state.get("current_trade_cycle_id", "UNKNOWN_CYCLE")
        logger.info(f"[{self.pair_symbol}][{trade_cycle_id}] Transition vers STATUT_2_ENTRY_FILLED_OCO_PENDING. Détails entrée : {filled_entry_details}, Prêt : {loan_info}")
        
        self.state["pending_entry_order_id"] = None
        self.state["pending_entry_client_order_id"] = None
        
        executed_qty = float(filled_entry_details.get("executedQty", 0.0))
        cummulative_quote_qty = float(filled_entry_details.get("cummulativeQuoteQty", 0.0))
        entry_price = cummulative_quote_qty / executed_qty if executed_qty > 0 else 0.0
        
        commission_total_usdc = 0.0
        if 'fills' in filled_entry_details and isinstance(filled_entry_details['fills'], list):
            for fill in filled_entry_details['fills']:
                if fill.get('commissionAsset', '').upper() == USDC_ASSET:
                    commission_total_usdc += float(fill.get('commission', 0.0))
        elif filled_entry_details.get('commissionAsset', '').upper() == USDC_ASSET:
             commission_total_usdc = float(filled_entry_details.get('commission', 0.0))


        self.state.update({
            "current_status": STATUT_2_ENTRY_FILLED_OCO_PENDING,
            "entry_order_details": filled_entry_details,
            "position_side": filled_entry_details.get("side"),
            "position_quantity": executed_qty,
            "position_entry_price": entry_price,
            "position_entry_timestamp": filled_entry_details.get("transactTime") or filled_entry_details.get("updateTime"),
            "position_total_commission_usdc_equivalent": commission_total_usdc,
            "loan_details": loan_info,
            "oco_params_to_place": {},
            "pending_oco_list_client_order_id": None,
            "pending_oco_order_list_id_api": None
        })
        self.state["last_status_update_timestamp"] = time.time() * 1000
        self._save_state()

    def prepare_for_oco_order(self, oco_params: Dict[str, Any], list_client_order_id: str, order_list_id_api: Optional[str] = None):
        trade_cycle_id = self.state.get("current_trade_cycle_id", "UNKNOWN_CYCLE")
        logger.info(f"[{self.pair_symbol}][{trade_cycle_id}] Préparation pour l'ordre OCO. Paramètres : {oco_params}, ListClientOrderID : {list_client_order_id}, OrderListId API (si connu) : {order_list_id_api}")
        self.update_specific_fields({
            "oco_params_to_place": oco_params,
            "pending_oco_list_client_order_id": list_client_order_id,
            "pending_oco_order_list_id_api": order_list_id_api, 
            "active_oco_details": {}, "active_oco_list_client_order_id": None,
            "active_oco_order_list_id": None, "active_sl_order_id": None,
            "active_tp_order_id": None, "active_sl_price": None,
            "active_tp_price": None, "oco_active_timestamp": None
        })

    def transition_to_status_3(self, active_oco_api_response: Dict[str, Any]):
        trade_cycle_id = self.state.get("current_trade_cycle_id", "UNKNOWN_CYCLE")
        logger.info(f"[{self.pair_symbol}][{trade_cycle_id}] Transition vers STATUT_3_OCO_ACTIVE. Détails OCO : {active_oco_api_response}")
        
        list_client_order_id = active_oco_api_response.get("listClientOrderId", self.state.get("pending_oco_list_client_order_id"))
        order_list_id = active_oco_api_response.get("orderListId")
        
        sl_order_id, tp_order_id = None, None
        sl_price, tp_price = None, None
        
        order_reports = active_oco_api_response.get("orders", []) 
        if not order_reports and "orderReports" in active_oco_api_response: 
            order_reports = active_oco_api_response.get("orderReports", [])

        if isinstance(order_reports, list):
            for report in order_reports:
                if report.get("type") in ["STOP_LOSS_LIMIT", "STOP_LOSS", "STOP_MARKET"]:
                    sl_order_id = str(report.get("orderId"))
                    sl_price = float(report.get("stopPrice", 0.0)) 
                elif report.get("type") in ["LIMIT", "TAKE_PROFIT_LIMIT", "TAKE_PROFIT", "LIMIT_MAKER"]:
                    tp_order_id = str(report.get("orderId"))
                    tp_price = float(report.get("price", 0.0)) 

        self.state.update({
            "current_status": STATUT_3_OCO_ACTIVE,
            "oco_params_to_place": {},
            "pending_oco_list_client_order_id": None,
            "pending_oco_order_list_id_api": None,
            "active_oco_details": active_oco_api_response,
            "active_oco_list_client_order_id": list_client_order_id,
            "active_oco_order_list_id": str(order_list_id) if order_list_id is not None else None,
            "active_sl_order_id": sl_order_id,
            "active_tp_order_id": tp_order_id,
            "active_sl_price": sl_price,
            "active_tp_price": tp_price,
            "oco_active_timestamp": active_oco_api_response.get("transactionTime") or (time.time() * 1000)
        })
        self.state["last_status_update_timestamp"] = time.time() * 1000
        self._save_state()

    def record_closed_trade(self, exit_reason: str, exit_price: Optional[float], pnl_usdc: Optional[float],
                              closed_order_details: Optional[Dict[str, Any]]):
        trade_cycle_id = self.state.get("current_trade_cycle_id", "UNKNOWN_CYCLE_CLOSURE")
        logger.info(f"[{self.pair_symbol}][{trade_cycle_id}] Enregistrement du trade clôturé. Raison : {exit_reason}, Prix de sortie : {exit_price}, PNL : {pnl_usdc}")
        
        entry_details = self.state.get("entry_order_details", {})
        commission_entry_usdc = self.state.get("position_total_commission_usdc_equivalent", 0.0)
        commission_exit_usdc = 0.0

        if closed_order_details and 'fills' in closed_order_details and isinstance(closed_order_details['fills'], list):
            for fill in closed_order_details['fills']:
                if fill.get('commissionAsset', '').upper() == USDC_ASSET:
                    commission_exit_usdc += float(fill.get('commission', 0.0))
        elif closed_order_details and closed_order_details.get('commissionAsset','').upper() == USDC_ASSET:
            commission_exit_usdc = float(closed_order_details.get('commission',0.0))


        self.state["last_closed_trade_info"] = {
            "trade_cycle_id": trade_cycle_id,
            "timestamp_closure_utc": time.strftime("%Y-%m-%dT%H:%M:%S.%fZ", time.gmtime()),
            "pair_symbol": self.pair_symbol,
            "entry_order_details": entry_details,
            "position_side": self.state.get("position_side"),
            "position_quantity": self.state.get("position_quantity"),
            "position_entry_price": self.state.get("position_entry_price"),
            "commission_entry_usdc": commission_entry_usdc,
            "loan_details_at_entry": self.state.get("loan_details", {}),
            "oco_details_at_placement": self.state.get("active_oco_details", {}),
            "exit_reason": exit_reason,
            "exit_price": exit_price,
            "commission_exit_usdc": commission_exit_usdc,
            "pnl_usdc_estimate_before_fees_and_funding": pnl_usdc, 
            "closed_order_details": closed_order_details
        }
        self._save_state()
