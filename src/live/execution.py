import logging
import os
import sys
import time
import json
from typing import Dict, Optional, Any, List, Union, Callable

import requests 

logger_execution_init = logging.getLogger(f"{__name__}_init")
if not logger_execution_init.handlers:
    handler_init = logging.StreamHandler(sys.stdout)
    formatter_init = logging.Formatter('%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - EXEC_PRECONFIG - %(message)s')
    handler_init.setFormatter(formatter_init)
    logger_execution_init.addHandler(handler_init)
    logger_execution_init.setLevel(logging.INFO)

logger_execution_init.info("Chargement de execution.py: Tentative d'import des bibliothèques Binance...")

try:
    import binance
    BINANCE_VERSION = getattr(binance, '__version__', 'unknown')
    logger_execution_init.info(f"Version de python-binance importée : {BINANCE_VERSION}.")
    from binance.client import Client
    from binance.exceptions import BinanceAPIException, BinanceRequestException, BinanceOrderException
    BINANCE_IMPORTS_OK = True
    logger_execution_init.info("Importations Binance (Client, Exceptions) réussies.")
except ImportError as e:
    logger_execution_init.critical(f"ÉCHEC CRITIQUE : L'importation des composants Binance a échoué : {e}. Assurez-vous que 'python-binance' (version >= 1.0.19) est installé.", exc_info=True)
    BINANCE_IMPORTS_OK = False
    BINANCE_VERSION = 'non_installé_ou_erreur_import'
    class BinanceAPIException(Exception): pass
    class BinanceRequestException(Exception): pass
    class BinanceOrderException(Exception): pass
    class Client: # type: ignore
        def __init__(self, api_key=None, api_secret=None, testnet=False, requests_params=None):
            logger_execution_init.critical("Client Binance factice utilisé en raison d'une erreur d'importation.")
        def ping(self): raise NotImplementedError("Client Binance factice")
        def get_server_time(self): return {'serverTime': int(time.time() * 1000)}
        def get_symbol_info(self, symbol): raise NotImplementedError("Client Binance factice")
        def get_isolated_margin_account(self, symbols): raise NotImplementedError("Client Binance factice")
        def get_margin_account(self): raise NotImplementedError("Client Binance factice")
        def get_open_margin_orders(self, **params): raise NotImplementedError("Client Binance factice")
        def get_all_oco_orders(self, **params): raise NotImplementedError("Client Binance factice")
        def get_margin_order(self, **params): raise NotImplementedError("Client Binance factice")
        def get_all_margin_orders(self, **params): raise NotImplementedError("Client Binance factice")
        def create_margin_order(self, **params): raise NotImplementedError("Client Binance factice")
        def create_margin_oco_order(self, **params): raise NotImplementedError("Client Binance factice")
        def repay_margin_loan(self, **params): raise NotImplementedError("Client Binance factice")
        def cancel_margin_order(self, **params): raise NotImplementedError("Client Binance factice")
        def cancel_margin_oco_order(self, **params): raise NotImplementedError("Client Binance factice")


except Exception as general_e:
    logger_execution_init.critical(f"ÉCHEC CRITIQUE : Erreur inattendue lors de l'importation de la bibliothèque Binance : {general_e}", exc_info=True)
    BINANCE_IMPORTS_OK = False
    BINANCE_VERSION = 'erreur_import_inconnue'
    class BinanceAPIException(Exception): pass
    class BinanceRequestException(Exception): pass
    class BinanceOrderException(Exception): pass
    class Client: # type: ignore
        def __init__(self, api_key=None, api_secret=None, testnet=False, requests_params=None):
            logger_execution_init.critical("Client Binance factice utilisé en raison d'une erreur d'importation générale.")
        def ping(self): raise NotImplementedError("Client Binance factice")
        def get_server_time(self): return {'serverTime': int(time.time() * 1000)}
        def get_symbol_info(self, symbol): raise NotImplementedError("Client Binance factice")
        def get_isolated_margin_account(self, symbols): raise NotImplementedError("Client Binance factice")
        def get_margin_account(self): raise NotImplementedError("Client Binance factice")
        def get_open_margin_orders(self, **params): raise NotImplementedError("Client Binance factice")
        def get_all_oco_orders(self, **params): raise NotImplementedError("Client Binance factice")
        def get_margin_order(self, **params): raise NotImplementedError("Client Binance factice")
        def get_all_margin_orders(self, **params): raise NotImplementedError("Client Binance factice")
        def create_margin_order(self, **params): raise NotImplementedError("Client Binance factice")
        def create_margin_oco_order(self, **params): raise NotImplementedError("Client Binance factice")
        def repay_margin_loan(self, **params): raise NotImplementedError("Client Binance factice")
        def cancel_margin_order(self, **params): raise NotImplementedError("Client Binance factice")
        def cancel_margin_oco_order(self, **params): raise NotImplementedError("Client Binance factice")

logger = logging.getLogger(__name__)

ACCOUNT_TYPE_MAP = {
    "SPOT": "SPOT",
    "MARGIN": "MARGIN",
    "ISOLATED_MARGIN": "ISOLATED_MARGIN",
    "FUTURES": "FUTURES_USD_M" 
}
USDC_ASSET = "USDC" 
DEFAULT_API_TIMEOUT_SECONDS = 10 
MAX_API_RETRIES = 3 
INITIAL_RETRY_DELAY_SECONDS = 1 

class OrderExecutionClient:
    def __init__(self,
                 api_key: Optional[str] = None,
                 api_secret: Optional[str] = None,
                 account_type: str = "MARGIN", 
                 is_testnet: bool = False):
        
        logger.info(f"[{self.__class__.__name__}] Initialisation pour type de compte : {account_type}, Testnet : {is_testnet}...")
        if not BINANCE_IMPORTS_OK:
            raise ImportError("Les composants de la bibliothèque Binance n'ont pas pu être importés lors de l'initialisation de OrderExecutionClient.")

        self.api_key = api_key or os.getenv("BINANCE_API_KEY_TESTNET" if is_testnet else "BINANCE_API_KEY")
        self.api_secret = api_secret or os.getenv("BINANCE_SECRET_KEY_TESTNET" if is_testnet else "BINANCE_SECRET_KEY")

        if not self.api_key or not self.api_secret:
            logger.error("Clé API/Secrète Binance non trouvée ou non fournie.")
            raise ValueError("Clé API/Secrète Binance non trouvée ou non fournie.")
        logger.info(f"Clés API pour {'Testnet' if is_testnet else 'Production'} chargées.")

        self.raw_account_type = account_type.upper()
        self.mapped_account_type = ACCOUNT_TYPE_MAP.get(self.raw_account_type)
        if not self.mapped_account_type:
            logger.warning(f"Type de compte non supporté '{account_type}'. Utilisation de MARGIN par défaut.")
            self.mapped_account_type = "MARGIN" 
        
        self.is_testnet = is_testnet
        
        try:
            requests_params = {'timeout': DEFAULT_API_TIMEOUT_SECONDS}
            self.client: Client = Client(self.api_key, self.api_secret, testnet=self.is_testnet, requests_params=requests_params)
            logger.info("Client Binance initialisé avec succès.")
        except Exception as e_init:
            logger.critical(f"Échec de l'initialisation du Client Binance : {e_init}", exc_info=True)
            raise ConnectionError(f"L'initialisation du Client Binance a échoué : {e_init}") from e_init

        self._symbol_info_cache: Dict[str, Dict[str, Any]] = {}
        logger.info(f"OrderExecutionClient initialisé pour {self.raw_account_type} (Testnet: {self.is_testnet}).")

    def _make_api_call(self, api_method: Callable, *args, **kwargs) -> Optional[Any]:
        num_retries = kwargs.pop('num_retries', MAX_API_RETRIES)
        initial_delay = kwargs.pop('initial_delay', INITIAL_RETRY_DELAY_SECONDS)
        log_context = kwargs.pop('log_context', api_method.__name__) 

        for attempt in range(num_retries):
            try:
                logger.debug(f"[{log_context}] Tentative d'appel API {attempt + 1}/{num_retries}. Args: {args}, Kwargs: {kwargs}")
                response = api_method(*args, **kwargs)
                logger.debug(f"[{log_context}] Réponse API reçue : {response}")
                return response
            except BinanceAPIException as e:
                logger.error(f"[{log_context}] Erreur API Binance (tentative {attempt + 1}/{num_retries}) : Code={e.code}, Msg='{e.message}'")
                if e.code == -1021: 
                    logger.warning(f"[{log_context}] Erreur de timestamp (-1021). Vérifiez la synchronisation de l'horloge système.")
                if attempt == num_retries - 1: 
                    logger.error(f"[{log_context}] Échec de l'appel API après {num_retries} tentatives.")
                    return None
                delay = initial_delay * (2 ** attempt)
                logger.info(f"[{log_context}] Nouvelle tentative dans {delay} secondes...")
                time.sleep(delay)
            except BinanceRequestException as e:
                logger.error(f"[{log_context}] Erreur de requête Binance (tentative {attempt + 1}/{num_retries}) : {e}")
                return None
            except requests.exceptions.Timeout: 
                logger.error(f"[{log_context}] Timeout de la requête API (tentative {attempt + 1}/{num_retries}).")
                if attempt == num_retries - 1: return None
                delay = initial_delay * (2 ** attempt)
                time.sleep(delay)
            except Exception as e_general:
                logger.error(f"[{log_context}] Erreur inattendue lors de l'appel API (tentative {attempt + 1}/{num_retries}) : {e_general}", exc_info=True)
                if attempt == num_retries - 1: return None
                delay = initial_delay * (2 ** attempt)
                time.sleep(delay)
        return None


    def test_connection(self) -> bool:
        try:
            self.client.ping()
            server_time_response = self._make_api_call(self.client.get_server_time, log_context="test_connection_get_server_time")
            if server_time_response:
                logger.info(f"Test de connexion API REST réussi. Heure du serveur Binance : {server_time_response.get('serverTime', 'N/A')}")
                return True
            else:
                logger.error("Test de connexion API REST échoué : impossible d'obtenir l'heure du serveur.")
                return False
        except Exception as e:
            logger.error(f"Test de connexion API REST échoué : {e}", exc_info=True)
            return False

    def get_symbol_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        symbol_upper = symbol.upper()
        if symbol_upper not in self._symbol_info_cache:
            logger.debug(f"Récupération des informations du symbole pour {symbol_upper} depuis l'API...")
            info = self._make_api_call(self.client.get_symbol_info, symbol_upper, log_context=f"get_symbol_info_{symbol_upper}")
            if info:
                self._symbol_info_cache[symbol_upper] = info
                logger.info(f"Informations du symbole mises en cache pour {symbol_upper}.")
            else:
                logger.error(f"Aucune information de symbole reçue (None ou vide) pour {symbol_upper} depuis l'API.")
                return None
        return self._symbol_info_cache.get(symbol_upper)

    def get_margin_asset_balance(self, asset: str, symbol_pair_for_isolated: Optional[str] = None) -> Optional[float]:
        asset_upper = asset.upper()
        log_ctx = f"get_margin_balance_{asset_upper}" + (f"_isolated_{symbol_pair_for_isolated.upper()}" if symbol_pair_for_isolated else "")
        logger.debug(f"Récupération du solde de marge pour l'actif : {asset_upper}, Paire isolée : {symbol_pair_for_isolated or 'N/A'}")
        
        try:
            asset_detail: Optional[Dict[str, Any]] = None
            if self.raw_account_type == "ISOLATED_MARGIN":
                if not symbol_pair_for_isolated:
                    logger.error(f"Le symbole de la paire isolée doit être fourni pour la vérification du solde ISOLATED_MARGIN (actif : {asset_upper}).")
                    return None
                
                account_details = self._make_api_call(
                    self.client.get_isolated_margin_account, 
                    symbols=symbol_pair_for_isolated.upper(), 
                    log_context=f"{log_ctx}_get_iso_account"
                )
                if not account_details or 'assets' not in account_details:
                    logger.warning(f"Aucun détail de compte de marge isolée trouvé pour le symbole {symbol_pair_for_isolated}.")
                    return 0.0

                pair_specific_assets_info = next((p_info for p_info in account_details.get('assets', []) if p_info.get('symbol') == symbol_pair_for_isolated.upper()), None)
                if not pair_specific_assets_info:
                    logger.warning(f"Aucun détail d'actif spécifique à la paire trouvé pour {symbol_pair_for_isolated} dans le compte de marge isolée.")
                    return 0.0

                if pair_specific_assets_info.get('baseAsset', {}).get('asset', '').upper() == asset_upper:
                    asset_detail = pair_specific_assets_info.get('baseAsset')
                elif pair_specific_assets_info.get('quoteAsset', {}).get('asset', '').upper() == asset_upper:
                    asset_detail = pair_specific_assets_info.get('quoteAsset')
            
            elif self.raw_account_type == "MARGIN": 
                account_details = self._make_api_call(self.client.get_margin_account, log_context=f"{log_ctx}_get_cross_account")
                if not account_details or 'userAssets' not in account_details:
                     logger.warning("Aucun détail de compte de marge croisée trouvé.")
                     return 0.0
                asset_detail = next((a_info for a_info in account_details.get('userAssets', []) if a_info.get('asset', '').upper() == asset_upper), None)
            else:
                logger.error(f"get_margin_asset_balance non supporté pour le type de compte : {self.raw_account_type}")
                return None
            
            free_balance = float(asset_detail.get('free', 0.0)) if asset_detail else 0.0
            logger.info(f"Solde 'free' pour {asset_upper} (Isolé: {symbol_pair_for_isolated or 'N/A'}): {free_balance}")
            return free_balance
        except Exception as e_gen: 
            logger.error(f"Erreur générale lors de la récupération du solde de marge pour {asset_upper} (Isolé : {symbol_pair_for_isolated}) : {e_gen}", exc_info=True)
            return None


    def get_margin_usdc_balance(self, symbol_pair_for_isolated: Optional[str] = None) -> Optional[float]:
        return self.get_margin_asset_balance(USDC_ASSET, symbol_pair_for_isolated=symbol_pair_for_isolated)

    def get_all_open_margin_orders(self, symbol: str, is_isolated: bool) -> List[Dict[str, Any]]:
        log_ctx = f"get_open_margin_orders_{symbol.upper()}" + ("_isolated" if is_isolated else "")
        logger.debug(f"Récupération de tous les ordres de marge ouverts pour le symbole : {symbol}, Isolé : {is_isolated}")
        
        params_api: Dict[str, Any] = {"symbol": symbol.upper()}
        if is_isolated:
            params_api["isIsolated"] = "TRUE" 
        
        open_orders = self._make_api_call(self.client.get_open_margin_orders, **params_api, log_context=log_ctx)
        return open_orders if isinstance(open_orders, list) else []

    def get_open_margin_oco_orders(self, symbol: str, is_isolated: bool) -> List[Dict[str, Any]]:
        log_ctx = f"get_open_margin_oco_{symbol.upper()}" + ("_isolated" if is_isolated else "")
        logger.debug(f"Récupération des ordres OCO de marge ouverts pour le symbole : {symbol}, Isolé : {is_isolated}")
        
        params_api: Dict[str, Any] = {}
        if is_isolated:
            params_api["symbol"] = symbol.upper() 
            params_api["isIsolated"] = "TRUE"
        
        all_open_ocos_raw = self._make_api_call(self.client.get_all_oco_orders, **params_api, log_context=log_ctx) 
        
        active_ocos = []
        if isinstance(all_open_ocos_raw, list):
            for oco in all_open_ocos_raw:
                symbol_match = (not is_isolated and oco.get('symbol', '').upper() == symbol.upper()) or is_isolated
                
                if symbol_match and oco.get('listOrderStatus') in ["EXECUTING", "ALL_DONE_PARTIALLY_FILLED"]: 
                    active_ocos.append(oco)
        logger.info(f"[{log_ctx}] Trouvé {len(active_ocos)} OCO actifs pour {symbol}.")
        return active_ocos


    def get_active_margin_loans(self, asset: Optional[str] = None, isolated_symbol_pair: Optional[str] = None) -> List[Dict[str, Any]]:
        asset_filter = asset.upper() if asset else None
        log_ctx = f"get_active_loans_{asset_filter or 'ALL'}" + (f"_isolated_{isolated_symbol_pair.upper()}" if isolated_symbol_pair else "")
        logger.debug(f"Récupération des prêts sur marge actifs. Actif : {asset_filter or 'Tous'}, Paire isolée : {isolated_symbol_pair or 'N/A'}")
        
        active_loans_found: List[Dict[str, Any]] = []
        try:
            if self.raw_account_type == "ISOLATED_MARGIN":
                if not isolated_symbol_pair:
                    logger.error("Le symbole de la paire isolée doit être fourni pour la vérification du prêt ISOLATED_MARGIN.")
                    return []
                
                account_details = self._make_api_call(
                    self.client.get_isolated_margin_account, 
                    symbols=isolated_symbol_pair.upper(), 
                    log_context=f"{log_ctx}_get_iso_account"
                )
                if not account_details or 'assets' not in account_details: return []

                pair_asset_info = next((p_info for p_info in account_details.get('assets', []) if p_info.get('symbol') == isolated_symbol_pair.upper()), None)
                if pair_asset_info:
                    base_asset_info = pair_asset_info.get('baseAsset', {})
                    quote_asset_info = pair_asset_info.get('quoteAsset', {})
                    if float(base_asset_info.get('borrowed', 0.0)) > 0 and (not asset_filter or base_asset_info.get('asset','').upper() == asset_filter):
                        active_loans_found.append(base_asset_info)
                    if float(quote_asset_info.get('borrowed', 0.0)) > 0 and (not asset_filter or quote_asset_info.get('asset','').upper() == asset_filter):
                        active_loans_found.append(quote_asset_info)
            
            elif self.raw_account_type == "MARGIN": 
                account_details = self._make_api_call(self.client.get_margin_account, log_context=f"{log_ctx}_get_cross_account")
                if not account_details or 'userAssets' not in account_details: return []
                
                for loan_info_item in account_details.get('userAssets', []):
                    if float(loan_info_item.get('borrowed', 0.0)) > 0 and (not asset_filter or loan_info_item.get('asset','').upper() == asset_filter):
                        active_loans_found.append(loan_info_item)
            else:
                logger.error(f"get_active_margin_loans non supporté pour le type de compte : {self.raw_account_type}")
                return []
            
            logger.info(f"[{log_ctx}] Trouvé {len(active_loans_found)} prêts actifs" + (f" pour l'actif {asset_filter}." if asset_filter else "."))
            return active_loans_found
        except Exception as e_gen:
            logger.error(f"Erreur générale lors de la récupération des prêts sur marge actifs : {e_gen}", exc_info=True)
            return []
            
    def get_margin_order_status(self, symbol: str, order_id: Optional[Union[int, str]] = None,
                                orig_client_order_id: Optional[str] = None, is_isolated: bool = False) -> Optional[Dict[str, Any]]:
        if not order_id and not orig_client_order_id:
            logger.error("order_id ou orig_client_order_id est requis pour get_margin_order_status.")
            return None
            
        params_api: Dict[str, Any] = {"symbol": symbol.upper()}
        if order_id: params_api["orderId"] = str(order_id) 
        if orig_client_order_id: params_api["origClientOrderId"] = orig_client_order_id
        
        params_api["isIsolated"] = "TRUE" if is_isolated else "FALSE"
        
        log_ctx = f"get_margin_order_{symbol.upper()}_id_{order_id or orig_client_order_id}" + ("_isolated" if is_isolated else "")
        logger.debug(f"Récupération du statut de l'ordre de marge pour {symbol} (Isolé : {is_isolated}) : ID={order_id}, ClientID={orig_client_order_id}")
        
        order_status = self._make_api_call(self.client.get_margin_order, **params_api, log_context=log_ctx)
        
        if order_status is None and order_id: 
             logger.warning(f"[{log_ctx}] Échec de get_margin_order, tentative de récupération via get_all_margin_orders...")
             all_orders = self._make_api_call(self.client.get_all_margin_orders, symbol=symbol.upper(), isIsolated=str(is_isolated).upper(), limit=10, log_context=f"{log_ctx}_fallback_all") 
             if isinstance(all_orders, list):
                 order_status = next((o for o in all_orders if str(o.get('orderId')) == str(order_id)), None)
                 if order_status: logger.info(f"[{log_ctx}] Ordre trouvé via fallback get_all_margin_orders.")
                 else: logger.warning(f"[{log_ctx}] Ordre non trouvé même avec fallback.")
        
        if order_status and isinstance(order_status, dict) and order_status.get("code") == -2013 : 
            logger.warning(f"[{log_ctx}] L'ordre (ID: {order_id}, ClientID: {orig_client_order_id}) n'existe pas sur l'exchange.")
            return {"status": "NOT_FOUND", "message": "Order does not exist."} 

        return order_status


    def place_margin_order(self, **params: Any) -> Dict[str, Any]:
        if self.raw_account_type not in ["MARGIN", "ISOLATED_MARGIN"]:
            return {"status": "ERROR", "message": f"Type de compte {self.raw_account_type} non supporté pour les ordres de marge."}
        
        final_params = params.copy()
        if 'isIsolated' not in final_params and self.raw_account_type == "ISOLATED_MARGIN":
            final_params['isIsolated'] = "TRUE"
        elif 'isIsolated' in final_params and isinstance(final_params['isIsolated'], bool):
            final_params['isIsolated'] = "TRUE" if final_params['isIsolated'] else "FALSE"

        symbol = str(final_params.get("symbol", "INCONNU"))
        log_ctx = f"place_margin_order_{symbol}_{final_params.get('side', '')}_{final_params.get('type', '')}"
        logger.info(f"[{log_ctx}] Tentative d'ordre de marge : {final_params.get('side')} {final_params.get('type')} (isIsolated pour méthode : {final_params.get('isIsolated')}) Qty : {final_params.get('quantity')} Px : {final_params.get('price', 'N/A')}")
        logger.debug(f"  Paramètres complets pour create_margin_order : {json.dumps(final_params)}")

        response = self._make_api_call(self.client.create_margin_order, **final_params, log_context=log_ctx)

        if response and (response.get("orderId") or response.get("clientOrderId")): 
            logger.info(f"[{log_ctx}] Réponse API de l'ordre de marge : {response}")
            return {"status": "SUCCESS", "data": response}
        else: 
            error_code = response.get("code") if isinstance(response, dict) else None
            error_msg = response.get("msg") if isinstance(response, dict) else str(response)
            logger.error(f"[{log_ctx}] Erreur API lors du placement de l'ordre de marge pour {symbol} : Code={error_code}, Msg='{error_msg}'. Paramètres : {final_params}")
            return {"status": "API_ERROR", "code": error_code, "message": error_msg, "params_sent": final_params}


    def place_margin_oco_order(self, **params: Any) -> Dict[str, Any]:
        if self.raw_account_type not in ["MARGIN", "ISOLATED_MARGIN"]:
            return {"status": "ERROR", "message": f"Type de compte {self.raw_account_type} non supporté pour les ordres OCO sur marge."}

        final_params = params.copy()
        if 'isIsolated' not in final_params and self.raw_account_type == "ISOLATED_MARGIN":
             final_params['isIsolated'] = "TRUE"
        elif 'isIsolated' in final_params and isinstance(final_params['isIsolated'], bool):
             final_params['isIsolated'] = "TRUE" if final_params['isIsolated'] else "FALSE"

        symbol = str(final_params.get("symbol", "INCONNU"))
        log_ctx = f"place_margin_oco_{symbol}_{final_params.get('side', '')}"
        logger.info(f"[{log_ctx}] Tentative d'ordre OCO sur marge (isIsolated pour méthode : {final_params.get('isIsolated')}). Side : {final_params.get('side')}, Qty : {final_params.get('quantity')}, TP Px : {final_params.get('price')}, SL TrigPx : {final_params.get('stopPrice')}")
        logger.debug(f"  Paramètres OCO complets pour create_margin_oco_order : {json.dumps(final_params)}")
        
        response = self._make_api_call(self.client.create_margin_oco_order, **final_params, log_context=log_ctx)

        if response and response.get("orderListId"): 
            logger.info(f"[{log_ctx}] Réponse API de l'ordre OCO sur marge : {response}")
            return {"status": "SUCCESS", "data": response}
        else:
            error_code = response.get("code") if isinstance(response, dict) else None
            error_msg = response.get("msg") if isinstance(response, dict) else str(response)
            logger.error(f"[{log_ctx}] Erreur API lors du placement de l'ordre OCO sur marge pour {symbol} : Code={error_code}, Msg='{error_msg}'. Paramètres : {final_params}")
            return {"status": "API_ERROR", "code": error_code, "message": error_msg, "params_sent": final_params}

    def repay_margin_loan(self, asset: str, amount: Union[float, str], isolated_symbol_pair: Optional[str] = None) -> Dict[str, Any]:
        if self.raw_account_type not in ["MARGIN", "ISOLATED_MARGIN"]:
            return {"status": "ERROR", "message": f"Type de compte {self.raw_account_type} non supporté pour le remboursement de prêt."}
        
        repay_params: Dict[str,Any] = {"asset": asset.upper(), "amount": str(amount)}
        is_isolated_repay = False
        if self.raw_account_type == "ISOLATED_MARGIN":
            if not isolated_symbol_pair:
                logger.error("Le symbole de la paire isolée est requis pour le remboursement de prêt sur marge ISOLATED_MARGIN.")
                return {"status": "ERROR", "message": "Symbole de paire isolée manquant pour le remboursement de prêt sur marge isolée."}
            repay_params["isIsolated"] = "TRUE" 
            repay_params["symbol"] = isolated_symbol_pair.upper()
            is_isolated_repay = True
        
        log_ctx = f"repay_margin_loan_{asset.upper()}_{amount}" + (f"_isolated_{isolated_symbol_pair.upper()}" if isolated_symbol_pair else "")
        logger.info(f"[{log_ctx}] Tentative de remboursement de prêt sur marge : {amount} {asset} (Contexte isolé : {is_isolated_repay}, Paire : {isolated_symbol_pair or 'N/A'})")
        logger.debug(f"  Paramètres de remboursement pour client.repay_margin_loan : {json.dumps(repay_params)}")
        
        response = self._make_api_call(self.client.repay_margin_loan, **repay_params, log_context=log_ctx)

        if response and response.get("tranId"): 
            logger.info(f"[{log_ctx}] Réponse API de remboursement de prêt sur marge pour {asset} : {response}")
            return {"status": "SUCCESS", "data": response}
        else:
            error_code = response.get("code") if isinstance(response, dict) else None
            error_msg = response.get("msg") if isinstance(response, dict) else str(response)
            logger.warning(f"[{log_ctx}] Réponse de remboursement pour {asset} sans tranId clair ou indication de succès : {response}")
            return {"status": "API_ERROR_OR_UNKNOWN", "code": error_code, "message": error_msg, "params_sent": repay_params}

    def cancel_margin_order(self, symbol: str, order_id: Optional[Union[int, str]] = None, 
                            orig_client_order_id: Optional[str] = None, is_isolated: bool = False) -> Dict[str, Any]:
        if not order_id and not orig_client_order_id:
            logger.error("order_id ou orig_client_order_id est requis pour cancel_margin_order.")
            return {"status": "ERROR", "message": "order_id ou orig_client_order_id manquant."}

        params_api: Dict[str, Any] = {"symbol": symbol.upper()}
        if order_id: params_api["orderId"] = str(order_id)
        if orig_client_order_id: params_api["origClientOrderId"] = orig_client_order_id
        params_api["isIsolated"] = "TRUE" if is_isolated else "FALSE"
        
        log_ctx = f"cancel_margin_order_{symbol.upper()}_id_{order_id or orig_client_order_id}" + ("_isolated" if is_isolated else "")
        logger.info(f"[{log_ctx}] Tentative d'annulation de l'ordre de marge : ID={order_id}, ClientID={orig_client_order_id}, Isolé={is_isolated}")

        response = self._make_api_call(self.client.cancel_margin_order, **params_api, log_context=log_ctx)

        if response and (response.get("orderId") or response.get("clientOrderId")): 
            logger.info(f"[{log_ctx}] Réponse API d'annulation d'ordre de marge : {response}")
            return {"status": "SUCCESS", "data": response}
        else:
            error_code = response.get("code") if isinstance(response, dict) else None
            error_msg = response.get("msg") if isinstance(response, dict) else str(response)
            if error_code == -2011: 
                 logger.warning(f"[{log_ctx}] Tentative d'annulation d'un ordre inconnu ou déjà traité : {error_msg}")
                 return {"status": "ORDER_NOT_FOUND_OR_ALREADY_PROCESSED", "code": error_code, "message": error_msg, "params_sent": params_api}
            logger.error(f"[{log_ctx}] Erreur API lors de l'annulation de l'ordre de marge : Code={error_code}, Msg='{error_msg}'. Paramètres : {params_api}")
            return {"status": "API_ERROR", "code": error_code, "message": error_msg, "params_sent": params_api}

    def cancel_margin_oco_order(self, symbol: str, order_list_id: Optional[int] = None, 
                                list_client_order_id: Optional[str] = None, is_isolated: bool = False) -> Dict[str, Any]:
        if not order_list_id and not list_client_order_id:
            logger.error("order_list_id ou list_client_order_id est requis pour cancel_margin_oco_order.")
            return {"status": "ERROR", "message": "order_list_id ou list_client_order_id manquant."}

        params_api: Dict[str, Any] = {"symbol": symbol.upper()}
        if order_list_id: params_api["orderListId"] = order_list_id
        if list_client_order_id: params_api["listClientOrderId"] = list_client_order_id
        params_api["isIsolated"] = "TRUE" if is_isolated else "FALSE"

        log_ctx = f"cancel_margin_oco_{symbol.upper()}_id_{order_list_id or list_client_order_id}" + ("_isolated" if is_isolated else "")
        logger.info(f"[{log_ctx}] Tentative d'annulation de l'ordre OCO sur marge : ListID={order_list_id}, ListClientID={list_client_order_id}, Isolé={is_isolated}")

        response = self._make_api_call(self.client.cancel_margin_oco_order, **params_api, log_context=log_ctx)
        
        if response and response.get("orderListId"): 
            logger.info(f"[{log_ctx}] Réponse API d'annulation d'ordre OCO sur marge : {response}")
            return {"status": "SUCCESS", "data": response}
        else:
            error_code = response.get("code") if isinstance(response, dict) else None
            error_msg = response.get("msg") if isinstance(response, dict) else str(response)
            if error_code == -2011: 
                 logger.warning(f"[{log_ctx}] Tentative d'annulation d'une liste OCO inconnue ou déjà traitée : {error_msg}")
                 return {"status": "ORDER_LIST_NOT_FOUND_OR_ALREADY_PROCESSED", "code": error_code, "message": error_msg, "params_sent": params_api}
            logger.error(f"[{log_ctx}] Erreur API lors de l'annulation de l'ordre OCO sur marge : Code={error_code}, Msg='{error_msg}'. Paramètres : {params_api}")
            return {"status": "API_ERROR", "code": error_code, "message": error_msg, "params_sent": params_api}


    def close(self):
        logger.info(f"Fermeture de OrderExecutionClient pour le type de compte {self.raw_account_type}...")
        logger.info(f"OrderExecutionClient pour le type de compte {self.raw_account_type} fermé.")

