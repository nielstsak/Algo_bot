# Fichier: test_otoco_execution.py (ou test_otoco_format.py)
# Description: Teste l'exécution réelle d'un ordre OTOCO Margin via l'API Testnet Binance.
# ATTENTION: Ce script INTERAGIT avec l'API Testnet Binance. Assurez-vous d'avoir
#            configuré les clés API Testnet et d'avoir des fonds Testnet suffisants.

import unittest
import logging
import math
import time
import uuid
import os
from pathlib import Path # Importer Path
from typing import Dict, Optional, Any

# --- Charger les variables d'environnement depuis .env ---
# Ajouter cet import
from dotenv import load_dotenv

# Charger le fichier .env situé à la racine du projet
# Détermine la racine du projet en remontant depuis ce fichier
# Ajustez le nombre de .parent si la structure est différente (ex: si ce fichier est dans src/tests/)
try:
    # Essayer de déterminer la racine du projet de manière robuste
    # Si ce script est à la racine:
    # project_root = Path(__file__).resolve().parent
    # Si ce script est dans un sous-dossier (ex: tests/):
    project_root = Path(__file__).resolve().parent
except NameError:
    # Fallback si __file__ n'est pas défini (ex: environnement interactif)
    project_root = Path(".").resolve()

env_path = project_root / '.env'
if env_path.is_file(): # Vérifier si c'est bien un fichier
    loaded = load_dotenv(dotenv_path=env_path, verbose=True) # verbose=True pour voir les détails
    if loaded:
        print(f"Loaded environment variables from: {env_path}")
    else:
        print(f"WARNING: .env file found at {env_path} but python-dotenv could not load it (check format?).")
else:
    print(f"WARNING: .env file not found at {env_path}. API keys must be set directly in the environment.")
# ---------------------------------------------------------


# Assurez-vous que le répertoire 'src' est dans le PYTHONPATH
try:
    # Tenter l'import relatif si ce fichier est dans un sous-répertoire de 'src'
    from strategies.base import BaseStrategy
    from live.execution import OrderExecutionClient # Importer le client d'exécution
    from utils.exchange_utils import adjust_precision, get_precision_from_filter, get_filter_value
except ImportError:
    # Tenter l'import absolu si lancé depuis la racine ou si 'src' est dans PYTHONPATH
    try:
        # Ajouter src au path si nécessaire
        src_path = project_root / "src"
        import sys
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))
            print(f"Added {src_path} to sys.path for imports.")

        from src.strategies.base import BaseStrategy
        from src.live.execution import OrderExecutionClient
        from src.utils.exchange_utils import adjust_precision, get_precision_from_filter, get_filter_value
    except ImportError as e_abs:
        print(f"ERREUR: Impossible d'importer les modules nécessaires: {e_abs}")
        print(f"Project Root: {project_root}, Src Path: {src_path}")
        print("Assurez-vous que le répertoire 'src' est accessible (PYTHONPATH).")
        exit(1) # Arrêter si les imports échouent

# Configuration du logging pour les tests
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Classe de Stratégie Minimale pour le Test ---
class MinimalStrategy(BaseStrategy):
    """Stratégie minimale juste pour pouvoir instancier et tester _build_otoco_order."""
    def generate_signals(self, data): pass
    def generate_order_request(self, data, symbol, current_position, available_capital, symbol_info): pass

# --- Données de Test ---
TEST_SYMBOL = "BTCUSDT" # Symbole valide sur Testnet Margin
# Les infos symbole sont récupérées dans setUpClass

# --- Classe de Test ---
class TestOtocoMarginExecution(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Initialisation unique avant tous les tests de la classe."""
        logger.info("Initialisation de la classe de test TestOtocoMarginExecution...")
        # Lire les clés API DEPUIS l'environnement (qui DOIT avoir été chargé par load_dotenv)
        cls.api_key = os.getenv("BINANCE_TESTNET_API_KEY")
        cls.api_secret = os.getenv("BINANCE_TESTNET_API_SECRET")

        # Vérification explicite que les clés ont été chargées
        if not cls.api_key or not cls.api_secret:
            logger.error("BINANCE_TESTNET_API_KEY ou BINANCE_TESTNET_API_SECRET n'ont pas été trouvées après load_dotenv.")
            logger.error(f"  Vérifiez le chemin du .env ({env_path}) et son contenu.")
            raise ValueError("Variables d'environnement BINANCE_TESTNET_API_KEY et BINANCE_TESTNET_API_SECRET doivent être définies (dans .env ou l'environnement système).")
        else:
            logger.info("Clés API Testnet lues depuis l'environnement.")


        try:
            cls.execution_client = OrderExecutionClient(
                api_key=cls.api_key,
                api_secret=cls.api_secret,
                account_type="MARGIN",
                is_testnet=True
            )
            if not cls.execution_client.test_connection():
                raise ConnectionError("Échec de la connexion à l'API Testnet Binance.")
            logger.info("Client d'exécution Testnet initialisé et connexion testée.")

            cls.symbol_info = cls.execution_client.get_symbol_info(TEST_SYMBOL)
            if not cls.symbol_info:
                raise RuntimeError(f"Impossible de récupérer les infos pour {TEST_SYMBOL} sur Testnet.")
            cls.price_precision = get_precision_from_filter(cls.symbol_info, 'PRICE_FILTER', 'tickSize')
            cls.qty_precision = get_precision_from_filter(cls.symbol_info, 'LOT_SIZE', 'stepSize')
            if cls.price_precision is None or cls.qty_precision is None:
                 raise RuntimeError(f"Impossible de déterminer la précision pour {TEST_SYMBOL} sur Testnet.")
            logger.info(f"Infos et précisions pour {TEST_SYMBOL} récupérées depuis Testnet (PricePrec: {cls.price_precision}, QtyPrec: {cls.qty_precision}).")

        except Exception as e:
            logger.critical(f"Erreur lors de l'initialisation de la classe de test: {e}", exc_info=True)
            raise

    def setUp(self):
        """Initialisation avant chaque test."""
        self.strategy = MinimalStrategy(params={
            'otoco_params': {
                'slOrderType': "STOP_LOSS",
                'tpOrderType': "LIMIT_MAKER"
            }
        })
        self.symbol = self.__class__.symbol_info["symbol"]
        self.symbol_info = self.__class__.symbol_info
        self.price_precision = self.__class__.price_precision
        self.qty_precision = self.__class__.qty_precision
        self.execution_client = self.__class__.execution_client

    def test_execute_buy_otoco_valid(self):
        """Teste l'exécution d'un ordre OTOCO BUY valide sur Testnet."""
        logger.info("--- Test Exécution OTOCO BUY (Testnet) ---")
        # Utiliser des prix très éloignés pour éviter exécution immédiate
        entry_price = 15000.0
        quantity = 0.001 # Vérifier minQty/minNotional sur Testnet
        stop_loss_price = 14500.0
        take_profit_price = 16000.0

        # 1. Construire la requête
        # Assurez-vous que la version de base.py utilisée ici est celle corrigée (v4 ou v5)
        order_request = self.strategy._build_otoco_order(
            symbol=self.symbol,
            side="BUY",
            entry_price=entry_price,
            quantity=quantity,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price,
            price_precision=self.price_precision,
            qty_precision=self.qty_precision,
            symbol_info=self.symbol_info
        )
        self.assertIsNotNone(order_request, "Échec de la construction de la requête OTOCO BUY")
        logger.info(f"Requête OTOCO BUY construite : {order_request}")

        # 2. Exécuter l'ordre via le client
        logger.info("Envoi de la requête OTOCO BUY à l'API Testnet...")
        execution_response = self.execution_client.execute_order(order_request)
        logger.info(f"Réponse de l'API Testnet : {execution_response}")

        # 3. Vérifier la réponse
        self.assertIsInstance(execution_response, dict)
        self.assertEqual(execution_response.get('status'), 'SUCCESS',
                         f"L'exécution OTOCO BUY a échoué: {execution_response.get('message', 'Erreur inconnue')}")
        self.assertIn('data', execution_response)
        self.assertIsInstance(execution_response['data'], dict)
        self.assertIn('orderListId', execution_response['data'])
        reports_key_found = 'orders' in execution_response['data'] or 'orderReports' in execution_response['data']
        self.assertTrue(reports_key_found, "La réponse devrait contenir 'orders' ou 'orderReports'")
        if 'orders' in execution_response['data']:
             self.assertIsInstance(execution_response['data']['orders'], list)
        if 'orderReports' in execution_response['data']:
             self.assertIsInstance(execution_response['data']['orderReports'], list)


        # --- Annulation Optionnelle ---
        order_list_id = execution_response.get('data', {}).get('orderListId')
        if order_list_id:
            logger.warning(f"Ordre OTOCO {order_list_id} placé sur Testnet. Pensez à l'annuler manuellement ou implémentez l'annulation.")
            # Implémenter l'annulation si nécessaire ici


    # Ajouter un test similaire pour OTOCO SELL si nécessaire


# --- Exécution des Tests ---
if __name__ == '__main__':
    print("Running OTOCO Margin EXECUTION Tests (interacting with Binance Testnet)...")
    print("\nIMPORTANT: Ces tests vont placer des ordres sur votre compte Testnet Binance Margin.")
    print("Assurez-vous d'avoir défini BINANCE_TESTNET_API_KEY/SECRET dans .env ou l'environnement,")
    print("et d'avoir des fonds Testnet et que les prix/quantités sont adaptés.\n")

    # Exécuter les tests en utilisant TestLoader pour éviter la dépréciation
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestOtocoMarginExecution)
    # suite = unittest.TestSuite()
    # suite.addTest(unittest.makeSuite(TestOtocoMarginExecution)) # Ancienne méthode dépréciée
    runner = unittest.TextTestRunner()
    runner.run(suite)
 