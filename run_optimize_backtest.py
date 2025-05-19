import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    PROJECT_ROOT = Path(__file__).resolve().parent
except NameError: 
    PROJECT_ROOT = Path(".").resolve()

SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# Initialisation basique du logger pour les erreurs d'importation précoces
# La configuration complète du logging sera faite dans main() après le chargement de la config.
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Logger spécifique pour ce module

try:
    from src.config.loader import load_all_configs, AppConfig
    from src.backtesting.wfo import WalkForwardOptimizer
    from src.utils.logging_setup import setup_logging 
except ImportError as e:
    logger.critical(f"ÉCHEC CRITIQUE: Impossible d'importer les modules nécessaires: {e}. Vérifiez PYTHONPATH et les installations.", exc_info=True)
    sys.exit(1)
except Exception as e:
    logger.critical(f"ÉCHEC CRITIQUE: Erreur inattendue lors des imports initiaux: {e}", exc_info=True)
    sys.exit(1)


def main():
    run_start_time = time.time()
    # Le logger est déjà initialisé au niveau du module, mais on va reconfigurer avec la config chargée.
    
    parser = argparse.ArgumentParser(description="Exécute l'Optimisation Walk-Forward pour les paires de cryptomonnaies.")
    parser.add_argument(
        '--pair', 
        type=str, 
        required=True, 
        help="Paire de cryptomonnaies à traiter (ex: AIXBTUSDC)"
    )
    parser.add_argument(
        '--tf', # 'tf' est souvent utilisé pour 'timeframe' ou 'test_focus', ici c'est un label de contexte
        type=str, 
        required=True, 
        help="Label de contexte pour cette exécution d'optimisation (ex: 5m_context, enriched_data_test, optimisation_ema_macd)."
    )
    parser.add_argument(
        '--config-global', 
        type=str, 
        default='config/config_global.json', 
        help="Chemin vers le fichier de configuration globale, relatif à la racine du projet."
    )
    parser.add_argument(
        '--config-strategies', 
        type=str, 
        default='config/config_strategies.json', 
        help="Chemin vers le fichier de configuration des stratégies, relatif à la racine du projet."
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Spécifier le répertoire racine du projet si le script n'est pas exécuté depuis la racine du projet.",
    )
    args = parser.parse_args()

    project_root_arg = args.root if args.root else str(PROJECT_ROOT)

    # Configurer le logging dès que possible avec la configuration chargée
    # Note: load_all_configs configure déjà le logging basé sur config_global.json.
    # On s'assure ici que c'est fait avant d'autres logs.
    app_config: Optional[AppConfig] = None
    try:
        app_config = load_all_configs(project_root=project_root_arg)
        # setup_logging est appelé dans load_all_configs, donc pas besoin de le rappeler ici
        # sauf si on veut surcharger spécifiquement pour ce script.
        logger.info("--- Démarrage du Script d'Optimisation Walk-Forward ---")
        logger.info(f"Arguments du script: Paire={args.pair}, Contexte (tf)='{args.tf}', Racine Projet='{project_root_arg}'")
        logger.info(f"Chemin config globale: '{args.config_global}', Chemin config stratégies: '{args.config_strategies}'")
        logger.info("Configuration de l'application chargée avec succès.")

    except FileNotFoundError as e:
         logger.critical(f"Fichier de configuration non trouvé: {e}. Vérifiez les chemins relatifs à la racine du projet: {project_root_arg}. Abandon.")
         sys.exit(1)
    except Exception as e_conf:
        logger.critical(f"Erreur critique lors du chargement de la configuration: {e_conf}", exc_info=True)
        sys.exit(1)

    if not app_config: # Vérification supplémentaire
        logger.critical("AppConfig n'a pas pu être chargée. Abandon.")
        sys.exit(1)

    try:
        logger.info(f"Initialisation de WalkForwardOptimizer pour la paire {args.pair} et le contexte '{args.tf}'...")
        wfo_runner = WalkForwardOptimizer(app_config=app_config)
        logger.info("WalkForwardOptimizer initialisé.")
        
        logger.info(f"Démarrage de wfo_runner.run pour la paire: {args.pair}, contexte: '{args.tf}'...")
        wfo_results = wfo_runner.run(pairs=[args.pair], context_labels=[args.tf])
        logger.info(f"wfo_runner.run terminé pour la paire: {args.pair}, contexte: '{args.tf}'.")
        
        if wfo_results:
            logger.info(f"Optimisation WFO terminée. Résultats obtenus pour {len(wfo_results)} configuration(s) de WFO.")
            # Ici, on pourrait logger un résumé des résultats si wfo_results contient des informations utiles.
            # Par exemple, si wfo_results est un dictionnaire de paramètres sélectionnés :
            # for key, params in wfo_results.items():
            #     logger.info(f"  - Résultat pour '{key}': {params if params else 'Aucun paramètre sélectionné'}")
        else:
            logger.warning("Optimisation WFO terminée, mais aucun résultat n'a été retourné par wfo_runner.run.")

    except FileNotFoundError as e: # Devrait être attrapé plus tôt, mais par sécurité
         logger.critical(f"Un fichier nécessaire n'a pas été trouvé pendant l'exécution de WFO: {e}. Abandon.", exc_info=True)
    except ImportError as e: # Erreurs d'importation dans les modules appelés
         logger.critical(f"Erreur d'importation de module pendant l'exécution de WFO: {e}. Abandon.", exc_info=True)
    except Exception as e:
        logger.critical(f"Une erreur non interceptée s'est produite pendant l'exécution de WFO: {e}", exc_info=True)
    finally:
        run_end_time = time.time()
        total_duration_seconds = run_end_time - run_start_time
        logger.info(f"Temps d'Exécution Total du Script: {total_duration_seconds:.2f} secondes")
        logger.info("--- Fin du Script d'Optimisation Walk-Forward ---")
        logging.shutdown() # S'assurer que tous les logs sont écrits

if __name__ == "__main__":
    # Note: Le logger global est configuré dans load_all_configs.
    # Si ce script est le point d'entrée, load_all_configs s'en chargera.
    # Si des logs sont nécessaires AVANT load_all_configs, une config basique est mise en place au début du module.
    main()
