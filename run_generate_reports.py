# Fichier: run_generate_reports.py

import argparse
import logging
import sys
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Type, TYPE_CHECKING, List

# --- Ajouter la racine du projet au PYTHONPATH ---
# Cela permet aux imports comme 'from src.reporting...' de fonctionner
# lorsque le script est lancé depuis la racine du projet.
project_root = Path(__file__).resolve().parent
src_path = project_root / "src"
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))
    print(f"Ajouté {src_path} au PYTHONPATH")

# --- Configuration initiale du Logging ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s:%(lineno)d - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# --- Importation de la fonction de génération ---
try:
    # Maintenant que src est dans le path, cet import devrait fonctionner
    from reporting.generator import generate_all_reports
    logger.info("Module 'reporting.generator' importé avec succès.")
except ImportError:
    logger.error("ERREUR : Impossible d'importer 'generate_all_reports' depuis 'src.reporting.generator'.")
    logger.error("Vérifiez que le fichier 'src/reporting/generator.py' existe et contient la fonction 'generate_all_reports'.")
    logger.error(f"Racine du projet détectée : {project_root}")
    logger.error(f"Chemin 'src' ajouté au sys.path : {src_path}")
    sys.exit(1)
except Exception as e:
     logger.error(f"Une erreur inattendue s'est produite lors de l'importation : {e}", exc_info=True)
     sys.exit(1)

# --- Logique Principale ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Génère les rapports (Markdown, Live Config JSON) à partir des artefacts d'un run WFO spécifique."
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        required=True, # Rend l'argument obligatoire
        help="Le nom du répertoire timestamp du run WFO à traiter (ex: '20250504_071117')."
    )
    # Optionnel : Ajouter des arguments pour spécifier les chemins de base si différents
    # parser.add_argument("--log-base", default="logs/backtest_optimization", help="Répertoire de base des logs WFO.")
    # parser.add_argument("--results-base", default="results", help="Répertoire de base pour sauvegarder les résultats générés.")

    args = parser.parse_args()

    # Définir les répertoires de base (relatifs à la racine du projet)
    log_base_dir = project_root / "logs" / "backtest_optimization"
    results_base_dir = project_root / "results"

    # Construire les chemins spécifiques au run
    run_log_dir = log_base_dir / args.timestamp
    run_results_dir = results_base_dir / args.timestamp

    logger.info(f"--- Lancement de la génération de rapports pour le run : {args.timestamp} ---")
    logger.info(f"Lecture des logs depuis : {run_log_dir.resolve()}")
    logger.info(f"Écriture des résultats dans : {run_results_dir.resolve()}")

    # Vérifier l'existence du répertoire de logs source
    if not run_log_dir.is_dir():
        logger.error(f"Le répertoire de logs spécifié n'existe pas : {run_log_dir}")
        logger.error("Vérifiez que le timestamp est correct et que le run WFO a bien généré des logs.")
        sys.exit(1)

    try:
        # Appeler la fonction principale de génération de rapports
        generate_all_reports(log_dir=run_log_dir, results_dir=run_results_dir)
        logger.info("Génération des rapports terminée avec succès.")

    except FileNotFoundError as fnf_err:
         logger.error(f"Erreur : Fichier non trouvé pendant la génération. {fnf_err}", exc_info=True)
         sys.exit(1)
    except Exception as e:
        logger.error(f"Une erreur s'est produite lors de la génération des rapports : {e}", exc_info=True)
        sys.exit(1)

    sys.exit(0) # Sortie normale
