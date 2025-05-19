import argparse
import logging
import sys
import time
import threading # Assurez-vous que threading est importé
from pathlib import Path
from typing import Any, Dict, List, Optional, Union # Union ajouté
from datetime import datetime, timezone # datetime et timezone ajoutés

try:
    PROJECT_ROOT = Path(__file__).resolve().parent
except NameError: 
    PROJECT_ROOT = Path(".").resolve()

SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

try:
    from src.config.loader import load_all_configs, AppConfig
    from src.live.manager import LiveTradingManager # Assurez-vous que LiveTradingManager est importé
    from src.utils.logging_setup import setup_logging 
except ImportError as e:
    logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.critical(f"Failed to import critical modules: {e}", exc_info=True)
    sys.exit(1)
except Exception as e:
    logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.critical(f"An unexpected error occurred during imports: {e}", exc_info=True)
    sys.exit(1)

logger = logging.getLogger(__name__)

# Variables globales pour gérer les managers et threads actifs
active_managers: Dict[str, LiveTradingManager] = {}
active_threads: Dict[str, threading.Thread] = {}
global_shutdown_event = threading.Event()


def signal_handler(signum, frame):
    """Gère les signaux d'arrêt (SIGINT, SIGTERM) pour un arrêt propre."""
    logger.info(f"Signal {signum} reçu. Demande d'arrêt global...")
    global_shutdown_event.set()
    # Donner un peu de temps aux managers pour s'arrêter avant de forcer
    # Cela est géré dans la boucle principale de l'orchestrateur maintenant.

def _run_single_trading_session_cycle(app_config: AppConfig, cli_args: argparse.Namespace, orchestrator_shutdown_event: threading.Event):
    """
    Exécute un cycle de session de trading : initialise et démarre les LiveTradingManagers.
    """
    global active_managers, active_threads # S'assurer qu'on utilise les globales
    logger.info(f"--- Orchestrator: Starting new trading session cycle at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} ---")

    try:
        current_project_root = cli_args.root if cli_args.root else str(PROJECT_ROOT)
        app_config = load_all_configs(project_root=current_project_root)
        
        if hasattr(app_config, 'live_config') and hasattr(app_config.live_config, 'live_logging'):
            live_log_conf = app_config.live_config.live_logging
            log_dir_live = app_config.global_config.paths.logs_live
            log_filename_live = getattr(live_log_conf, 'log_filename_live', 'live_trading_run.log')
            live_log_level_str = getattr(live_log_conf, 'level', 'INFO').upper()
            live_log_level_val = getattr(logging, live_log_level_str, logging.INFO)
            setup_logging(live_log_conf, log_dir_live, log_filename_live, root_level=live_log_level_val)
        logger.info("Full configuration (re)loaded and logging reconfigured for new cycle.")
    except Exception as e_conf:
        logger.critical(f"CRITICAL ERROR reloading configuration: {e_conf}", exc_info=True)
        orchestrator_shutdown_event.set()
        return

    # Nettoyer les managers et threads qui ne sont plus actifs (terminés ou en erreur)
    # Cela est important pour permettre le redémarrage ou la gestion des erreurs.
    inactive_manager_ids = [mid for mid, th in active_threads.items() if not th.is_alive()]
    for mid in inactive_manager_ids:
        logger.info(f"Manager thread {mid} is no longer alive. Cleaning up.")
        if mid in active_managers:
            # S'assurer que le manager est bien arrêté (au cas où le thread s'est terminé anormalement)
            try:
                active_managers[mid].stop_trading() 
            except Exception as e_stop:
                logger.error(f"Error ensuring manager {mid} is stopped: {e_stop}")
            del active_managers[mid]
        del active_threads[mid]
    if inactive_manager_ids:
        logger.info(f"Cleaned up {len(inactive_manager_ids)} inactive manager(s).")


    pairs_to_trade_live_config = app_config.live_config.live_fetch.crypto_pairs
    pairs_to_trade_cli = cli_args.pair if isinstance(cli_args.pair, list) else ([cli_args.pair] if cli_args.pair else None)
    
    pairs_to_process = pairs_to_trade_cli if pairs_to_trade_cli else pairs_to_trade_live_config
    if not isinstance(pairs_to_process, list): # S'assurer que c'est une liste
        pairs_to_process = [pairs_to_process]

    if not pairs_to_process:
        logger.warning("No crypto pairs specified either via --pair or in config_live.json. Nothing to trade for this cycle.")
        return

    general_context_label_from_cli = cli_args.tf
    
    logger.info(f"Processing pairs for this cycle: {pairs_to_process}")
    
    managers_started_this_cycle = 0
    for pair_symbol_to_trade in pairs_to_process:
        for deployment_config in app_config.live_config.strategy_deployments:
            if not getattr(deployment_config, 'active', False):
                continue

            strategy_id_from_config = getattr(deployment_config, 'strategy_id', '')
            
            # Déterminer le context_label à utiliser pour ce manager
            current_context_label_for_manager = general_context_label_from_cli
            if not current_context_label_for_manager:
                results_path = Path(getattr(deployment_config, 'results_config_path', ''))
                # Essayer de déduire le contexte du chemin des résultats
                # Le parent du fichier live_config.json est le dossier du contexte
                if results_path.parent.name and results_path.parent.name != pair_symbol_to_trade.upper() and \
                   (not hasattr(deployment_config,'strategy_name_base') or results_path.parent.name != getattr(deployment_config,'strategy_name_base','')):
                    current_context_label_for_manager = results_path.parent.name
                else:
                    current_context_label_for_manager = "default_live_context"
                logger.info(f"No --tf provided. Using context label for manager: '{current_context_label_for_manager}' for deployment ID '{strategy_id_from_config}'")

            # La logique de correspondance de la paire et du contexte est maintenant gérée en partie par LiveTradingManager
            # Ici, on s'assure juste qu'on tente de lancer un manager si la paire correspond
            # et que le manager n'est pas déjà actif pour cette combinaison unique.
            # Le strategy_id dans config_live.json doit être unique par déploiement.
            
            # Construire un ID unique pour le manager basé sur strategy_id et context_label
            # pour permettre plusieurs instances de la même strat/paire avec des contextes différents.
            manager_unique_id = f"{strategy_id_from_config}_ctx_{current_context_label_for_manager.replace(' ','_')}"

            # Vérifier si le déploiement actuel est pour la paire en cours de traitement.
            # Le strategy_id peut avoir différents formats, ex: "stratName_PAIR_timeframe" ou "stratName_PAIR".
            # Une vérification simple est de s'assurer que la paire est dans l'ID.
            if f"_{pair_symbol_to_trade.upper()}_" in strategy_id_from_config or \
               strategy_id_from_config.endswith(f"_{pair_symbol_to_trade.upper()}") or \
               strategy_id_from_config.startswith(f"{pair_symbol_to_trade.upper()}_") or \
               pair_symbol_to_trade.upper() in strategy_id_from_config : # Plus permissif

                if manager_unique_id in active_managers and active_threads[manager_unique_id].is_alive():
                    logger.debug(f"Manager {manager_unique_id} already active and alive. Skipping re-initialization.")
                    managers_started_this_cycle +=1 
                    continue
                elif manager_unique_id in active_managers and not active_threads[manager_unique_id].is_alive():
                    logger.warning(f"Manager thread {manager_unique_id} found inactive. Will attempt to restart.")
                    del active_managers[manager_unique_id]
                    del active_threads[manager_unique_id]


                logger.info(f"--- Initializing manager for deployment ID '{strategy_id_from_config}' (Pair: {pair_symbol_to_trade}, ContextLabel: {current_context_label_for_manager}) ---")
                try:
                    manager = LiveTradingManager(
                        app_config=app_config,
                        pair_to_trade=pair_symbol_to_trade,
                        context_label=current_context_label_for_manager 
                    )
                    
                    thread = threading.Thread(target=manager.run, name=manager_unique_id)
                    active_managers[manager_unique_id] = manager
                    active_threads[manager_unique_id] = thread
                    thread.start()
                    logger.info(f"LiveTradingManager thread started for {manager_unique_id}")
                    managers_started_this_cycle +=1
                except Exception as e_mgr_init:
                    logger.critical(f"CRITICAL FAILURE initializing LiveTradingManager for deployment ID '{strategy_id_from_config}' with context '{current_context_label_for_manager}': {e_mgr_init}", exc_info=True)
            else:
                 logger.debug(f"Skipping deployment ID '{strategy_id_from_config}' as it does not seem to match current pair '{pair_symbol_to_trade}' for direct manager instantiation.")

    if managers_started_this_cycle == 0 and any(getattr(d, 'active', False) for d in app_config.live_config.strategy_deployments):
         logger.error("No managers could be started this session cycle, though active deployments are configured. Check pair/context matching and strategy_id formats in config_live.json.")
    elif managers_started_this_cycle > 0:
        logger.info(f"{managers_started_this_cycle} manager thread(s) are active or were started this cycle.")
    else:
        logger.info("No active deployments found in config or no matching pairs/contexts for CLI args. Orchestrator idle for this cycle.")


def main_loop_orchestrator(args: argparse.Namespace):
    """Boucle principale de l'orchestrateur pour gérer les sessions de trading."""
    global global_shutdown_event, active_managers, active_threads
    
    # Charger la configuration initiale une fois pour obtenir les paramètres globaux comme le cycle_interval
    try:
        current_project_root = args.root if args.root else str(PROJECT_ROOT)
        initial_app_config = load_all_configs(project_root=current_project_root)
        # Configurer le logging global initial (sera potentiellement reconfiguré par cycle)
        setup_logging(initial_app_config.global_config.logging, 
                      initial_app_config.global_config.paths.logs_live, # Log dans le dossier live
                      initial_app_config.global_config.logging.log_filename_global, # Nom de fichier global
                      root_level=getattr(logging, initial_app_config.global_config.logging.level.upper(), logging.INFO))
    except Exception as e_init_conf:
        logger.critical(f"CRITICAL ERROR during initial configuration load: {e_init_conf}", exc_info=True)
        return # Ne peut pas continuer sans configuration initiale

    session_cycle_interval_seconds = getattr(initial_app_config.live_config.global_live_settings, 'session_cycle_interval_seconds', 60)
    if not isinstance(session_cycle_interval_seconds, (int, float)) or session_cycle_interval_seconds <= 0:
        logger.warning(f"Invalid session_cycle_interval_seconds ({session_cycle_interval_seconds}). Defaulting to 60 seconds.")
        session_cycle_interval_seconds = 60


    while not global_shutdown_event.is_set():
        # Nettoyer les listes globales avant de démarrer un nouveau cycle de session
        # Cela est important si _run_single_trading_session_cycle est appelé plusieurs fois
        # et que les managers peuvent s'arrêter/démarrer.
        # Cependant, la logique de nettoyage des threads inactifs est maintenant DANS _run_single_trading_session_cycle.
        # active_managers.clear() # Ne pas clearer ici, le cycle interne gère les threads morts.
        # active_threads.clear()
        # logger.info("Global active_managers and active_threads lists cleared before new session cycle.")

        try:
            # Passer l'objet app_config actuel (qui sera rechargé à l'intérieur)
            # et l'événement d'arrêt de l'orchestrateur.
            _run_single_trading_session_cycle(initial_app_config, args, global_shutdown_event)
        except Exception as e_cycle:
            logger.critical(f"Unhandled error in main_loop_orchestrator's call to _run_single_trading_session_cycle: {e_cycle}", exc_info=True)
            # Envisager un backoff ou une stratégie de redémarrage plus robuste ici si nécessaire.
            # Pour l'instant, on continue la boucle après un délai.
            if global_shutdown_event.wait(timeout=session_cycle_interval_seconds): # Attendre avant de réessayer ou de quitter
                break 
            continue # Essayer un nouveau cycle après le délai


        if global_shutdown_event.is_set():
            logger.info("Orchestrator shutdown event received during cycle. Exiting main loop.")
            break
        
        logger.info(f"Next trading session cycle in {session_cycle_interval_seconds} seconds (unless shutdown is requested)...")
        if global_shutdown_event.wait(timeout=session_cycle_interval_seconds):
            logger.info("Orchestrator shutdown event received during wait. Exiting main loop.")
            break # Sortir de la boucle while si l'événement est signalé pendant l'attente

    logger.info("--- Orchestrator: Main loop finished. Proceeding to shutdown active managers. ---")
    # Arrêter tous les managers actifs
    active_manager_ids_at_shutdown = list(active_managers.keys()) # Copier les clés car le dict peut changer
    for manager_id in active_manager_ids_at_shutdown:
        manager = active_managers.get(manager_id)
        thread = active_threads.get(manager_id)
        if manager:
            logger.info(f"Requesting stop for manager: {manager_id}")
            try:
                manager.stop_trading() # Signale l'arrêt au manager
            except Exception as e_stop_mgr:
                logger.error(f"Error signaling stop to manager {manager_id}: {e_stop_mgr}")
        if thread and thread.is_alive():
            logger.info(f"Waiting for manager thread {manager_id} to join...")
            thread.join(timeout=30) # Attendre avec un timeout
            if thread.is_alive():
                logger.warning(f"Manager thread {manager_id} did not join in time.")
        if manager_id in active_managers: del active_managers[manager_id]
        if manager_id in active_threads: del active_threads[manager_id]

    logger.info("--- All active managers requested to stop. Orchestrator shutdown complete. ---")


def main():
    """Fonction principale pour lancer l'orchestrateur de trading en direct."""
    run_start_time = time.time()
    # Le logging initial sera configuré par main_loop_orchestrator après chargement du config global

    parser = argparse.ArgumentParser(description="Run Live Trading Bot Orchestrator.")
    parser.add_argument(
        "--pair",
        type=str,
        action='append', # Permet de spécifier --pair plusieurs fois
        help="Specific crypto pair(s) to trade (e.g., AIXBTUSDC). If not provided, uses pairs from config_live.json."
    )
    parser.add_argument(
        "--tf", "--context", dest="tf", # Accepter --tf ou --context
        type=str,
        help="Context label for this run (e.g., 'MyOptimisationContext', 'DefaultLive'). Used for selecting specific parameter sets and naming log/state files."
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Specify the project root directory if the script is not run from the project root.",
    )
    args = parser.parse_args()

    # Configurer les gestionnaires de signaux globaux
    try:
        import signal
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        logger.info("Global signal handlers configured.")
    except ImportError: # signal n'est pas dispo sur toutes les plateformes (ex: Windows pour SIGTERM parfois)
        logger.warning("Module 'signal' non disponible. Les signaux SIGINT/SIGTERM pourraient ne pas être gérés gracieusement sur cette plateforme.")
    except Exception as e_signal:
        logger.error(f"Erreur lors de la configuration des gestionnaires de signaux : {e_signal}")


    logger.info("--- Starting Script run_live.py ---")
    
    try:
        main_loop_orchestrator(args)
    except Exception as e_main:
        logger.critical(f"CRITICAL Unhandled error in run_live.py main execution: {e_main}", exc_info=True)
    finally:
        run_end_time = time.time()
        total_duration_seconds = run_end_time - run_start_time
        logger.info(f"--- Script run_live.py Finished --- Total Execution Time: {total_duration_seconds:.2f} seconds")
        logging.shutdown() # S'assurer que tous les logs sont écrits

if __name__ == "__main__":
    main()
