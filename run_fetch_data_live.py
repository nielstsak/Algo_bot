# -*- coding: utf-8 -*-
"""
Main script to run the live data fetching and preprocessing loop.
1. Initializes data files and fetches initial history if needed (calls acquisition_live).
2. Starts WebSocket listeners in the background (calls acquisition_live).
3. Periodically reads raw data, preprocesses it (calls preprocessing_live), and saves processed data.
4. Sleeps based on the configured interval.
"""

import json
import logging
import os
import signal
import sys
import threading
import time
from pathlib import Path

import pandas as pd

# --- Add project root to sys.path ---
# This allows importing modules from src.*
# *** CORRECTED project_root definition ***
# Assumes this script is run from the project's root directory
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# --- Import project modules ---
try:
    from src.data import acquisition_live
    from src.data import preprocessing_live
    # Make sure logging_setup is imported correctly
    # The setup_logging function expects a config object, but we'll pass the dict for now
    # It might be better to import the config loader if available
    from src.utils.logging_setup import setup_logging
    # from src.config.loader import load_live_config # Example if you have a specific loader
except ImportError as e:
    print(f"Error importing project modules: {e}", file=sys.stderr)
    print(f"Project root calculated as: {project_root}", file=sys.stderr)
    print("Ensure the script is run from the project root or the PYTHONPATH is set correctly.", file=sys.stderr)
    sys.exit(1)

# --- Global Variables ---
logger = logging.getLogger(__name__)
shutdown_flag = threading.Event()
ws_clients = [] # To keep track of websocket clients for shutdown

# Define base columns expected (consistent with acquisition_live)
BASE_COLUMNS = ['timestamp', 'open', 'high', 'low', 'close', 'volume']

# --- Configuration Loading ---
def load_config(config_path: str = "config/config_live.json") -> dict:
    """Loads the live configuration file."""
    global project_root # Ensure we use the globally defined project_root
    try:
        # config_path is relative to project_root
        config_full_path = project_root / config_path
        logger.debug(f"Attempting to load configuration from: {config_full_path}")
        with open(config_full_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Configuration loaded from {config_full_path}")
        return config
    except FileNotFoundError:
        logger.critical(f"CRITICAL: Configuration file not found at {config_full_path}")
        raise
    except json.JSONDecodeError:
        logger.critical(f"CRITICAL: Error decoding JSON from {config_full_path}")
        raise
    except Exception as e:
        logger.critical(f"CRITICAL: Failed to load config from {config_full_path}: {e}")
        raise

# --- Interval Conversion ---
def parse_interval_to_seconds(interval_str: str) -> int:
    """Converts Binance interval string (e.g., '1m', '5m', '1h', '1d') to seconds."""
    unit = interval_str[-1].lower()
    try:
        value = int(interval_str[:-1])
    except ValueError:
        logger.error(f"Invalid interval format: {interval_str}. Cannot parse value.")
        return 60 # Default to 1 minute on error

    if unit == 'm':
        return value * 60
    elif unit == 'h':
        return value * 60 * 60
    elif unit == 'd':
        return value * 60 * 60 * 24
    elif unit == 's': # Support seconds?
        return value
    else:
        logger.error(f"Unsupported interval unit: {unit} in {interval_str}. Defaulting to 60s.")
        return 60

# --- Signal Handling for Graceful Shutdown ---
def signal_handler(signum, frame):
    """Sets the shutdown flag when SIGINT (Ctrl+C) or SIGTERM is received."""
    logger.warning(f"Received signal {signum}. Initiating graceful shutdown...")
    shutdown_flag.set()
    # Stop websocket clients
    logger.info("Stopping WebSocket clients...")
    active_clients = [c for c in ws_clients if c.thread and c.thread.is_alive()]
    if not active_clients:
        logger.info("No active WebSocket clients to stop.")
        return

    for client in active_clients:
        try:
            logger.debug(f"Sending stop signal to client: {client.symbol} {client.interval}")
            client.stop()
        except Exception as e:
            logger.error(f"Error signalling stop for client {client.symbol} {client.interval}: {e}")

    # Optional: Wait for threads to join (add timeout)
    # logger.info("Waiting for WebSocket threads to join...")
    # for client in active_clients:
    #     try:
    #         if client.thread:
    #              client.thread.join(timeout=5) # 5 second timeout per thread
    #              if client.thread.is_alive():
    #                   logger.warning(f"WebSocket thread for {client.symbol} {client.interval} did not exit cleanly.")
    #     except Exception as e:
    #          logger.error(f"Error joining thread for {client.symbol} {client.interval}: {e}")

    logger.info("All WebSocket stop signals sent.")


# --- Main Loop ---
def main():
    """Main execution function."""
    global ws_clients, project_root # Allow modifying the global list, ensure project_root is accessible

    # Initial basic config in case setup_logging fails
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info(f"Project root set to: {project_root}") # Log the calculated root


    # --- Configuration should be loaded first ---
    try:
        config = load_config("config/config_live.json")
    except Exception as e:
        # Use basicConfig logger if setup hasn't happened
        logging.critical(f"CRITICAL: Failed to load configuration: {e}. Exiting.", exc_info=True)
        sys.exit(1)

    # --- Setup Logging ---
    log_config_dict = config.get('live_logging', {}) # Get logging section
    log_dir = project_root / "logs" / "live_trading" # Use corrected project_root
    log_dir.mkdir(parents=True, exist_ok=True)
    # Use filename from config if available, otherwise default
    log_filename = log_config_dict.get('log_filename_live', 'run_fetch_data_live.log')
    log_file = log_dir / log_filename

    try:
        # Pass the dictionary from config, the calculated log directory, and log filename
        setup_logging(
            log_config=log_config_dict,
            log_dir=str(log_dir),
            log_filename=log_filename
        )
        logger.info("--- Starting run_fetch_data_live.py ---") # Now uses configured logger
    except Exception as log_setup_err:
         logger.error(f"Error setting up custom logging: {log_setup_err}. Continuing with potentially basic logging.", exc_info=True)


    # Register signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


    # 2. Parse relevant config sections
    try:
        live_fetch_config = config.get('live_fetch', {})
        pairs = live_fetch_config.get('crypto_pairs', [])
        intervals = live_fetch_config.get('intervals', []) # Should contain only one for this script's logic
        if not pairs or not intervals:
             logger.critical("CRITICAL: 'crypto_pairs' or 'intervals' not found in config['live_fetch']. Exiting.")
             sys.exit(1)
        if len(intervals) > 1:
             logger.warning(f"Multiple intervals found in config: {intervals}. Using the first one: '{intervals[0]}' for main loop timing.")
        interval_str = intervals[0] # Assume only one interval is relevant for the sleep cycle
        interval_seconds = parse_interval_to_seconds(interval_str)
        logger.info(f"Using interval: {interval_str} ({interval_seconds} seconds) for processing cycle.")

    except Exception as e:
        logger.critical(f"CRITICAL: Failed to parse configuration sections: {e}. Exiting.", exc_info=True)
        sys.exit(1)


    # Define base data directories
    base_data_dir = project_root / "data" / "live" # Use corrected project_root
    raw_dir = base_data_dir / "raw"
    processed_dir = base_data_dir / "processed"


    # 3. Run Initial Data Acquisition Setup
    logger.info("Step 1: Running data initialization...")
    try:
        # Ensure acquisition_live also uses the correct project root if needed
        # (It calculates its own base_dir currently, which should be fine if run from root)
        acquisition_live.run_initialization(config, BASE_COLUMNS)
        logger.info("Data initialization finished.")
    except Exception as e:
        logger.critical(f"CRITICAL: Error during data initialization: {e}. Exiting.", exc_info=True)
        sys.exit(1)


    # 4. Start WebSocket Listeners
    logger.info("Step 2: Starting WebSocket listeners...")
    try:
        # Pass the global ws_clients list to be populated
        ws_clients = acquisition_live.start_websockets(config, BASE_COLUMNS)
        if not ws_clients:
             logger.warning("No WebSocket clients were started. Check configuration and initialization logs.")
             # Decide if script should exit or continue without live updates
             # sys.exit(1) # Uncomment to exit if websockets are crucial

        logger.info("WebSocket listeners started in background threads.")
    except Exception as e:
        logger.critical(f"CRITICAL: Error starting websockets: {e}. Exiting.", exc_info=True)
        sys.exit(1)


    # 5. Main Processing Loop
    logger.info("Step 3: Entering main processing loop...")
    while not shutdown_flag.is_set():
        loop_start_time = time.monotonic()
        logger.info("--- Starting new processing cycle ---")

        # Process each pair defined in the config for the primary interval
        for pair in pairs:
            if shutdown_flag.is_set(): break # Check shutdown flag frequently

            logger.debug(f"Processing pair: {pair} with interval: {interval_str}")
            raw_path = raw_dir / f"{pair}_{interval_str}.csv"
            processed_path = processed_dir / f"{pair}_{interval_str}.csv"

            try:
                # Read Raw Data
                if not raw_path.exists():
                    logger.warning(f"Raw file not found for {pair} {interval_str}: {raw_path}. Skipping processing.")
                    continue

                try:
                    # Specify low_memory=False for potentially mixed types or large files
                    df_raw = pd.read_csv(raw_path, low_memory=False)
                except pd.errors.EmptyDataError:
                    logger.warning(f"Raw file is empty for {pair} {interval_str}: {raw_path}. Skipping processing.")
                    continue
                except Exception as read_err:
                     logger.error(f"Error reading raw CSV {raw_path}: {read_err}. Skipping processing.")
                     continue


                if df_raw.empty:
                     logger.warning(f"Raw DataFrame is empty after loading for {pair} {interval_str}. Skipping processing.")
                     continue

                # Ensure base columns exist before preprocessing
                # Use a consistent check method (lowercase)
                current_columns = set(c.lower() for c in df_raw.columns)
                required_columns = set(BASE_COLUMNS)
                if not required_columns.issubset(current_columns):
                     missing = required_columns - current_columns
                     logger.error(f"Raw DataFrame for {pair} {interval_str} missing required columns: {missing}. Raw columns: {df_raw.columns}. Skipping preprocessing.")
                     continue


                # Call Preprocessing
                logger.debug(f"Calling preprocessing for {pair} {interval_str} (rows: {len(df_raw)})")
                # Pass a copy to avoid modifying the raw df if preprocessing does it inplace
                df_processed = preprocessing_live.preprocess_live_data(df_raw.copy())

                if df_processed is None or df_processed.empty:
                    logger.warning(f"Preprocessing returned empty or None for {pair} {interval_str}. Not saving processed file.")
                    continue

                # Save Processed Data
                try:
                    df_processed.to_csv(processed_path, index=False)
                    logger.debug(f"Saved processed data for {pair} {interval_str} to {processed_path} (shape: {df_processed.shape})")
                except Exception as save_err:
                     logger.error(f"Error saving processed CSV {processed_path}: {save_err}")


            except Exception as e:
                logger.error(f"Error processing {pair} {interval_str}: {e}", exc_info=True)
                # Continue to next pair even if one fails

        if shutdown_flag.is_set(): break # Check after loop finishes

        # Calculate Sleep Time
        loop_end_time = time.monotonic()
        elapsed_time = loop_end_time - loop_start_time
        sleep_time = interval_seconds - elapsed_time
        logger.info(f"Processing cycle took {elapsed_time:.2f} seconds.")

        if sleep_time > 0:
            logger.info(f"Sleeping for {sleep_time:.2f} seconds...")
            # Use wait with timeout for faster shutdown response
            interrupted = shutdown_flag.wait(timeout=sleep_time)
            if interrupted:
                 logger.info("Sleep interrupted by shutdown signal.")
                 break # Exit loop immediately if shutdown during sleep
        else:
            logger.warning(f"Processing time ({elapsed_time:.2f}s) exceeded interval ({interval_seconds}s). Running next cycle immediately.")
            # Small sleep to prevent tight loop on consistent overruns and allow signal check
            time.sleep(0.1)


    logger.info("--- Shutdown signal received or loop exited ---")
    # Final cleanup (websockets are stopped by signal handler)
    logger.info("Waiting briefly for threads to potentially finish...")
    time.sleep(2) # Give threads a moment
    # Attempt to join threads again as a final measure
    final_active_clients = [c for c in ws_clients if c.thread and c.thread.is_alive()]
    if final_active_clients:
        logger.info(f"Attempting final join for {len(final_active_clients)} potentially active threads...")
        for client in final_active_clients:
            try:
                 if client.thread:
                      client.thread.join(timeout=1)
            except Exception:
                 pass # Ignore errors during final cleanup join

    logger.info("--- run_fetch_data_live.py finished ---")
    # Use os._exit for a faster exit if threads are stuck, otherwise use sys.exit
    # os._exit(0)
    sys.exit(0) # Ensure clean exit


if __name__ == "__main__":
    main()