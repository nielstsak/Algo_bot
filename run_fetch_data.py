import logging
import os
import sys
import time
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import numpy as np

try:
    PROJECT_ROOT = str(Path(__file__).resolve().parent)
except NameError:
    PROJECT_ROOT = str(Path('.').resolve())

SRC_DIR = os.path.join(PROJECT_ROOT, 'src')
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

try:
    from src.config.loader import load_all_configs, AppConfig
    from src.data.acquisition import fetch_all_historical_data
    from src.data import data_utils 
except ImportError as e:
    logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.critical(f"ERROR: Failed to import necessary modules: {e}")
    logging.critical(f"Attempted import from SRC_DIR: {SRC_DIR}")
    logging.critical("Ensure the script is run from the project root directory and 'src' contains the modules.")
    sys.exit(1)

logger = logging.getLogger(__name__)

def main():
    script_start_time = time.time()
    logging.info("--- Starting Data Fetch, Clean, and Enrichment Script ---")

    parser = ArgumentParser(description="Fetch, clean, and enrich historical market data.")
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Specify the project root directory if not running from the root.",
    )
    args = parser.parse_args()

    project_root_arg = args.root if args.root else PROJECT_ROOT
    
    try:
        config: AppConfig = load_all_configs(project_root=project_root_arg)
        logger.info(f"Using project root: {project_root_arg}")
        logger.info("Application configuration loaded successfully.")

        logger.info("Starting historical 1-minute data fetching and cleaning...")
        try:
            raw_data_output_dir_path_str = fetch_all_historical_data(config)
            
            if not raw_data_output_dir_path_str or not os.path.isdir(raw_data_output_dir_path_str):
                logger.error("Historical data fetching and cleaning did not return a valid raw data directory path. Aborting enrichment.")
                return
            
            cleaned_data_path_str = config.global_config.paths.data_historical_processed_cleaned
            logger.info(f"Historical 1-minute data fetching and cleaning complete.")
            logger.info(f"Raw data saved in subdirectories of: {raw_data_output_dir_path_str}")
            logger.info(f"Cleaned 1-minute data available in: {cleaned_data_path_str}")

        except Exception as fetch_exc:
            logger.exception("An error occurred during historical data fetching and cleaning.", exc_info=True)
            logger.error("Aborting script due to fetching/cleaning error.")
            return

        logger.info("Starting data enrichment process (aggregation and ATR calculation)...")
        
        cleaned_1min_data_dir = Path(cleaned_data_path_str)
        enriched_data_dir = cleaned_1min_data_dir.parent / "enriched"
        enriched_data_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Enriched data will be saved to: {enriched_data_dir}")

        pairs = config.data_config.assets_and_timeframes.pairs
        timeframes_to_aggregate_minutes = [3, 5, 15, 30, 60] 

        for pair_symbol in pairs:
            logger.info(f"Processing enrichment for {pair_symbol}...")
            cleaned_1min_file = cleaned_1min_data_dir / f"{pair_symbol}_1min_cleaned_with_taker.parquet"

            if not cleaned_1min_file.exists():
                logger.warning(f"Cleaned 1-minute data for {pair_symbol} not found at {cleaned_1min_file}. Skipping.")
                continue

            try:
                df_1min = pd.read_parquet(cleaned_1min_file)
                if 'timestamp' not in df_1min.columns:
                    logger.error(f"'timestamp' column missing in {cleaned_1min_file}. Skipping {pair_symbol}.")
                    continue
                
                df_1min['timestamp'] = pd.to_datetime(df_1min['timestamp'])
                df_1min = df_1min.set_index('timestamp', drop=False) 
                df_1min = df_1min.sort_index()
                if not df_1min.index.is_unique:
                    df_1min = df_1min[~df_1min.index.duplicated(keep='first')]

            except Exception as e:
                logger.error(f"Error loading or pre-processing {cleaned_1min_file} for {pair_symbol}: {e}", exc_info=True)
                continue
            
            if df_1min.empty:
                logger.warning(f"DataFrame for {pair_symbol} is empty after loading. Skipping.")
                continue

            final_df = df_1min.copy()

            for tf_minutes in timeframes_to_aggregate_minutes:
                logger.info(f"Aggregating {pair_symbol} to {tf_minutes}-minute K-lines...")
                
                df_1min_for_agg = df_1min.set_index(pd.DatetimeIndex(df_1min['timestamp']))
                
                df_aggregated = data_utils.aggregate_klines_to_dataframe(df_1min_for_agg, tf_minutes)

                if df_aggregated.empty:
                    logger.warning(f"Aggregation to {tf_minutes}-min for {pair_symbol} resulted in empty DataFrame. Skipping ATRs for this timeframe.")
                    continue
                
                logger.info(f"Calculating ATRs for {tf_minutes}-minute K-lines of {pair_symbol}...")
                df_aggregated_with_atr = data_utils.calculate_atr_for_dataframe(
                    df_aggregated, 
                    atr_low=10, 
                    atr_high=21, 
                    atr_step=1
                )

                new_column_names = {}
                for col in df_aggregated_with_atr.columns:
                    if col in ['open', 'high', 'low', 'close', 'volume']:
                        new_column_names[col] = f"Klines_{tf_minutes}min_{col}"
                    elif col.startswith('ATR_'):
                        new_column_names[col] = f"Klines_{tf_minutes}min_{col}"
                    else:
                        new_column_names[col] = f"Klines_{tf_minutes}min_{col}" 
                
                df_aggregated_with_atr = df_aggregated_with_atr.rename(columns=new_column_names)
                
                if not df_aggregated_with_atr.empty:
                    df_reindexed = df_aggregated_with_atr.reindex(final_df.index, method='ffill')
                    for col in df_reindexed.columns:
                        if col not in final_df.columns:
                            final_df[col] = df_reindexed[col]
                        else: # Handle potential column name conflicts if any other than originals
                            final_df[col] = df_reindexed[col] # Overwrite for simplicity, or use unique names

            output_file = enriched_data_dir / f"{pair_symbol}_enriched.parquet"
            try:
                final_df_to_save = final_df.reset_index(drop=True)
                final_df_to_save.to_parquet(output_file, index=False)
                logger.info(f"Saved enriched data for {pair_symbol} to {output_file}")
            except Exception as e_save:
                logger.error(f"Error saving enriched data for {pair_symbol} to {output_file}: {e_save}", exc_info=True)

    except FileNotFoundError as e:
        logger.error(f"Configuration file not found: {e}. Ensure config files exist relative to project root.")
    except ValueError as e:
        logger.error(f"Configuration or prerequisite error: {e}")
    except Exception as e:
        logger.exception("An unexpected critical error occurred in the main script.", exc_info=True)

    finally:
        script_end_time = time.time()
        logger.info(f"--- Data Fetch, Clean, and Enrichment Script Finished ---")
        logger.info(f"Total execution time: {script_end_time - script_start_time:.2f} seconds")

if __name__ == "__main__":
    main()
