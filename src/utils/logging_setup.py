import logging
import logging.handlers
import os
import sys
from typing import Dict, Optional, Any, List, Union, TYPE_CHECKING, Union
from pathlib import Path

if TYPE_CHECKING:
    from src.config.loader import LoggingConfig, LiveLoggingConfig

logger = logging.getLogger(__name__)

_handlers_managed_by_setup_logging: List[logging.Handler] = []

def setup_logging(
    log_config: Union['LoggingConfig', 'LiveLoggingConfig'],
    log_dir: str,
    log_filename: str,
    root_level: int = logging.INFO
) -> None:
    global _handlers_managed_by_setup_logging
    
    try:
        log_level_str = getattr(log_config, 'level', 'INFO')
        log_level = getattr(logging, log_level_str.upper(), logging.INFO)
        log_format_str = getattr(log_config, 'format', '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s')
        log_to_file_flag = getattr(log_config, 'log_to_file', False)

        root_logger = logging.getLogger()
        
        for handler_instance in _handlers_managed_by_setup_logging:
            if handler_instance in root_logger.handlers:
                root_logger.removeHandler(handler_instance)
        _handlers_managed_by_setup_logging.clear()

        root_logger.setLevel(root_level)
        formatter = logging.Formatter(log_format_str)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        _handlers_managed_by_setup_logging.append(console_handler)

        if log_to_file_flag:
            try:
                log_dir_path = Path(log_dir)
                log_dir_path.mkdir(parents=True, exist_ok=True)
                log_file_path = log_dir_path / log_filename

                file_handler = logging.handlers.RotatingFileHandler(
                    log_file_path, maxBytes=10*1024*1024, backupCount=5, encoding='utf-8'
                )
                file_handler.setLevel(log_level)
                file_handler.setFormatter(formatter)
                root_logger.addHandler(file_handler)
                _handlers_managed_by_setup_logging.append(file_handler)
                # Initial log message after setup
                logging.getLogger(__name__).info(f"Logging to console and file: {log_file_path} at level {log_level_str}")
            except OSError as e:
                logging.getLogger(__name__).error(f"Failed to create log directory or file handler for {log_filename} in {log_dir}: {e}")
                logging.getLogger(__name__).info("Logging to console only due to file handler error.")
            except Exception as file_log_e:
                 logging.getLogger(__name__).error(f"Unexpected error setting up file logging: {file_log_e}", exc_info=True)
                 logging.getLogger(__name__).info("Logging to console only due to unexpected file handler error.")
        else:
            logging.getLogger(__name__).info(f"Logging to console only at level {log_level_str}")

    except AttributeError as e:
         logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - FALLBACK_LOGGING - %(message)s')
         logging.getLogger(__name__).error(f"Logging setup failed due to AttributeError: {e}. Falling back to basic console logging.", exc_info=True)
    except Exception as e:
        logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - FALLBACK_LOGGING - %(message)s')
        logging.getLogger(__name__).exception(f"Critical error in logging setup: {e}. Falling back to basic console logging.", exc_info=True)

