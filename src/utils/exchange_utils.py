# Fichier: src/utils/exchange_utils.py
"""
Utilitaires pour gérer les filtres et précisions des exchanges (ex: Binance).
"""

import logging
import math
import numpy as np
from typing import Dict, Optional, Any, Union

# Configure logging pour ce module
logger = logging.getLogger(__name__)

def adjust_precision(value: Optional[float], precision: Optional[int], rounding_method=math.floor) -> Optional[float]:
    """
    Ajuste une valeur flottante à un nombre spécifié de décimales.

    Args:
        value (Optional[float]): La valeur à ajuster.
        precision (Optional[int]): Le nombre de décimales souhaité. Si None, retourne la valeur originale.
        rounding_method (function): La méthode d'arrondi (math.floor, math.ceil, round).

    Returns:
        Optional[float]: La valeur ajustée, ou None si l'entrée est None.
    """
    if value is None or np.isnan(value):
        return value # Retourner None ou NaN tel quel
    if precision is None or not isinstance(precision, int) or precision < 0:
        logger.error(f"Précision invalide ou manquante fournie: {precision}. Retour de la valeur originale.")
        return value
    if precision == 0:
         # Gérer le cas de précision 0 explicitement
         return float(rounding_method(value))

    factor = 10**precision
    # Utiliser une petite tolérance pour éviter les problèmes d'arrondi flottant avant l'arrondi final
    epsilon = 1e-9
    try:
        if rounding_method == math.floor:
            # Ajouter epsilon avant floor pour gérer les cas limites (ex: 2.9999999999999996 doit devenir 3 si precision=0)
            return math.floor(value * factor + epsilon) / factor
        elif rounding_method == math.ceil:
             # Soustraire epsilon avant ceil
            return math.ceil(value * factor - epsilon) / factor
        else: # round (arrondi standard)
            # La fonction round native de Python gère l'arrondi à la décimale près
            return round(value, precision)
    except OverflowError:
         logger.error(f"OverflowError lors de l'ajustement de {value} avec précision {precision}. Retour de la valeur originale.")
         return value


def get_precision_from_filter(symbol_info: dict, filter_type: str, filter_key: str) -> Optional[int]:
    """
    Extrait la précision (nombre de décimales) des filtres de symbole (ex: Binance).
    Gère les formats numériques et les notations comme '1e-5'.

    Args:
        symbol_info (dict): Dictionnaire contenant les informations du symbole (ex: de client.get_symbol_info).
        filter_type (str): Le type de filtre (ex: 'LOT_SIZE', 'PRICE_FILTER').
        filter_key (str): La clé dans le filtre contenant la valeur de précision (ex: 'stepSize', 'tickSize').

    Returns:
        Optional[int]: Le nombre de décimales, ou None si non trouvé ou invalide.
    """
    try:
        if not isinstance(symbol_info, dict): return None
        filters = symbol_info.get('filters', [])
        if not isinstance(filters, list): return None

        target_filter = next((f for f in filters if isinstance(f, dict) and f.get('filterType') == filter_type), None)
        if target_filter:
            size_str = target_filter.get(filter_key)
            if isinstance(size_str, str) and size_str:
                try:
                    size_float = float(size_str)
                    if size_float > 0:
                        # Gérer les nombres comme 1.0, 10.0 etc. -> précision 0
                        if size_float >= 1 and '.' not in size_str.rstrip('0'):
                            return 0
                        # Gérer les nombres comme 0.1, 0.01, 0.0001 etc.
                        if size_float < 1:
                             # Compter les décimales après avoir retiré les zéros de fin inutiles
                             if '.' in size_str:
                                  # Retirer les zéros de fin avant de compter
                                  decimal_part_stripped = size_str.split('.')[-1].rstrip('0')
                                  # Trouver la position du '1' s'il n'y a que des zéros après
                                  non_zero_index = next((i for i, char in enumerate(decimal_part_stripped) if char != '0'), -1)
                                  if non_zero_index != -1 and decimal_part_stripped[non_zero_index] == '1' and all(c == '0' for c in decimal_part_stripped[non_zero_index+1:]):
                                       # Cas comme 0.00100 -> 0.001 -> précision 3
                                       return non_zero_index + 1
                                  else:
                                       # Cas comme 0.01500 -> 0.015 -> précision 3
                                       # Retourne la longueur de la partie décimale après suppression des zéros finaux
                                       return len(decimal_part_stripped) if decimal_part_stripped else 0
                             else: # Devrait avoir un '.' si < 1, mais sécurité
                                  return 0
                        # Gérer les nombres comme 1.5, 0.5 etc.
                        else: # size_float >= 1 and '.' in size_str
                             # Retirer les zéros de fin avant de compter
                             decimal_part_stripped = size_str.split('.')[-1].rstrip('0')
                             return len(decimal_part_stripped) if decimal_part_stripped else 0

                    else: return None # Taille nulle ou négative
                except ValueError:
                    logger.warning(f"Impossible de convertir '{filter_key}' ('{size_str}') en float pour le filtre {filter_type}.")
                    return None # Non convertible en float
        return None # Filtre ou clé non trouvé
    except Exception as e:
        logger.error(f"Erreur dans get_precision_from_filter ({filter_type}/{filter_key}): {e}", exc_info=True)
        return None

def get_filter_value(symbol_info: dict, filter_type: str, filter_key: str) -> Optional[float]:
    """
    Extrait une valeur numérique spécifique d'un filtre de symbole (ex: Binance).

    Args:
        symbol_info (dict): Dictionnaire contenant les informations du symbole.
        filter_type (str): Le type de filtre (ex: 'LOT_SIZE', 'MIN_NOTIONAL').
        filter_key (str): La clé dans le filtre contenant la valeur (ex: 'minQty', 'minNotional').

    Returns:
        Optional[float]: La valeur flottante du filtre, ou None si non trouvé ou invalide.
    """
    try:
        if not isinstance(symbol_info, dict): return None
        filters = symbol_info.get('filters', [])
        if not isinstance(filters, list): return None

        target_filter = next((f for f in filters if isinstance(f, dict) and f.get('filterType') == filter_type), None)
        if target_filter:
            value_str = target_filter.get(filter_key)
            if isinstance(value_str, (str, int, float)): # Accepter aussi les nombres directement
                try:
                    return float(value_str)
                except (ValueError, TypeError):
                     logger.warning(f"Impossible de convertir '{filter_key}' ('{value_str}') en float pour le filtre {filter_type}.")
                     return None
        return None # Filtre ou clé non trouvé
    except Exception as e:
        logger.error(f"Erreur dans get_filter_value ({filter_type}/{filter_key}): {e}", exc_info=True)
        return None

# Fin du fichier src/utils/exchange_utils.py
