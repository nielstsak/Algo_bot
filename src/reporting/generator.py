import pathlib
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, Optional, Any, List, Union, Tuple

logger = logging.getLogger(__name__)

def _format_metric(value: Any, precision: int = 4) -> str:
    if isinstance(value, (float, np.floating, np.float64)):
        if np.isnan(value):
            return "NaN"
        elif np.isinf(value):
            return "Infinity" if value > 0 else "-Infinity"
        else:
            return f"{value:.{precision}f}"
    elif isinstance(value, (int, np.integer)):
        return str(value)
    elif isinstance(value, str):
        return value
    elif value is None:
        return "N/A"
    else:
        return str(value)

def _generate_markdown_report(
    wfo_summary_data: Dict[str, Any],
    base_fold_log_path: pathlib.Path, # Path to logs/.../strategy/pair/context_label/
    report_file: pathlib.Path,
    live_config_params_selected: Optional[Dict[str, Any]],
    selection_fold_id: Optional[int],
    selection_oos_metric_value: Optional[float],
    selection_metric_name: str
    ):
    try:
        strategy_name = wfo_summary_data.get("strategy_name", "N/A")
        pair_symbol = wfo_summary_data.get("pair_symbol", "N/A")
        context_label = wfo_summary_data.get("context_label", "N/A")
        fold_results_from_wfo = wfo_summary_data.get("folds_data", [])

        with open(report_file, "w", encoding="utf-8") as f:
            f.write(f"# WFO Performance Report: {strategy_name} - {pair_symbol} ({context_label})\n\n")
            f.write("## WFO Configuration\n")
            f.write(f"- Strategy: {strategy_name}\n")
            f.write(f"- Pair: {pair_symbol}\n")
            f.write(f"- Context: {context_label}\n")
            f.write(f"- WFO Run Timestamp: {wfo_summary_data.get('wfo_run_timestamp', 'N/A')}\n\n")

            f.write("## Overall WFO OOS Performance (Aggregated - if available in wfo_summary.json)\n")
            f.write("*Detailed OOS performance is shown per fold below.*\n\n")

            f.write("## Parameters Selected for Live Configuration\n")
            if live_config_params_selected:
                f.write(f"*Selected from Fold {selection_fold_id} based on best OOS '{selection_metric_name}': {_format_metric(selection_oos_metric_value)}*\n")
                f.write("```json\n")
                f.write(json.dumps(live_config_params_selected, indent=2, default=str))
                f.write("\n```\n\n")
            else:
                f.write("*No parameters were selected for live configuration (e.g., no successful OOS trials found).*\n\n")

            f.write("## Fold-by-Fold OOS Results\n\n")
            if not fold_results_from_wfo:
                f.write("No fold results available in WFO summary.\n\n")
            else:
                f.write("| Fold | IS Status | IS Params (Selected for OOS) | Top OOS Trials Summary |\n")
                f.write("| :--- | :-------- | :--------------------------- | :----------------------- |\n")
                for fold_wfo_data in fold_results_from_wfo:
                    fold_id = fold_wfo_data.get('fold_index', 'N/A')
                    is_status = fold_wfo_data.get('status', 'UNKNOWN')
                    selected_is_params_for_fold = fold_wfo_data.get('selected_params_for_fold', {})
                    
                    params_str_parts = []
                    if selected_is_params_for_fold:
                        for k, v in selected_is_params_for_fold.items():
                            if "frequence" in k: # More generic check for frequency params
                                param_name_short = k.split('_')[-1] if '_' in k else k
                                params_str_parts.append(f"{param_name_short}:{v}")
                            elif any(sub in k for sub in ["period", "step", "mult", "threshold", "ema", "allocation", "leverage"]):
                                param_name_short = k.split('_')[0] if '_' in k else k 
                                param_name_short = param_name_short[:4] # Abbreviate
                                params_str_parts.append(f"{param_name_short}:{_format_metric(v,2)}")
                            # Add other specific formatting if needed
                    params_display = ", ".join(params_str_parts) if params_str_parts else "N/A"


                    # base_fold_log_path is now logs/.../strategy/pair/context_label/
                    oos_summary_file_for_fold = base_fold_log_path / f"fold_{fold_id}" / "oos_validation_summary_TOP_N_TRIALS.json"
                    oos_trials_summary_str = "OOS summary file not found or N/A."
                    if oos_summary_file_for_fold.exists():
                        try:
                            with open(oos_summary_file_for_fold, 'r', encoding='utf-8') as oos_f:
                                oos_trials_data = json.load(oos_f)
                            if oos_trials_data:
                                top_oos_pnls = sorted(
                                    [t.get('oos_metrics', {}).get('Total Net PnL USDC', -float('inf')) for t in oos_trials_data if t.get('oos_metrics')],
                                    reverse=True
                                )
                                oos_trials_summary_str = f"{len(oos_trials_data)} OOS trials. Best PnL: {_format_metric(top_oos_pnls[0] if top_oos_pnls else np.nan, 2)}. PnLs: [{', '.join([_format_metric(p,0) for p in top_oos_pnls[:3]])}...]"
                            else:
                                oos_trials_summary_str = "OOS trials run, but no data."
                        except Exception as e_oos_load:
                            oos_trials_summary_str = f"Error loading OOS summary: {e_oos_load}"
                    elif is_status != "COMPLETED":
                         oos_trials_summary_str = f"(IS Status: {is_status})"
                    
                    f.write(f"| {fold_id:<4} | {is_status:<9} | {params_display:<60} | {oos_trials_summary_str} |\n")
                f.write("\n")
        logger.info(f"Generated Markdown report: {report_file}")
    except Exception as e:
        logger.error(f"Failed to generate Markdown report for {strategy_name}/{pair_symbol}: {e}", exc_info=True)

def _generate_live_config(
    wfo_summary_data: Dict[str, Any], 
    base_fold_log_path: pathlib.Path, # Path to logs/.../strategy/pair/context_label/
    live_config_output_file: pathlib.Path,
    selection_metric: str = "Total Net PnL USDC" 
    ) -> Tuple[Optional[Dict[str, Any]], Optional[int], Optional[float]]:

    strategy_name = wfo_summary_data.get("strategy_name")
    pair_symbol = wfo_summary_data.get("pair_symbol")
    context_label = wfo_summary_data.get("context_label")
    
    if not strategy_name or not pair_symbol or not context_label:
        logger.error("Strategy name, pair symbol, or context_label missing in WFO summary. Cannot generate live config.")
        return None, None, None

    fold_results_from_wfo = wfo_summary_data.get("folds_data", [])
    last_successful_fold_id: Optional[int] = None
    
    for fold_data in reversed(fold_results_from_wfo):
        if fold_data.get("status") == "COMPLETED" and fold_data.get("selected_params_for_fold") is not None:
            last_successful_fold_id = fold_data.get("fold_index")
            break
            
    if last_successful_fold_id is None:
        logger.warning(f"No successful IS optimization fold found for {strategy_name}/{pair_symbol}/{context_label}. Cannot generate live config from OOS results.")
        return None, None, None

    logger.info(f"Attempting to generate live_config from OOS results of Fold {last_successful_fold_id} for {strategy_name}/{pair_symbol}/{context_label}.")
    
    # base_fold_log_path is now logs/.../strategy/pair/context_label/
    oos_summary_file_path = base_fold_log_path / f"fold_{last_successful_fold_id}" / "oos_validation_summary_TOP_N_TRIALS.json"

    if not oos_summary_file_path.exists():
        logger.warning(f"OOS validation summary file not found for Fold {last_successful_fold_id}: {oos_summary_file_path}. Cannot generate live config from OOS.")
        return None, None, None

    try:
        with open(oos_summary_file_path, 'r', encoding='utf-8') as f:
            oos_trials_data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load OOS validation summary {oos_summary_file_path}: {e}", exc_info=True)
        return None, None, None

    if not oos_trials_data:
        logger.warning(f"OOS validation summary for Fold {last_successful_fold_id} is empty. Cannot select parameters.")
        return None, None, None

    best_oos_trial_for_live = None
    best_oos_metric_value = -float('inf')

    for oos_trial_run in oos_trials_data:
        current_metric_value = oos_trial_run.get("oos_metrics", {}).get(selection_metric)
        if current_metric_value is not None and isinstance(current_metric_value, (int, float)) and np.isfinite(current_metric_value):
            if current_metric_value > best_oos_metric_value:
                best_oos_metric_value = current_metric_value
                best_oos_trial_for_live = oos_trial_run
    
    if best_oos_trial_for_live and best_oos_trial_for_live.get("is_trial_params"):
        selected_params = best_oos_trial_for_live["is_trial_params"]
        # Construct live_strategy_id using context_label for uniqueness if it's not just "live" or "default"
        live_strategy_id_suffix = context_label if context_label and context_label not in ["live", "default_context", "default"] else wfo_summary_data.get('timeframe_context', 'live') # Fallback
        live_strategy_id = f"{strategy_name}_{pair_symbol}_{live_strategy_id_suffix}"


        live_config_content = {
            "strategy_id": live_strategy_id,
            "strategy_name_base": strategy_name,
            "pair_symbol": pair_symbol,
            "timeframe_context": context_label, 
            "parameters": selected_params,
            "source_wfo_run": wfo_summary_data.get('wfo_run_timestamp'),
            "source_fold_id": last_successful_fold_id,
            "source_is_trial_number": best_oos_trial_for_live.get("is_trial_number"),
            "selection_oos_metric": selection_metric,
            "selection_oos_metric_value": best_oos_metric_value
        }
        with open(live_config_output_file, "w", encoding="utf-8") as f:
            json.dump(live_config_content, f, indent=4, default=str)
        logger.info(f"Generated Live JSON config using best OOS trial from Fold {last_successful_fold_id}: {live_config_output_file}")
        return selected_params, last_successful_fold_id, best_oos_metric_value
    else:
        logger.warning(f"Could not find a suitable OOS trial to select parameters for live config from Fold {last_successful_fold_id} for {strategy_name}/{pair_symbol}/{context_label}.")
        return None, None, None


def generate_all_reports(log_dir: pathlib.Path, results_dir: pathlib.Path):
    logger.info(f"Starting report generation from WFO log directory: {log_dir}")
    logger.info(f"Saving reports to results directory: {results_dir}")

    if not log_dir.is_dir():
        logger.error(f"Base log directory not found: {log_dir}")
        return

    results_dir.mkdir(parents=True, exist_ok=True)
    all_runs_summary_for_global_csv: List[Dict[str, Any]] = []
    
    for strategy_dir in log_dir.iterdir():
        if not strategy_dir.is_dir():
            continue
        
        for pair_symbol_dir in strategy_dir.iterdir():
            if not pair_symbol_dir.is_dir():
                continue
            
            # MODIFICATION: Iterate one level deeper for context_label_dir
            for context_label_dir in pair_symbol_dir.iterdir():
                if not context_label_dir.is_dir():
                    continue

                wfo_summary_file = context_label_dir / "wfo_strategy_pair_summary.json"
                if not wfo_summary_file.is_file():
                    logger.debug(f"WFO summary file not found: {wfo_summary_file}. Skipping (this might be a fold_X dir).")
                    continue
                
                try:
                    with open(wfo_summary_file, 'r', encoding='utf-8') as f:
                        wfo_summary_content = json.load(f)

                    strategy_name_json = wfo_summary_content.get("strategy_name", strategy_dir.name)
                    pair_symbol_json = wfo_summary_content.get("pair_symbol", pair_symbol_dir.name)
                    context_label_json = wfo_summary_content.get("context_label", context_label_dir.name)


                    results_output_subdir = results_dir / strategy_name_json / pair_symbol_json / context_label_json
                    results_output_subdir.mkdir(parents=True, exist_ok=True)
                    
                    report_md_path = results_output_subdir / "performance_report.md"
                    live_config_json_path = results_output_subdir / "live_config.json"

                    selected_live_params, sel_fold, sel_oos_metric, sel_metric_name = None, None, None, "Total Net PnL USDC"
                    
                    # base_fold_log_path for _generate_live_config and _generate_markdown_report
                    # should be the context_label_dir, as fold_X directories are inside it.
                    selected_live_params, sel_fold, sel_oos_metric = _generate_live_config(
                        wfo_summary_data=wfo_summary_content,
                        base_fold_log_path=context_label_dir, # Pass the context_label_dir
                        live_config_output_file=live_config_json_path,
                        selection_metric=sel_metric_name
                    )
                    
                    _generate_markdown_report(
                        wfo_summary_data=wfo_summary_content,
                        base_fold_log_path=context_label_dir, # Pass the context_label_dir
                        report_file=report_md_path,
                        live_config_params_selected=selected_live_params,
                        selection_fold_id=sel_fold,
                        selection_oos_metric_value=sel_oos_metric,
                        selection_metric_name=sel_metric_name
                    )

                    num_folds_run = len(wfo_summary_content.get("folds_data", []))
                    successful_folds = sum(1 for f_data in wfo_summary_content.get("folds_data", []) if f_data.get("status") == "COMPLETED")
                    
                    global_entry = {
                        "WFO Run Timestamp": wfo_summary_content.get('wfo_run_timestamp', log_dir.name),
                        "Strategy": strategy_name_json,
                        "PairSymbol": pair_symbol_json,
                        "ContextLabel": context_label_json,
                        "Total Folds": num_folds_run,
                        "Successful Folds": successful_folds,
                        "Live Params Generated": selected_live_params is not None,
                        "Selected Fold for Live": sel_fold,
                        f"Selected OOS {sel_metric_name}": _format_metric(sel_oos_metric, 2 if "PnL" in sel_metric_name else 4)
                    }
                    all_runs_summary_for_global_csv.append(global_entry)

                except json.JSONDecodeError:
                    logger.error(f"Failed to decode JSON from {wfo_summary_file}. Skipping.", exc_info=True)
                except Exception as e_proc:
                    logger.error(f"Error processing summary {wfo_summary_file}: {e_proc}. Skipping.", exc_info=True)

    if all_runs_summary_for_global_csv:
        try:
            global_df = pd.DataFrame(all_runs_summary_for_global_csv)
            cols_order = ["WFO Run Timestamp", "Strategy", "PairSymbol", "ContextLabel", 
                          "Total Folds", "Successful Folds", "Live Params Generated", 
                          "Selected Fold for Live", f"Selected OOS {sel_metric_name}"]
            
            final_cols = [c for c in cols_order if c in global_df.columns] + \
                         [c for c in global_df.columns if c not in cols_order]
            global_df = global_df[final_cols]

            global_csv_path = results_dir.parent / f"global_wfo_summary_for_run_{log_dir.name}.csv"
            global_df.to_csv(global_csv_path, index=False, float_format='%.4f')
            logger.info(f"Generated global WFO summary CSV for run {log_dir.name}: {global_csv_path}")
        except Exception as e_csv:
            logger.error(f"Failed to generate global WFO summary CSV: {e_csv}", exc_info=True)
    else:
        logger.warning(f"No WFO results found in {log_dir} to generate a global summary.")

    logger.info(f"Report generation process finished for run: {log_dir.name}")
