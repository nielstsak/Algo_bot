import logging
from typing import Any, Dict, Union, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

def calculate_performance_metrics(
    trades_df: Optional[pd.DataFrame], 
    equity_curve: Optional[pd.Series],
    initial_capital: float
    ) -> Dict[str, Union[float, int, str, None]]:
    
    metrics: Dict[str, Union[float, int, str, None]] = {}

    if trades_df is None or trades_df.empty or equity_curve is None or equity_curve.empty or equity_curve.isnull().all():
        logger.warning("No trades executed or equity curve empty/NaN. Returning default metrics.")
        metrics["Start Date"] = equity_curve.index.min().isoformat() if equity_curve is not None and not equity_curve.empty and len(equity_curve.index) > 0 else None
        metrics["End Date"] = equity_curve.index.max().isoformat() if equity_curve is not None and not equity_curve.empty and len(equity_curve.index) > 0 else None
        
        duration_days_val = 0.0
        if equity_curve is not None and not equity_curve.empty and len(equity_curve.index) > 1:
            duration = equity_curve.index.max() - equity_curve.index.min()
            duration_days_val = duration.days + duration.seconds / (24 * 3600)
        metrics["Duration Days"] = duration_days_val

        metrics["Initial Capital USDC"] = initial_capital
        final_equity = equity_curve.iloc[-1] if equity_curve is not None and not equity_curve.empty and pd.notna(equity_curve.iloc[-1]) else initial_capital
        metrics["Final Equity USDC"] = final_equity
        metrics["Total Net PnL USDC"] = final_equity - initial_capital
        metrics["Total Net PnL Pct"] = ((final_equity - initial_capital) / initial_capital * 100) if initial_capital > 0 else 0.0
        metrics["Total Trades"] = 0
        metrics["Winning Trades"] = 0
        metrics["Losing Trades"] = 0
        metrics["Win Rate Pct"] = np.nan
        metrics["Average Trade PnL USDC"] = np.nan
        metrics["Average Winning Trade USDC"] = np.nan
        metrics["Average Losing Trade USDC"] = np.nan
        metrics["Payoff Ratio"] = np.nan
        metrics["Profit Factor"] = np.nan
        metrics["Max Drawdown Pct"] = 0.0
        metrics["Max Drawdown USDC"] = 0.0
        metrics["Annualized Return Pct"] = 0.0
        metrics["Annualized Volatility Pct"] = 0.0
        metrics["Sharpe Ratio"] = np.nan
        metrics["Sortino Ratio"] = np.nan
        metrics["Calmar Ratio"] = np.nan
        metrics["Longest Winning Streak"] = 0
        metrics["Longest Losing Streak"] = 0
        metrics["Average Holding Period"] = None 
        metrics["Status"] = "No Trades or Data"
        return metrics

    if 'pnl_net_usd' not in trades_df.columns:
         logger.error("trades_df must contain 'pnl_net_usd' column.")
         metrics["Status"] = "Error: pnl_net_usd missing"
         metrics["Total Net PnL USDC"] = 0.0
         metrics["Win Rate Pct"] = np.nan
         return metrics

    trades_df_cleaned = trades_df.dropna(subset=['pnl_net_usd'])
    trades_df_to_use = trades_df_cleaned if not trades_df_cleaned.empty else trades_df
    
    metrics["Start Date"] = equity_curve.index.min().isoformat()
    metrics["End Date"] = equity_curve.index.max().isoformat()
    duration = equity_curve.index.max() - equity_curve.index.min()
    metrics["Duration Days"] = duration.days + duration.seconds / (24 * 3600) if duration.days + duration.seconds > 0 else 0

    metrics["Initial Capital USDC"] = initial_capital
    final_equity_val = equity_curve.iloc[-1] if pd.notna(equity_curve.iloc[-1]) else initial_capital
    metrics["Final Equity USDC"] = final_equity_val
    metrics["Total Net PnL USDC"] = final_equity_val - initial_capital
    metrics["Total Net PnL Pct"] = (metrics["Total Net PnL USDC"] / initial_capital * 100) if initial_capital > 0 else 0.0

    metrics["Total Trades"] = len(trades_df_to_use)
    if metrics["Total Trades"] > 0:
        pnl_series = trades_df_to_use['pnl_net_usd']
        winning_trades_series = pnl_series[pnl_series > 0]
        losing_trades_series = pnl_series[pnl_series < 0]
        metrics["Winning Trades"] = len(winning_trades_series)
        metrics["Losing Trades"] = len(losing_trades_series)

        metrics["Win Rate Pct"] = (metrics["Winning Trades"] / metrics["Total Trades"] * 100) if metrics["Total Trades"] > 0 else np.nan
        metrics["Average Trade PnL USDC"] = pnl_series.mean() if not pnl_series.empty else np.nan
        metrics["Average Winning Trade USDC"] = winning_trades_series.mean() if metrics["Winning Trades"] > 0 else 0.0
        metrics["Average Losing Trade USDC"] = losing_trades_series.mean() if metrics["Losing Trades"] > 0 else 0.0 

        avg_win = metrics["Average Winning Trade USDC"]
        avg_loss = metrics["Average Losing Trade USDC"]
        metrics["Payoff Ratio"] = abs(avg_win / avg_loss) if avg_loss != 0 and metrics["Losing Trades"] > 0 else np.inf if avg_win > 0 else np.nan

        gross_profit = winning_trades_series.sum()
        gross_loss = abs(losing_trades_series.sum()) 
        metrics["Profit Factor"] = gross_profit / gross_loss if gross_loss > 0 else np.inf if gross_profit > 0 else np.nan
    else:
        metrics["Winning Trades"] = 0
        metrics["Losing Trades"] = 0
        metrics["Win Rate Pct"] = np.nan
        metrics["Average Trade PnL USDC"] = np.nan
        metrics["Average Winning Trade USDC"] = np.nan
        metrics["Average Losing Trade USDC"] = np.nan
        metrics["Payoff Ratio"] = np.nan
        metrics["Profit Factor"] = np.nan

    returns = equity_curve.pct_change().dropna()
    trading_days_per_year = 252 
    periods_per_year = np.nan

    if len(returns) >=2 :
        median_delta_seconds = equity_curve.index.to_series().diff().median().total_seconds()
        if pd.notna(median_delta_seconds) and median_delta_seconds > 0:
             seconds_in_year = trading_days_per_year * 24 * 3600
             periods_per_year = seconds_in_year / median_delta_seconds
        else: # Fallback if median_delta is not usable
            total_duration_days = metrics["Duration Days"]
            if total_duration_days and total_duration_days > 0:
                 periods_per_year = len(returns) / (total_duration_days / trading_days_per_year) if total_duration_days > 0 else len(returns) * trading_days_per_year # Crude estimate


    if pd.isna(periods_per_year) or len(returns) < 2:
        metrics["Annualized Return Pct"] = np.nan
        metrics["Annualized Volatility Pct"] = np.nan
        metrics["Sharpe Ratio"] = np.nan
        metrics["Sortino Ratio"] = np.nan
    else:
        total_return_overall = (equity_curve.iloc[-1] / initial_capital) - 1
        years = metrics["Duration Days"] / trading_days_per_year if metrics["Duration Days"] is not None and metrics["Duration Days"] > 0 else (len(returns) / periods_per_year if periods_per_year > 0 else 1)
        
        if years > 1e-6 : # Avoid division by zero or very small numbers
            metrics["Annualized Return Pct"] = ((1 + total_return_overall) ** (1 / years) - 1) * 100 if total_return_overall > -1 else -100.0
        elif total_duration_days and total_duration_days > 0 : # if less than a year, project return
            metrics["Annualized Return Pct"] = total_return_overall * (trading_days_per_year / total_duration_days) * 100
        else: # Default to non-annualized if duration is too short or zero
             metrics["Annualized Return Pct"] = total_return_overall * 100


        volatility = returns.std() * np.sqrt(periods_per_year)
        metrics["Annualized Volatility Pct"] = volatility * 100 if pd.notna(volatility) else np.nan

        mean_return_period = returns.mean()
        std_dev_return_period = returns.std()
        
        if pd.notna(std_dev_return_period) and std_dev_return_period > 1e-9 and pd.notna(mean_return_period): 
            metrics["Sharpe Ratio"] = (mean_return_period / std_dev_return_period) * np.sqrt(periods_per_year)
        else:
            metrics["Sharpe Ratio"] = np.nan if pd.notna(mean_return_period) and mean_return_period == 0 else (np.inf * np.sign(mean_return_period) if pd.notna(mean_return_period) else np.nan)
        
        negative_returns = returns[returns < 0]
        downside_deviation_period = negative_returns.std() if not negative_returns.empty else 0.0
        if pd.notna(downside_deviation_period) and downside_deviation_period > 1e-9 and pd.notna(mean_return_period):
            metrics["Sortino Ratio"] = (mean_return_period / downside_deviation_period) * np.sqrt(periods_per_year)
        elif pd.notna(mean_return_period):
             metrics["Sortino Ratio"] = np.inf if mean_return_period > 0 else (0.0 if mean_return_period == 0 else -np.inf if mean_return_period < 0 else np.nan)
        else:
            metrics["Sortino Ratio"] = np.nan


    cumulative_max_equity = equity_curve.cummax()
    drawdown_series = (equity_curve - cumulative_max_equity) / cumulative_max_equity
    max_drawdown_pct_val = drawdown_series.min()
    metrics["Max Drawdown Pct"] = max_drawdown_pct_val * 100 if pd.notna(max_drawdown_pct_val) else 0.0
    
    peak_before_max_dd_val = initial_capital
    if pd.notna(max_drawdown_pct_val) and max_drawdown_pct_val < 0 :
        idx_min_drawdown = drawdown_series.idxmin()
        if idx_min_drawdown in cumulative_max_equity.index:
            peak_before_max_dd_val = cumulative_max_equity[idx_min_drawdown]
    metrics["Max Drawdown USDC"] = peak_before_max_dd_val * (metrics["Max Drawdown Pct"] / 100.0)


    annual_return_pct_val = metrics.get("Annualized Return Pct")
    max_dd_pct_val = metrics.get("Max Drawdown Pct")
    if isinstance(annual_return_pct_val, (float, int)) and pd.notna(annual_return_pct_val) and \
       isinstance(max_dd_pct_val, (float, int)) and pd.notna(max_dd_pct_val) and max_dd_pct_val < -1e-9 :
        metrics["Calmar Ratio"] = annual_return_pct_val / abs(max_dd_pct_val)
    else:
        metrics["Calmar Ratio"] = np.nan

    win_streak = 0; loss_streak = 0
    max_win_streak = 0; max_loss_streak = 0
    if metrics["Total Trades"] > 0 and 'pnl_net_usd' in trades_df_to_use.columns:
        for pnl in trades_df_to_use['pnl_net_usd']:
            if pd.notna(pnl):
                if pnl > 0:
                    win_streak += 1; loss_streak = 0
                    max_win_streak = max(max_win_streak, win_streak)
                elif pnl < 0:
                    loss_streak += 1; win_streak = 0
                    max_loss_streak = max(max_loss_streak, loss_streak)
                else: 
                     win_streak = 0; loss_streak = 0 
    metrics["Longest Winning Streak"] = max_win_streak
    metrics["Longest Losing Streak"] = max_loss_streak

    if metrics["Total Trades"] > 0 and 'entry_timestamp' in trades_df_to_use.columns and 'exit_timestamp' in trades_df_to_use.columns:
        entry_ts = pd.to_datetime(trades_df_to_use['entry_timestamp'], errors='coerce', utc=True)
        exit_ts = pd.to_datetime(trades_df_to_use['exit_timestamp'], errors='coerce', utc=True)
        valid_timestamps = entry_ts.notna() & exit_ts.notna()
        if valid_timestamps.any():
            holding_periods = (exit_ts[valid_timestamps] - entry_ts[valid_timestamps])
            avg_holding_timedelta = holding_periods.mean()
            metrics["Average Holding Period"] = str(avg_holding_timedelta) if pd.notna(avg_holding_timedelta) else None
        else:
            metrics["Average Holding Period"] = None
    else:
        metrics["Average Holding Period"] = None

    for key, value in metrics.items():
        if isinstance(value, float) and pd.notna(value):
            if "Pct" in key or "Ratio" in key or "Factor" in key:
                 metrics[key] = round(value, 4)
            elif "USDC" in key:
                 metrics[key] = round(value, 2)
    
    metrics["Status"] = "Completed"
    return metrics
