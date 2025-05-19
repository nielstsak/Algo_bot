# WFO Performance Report: uo_threshold_otoco - PENGUUSDC_5m

## WFO Configuration

*Note: Detailed WFO settings were not found in the summary file.*

## Out-of-Sample (OOS) Performance Summary

- **Average OOS Sharpe Ratio:** -2.8274
- **Total OOS PnL (All Folds):** -51.16 USDC
- **Total Folds Run:** 2
- **Successful OOS Folds:** 2
## Fold-by-Fold OOS Results

*(Displaying primary metric: Sharpe Ratio)*

| Fold | Status    | Best IS Params (kama_period, fast, slow) | OOS Metric Value |
| :--- | :-------- | :--------------------------------------- | :--------------- |
| 0    | COMPLETED | p=?, f=?, s=?                            | N/A              |
| 1    | COMPLETED | p=?, f=?, s=?                            | -2.8274          |

## Parameters Selected for Live Config

*Parameters selected from the last completed fold (Fold 1)*

```json
{
  "uo_short": 10,
  "uo_medium": 17,
  "uo_long": 28,
  "uo_low": 20,
  "uo_high": 78,
  "atr_period": 18,
  "sl_atr_mult": 3.0,
  "tp_atr_mult": 4.5
}
```
