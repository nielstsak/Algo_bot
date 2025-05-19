# WFO Performance Report: psar_reversal_otoco - AIXBTUSDC_5m

## WFO Configuration

*Note: Detailed WFO settings were not found in the summary file.*

## Out-of-Sample (OOS) Performance Summary

- **Average OOS Sharpe Ratio:** 3.8630
- **Total OOS PnL (All Folds):** 6096.18 USDC
- **Total Folds Run:** 3
- **Successful OOS Folds:** 3
## Fold-by-Fold OOS Results

*(Displaying primary metric: Sharpe Ratio)*

| Fold | Status    | Best IS Params (kama_period, fast, slow) | OOS Metric Value |
| :--- | :-------- | :--------------------------------------- | :--------------- |
| 0    | COMPLETED | p=?, f=?, s=?                            | 3.8630           |
| 1    | COMPLETED | p=?, f=?, s=?                            | 3.8630           |
| 2    | COMPLETED | p=?, f=?, s=?                            | 3.8630           |

## Parameters Selected for Live Config

*Parameters selected from the last completed fold (Fold 2)*

```json
{
  "psar_step": 0.04,
  "psar_max_step": 0.2,
  "atr_period": 20,
  "sl_atr_mult": 2.0,
  "tp_atr_mult": 3.0
}
```
