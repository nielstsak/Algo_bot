# WFO Performance Report: kama_crossover_otoco - PENGUUSDC_5m

## WFO Configuration

*Note: Detailed WFO settings were not found in the summary file.*

## Out-of-Sample (OOS) Performance Summary

- **Average OOS Sharpe Ratio:** 15.8798
- **Total OOS PnL (All Folds):** 4145.20 USDC
- **Total Folds Run:** 2
- **Successful OOS Folds:** 2
## Fold-by-Fold OOS Results

*(Displaying primary metric: Sharpe Ratio)*

| Fold | Status    | Best IS Params (kama_period, fast, slow) | OOS Metric Value |
| :--- | :-------- | :--------------------------------------- | :--------------- |
| 0    | COMPLETED | p=9, f=2, s=39                           | 19.6524          |
| 1    | COMPLETED | p=9, f=2, s=40                           | 12.1071          |

## Parameters Selected for Live Config

*Parameters selected from the last completed fold (Fold 1)*

```json
{
  "kama_period": 9,
  "kama_fast_ema": 2,
  "kama_slow_ema": 40,
  "atr_period": 21,
  "sl_atr_mult": 2.0,
  "tp_atr_mult": 2.0
}
```
