# WFO Performance Report: kama_crossover_otoco - AIXBTUSDC_5m

## WFO Configuration

*Note: Detailed WFO settings were not found in the summary file.*

## Out-of-Sample (OOS) Performance Summary

- **Average OOS Sharpe Ratio:** 16.3000
- **Total OOS PnL (All Folds):** 34374.04 USDC
- **Total Folds Run:** 3
- **Successful OOS Folds:** 3
## Fold-by-Fold OOS Results

*(Displaying primary metric: Sharpe Ratio)*

| Fold | Status    | Best IS Params (kama_period, fast, slow) | OOS Metric Value |
| :--- | :-------- | :--------------------------------------- | :--------------- |
| 0    | COMPLETED | p=5, f=2, s=38                           | 17.2285          |
| 1    | COMPLETED | p=6, f=2, s=29                           | 15.7219          |
| 2    | COMPLETED | p=6, f=2, s=24                           | 15.9496          |

## Parameters Selected for Live Config

*Parameters selected from the last completed fold (Fold 2)*

```json
{
  "kama_period": 6,
  "kama_fast_ema": 2,
  "kama_slow_ema": 24,
  "atr_period": 20,
  "sl_atr_mult": 2.0,
  "tp_atr_mult": 2.5
}
```
