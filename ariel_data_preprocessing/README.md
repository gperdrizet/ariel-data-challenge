# Ariel Data Preprocessing

This module contains the FGS1 and AIRS-CH0 signal data preprocessing tools.

## Submodules

1. Signal correction (implemented)
2. Data reduction (planned)
3. Signal extraction (planned)

## 1. Signal correction

Implements the six signal correction steps outline in the [Calibrating and Binning Ariel Data](https://www.kaggle.com/code/gordonyip/calibrating-and-binning-ariel-data) shared by the contest organizers.

Example use:

```python
from ariel-data-preprocessing.signal_correction import SignalCorrection

signal_correction = SignalCorrection(
    input_data_path='data/raw',
    output_data_path='data/corrected',
    gain=0.4369,
    offset=-1000.0,
    n_cpus=16
)

signal_correction.run()
```
