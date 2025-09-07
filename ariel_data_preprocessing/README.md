# Ariel Data Preprocessing

[![PyPI release](https://github.com/gperdrizet/ariel-data-challenge/actions/workflows/pypi_release.yml/badge.svg)](https://github.com/gperdrizet/ariel-data-challenge/actions/workflows/pypi_release.yml)
[![Unittest](https://github.com/gperdrizet/ariel-data-challenge/actions/workflows/unittest.yml/badge.svg)](https://github.com/gperdrizet/ariel-data-challenge/actions/workflows/unittest.yml)

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
    n_planets=10
)

signal_correction.run()
```

The signal preprocessing pipeline will write the corrected frames as an hdf5 archive called `train.h5` with the following structure:

```text
├── planet_1
|   ├── AIRS-CH0_signal
│   └── FGS1_signal
│
├── planet_1
|   ├── AIRS-CH0_signal
│   └── FGS1_signal
│
.
.
.
└── planet_n
    ├── AIRS-CH0_signal
    └── FGS1_signal
```
