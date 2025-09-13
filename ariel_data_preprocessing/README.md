# Ariel Data Preprocessing

[![PyPI release](https://github.com/gperdrizet/ariel-data-challenge/actions/workflows/pypi_release.yml/badge.svg)](https://github.com/gperdrizet/ariel-data-challenge/actions/workflows/pypi_release.yml)
[![Unittest](https://github.com/gperdrizet/ariel-data-challenge/actions/workflows/unittest.yml/badge.svg)](https://github.com/gperdrizet/ariel-data-challenge/actions/workflows/unittest.yml)

This module contains the FGS1 and AIRS-CH0 signal data preprocessing tools.

## Submodules

1. Signal correction (implemented)
3. Signal extraction (partially implemented - AIRS-CH0 data only)

## 1. Signal correction

Implements the six signal correction steps outline in the [Calibrating and Binning Ariel Data](https://www.kaggle.com/code/gordonyip/calibrating-and-binning-ariel-data) notebook shared by the contest organizers.

See the following notebooks for implementation details and plots:

1. [Signal correction](https://github.com/gperdrizet/ariel-data-challenge/blob/main/notebooks/02.1-signal_correction.ipynb)
2. [Signal correction optimization](https://github.com/gperdrizet/ariel-data-challenge/blob/main/notebooks/02.2-signal_correction_optimization.ipynb)

**Example use:**

```python
from ariel-data-preprocessing.signal_correction import SignalCorrection

signal_correction = SignalCorrection(
    input_data_path='data/raw',
    output_data_path='data/corrected',
    n_planets=10
)

signal_correction.run()
```

The signal preprocessing pipeline will write the corrected frames as an HDF5 archive called `train.h5` with the following structure:

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

## 2. Signal extraction

Takes signal corrected data HDF5 output from `SignalCorrection()`.

Selects top n brightest rows of pixels from AIRS-CH0 spectrogram and sums them. Then applies moving average smoothing for each wavelength index across the frames.

See the following notebooks for implementation details and plots:

1. [Signal extraction](https://github.com/gperdrizet/ariel-data-challenge/blob/main/notebooks/02.3-signal_extraction.ipynb)
2. [Wavelength smoothing](https://github.com/gperdrizet/ariel-data-challenge/blob/main/notebooks/02.4-wavelength_smoothing.ipynb)

**Example usage:**

```python
from ariel-data-preprocessing.signal_correction import SignalExtraction

signal_extraction = SignalExtraction(
    input_data_path='data/corrected',
    output_data_path='data/extracted',
    inclusion_threshold=0.95
)

signal_extraction.run()
```

Output data will be written to `train.h5` in the directory passed to `output_data_path`. The structure of the HDF5 archive matches the output from `SignalCorrection()`.