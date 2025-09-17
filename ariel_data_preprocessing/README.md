# Ariel Data Preprocessing

[![PyPI release](https://github.com/gperdrizet/ariel-data-challenge/actions/workflows/pypi_release.yml/badge.svg)](https://github.com/gperdrizet/ariel-data-challenge/actions/workflows/pypi_release.yml)
[![Unittest](https://github.com/gperdrizet/ariel-data-challenge/actions/workflows/unittest.yml/badge.svg)](https://github.com/gperdrizet/ariel-data-challenge/actions/workflows/unittest.yml)

This module contains the complete FGS1 and AIRS-CH0 signal data preprocessing pipeline for the Ariel Data Challenge.

## Overview

The `DataProcessor` class provides an integrated pipeline that combines:

1. **Signal correction** - Complete 6-step calibration pipeline for raw telescope data
2. **Signal extraction** - Intelligent data reduction and spectral signal extraction

This unified approach transforms raw Ariel telescope data directly into extracted, science-ready spectral time series in a single processing step.

## 1. Signal correction

Implements the complete six-step signal correction pipeline outlined in the [Calibrating and Binning Ariel Data](https://www.kaggle.com/code/gordonyip/calibrating-and-binning-ariel-data) notebook shared by the contest organizers. This module transforms raw Ariel telescope data into science-ready corrected signals.

**Processing Pipeline:**
1. **ADC Conversion** - Convert raw counts to physical units
2. **Hot/Dead Pixel Masking** - Remove problematic detector pixels
3. **Linearity Correction** - Account for non-linear detector response
4. **Dark Current Subtraction** - Remove thermal background noise
5. **Correlated Double Sampling (CDS)** - Reduce read noise via paired exposures
6. **Flat Field Correction** - Normalize pixel-to-pixel sensitivity variations

**Key Features:**
- Multiprocessing support for parallel planet processing
- Optional FGS1 downsampling to match AIRS-CH0 cadence (83% data reduction)
- Configurable processing steps (enable/disable individual corrections)
- HDF5 output for efficient large dataset storage

See the following notebooks for implementation details and performance analysis:

1. [Signal correction](https://github.com/gperdrizet/ariel-data-challenge/blob/main/notebooks/02.1-signal_correction.ipynb)
2. [Signal correction optimization](https://github.com/gperdrizet/ariel-data-challenge/blob/main/notebooks/02.2-signal_correction_optimization.ipynb)

**Example use:**

```python
from ariel_data_preprocessing.data_preprocessing import DataProcessor

data_processor = DataProcessor(
    input_data_path='data/raw',
    output_data_path='data/corrected',
    n_cpus=4,
    downsample_fgs=True,
    n_planets=100
)

data_processor.run()
```

The signal correction pipeline will write the corrected frames and hot/dead pixel masks as an HDF5 archive called `train.h5` by default with the following structure:

```text
    train.h5:
    │
    ├── planet_id_1/
    │   ├── signal  # Combined corrected/extracted spectral time series
    │   └── mask    # Dead/hot pixel mask for spectra
    |
    ├── planet_id_2/
    │   ├── signal  
    │   └── mask    
    |
    └── planet_id_n/
```

## 2. Signal extraction

**Complete extraction pipeline integrated with DataProcessor**

The signal extraction functionality is integrated within the `DataProcessor` class, which handles both signal correction and extraction in a unified pipeline. This approach transforms 3D detector arrays into focused time series suitable for exoplanet atmospheric analysis.

**Processing Features:**
- **AIRS-CH0 Extraction**: Selects brightest detector rows containing spectral traces, sums to create 1D spectra per frame
- **FGS1 Extraction**: Uses 2D block extraction to identify signal regions, collapses to single brightness value per frame  
- **Combined Output**: Merges FGS1 and AIRS-CH0 signals (FGS1 as first column for transit detection)
- **Adaptive Thresholding**: Automatically selects signal-bearing pixels based on configurable intensity thresholds
- **Optional Smoothing**: Applies moving average filtering across wavelengths to reduce noise
- **Massive Data Reduction**: Achieves ~97-98% volume reduction while preserving transit signals

**Key Benefits:**
- Dramatically faster downstream processing due to reduced data volume
- Improved signal-to-noise ratio by focusing on high-signal detector regions
- Preserved exoplanet transit signatures with cleaner temporal structure
- Unified processing for both instrument types

See the following notebooks for implementation details and analysis:

1. [AIRS-CH0 signal extraction](https://github.com/gperdrizet/ariel-data-challenge/blob/main/notebooks/02.3-AIRS_signal_extraction.ipynb)
2. [FGS1 signal extraction](https://github.com/gperdrizet/ariel-data-challenge/blob/main/notebooks/02.4-FGS_signal_extraction.ipynb)
3. [Wavelength smoothing](https://github.com/gperdrizet/ariel-data-challenge/blob/main/notebooks/02.5-wavelength_smoothing.ipynb)

**Example usage:**

The signal extraction is performed as part of the integrated `DataProcessor` pipeline:

```python
from ariel_data_preprocessing.data_preprocessing import DataProcessor

data_processor = DataProcessor(
    input_data_path='data/raw',
    output_data_path='data/processed',
    inclusion_threshold=0.75,
    smooth=True,
    smoothing_window=200
)

data_processor.run()
```

Output data will be written to `train.h5` by default in the directory passed to `output_data_path`. The structure of the HDF5 archive combines both AIRS-CH0 and FGS1 signals:

```text
    train.h5
    |
    ├── planet_1/
    │   ├── signal  # Shape: (n_frames, n_wavelengths + 1) - FGS1 + AIRS-CH0
    │   └── mask    # Shape: (n_wavelengths + 1,) - combined mask
    │
    ├── planet_2/
    │   ├── signal  # Shape: (n_frames, n_wavelengths + 1) - FGS1 + AIRS-CH0  
    │   └── mask    # Shape: (n_wavelengths + 1,) - combined mask
    │
    └── planet_n/
```

**Note**: The first column of the `signal` dataset contains the extracted FGS1 brightness time series, followed by the AIRS-CH0 spectral channels. This structure facilitates easy transit detection using FGS1 data while providing wavelength-dependent atmospheric information from AIRS-CH0.