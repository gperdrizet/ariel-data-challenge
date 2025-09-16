---
layout: post
title: "From Notebook to Package: Refactoring and Deploying the Signal Correction Pipeline"
---

Yesterday's challenge was getting the signal correction pipeline to work. Today's challenge? Making it production-ready. Time to refactor the preprocessing code into a proper Python package, add comprehensive testing, and set up automated CI/CD for deployment to PyPI.

TLDR: here is the [PyPi package](https://pypi.org/project/ariel-data-preprocessing)

## 1. Refactoring the signal correction pipeline

The original signal correction pipeline worked great in a Jupyter notebook, but notebook code doesn't scale well. Here's what needed to happen:

1. **Extract the logic** from notebook cells into a clean, reusable class
2. **Add proper documentation** with docstrings for every method
3. **Implement comprehensive unit tests** to catch bugs and regressions
4. **Set up CI/CD workflows** for automated testing and deployment
5. **Package for PyPI** so anyone can install with `pip install ariel-data-challenge`

## 2. The SignalCorrection() class

The refactored `SignalCorrection()` class in `ariel_data_preprocessing/signal_correction.py` implements the complete 6-step pipeline:

```python
class SignalCorrection:
    '''
    Complete signal correction and calibration pipeline for Ariel telescope data.
    
    This class implements the full 6-step preprocessing pipeline required to transform
    raw Ariel telescope detector outputs into science-ready data suitable for exoplanet
    atmospheric analysis. The pipeline handles both AIRS-CH0 (infrared spectrometer) 
    and FGS1 (guidance camera) data with parallel processing capabilities.
    
    Processing Pipeline:
        1. Analog-to-Digital Conversion (ADC) - Convert raw counts to physical units
        2. Hot/Dead Pixel Masking - Remove problematic detector pixels
        3. Linearity Correction - Account for non-linear detector response
        4. Dark Current Subtraction - Remove thermal background noise
        5. Correlated Double Sampling (CDS) - Reduce read noise via paired exposures
        6. Flat Field Correction - Normalize pixel-to-pixel sensitivity variations
    
    Key Features:
        - Multiprocessing support for parallel planet processing
        - Optional FGS1 downsampling to match AIRS-CH0 cadence
        - Configurable processing steps (can enable/disable individual corrections)
        - Automatic calibration data loading and management
        - HDF5 output for efficient large dataset storage
    
    Performance Optimizations:
        - Process-level parallelization across planets
        - Intelligent FGS downsampling (83% data reduction)
    
    Example:
        >>> corrector = SignalCorrection(
        ...     input_data_path='data/raw',
        ...     output_data_path='data/corrected',
        ...     n_cpus=4,
        ...     downsample_fgs=True,
        ...     n_planets=100
        ... )
        >>> corrector.run()
    
    Input Requirements:
        - Works with Ariel Data Challenge (2025) dataset from Kaggle
        - Raw Ariel telescope data in parquet format
        - Calibration data (dark, dead, flat, linearity correction files)
        - ADC conversion parameters
        - Axis info metadata for timing
    
    Output:
        - HDF5 file with corrected AIRS-CH0 and FGS1 signals and hot/dead pixel masks
        - Organized by planet ID for easy access
        - Reduced data volume (50% reduction from CDS, optional 83% FGS reduction)
        - Science-ready data for downstream analysis
        - Output structure:
        
            HDF5 file structure:
            ├── planet_id_1/
            │   ├── AIRS-CH0_signal       # Corrected spectrometer data
            │   ├── AIRS-CH0_signal_mask  # Mask for spectrometer data
            │   ├── FGS1_signal           # Corrected guidance camera data
            │   └── FGS1_signal_mask      # Mask for guidance camera data
            |
            ├── planet_id_2/
            │   ├── AIRS-CH0_signal       # Corrected spectrometer data
            │   ├── AIRS-CH0_signal_mask  # Mask for spectrometer data
            │   ├── FGS1_signal           # Corrected guidance camera data
            │   └── FGS1_signal_mask      # Mask for guidance camera data
            |
            └── ...
    '''
```

Each step is now a private method with clear documentation:
- `_ADC_convert()` - Applies gain and offset corrections
- `_mask_hot_dead()` - Uses sigma clipping to identify hot pixels and masks dead pixels
- `_apply_linear_corr()` - Applies polynomial corrections pixel by pixel
- `_clean_dark()` - Subtracts scaled dark current
- `_get_cds()` - Performs correlated double sampling
- `_correct_flat_field()` - Normalizes pixel sensitivity

The class is configurable with ADC parameters, CPU count for parallel processing, and input/output data paths.

## 3. Comprehensive unit testing

Testing a signal processing pipeline requires careful validation of each step. The test suite in `tests/test_preprocessing.py` covers:

- **Shape preservation** - Ensuring array dimensions are maintained through each step
- **Data type handling** - Verifying float64 conversion and masked array creation
- **CDS frame reduction** - Confirming the frame count is halved correctly
- **Integration with real data** - Using actual calibration files and signal data

Each test uses a subset of real Ariel data to ensure the corrections work with actual telescope outputs, not just synthetic test cases.

## 4. Automated CI/CD pipeline

Three GitHub workflows handle different aspects of the development pipeline:

### 4.1. Unit testing ([`unittest.yml`](https://github.com/gperdrizet/ariel-data-challenge/blob/main/.github/workflows/unittest.yml))
Triggered on every pull request to main:

- Sets up Python 3.8 environment
- Installs dependencies
- Runs the complete test suite
- Prevents merging if any tests fail

### 4.2. Test PyPI release ([`test_pypi_release.yml`](https://github.com/gperdrizet/ariel-data-challenge/blob/main/.github/workflows/test_pypi_release.yml))
Triggered when pushing tags to the dev branch:

- Builds the package distribution
- Runs unit tests to ensure quality
- Publishes to Test PyPI for validation
- Allows testing the installation process before production release

### 4.3. Production PyPI release ([`pypi_release.yml`](https://github.com/gperdrizet/ariel-data-challenge/blob/main/.github/workflows/pypi_release.yml))
Triggered when creating a GitHub release:

- Builds the final distribution
- Runs comprehensive tests
- Publishes to the main PyPI repository
- Makes the package publicly available via `pip install`

## 5. The Benefits

This refactoring effort pays dividends in multiple ways:

### 5.1. **Reproducibility**
The Ariel Data Challenge isn't just about building a working solution - it's about creating tools that the broader astronomical community can use and improve. Anyone can now install and use the exact same preprocessing pipeline:

```
pip install ariel-data-preprocessing
```

### 5.2. **Reliability** 
Automated testing catches bugs before they reach production. Every code change is validated against real data.

### 5.3. **Maintainability**
Clean class structure with documented methods makes the code much easier to understand and modify.

### 5.4. **Collaboration**
Other researchers can easily build on this work, contribute improvements, or adapt the pipeline for their own projects.

With the preprocessing pipeline now available as a proper Python package, complete with automated testing and continuous deployment, the foundation is solid for the next phase: building machine learning models to extract exoplanet atmospheric spectra.

## 6. Next steps

With the infrastructure in place, the focus shifts back to science:

1. **Integrate the package** into the main analysis workflow
2. **Optimize performance** for batch processing of multiple planets
3. **Build the spectral extraction pipeline** using the cleaned data
4. **Develop machine learning models** for atmospheric parameter estimation

The engineering detour is complete - time to get back to hunting for exoplanet atmospheres!
