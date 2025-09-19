---
layout: post
title: "Wavelength Smoothing: Taming Spectral Noise for Clean Time Series"
---

With clean spectral signals extracted from the AIRS-CH0 and FGS1 detectors, the next challenge emerges: individual wavelength channels are incredibly noisy. Each extracted time series shows significant frame-to-frame variations that could mask the subtle exoplanet atmospheric signals we're trying to detect. Time for some smoothing.

Checkout the [wavelength smoothing notebook](https://github.com/gperdrizet/ariel-data-challenge/blob/main/notebooks/02.5-wavelength_smoothing.ipynb) on GitHub.

## 1. The noise problem

After signal correction and extraction, the individual frame spectra and the overall transit signal look great. But I still have a lot of frame-to-frame noise in the individual wavelength channels.

The solution requires intelligent smoothing that preserves real signals while reducing noise. Three different approaches were tested on the wavelength time series:

<p align="center">
  <img src="https://raw.githubusercontent.com/gperdrizet/ariel-data-challenge/refs/heads/main/figures/signal_extraction/02.5.1-wavelength_smoothing.jpg" alt="Comparison of smoothing methods">
</p>

The comparison reveals important trade-offs:

- **Savitzky-Golay filtering**: Excellent signal preservation but computationally expensive
- **Simple convolution**: Good smoothing but can introduce artifacts at edges  
- **Moving average**: Clean results with optimal performance for large datasets

For processing 1100+ planets within Kaggle's time constraints, the moving average emerges as the clear winner - it provides excellent noise reduction while being computationally efficient enough for production use.

## 2. Efficient moving average implementation

The key insight is using cumulative sums for O(n) moving average computation instead of O(n×w) sliding window calculations:

```python
def moving_average_rows(a: np.ndarray, n: int) -> np.ndarray:
    '''Moving average smoothing

    Args:
      a: frame time series for one planet
      n: smoothing window width

    Returns:
      Smoothed data, number of frames will be less by 0.5 * n
    '''

    # Compute cumulative sum along axis 1 (across columns)
    cumsum_vec = np.cumsum(a, axis=1, dtype=float)
    
    # Subtract cumulative sum at window start from window end
    cumsum_vec[:, n:] = cumsum_vec[:, n:] - cumsum_vec[:, :-n]
    
    # Return averages for each window
    return cumsum_vec[:, n - 1:] / n
```

This approach scales beautifully - it can smooth all wavelength channels across a sequence of frames simultaneously with minimal computational overhead.

## 3. Spectral time series results

Applying the moving average smoothing to the extracted data and standardizing each wavelength produces remarkably clean spectral time series:

<p align="center">
  <img src="https://raw.githubusercontent.com/gperdrizet/ariel-data-challenge/refs/heads/main/figures/signal_extraction/02.5.2-smoothed_wavelength_spectrogram.jpg" alt="Smoothed spectral time series">
</p>

The smoothed spectrogram reveals several important features:

- **Clear temporal structure**: Systematic variations that correspond to the exoplanet transit
- **Reduced noise floor**: Frame-to-frame variations are dramatically suppressed

## 4. Exoplanet transit signal

The total signal per frame now looks even better - yes that is a scatter plot on the right!

<p align="center">
  <img src="https://raw.githubusercontent.com/gperdrizet/ariel-data-challenge/refs/heads/main/figures/signal_extraction/02.5.3-transit_plot_total_vs_wavelength_smoothed.jpg" alt="Smoothed spectral time series">
</p>

## 5. Parameter optimization

The smoothing window size becomes a critical parameter balancing noise reduction against signal preservation. This is included as a tunable parameter in the signal extraction pipeline. In limited experimentation thus far, a window size of ~200 appears to be the sweet spot - but this should be rigorously optimized later.

## 6. Integration with signal extraction

The wavelength smoothing integrates seamlessly into the signal extraction pipeline as an optional final step:

```python
from ariel_data_preprocessing import SignalExtraction

extractor = SignalExtraction(
    input_data_path='data/corrected',
    output_data_path='data/extracted',
    inclusion_threshold=0.8,
    smooth=True,              # Enable wavelength smoothing
    smoothing_window=200      # Optimize for transit timescales
)

output_file = extractor.run()
```

When enabled, the smoothing is applied to each wavelength channel independently, preserving the full spectral information while dramatically improving signal quality.

Check it out on PyPI: [ariel-data-preprocessing](https://pypi.org/project/ariel-data-preprocessing/)

The journey from raw detector counts to science-ready spectral time series almost complete. Next up, a cool trick to generate more diversity in dataset and a path to uncertainty estimation...
