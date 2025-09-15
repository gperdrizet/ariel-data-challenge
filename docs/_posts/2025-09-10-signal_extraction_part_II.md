---
layout: post
title: "Signal Extraction Part II: FGS1 Data Reduction"
---

Building on the success of AIRS-CH0 signal extraction, let's apply the same intelligent data reduction approach to the FGS1 guidance camera data. The goal is to identify and extract just the signal-bearing pixels from the 2D frames, reducing data volume while preserving the exoplanet transit signatures.

Checkout the [FGS1 signal extraction notebook](https://github.com/gperdrizet/ariel-data-challenge/blob/main/notebooks/02.4-FGS_signal_extraction.ipynb) on GitHub.

## 1. FGS1 signal structure

Unlike AIRS-CH0's spectral strips, FGS1 frames contain a more compact signal region that extends across both rows and columns. The signal analysis reveals a clear pattern:

<p align="center">
  <img src="https://raw.githubusercontent.com/gperdrizet/ariel-data-challenge/refs/heads/main/figures/signal_extraction/02.4.3-total_flux_by_row_spectrogram.jpg" alt="FGS1 signal distribution">
</p>

The signal is concentrated in a roughly square region around the center of the detector, with the brightest pixels clustered between rows and columns 10-20. This makes sense for a guidance camera - the star's point spread function creates a compact brightness distribution on the detector.

## 2. 2D signal extraction approach

The extraction strategy extends the 1D row-based approach used for AIRS-CH0 to work in two dimensions:

1. **Identify bright rows**: Find the top N rows with highest total signal
2. **Identify bright columns**: Find the top N columns with highest total signal  
3. **Extract signal block**: Select the intersection of bright rows and columns
4. **Sum to single value**: Collapse the extracted block to one brightness measurement per frame

This creates a "signal block" rather than a "signal strip", capturing the 2D nature of the FGS1 point source.

## 3. Extraction results

The results demonstrate excellent performance with dramatic data reduction:

<p align="center">
  <img src="https://raw.githubusercontent.com/gperdrizet/ariel-data-challenge/refs/heads/main/figures/signal_extraction/02.4.4-transit_plot_total_vs_strip.jpg" alt="FGS1 extracted vs total signal">
</p>

**Key findings:**

- **Transit preservation**: The extracted signal shows virtually identical transit signatures to the full frame data
- **Data reduction**: Using a 6×6 pixel block achieves ~98% data reduction while preserving signal quality
- **Signal consistency**: The extracted signal maintains the same relative brightness and noise characteristics
- **Processing efficiency**: Dramatically smaller datasets enable much faster downstream analysis

## 4. Optimal extraction parameters

Testing different block sizes reveals that a 4×4 to 6×6 pixel extraction region hits the sweet spot:
- Smaller blocks risk losing signal from the point spread function wings
- Larger blocks start including more background noise than useful signal
- The 6×6 approach provides robust signal capture with excellent noise rejection

## 5. Implementation ready

The FGS1 signal extraction approach mirrors the adaptive threshold method developed for AIRS-CH0, making it straightforward to integrate into the preprocessing pipeline. The two-dimensional extraction naturally handles variations in star brightness and detector response across different planets.

This completes the signal extraction development - we now have intelligent data reduction methods for both instruments that preserve transit signals while achieving ~98% data volume reduction. The next step is integrating these methods into the full preprocessing pipeline for production use.

Just like with AIRS-CH0, sometimes the best signal processing isn't about more sophisticated algorithms - it's about intelligently identifying and keeping only the data that matters.