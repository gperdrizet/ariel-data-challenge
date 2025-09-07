---
layout: post
title: "Signal Correction Pipeline: From Raw Counts to Science-Ready Data"
---

Time to tackle the full signal correction pipeline! After understanding the timing structure and CDS basics, it's time to implement all six preprocessing steps to turn noisy detector outputs into clean, calibrated data suitable for exoplanet analysis.

## The Six-Step Pipeline

Following the competition organizers' guidance, here's the complete preprocessing workflow:

1. **Analog-to-Digital Conversion** - Convert raw detector counts to physical units
2. **Hot/Dead Pixel Masking** - Remove problematic pixels using sigma clipping
3. **Linearity Correction** - Apply polynomial corrections for detector non-linearity
4. **Dark Current Subtraction** - Remove thermal background noise
5. **Correlated Double Sampling (CDS)** - Subtract paired exposures to reduce read noise
6. **Flat Field Correction** - Normalize pixel-to-pixel sensitivity variations

## Step-by-Step Results

Here's how the AIRS-CH0 signal evolves through each correction step:

<p align="center">
  <img src="https://raw.githubusercontent.com/gperdrizet/ariel-data-challenge/refs/heads/main/figures/data_preprocessing/02.1-AIRS_signal_correction_steps.jpg" alt="AIRS signal correction pipeline steps">
</p>

Each step addresses specific detector artifacts:
- **ADC conversion** transforms raw counts using gain and offset corrections
- **Masking** removes hot pixels (identified via sigma clipping) and known dead pixels
- **Linearity correction** applies polynomial fits to account for non-linear detector response
- **Dark subtraction** removes thermal background scaled by integration time
- **CDS** subtracts the short/long exposure pairs to reduce read noise
- **Flat fielding** normalizes pixel-to-pixel sensitivity differences

## Final Results

After the complete pipeline, both instruments produce much cleaner data:

<div style="display: flex; justify-content: space-around; align-items: flex-start; gap: 20px; margin: 20px 0;">
  <div style="flex: 1; text-align: center;">
    <img src="https://raw.githubusercontent.com/gperdrizet/ariel-data-challenge/refs/heads/main/figures/data_preprocessing/02.1-corrected_AIRS_CDS_sample_frames.jpg" alt="Corrected AIRS CDS frames" style="max-width: 100%; height: auto;">
    <p style="margin-top: 10px; font-style: italic;">Corrected AIRS-CH0 frames</p>
  </div>
  <div style="flex: 1; text-align: center;">
    <img src="https://raw.githubusercontent.com/gperdrizet/ariel-data-challenge/refs/heads/main/figures/data_preprocessing/02.1-corrected_FGS1_CDS_sample_frames.jpg" alt="Corrected FGS1 CDS frames" style="max-width: 100%; height: auto;">
    <p style="margin-top: 10px; font-style: italic;">Corrected FGS1 frames</p>
  </div>
</div>

The masked hot/dead pixels appear as dark blobs, but the overall signal is much cleaner and more uniform.

## Transit Detection

The real test: can we still see exoplanet transits after all this processing? 

<div style="display: flex; justify-content: space-around; align-items: flex-start; gap: 20px; margin: 20px 0;">
  <div style="flex: 1; text-align: center;">
    <img src="https://raw.githubusercontent.com/gperdrizet/ariel-data-challenge/refs/heads/main/figures/data_preprocessing/02.1-corrected_AIRS_CDS_transit.jpg" alt="Corrected AIRS transit" style="max-width: 100%; height: auto;">
    <p style="margin-top: 10px; font-style: italic;">AIRS-CH0 total flux over time</p>
  </div>
  <div style="flex: 1; text-align: center;">
    <img src="https://raw.githubusercontent.com/gperdrizet/ariel-data-challenge/refs/heads/main/figures/data_preprocessing/02.1-corrected_FGS_CDS_transit.jpg" alt="Corrected FGS transit" style="max-width: 100%; height: auto;">
    <p style="margin-top: 10px; font-style: italic;">FGS1 total flux over time</p>
  </div>
</div>

Excellent! The transit signal is clearly visible in both instruments after correction. Surprisingly, the AIRS data shows the transit even more clearly than the FGS data - apparently it's a proper science instrument, not just an alignment camera.

## Performance Considerations

The full six-step pipeline works, but it's computationally expensive. Processing one planet takes significant time, and with Kaggle's 4-core limit, we need to think about optimization strategies:

1. **Refactor into a clean module** - Package the preprocessing into a reusable module (maybe even deploy to PyPI for easy installation)
2. **Smart data reduction** - Crop signals and downsample FGS data to match AIRS timing
3. **Parallelize where possible** - Take advantage of multiple cores for batch processing
4. **Order of operations** - Apply data reduction steps early to minimize processing overhead

## Next Steps

With the signal correction pipeline working and transits clearly visible in the processed data, the foundation is solid. The next priorities are:

1. **Optimize the preprocessing workflow** for speed and reliability
2. **Implement intelligent cropping** to focus on the actual signal regions
3. **Develop transit detection algorithms** to automatically identify and extract relevant time windows
4. **Build the spectral extraction pipeline** to turn corrected AIRS data into planetary spectra

Next up, refactor the signal preprocessing pipeline. The next day is going to feel more like engineering than science, but stay tuned - we are making progress!
