---
layout: post
title: "Axis Info & Signal Preprocessing Deep Dive"
---

Time to dive deeper into the data structure and figure out how these instruments actually work. Today's focus: understanding the axis_info metadata and building a proper preprocessing pipeline.

## Understanding the Axis Info

Initially, I was confused about what "axis info" meant - spatial alignment? Star system geometry? Turns out it's much more practical: it's the metadata for each axis of the signal matrices. Here's what we're working with:

1. **AIRS-CH0-axis0-h**: Time index for AIRS readings (hours, ~0.1 second intervals)
2. **AIRS-CH0-axis2-um**: Wavelength across frames in micrometers (~1.6-4.0 μm IR range)
3. **AIRS-CH0-integration_time**: Detector accumulation time in seconds
4. **FGS1-axis0-h**: Time index for FGS1 readings

The timing data reveals the correlated double sampling (CDS) strategy both instruments use:

<p align="center">
  <img src="https://github.com/gperdrizet/ariel-data-challenge/raw/main/figures/EDA/01.5-captures_over_time.jpg" alt="Instrument capture timing">
</p>

You can clearly see the exposure pairs for each instrument. The timing is intricate: AIRS accumulates for 4.5 seconds, reads, then starts the next exposure, while FGS does quick 0.2-second exposures. This whole dance is made confusing by the fact that the reported times represent different phases of the collection cycle for short vs. long exposures.

## The Preprocessing Pipeline

Raw detector data needs significant correction before we can extract meaningful signals. Here's the full six-step pipeline:

1. **Analog-to-Digital Conversion**: Apply gain and offset corrections
2. **Hot/Dead Pixel Masking**: Remove problematic detector elements  
3. **Linearity Correction**: Account for non-linear detector response
4. **Dark Current Subtraction**: Remove thermal background
5. **Correlated Double Sampling (CDS)**: Subtract exposure pairs to remove read noise
6. **Flat Field Correction**: Normalize pixel-to-pixel sensitivity variations

### Before and After: AIRS-CH0 Frames

<div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 10px;">
  <div style="flex: 1;">
    <img src="https://github.com/gperdrizet/ariel-data-challenge/raw/main/figures/EDA/01.5-uncorrected_AIRS_CDS_sample_frames.jpg" alt="Uncorrected AIRS frames" style="width: 100%; max-width: 600px; float: right;">
  </div>
  <div style="flex: 1;">
    <img src="https://github.com/gperdrizet/ariel-data-challenge/raw/main/figures/EDA/01.5-corrected_AIRS_CDS_sample_frames.jpg" alt="Corrected AIRS frames" style="width: 100%; max-width: 600px; float: left;">
  </div>
</div>

<p></p>

Much better! The dark blobs in the corrected frames are masked hot/dead pixels, and the spectral traces are now clearly visible and properly calibrated.

### FGS1 Preprocessing Results

Corrected FGS1 frames:
<p align="center">
  <img src="https://github.com/gperdrizet/ariel-data-challenge/raw/main/figures/EDA/01.5-corrected_FGS1_CDS_sample_frames.jpg" alt="Corrected FGS1 frames">
</p>

## Transit Detection After Preprocessing

The real test: can we still see exoplanet transits after all this processing?

<div style="display: flex; justify-content: space-between; align-items: flex-start; gap: 10px;">
  <div style="flex: 1; text-align: right;">
    <img src="https://github.com/gperdrizet/ariel-data-challenge/raw/main/figures/EDA/01.5-corrected_FGS_CDS_transit.jpg" alt="FGS1 transit detection" style="width: 100%; max-width: 400px;">
  </div>
  <div style="flex: 1; text-align: left;">
    <img src="https://github.com/gperdrizet/ariel-data-challenge/raw/main/figures/EDA/01.5-corrected_AIRS_CDS_transit.jpg" alt="AIRS transit detection" style="width: 100%; max-width: 400px;">
  </div>
</div>
<p></p>

Excellent! The transit is clearly visible in both instruments after preprocessing. Surprisingly, the AIRS data shows an even cleaner transit signal than FGS1 - I guess it really is a precision science instrument, not just an alignment camera.

## Next Steps & Strategy

The good news: we don't need to use FGS data as a proxy to find transits in AIRS data. Both instruments show clear transit signatures.

The challenge: this six-step preprocessing pipeline is computationally expensive, and we'll be limited to 4 cores in the final Kaggle environment.

**Revised plan:**

1. **Refactor preprocessing** into a clean module for the final submission
2. **Downsample FGS data** to match AIRS timing (why process more data than needed?)
3. **Optimize signal extraction** - crop more aggressively around the actual spectral traces
4. **Develop transit windowing** - identify in-transit vs. out-of-transit periods for normalization

The order of operations matters here. Steps 2-4 could significantly reduce data volume and speed up correction preprocessing, but it might be harder to identify signals and transits in uncorrected data. Some experimentation needed to find the optimal workflow.

This preprocessing deep dive has been illuminating - we're definitely dealing with real astronomical data processing challenges, not just a cleaned-up machine learning problem!
