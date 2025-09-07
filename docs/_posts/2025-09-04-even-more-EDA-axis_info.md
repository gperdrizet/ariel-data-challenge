---
layout: post
title: "Understanding Timing and CDS: Making Sense of the Axis Info"
---

Time to dig into the timing metadata and figure out how these instruments actually work together. The axis info data turned out to be much more useful than I initially thought - it's not about satellite alignment at all, but about the structure of the signal matrices themselves.

## What's in the Axis Info?

The axis info metadata gives us the key to understanding how the telescope data is organized:

1. **AIRS-CH0-axis0-h**: Time index for AIRS readings (in hours, ~0.1 second resolution)
2. **AIRS-CH0-axis2-um**: Wavelength across frames in micrometers (~1.6 μm range, high-energy IR)
3. **AIRS-CH0-integration_time**: Detector accumulation time in seconds
4. **FGS1-axis0-h**: Time index for FGS1 readings

Converting to seconds makes the timing much clearer - we can see exactly how the two instruments are synchronized.

## Exposure Timing Patterns

Plotting the capture timing reveals the correlated double sampling (CDS) strategy:

<p align="center">
  <img src="https://raw.githubusercontent.com/gperdrizet/ariel-data-challenge/refs/heads/main/figures/EDA/01.5-captures_over_time.jpg" alt="Instrument captures over time">
</p>

The pattern is clear once you know what to look for. Each instrument takes paired exposures:
- **FGS1**: Short (0.1s) + long (0.2s) exposures every ~0.5 seconds
- **AIRS-CH0**: Short (0.1s) + long (4.5s) exposures every ~5 seconds

The timing is deliberately offset so that the "short" exposure ends just before the "long" exposure begins. This CDS approach helps reduce read noise by subtracting the paired frames.

## Implementing CDS

The key insight from the timing analysis: subtract the second (long) exposure from the first (short) exposure to get the cleaned signal. This halves the number of frames but should significantly reduce noise.

Testing this on a sample planet:

<div style="display: flex; justify-content: space-around; align-items: flex-start; gap: 20px; margin: 20px 0;">
  <div style="flex: 1; text-align: center;">
    <img src="https://raw.githubusercontent.com/gperdrizet/ariel-data-challenge/refs/heads/main/figures/EDA/01.5-uncorrected_AIRS_CDS_sample_frames.jpg" alt="AIRS CDS sample frames" style="max-width: 100%; height: auto;">
    <p style="margin-top: 10px; font-style: italic;">AIRS-CH0 CDS frames</p>
  </div>
  <div style="flex: 1; text-align: center;">
    <img src="https://raw.githubusercontent.com/gperdrizet/ariel-data-challenge/refs/heads/main/figures/EDA/01.5-uncorrected_FGS1_CDS_sample_frames.jpg" alt="FGS1 CDS sample frames" style="max-width: 100%; height: auto;">
    <p style="margin-top: 10px; font-style: italic;">FGS1 CDS frames</p>
  </div>
</div>

The CDS processing works - we get positive signal values and the frame count is exactly halved. However, the raw CDS frames still show significant background noise and artifacts. This makes sense because CDS is just one step in the full calibration pipeline.

## The Full Calibration Picture

The analysis confirms what the competition organizers hinted at: proper data reduction is crucial. The complete preprocessing pipeline needs to handle:

1. **Analog-to-Digital Conversion** - Convert raw counts to physical units
2. **Hot/Dead Pixel Masking** - Remove problematic detector pixels  
3. **Linearity Correction** - Account for non-linear detector response
4. **Dark Current Subtraction** - Remove thermal background
5. **Correlated Double Sampling** - Reduce read noise (what we just implemented)
6. **Flat Field Correction** - Normalize pixel-to-pixel sensitivity

CDS alone isn't enough - we need the full treatment to get clean, calibrated data suitable for exoplanet analysis.

## Next Steps

With the timing structure decoded and CDS implemented, the next priority is building out the complete signal correction pipeline. The raw data is definitely there, but it needs serious cleanup before we can reliably extract planetary spectra.

The good news: the instruments are well-synchronized and the CDS strategy is working as designed. The challenge now is implementing the remaining calibration steps to turn these noisy detector outputs into science-ready data.
