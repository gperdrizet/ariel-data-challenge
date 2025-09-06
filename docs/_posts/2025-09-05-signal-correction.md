---
layout: post
title: "Signal Correction Module: Cleaning Up the Data"
---

Next up in the pipeline: making sure our raw telescope data is actually usable. The signal correction module is now in place, and it's doing the heavy lifting to turn noisy detector outputs into something we can analyze without losing sleep (or at least, not as much sleep).

## What Does the Signal Correction Module Do?

This module implements the full six-step preprocessing pipeline for Ariel telescope data:

1. **Analog-to-Digital Conversion (ADC)** – Converts raw detector counts to physical units using gain and offset corrections.
2. **Hot/Dead Pixel Masking** – Identifies and masks problematic pixels using calibration data and sigma clipping.
3. **Linearity Correction** – Applies polynomial corrections to account for non-linear detector response.
4. **Dark Current Subtraction** – Removes thermal background noise, scaled by integration time.
5. **Correlated Double Sampling (CDS)** – Subtracts paired exposures to reduce read noise and halve the number of frames.
6. **Flat Field Correction** – Normalizes pixel-to-pixel sensitivity using flat field calibration.

Each step is implemented as a method in the `SignalCorrection` class, with clear docstrings and comments to make the code readable (for future-me, who will inevitably forget how any of this works).

## Automated CI/CD: Testing and Deployment

To keep the codebase reliable and up-to-date, we've set up automated CI/CD using GitHub workflows. Every time changes are pushed, the following steps are triggered:

1. **Unit Testing** – All unittests (including those for the signal correction module) are automatically run to catch bugs early and ensure new changes don't break existing functionality.
2. **Continuous Integration** – The code is checked for style, linting, and compatibility across Python versions.
3. **Deployment to PyPI** – When a new release is tagged, the module is packaged and published to PyPI, making it easy to install and use in other projects with a simple `pip install ariel-data-challenge`.

This automated pipeline helps maintain code quality, reproducibility, and makes sharing updates with the community seamless. For more details, see the workflow files in `.github/workflows/`.

## Why Bother?

Raw space telescope data is messy. If you want to spot exoplanet transits or extract meaningful spectra, you need to clean up everything from dead pixels to non-linearities and background noise. This module makes sure that by the time we get to the analysis and machine learning stages, we're working with data that's as clean and calibrated as possible.

## Next Steps

With the signal correction pipeline in place, the next job is to integrate it into the full data processing workflow and start running it on real (well, simulated) planetary observations. If all goes well, the rest of the analysis should be a lot less painful.

Stay tuned for more updates as the pipeline comes together and the project starts to feel less like engineering and more like science!