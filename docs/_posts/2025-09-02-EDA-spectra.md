---
layout: post
title: "Project Introduction & Initial EDA"
---

Welcome to my exploration of the [Ariel Data Challenge 2025](https://www.kaggle.com/competitions/ariel-data-challenge-2025)! This Kaggle competition presents a fascinating problem: extracting planetary atmospheric spectra from simulated space telescope observations.

## The Challenge

The Ariel space mission, scheduled to launch in 2029, will study the atmospheres of approximately 1000 exoplanets through transit spectroscopy. This competition gives us a taste of what that data analysis will look like, complete with realistic instrumental noise, calibration challenges, and systematic effects.

## Initial Data Exploration

I've started with an exploratory data analysis of the ground truth spectra to understand the dataset structure and characteristics. One of the key insights comes from examining how flux values are distributed across different wavelengths.

<p align="center">
  <img src="https://raw.githubusercontent.com/gperdrizet/ariel-data-challenge/refs/heads/main/figures/EDA/01.2-flux_distribution_by_wavelength.jpg" alt="Planet-Standardized Flux Distribution Heatmap">
</p>

This heatmap shows the distribution of planet-standardized flux values across wavelengths. Each planet's spectrum has been standardized using its own mean and standard deviation, revealing the underlying spectral patterns independent of individual planet brightness levels. The x-axis shows wavelength indices, while the y-axis represents standardized flux values (Z-scores).

<p align="center">
  <img src="https://raw.githubusercontent.com/gperdrizet/ariel-data-challenge/refs/heads/main/figures/EDA/01.2-hierarchical_clustered_spectra.jpg" alt="Hierarchically Clustered Spectra">
</p>

This visualization shows all 1100 planets ordered by hierarchical clustering based on their standardized spectral signatures. Each row represents a planet, and each column represents a wavelength. The planets have been grouped by spectral similarity using Ward linkage clustering, revealing distinct groups with similar atmospheric characteristics.

## Next Steps

The initial EDA has revealed interesting spectral features and distribution patterns; as you can see from the figures, there is a lot going on! Moving forward, I'll be diving into:

- Analysis of the raw detector images from both FGS1 (guidance camera) and AIRS-CH0 (science instrument)
- Transit detection algorithms
- Spectral extraction techniques
- Machine learning approaches for robust spectrum recovery

Stay tuned for more updates as I work through this fascinating intersection of astrophysics and data science!
