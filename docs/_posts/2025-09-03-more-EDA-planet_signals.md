---
layout: post
title: "AIRS-CH0 & FGS1 signal EDA"
---

Next up - let's take a look at some of the signal data from both instruments: FGS1 (the guidance/alignment camera) and AIRS-CH0 (the IR spectrometer),

## AIRS-CH0 signal

Here are some example frames from AIRS-CH0 for a random planet:

<p align="center">
  <img src="https://raw.githubusercontent.com/gperdrizet/ariel-data-challenge/refs/heads/main/figures/EDA/01.3-AIRS_sample_frames.jpg" alt="AIRS-CH0 sample frames">
</p>

I think I'm starting to get it. Those smears are spectra (duh) - Ariel must put the signal through a grism or prism or something. Nice hint from the competition organizers about cropping the frames with `[:, :, 39:321]`. Now we have a 1 to 1 correspondence between the pixels across the observed spectra and the wavelengths (though let's not assume no overlap). One **more** hint the organizers dropped in the [preprocessing notebook](https://www.kaggle.com/code/gordonyip/calibrating-and-binning-ariel-data): '*Data reduction is crucial in astronomical observations*'. We should be able to do much better than just cropping off the right hand part of each frame. I bet we can isolate just the 'strip' that contains the actual data. I also now understand why they included the guidance camera frames - it will probably make it much easier to spot exoplanet transits than trying to use these IR spectra smears.

### FGS1 signal

<p align="center">
  <img src="https://raw.githubusercontent.com/gperdrizet/ariel-data-challenge/refs/heads/main/figures/EDA/01.4-FGS1_sample_frames.jpg" alt="FGS1 sample frames">
</p>

Yep - just what I thought. These are images, not spectrograms. The FGS1 frames are much smaller and could be cropped even smaller. We can use the total brightness to see the transits. In fact, I bet they jump right out...

<p align="center">
  <img src="https://raw.githubusercontent.com/gperdrizet/ariel-data-challenge/refs/heads/main/figures/EDA/01.4-example_raw_transit.jpg" alt="Example exoplanet transit in total flux data">
</p>

Bingo - definitely need some calibration, etc., but the transit is obvious. We can probably dump a lot of data here too. I think the two traces are probably the alternating detector read sampling strategy (correlated double sampling - CDS) mentioned in the data page on the [competition data page](https://www.kaggle.com/competitions/ariel-data-challenge-2025/data). We could probably just take the high signal trace. That by itself will cut the data in half. Then, since we are only interested in the spectrum of the planet we can take pieces of the time series. Maybe just the bottom of the well and one of the limbs. Use those parts of the time series for signal and background.

<p align="center">
  <img src="https://raw.githubusercontent.com/gperdrizet/ariel-data-challenge/refs/heads/main/figures/EDA/01.4-transit_high_trace.jpg" alt="Example exoplanet transit in total flux data, high signal trace only">
</p>

Wow, this is a cool problem. One last thing before we call it a night - can we see the transit visually?

<p align="center">
  <img src="https://raw.githubusercontent.com/gperdrizet/ariel-data-challenge/refs/heads/main/figures/EDA/01.4-example_transit_frames.jpg" alt="Selected frames from exoplanet transit">
</p>

Nope, I can't convince myself that I see a transit in the images. But the total flux plot is pretty unambiguous. Still not sure how exactly to get the uncertainties - some kind of bootstrapping probably.

Here's the tentative plan:

1. Figure out how to to efficiently spot transits in the FGS data.
2. Use transit info from FGS data to isolate transit spectral data from AIRS.
3. Use some kind of bootstrapping or ensemble method to learn the spectrum and get uncertainties.

I think the next thing to focus on is figuring out how to/if we can align the FGS1 and AIRS signals in time. The time series have different numbers of frames, but I think there is probably going to be some metadata we can use in the ancillary data files. Tomorrow.