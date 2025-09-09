---
layout: post
title: "Performance Optimization: Making the Pipeline Kaggle-Ready"
---

The signal correction pipeline works beautifully, but there's one small problem: it takes forever to run. With 1100 planets to process and a 9-hour runtime limit on Kaggle submission notebooks, we needed some serious performance optimization. Time to make this thing fast.

## The Performance Challenge

Initial testing revealed a harsh reality: just signal correcting the full dataset would take longer than the competition's 9-hour runtime limit on Kaggle notebooks. We needed two key optimizations:

1. **Data parallelization** - Use multiple CPU cores to process planets simultaneously
2. **FGS downsampling** - Reduce the FGS1 data volume by selecting only frames which match the AIRS-CH0 capture cadence

Both changes target the biggest bottlenecks: computation time and data volume.

## FGS1 Downsampling

The FGS1 guidance camera captures 135,000 frames per planet, while AIRS-CH0 only captures 11,250. We reduce the amount of data significantly by downsampling to match the AIRS-CH0 timing.

The downsampling strategy takes every 24th frame pair from the FGS1 data, reducing the number of frames to ~15% of the raw count, while preserving the correlated double sampling structure. 

<p align="center">
  <img src="https://raw.githubusercontent.com/gperdrizet/ariel-data-challenge/refs/heads/main/figures/signal_correction/02.2-predicted_runtime_vs_planets_downsampling.jpg" alt="Runtime comparison with and without downsampling">
</p>

**Results:** Downsampling cuts predicted runtime almost in half. For 1100 planets:
- **Without downsampling**: ~15 hours (major fail)
- **With downsampling**: ~8 hours (under the time limit, but could be better)

## Multiprocessing: Divide and Conquer

With downsampling we make the time cutoff, but it would be great to have more time left over after signal correction. Time to parallelize across multiple CPU cores. The pipeline now uses multiprocessing to handle multiple planets simultaneously, with separate worker processes for signal correction and a dedicated process for saving results.

<p align="center">
  <img src="https://raw.githubusercontent.com/gperdrizet/ariel-data-challenge/refs/heads/main/figures/signal_correction/02.2-predicted_runtime_vs_planets_cpu_count.jpg" alt="Runtime scaling with CPU count">
</p>

**Results:** The speedup is significant but shows diminishing returns:
- **1 CPU**: 8.3 hours (downsampled baseline)
- **2 CPUs**: 4.1 hours (2.0x speedup)
- **3 CPUs**: 2.6 hours (3.2x speedup)
- **4 CPUs**: 2.0 hours (4.1x speedup)

## Memory Usage: The Hidden Cost

Parallelization comes with a memory cost. More worker processes mean more data in memory. Let's also make sure we can run all 4 avalible cores in a free tier Kaggle notebook simultaneously and not run out of memory:

<p align="center">
  <img src="https://raw.githubusercontent.com/gperdrizet/ariel-data-challenge/refs/heads/main/figures/signal_correction/02.2-memory_vs_cpu_count.jpg" alt="Memory usage scaling">
</p>

**Memory scaling for 32 planets:**
- **1 CPU**: 9.1 GB
- **2 CPUs**: 12.2 GB  
- **3 CPUs**: 16.8 GB
- **4 CPUs**: 22.3 GB

The memory usage grows roughly linearly with CPU count at ~4.4 GB per additional core, up to ~20 GB with 4 cores. Also, the memory footprint is independent of how many planets we process. This makes sense, since each worker processes one planet at a time, the total memory should only depend on the number of workers. Conclusion: were safe - the data will fit in a Kaggle notebook's memory, even with all avalible cores pegged.

## The Sweet Spot: 3-4 CPUs

The performance analysis reveals the optimal configuration:

- **3 CPUs**: 2.6 hours, ~13 GB memory (safe choice)
- **4 CPUs**: 2.0 hours, ~20 GB memory (maximum performance)

Both configurations beat the 9-hour limit without exceeding the memory avalible in a Kaggle notebook. The 4-CPU setup gives the best performance with a comfortable 5.8-hour margin for safety. But, I think three cores is going to be the sweet spot. These tests were run on a machine with more than 4 cores, while the Kaggle notebook environment has ONLY 4. Trying to run signal correction workers on all 4 cores will leave none for the output worker - to say nothing of the host OS. Either way, we can comfortably preprocess the data within the time limit.

## Implementation Details

The optimized pipeline uses:
- **Process pools** for planet-level parallelization
- **Shared queues** for inter-process communication  
- **Memory monitoring** to track resource usage
- **Smart downsampling** that preserves CDS structure

All of this is now packaged and available via `pip install ariel-data-challenge`, making it easy to deploy on Kaggle or any other platform.

## Result summary

What started as a 15-hour processing job is now down to about 2 hours with 4 CPUs - a **7.5x speedup** at the cost of a less than 2x increase in total memory footprint.

Next up, let's try and reduce the size of the data and the noise further by extracting just the signal from each frame.
