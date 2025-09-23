'''Functions to isolate transit spectra and generate spectral difference pairs.'''

# Standard library imports
import random

# Third-party imports
import numpy as np
from scipy.ndimage import gaussian_filter1d


def spectral_difference_pairs(signal: np.ndarray, pairs_per_planet: int) -> np.ndarray:
    '''Main function to take a 2D array of shape (timepoints, wavelengths), 
    identify transit and non-transit spectra, and generate spectral difference pairs
    by subtracting each non-transit spectrum from each transit spectrum.'''


    transit_signal, non_transit_signal = extract_transit_regions(signal)
    difference_pairs = _generate_spectrum_subtraction_dataset(
        transit_signal,
        non_transit_signal,
        pairs_per_planet
    )

    return difference_pairs


def extract_transit_regions(signal):
    """
    Extract transit and non-transit regions from a signal array.
    
    Parameters:
    - signal: 2D array of shape (time_points, wavelengths)
    
    Returns:
    - transit_signal: Array containing transit frames
    - non_transit_signal: Array containing non-transit frames
    - metadata: Dictionary with detection information
    """
    
    # Calculate total signal across wavelengths
    total_signal = np.sum(signal, axis=1)
    
    # Derivative-based boundary detection
    slope = np.gradient(total_signal)
    smoothed_slope = gaussian_filter1d(slope, sigma=30)
    slope_change = np.gradient(smoothed_slope)
    smoothed_slope_change = gaussian_filter1d(slope_change, sigma=30)
    slope_change_threshold = np.std(smoothed_slope_change) * 2
    transit_boundaries = np.where(smoothed_slope_change < -slope_change_threshold)[0]
    
    # # Initialize metadata
    # metadata = {
    #     'boundaries': transit_boundaries,
    #     'total_signal': total_signal,
    #     'boundary_point': None,
    #     'transit_type': None,
    #     'left_mean': None,
    #     'right_mean': None
    # }
    
    if len(transit_boundaries) == 0:

        # No boundaries detected
        return np.array([]), np.array([])
    
    # Calculate boundary differences
    boundary_diffs = np.diff(transit_boundaries) if len(transit_boundaries) > 1 else [0]
    max_boundary_diff = max(boundary_diffs) if len(boundary_diffs) > 0 else 0
    
    if max_boundary_diff > 20:
        # Clean transit - both start and end detected
        # Use first and last boundary indices for transit region
        transit_start = transit_boundaries[0]
        transit_end = transit_boundaries[-1]
        
        transit_signal = signal[transit_start:transit_end]
        non_transit_signal = np.vstack([
            signal[:transit_start],
            signal[transit_end:]
        ])
        
        # metadata.update({
        #     'transit_type': 'Clean transit',
        #     'boundary_point': (transit_start, transit_end),
        #     'left_mean': np.mean(total_signal[:transit_start]),
        #     'right_mean': np.mean(total_signal[transit_end:])
        # })
        
    else:
        # Incomplete transit - missing start or end
        # Determine which side has lower signal to identify transit region
        midpoint = len(total_signal) // 2
        left_half_mean = np.mean(total_signal[:midpoint])
        right_half_mean = np.mean(total_signal[midpoint:])
        
        # Select appropriate boundary based on where transit is located
        if left_half_mean < right_half_mean:
            # Transit is on left - use last boundary index
            boundary_point = transit_boundaries[-1]
            transit_signal = signal[:boundary_point]
            non_transit_signal = signal[boundary_point:]
            # transit_type = "Entry missing - transit on left"
        else:
            # Transit is on right - use first boundary index  
            boundary_point = transit_boundaries[0]
            transit_signal = signal[boundary_point:]
            non_transit_signal = signal[:boundary_point]
            # transit_type = "Exit missing - transit on right"
        
        # # Calculate final region means
        # left_mean = np.mean(total_signal[:boundary_point])
        # right_mean = np.mean(total_signal[boundary_point:])
        
        # metadata.update({
        #     'transit_type': transit_type,
        #     'boundary_point': boundary_point,
        #     'left_mean': left_mean,
        #     'right_mean': right_mean
        # })
    
    return transit_signal, non_transit_signal, #metadata


def _generate_spectrum_subtraction_dataset(
        transit_signal: np.ndarray,
        non_transit_signal: np.ndarray,
        pairs_per_planet: int
):
    """
    Generate final dataset by subtracting each non-transit spectrum from each transit spectrum.
    
    Parameters:
    - transit_signal: Array of shape (n_transit_frames, n_wavelengths)
    - non_transit_signal: Array of shape (n_non_transit_frames, n_wavelengths)
    
    Returns:
    - dataset: Array of shape (n_transit_frames * n_non_transit_frames, n_wavelengths)
               Each row represents one transit spectrum minus one non-transit spectrum
    - metadata: Dictionary containing information about the dataset generation
    """
    
    # Check if we have both transit and non-transit signals
    if transit_signal.size == 0:
        # print("Warning: No transit signal available")
        return np.array([])
    
    if non_transit_signal.size == 0:
        # print("Warning: No non-transit signal available")
        return np.array([])
    
    samples = pairs_per_planet ** 0.5
    
    if samples % 1 != 0:
        raise ValueError('Pairs per planet must be a perfect square')

    
    # Get dimensions
    n_wavelengths = transit_signal.shape[1]
    n_non_transit_frames = non_transit_signal.shape[0]
    n_transit_frames = transit_signal.shape[0]

    non_transit_indices = random.sample(range(0, n_non_transit_frames), int(samples))
    transit_indices = random.sample(range(0, n_transit_frames), int(samples))

    # Randomly sample non-transit indices
    sampled_non_transit = non_transit_signal[non_transit_indices]
    sampled_transit = transit_signal[transit_indices]
    
    # print(f"Transit frames: {n_transit_frames}")
    # print(f"Non-transit frames: {n_non_transit_frames}")
    # print(f"Wavelength channels: {n_wavelengths}")
    # print(f"Total combinations: {n_transit_frames * n_non_transit_frames}")
    
    # Use broadcasting to perform all subtractions at once
    # Reshape transit_signal to (n_transit_frames, 1, n_wavelengths)
    # Reshape non_transit_signal to (1, n_non_transit_frames, n_wavelengths)
    # Broadcasting will create (n_transit_frames, n_non_transit_frames, n_wavelengths)
    transit_expanded = sampled_transit[:, np.newaxis, :]  # Shape: (n_transit, 1, n_wavelengths)
    non_transit_expanded = sampled_non_transit[np.newaxis, :, :]  # Shape: (1, n_non_transit, n_wavelengths)
    
    # Subtract non-transit from transit using broadcasting
    subtracted_spectra = transit_expanded - non_transit_expanded  # Shape: (n_transit, n_non_transit, n_wavelengths)
    
    # Reshape to final format: (n_transit * n_non_transit, n_wavelengths)
    dataset = subtracted_spectra.reshape(-1, n_wavelengths)
    
    # # Create metadata
    # metadata = {
    #     'n_transit_frames': n_transit_frames,
    #     'n_non_transit_frames': n_non_transit_frames,
    #     'n_wavelengths': n_wavelengths,
    #     'total_spectra': dataset.shape[0],
    #     'transit_indices': np.repeat(np.arange(n_transit_frames), n_non_transit_frames),
    #     'non_transit_indices': np.tile(np.arange(n_non_transit_frames), n_transit_frames)
    # }
    
    # print(f"Generated dataset shape: {dataset.shape}")
    
    return dataset #, metadata