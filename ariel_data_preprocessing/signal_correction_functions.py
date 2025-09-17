'''Individual signal correction functions for data preprocessing pipeline'''

# Standard library imports
import itertools

# Third-party imports
import numpy as np
from astropy.stats import sigma_clip


def ADC_convert(signal, gain, offset):
    '''
    Step 1: Convert raw detector counts to physical units.
    
    Applies analog-to-digital conversion correction using gain and offset
    values from the adc_info.csv file.
    
    Args:
        signal (np.ndarray): Raw detector signal
        gain (float): ADC gain factor for conversion
        offset (float): ADC offset value for conversion
        
    Returns:
        np.ndarray: ADC-corrected signal
    '''

    # Convert to float64 for precision
    signal = signal.astype(np.float64)
    signal /= gain    # Apply gain correction
    signal += offset  # Apply offset correction

    return signal


def mask_hot_dead(signal, dead, dark):
    '''
    Step 2: Mask hot and dead pixels in the detector.
    
    Hot pixels are identified using sigma clipping on dark frames.
    Dead pixels are provided in the calibration data.
    
    Args:
        signal (np.ndarray): Input signal array
        dead (np.ndarray): Dead pixel mask from calibration
        dark (np.ndarray): Dark frame for hot pixel detection
        
    Returns:
        np.ma.MaskedArray: Signal with hot/dead pixels masked
    '''

    # Identify hot pixels using 5-sigma clipping on dark frame
    hot = sigma_clip(
        dark, sigma=5, maxiters=5
    ).mask
    
    # Tile masks to match signal dimensions
    hot = np.tile(hot, (signal.shape[0], 1, 1))
    dead = np.tile(dead, (signal.shape[0], 1, 1))
    
    # Apply masks to signal
    signal = np.ma.masked_where(dead, signal)
    signal = np.ma.masked_where(hot, signal)

    return signal


def apply_linear_corr(linear_corr, signal):
    '''
    Step 3: Apply linearity correction to detector response.
    
    Corrects for non-linear detector response using polynomial
    coefficients from calibration data.
    
    Args:
        linear_corr (np.ndarray): Polynomial coefficients for linearity correction
        signal (np.ndarray): Input signal array
        
    Returns:
        np.ndarray: Linearity-corrected signal
    '''

    # Flip coefficients for correct polynomial order
    linear_corr = np.flip(linear_corr, axis=0)

    axis_one = signal.shape[1]
    axis_two = signal.shape[2]
    
    # Apply polynomial correction pixel by pixel
    for x, y in itertools.product(range(axis_one), range(axis_two)):
        poli = np.poly1d(linear_corr[:, x, y])
        signal[:, x, y] = poli(signal[:, x, y])

    return signal


def clean_dark(signal, dead, dark, dt):
    '''
    Step 4: Subtract dark current from signal.
    
    Removes thermal background scaled by integration time.
    
    Args:
        signal (np.ndarray): Input signal array
        dead (np.ndarray): Dead pixel mask
        dark (np.ndarray): Dark frame
        dt (np.ndarray): Integration time for each frame
        
    Returns:
        np.ndarray: Dark-corrected signal
    '''

    # Mask dead pixels in dark frame
    dark = np.ma.masked_where(dead, dark)
    dark = np.tile(dark, (signal.shape[0], 1, 1))

    # Subtract scaled dark current
    signal -= dark * dt[:, np.newaxis, np.newaxis]

    return signal


def get_cds(signal):
    '''
    Step 5: Apply Correlated Double Sampling (CDS).
    
    Subtracts alternating exposure pairs to remove read noise.
    This reduces the number of frames by half.
    
    Args:
        signal (np.ndarray): Input signal array
        
    Returns:
        np.ndarray: CDS-processed signal (half the input frames)
    '''

    # Subtract even frames from odd frames
    cds = signal[1::2,:,:] - signal[::2,:,:]

    return cds


def correct_flat_field(signal, flat, dead):
    '''
    Step 6: Apply flat field correction.
    
    Normalizes pixel-to-pixel sensitivity variations using
    flat field calibration data.
    
    Args:
        signal (np.ndarray): Input signal array
        flat (np.ndarray): Flat field frame
        dead (np.ndarray): Dead pixel mask
        
    Returns:
        np.ndarray: Flat field corrected signal
    '''

    # Transpose flat field to match signal orientation
    signal = signal.transpose(0, 2, 1)
    flat = flat.transpose(1, 0)
    dead = dead.transpose(1, 0)
    
    # Mask dead pixels in flat field
    flat = np.ma.masked_where(dead, flat)
    flat = np.tile(flat, (signal.shape[0], 1, 1))
    
    # Apply flat field correction
    signal = signal / flat

    return signal.transpose(0, 2, 1)


def fgs_downsamples(fgs_frames):
    '''
    Generate downsampling indices for FGS signal to match AIRS cadence.
    
    Creates an index array for downsampling FGS1 data from 135,000 frames
    to match the AIRS-CH0 frame rate. Preserves the correlated double sampling
    (CDS) structure by taking frame pairs at regular intervals.
    
    Downsampling Strategy:
        - Take every 24th frame pair (frames n and n+1)
        - Reduces data volume by ~83% (135k → 22.5k frames)
        - Maintains temporal alignment with AIRS-CH0 observations
        - Preserves CDS structure for proper noise reduction
    
    Returns:
        np.ndarray: Sorted array of frame indices to extract from FGS data
        
    Example:
        For n=24, generates indices: [0, 1, 24, 25, 48, 49, ...]
        This creates pairs for CDS while dramatically reducing data volume.
    '''

    n = 24  # Take 2 elements, skip 20
    indices_to_take = np.arange(0, fgs_frames, n)  # Start from 0, step by n
    indices_to_take = np.concatenate([  # Add the next index
        indices_to_take,
        indices_to_take + 1
    ])

    indices_to_take = np.sort(indices_to_take).astype(int)

    return indices_to_take