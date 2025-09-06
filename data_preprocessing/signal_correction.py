'''Signal correction pipeline for Ariel Data Challenge

This module implements the complete preprocessing pipeline for Ariel telescope data,
including ADC conversion, pixel masking, linearity correction, dark current subtraction,
correlated double sampling (CDS), and flat field correction.
'''

# Standard library imports
import itertools

# Third party imports
from astropy.stats import sigma_clip
import numpy as np


class SignalCorrection:
    '''
    Class to handle signal correction for Ariel Data Challenge.
    
    Implements the complete 6-step preprocessing pipeline:
    1. Analog-to-Digital Conversion (ADC)
    2. Hot/dead pixel masking
    3. Linearity correction
    4. Dark current subtraction
    5. Correlated Double Sampling (CDS)
    6. Flat field correction
    '''

    def __init__(
            self,
            input_data_path: str = None,
            output_data_path: str = None,
            gain: float = 0.4369,
            offset: float = -1000.0,
            n_cpus: int = 1,
    ):
        '''
        Initialize the SignalCorrection class.
        
        Args:
            input_data_path (str): Path to input data directory
            output_data_path (str): Path to output data directory
            gain (float): ADC gain factor (default: 0.4369)
            offset (float): ADC offset value (default: -1000.0)
            n_cpus (int): Number of CPUs for parallel processing (default: 1)
            
        Raises:
            ValueError: If input or output data paths are not provided
        '''
        
        if input_data_path is None or output_data_path is None:
            raise ValueError("Input and output data paths must be provided.")
        
        self.input_data_path = input_data_path
        self.output_data_path = output_data_path
        self.gain = gain
        self.offset = offset
        self.n_cpus = n_cpus


    def _ADC_convert(self, signal, gain, offset):
        '''
        Step 1: Convert raw detector counts to physical units.
        
        Applies analog-to-digital conversion correction using gain and offset
        values from the adc_info.csv file.
        
        Args:
            signal (np.ndarray): Raw detector signal
            
        Returns:
            np.ndarray: ADC-corrected signal
        '''
        signal = signal.astype(np.float64)
        signal /= gain    # Apply gain correction
        signal += offset  # Apply offset correction

        return signal


    def _mask_hot_dead(self, signal, dead, dark):
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
    

    def _apply_linear_corr(self, linear_corr, signal):
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
    

    def _clean_dark(self, signal, dead, dark, dt):
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
    

    def _get_cds(self, signal):
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


    def _correct_flat_field(self, signal, flat, dead):
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