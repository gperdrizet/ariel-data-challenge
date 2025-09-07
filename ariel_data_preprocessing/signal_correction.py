'''Signal correction pipeline for Ariel Data Challenge

This module implements the complete preprocessing pipeline for Ariel telescope data,
including ADC conversion, pixel masking, linearity correction, dark current subtraction,
correlated double sampling (CDS), and flat field correction.
'''

# Standard library imports
from fileinput import filename
import itertools
import os

# Third party imports
import h5py
import numpy as np
import pandas as pd

from astropy.stats import sigma_clip


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
            adc_conversion: bool = True,
            masking: bool = True,
            linearity_correction: bool = True,
            dark_subtraction: bool = True,
            cds_subtraction: bool = True,
            flat_field_correction: bool = True,
            cut_inf: int = 39,
            cut_sup: int = 321,
            gain: float = 0.4369,
            offset: float = -1000.0,
            n_cpus: int = 1,
            n_planets: int = -1
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
        self.adc_conversion = adc_conversion
        self.masking = masking
        self.linearity_correction = linearity_correction
        self.dark_subtraction = dark_subtraction
        self.cds_subtraction = cds_subtraction
        self.flat_field_correction = flat_field_correction
        self.gain = gain
        self.offset = offset
        self.cut_inf = cut_inf
        self.cut_sup = cut_sup
        self.n_cpus = n_cpus
        self.n_planets = n_planets

        # Make sure output directory exists
        os.makedirs(self.output_data_path, exist_ok=True)

        # Remove hdf5 files from previous runs
        filename = (f'{self.output_data_path}/train.h5')
        
        try:
            os.remove(filename)

        except OSError:
            pass

        # Get planet list from input data
        self.planet_list = self._get_planet_list()

        if self.n_planets != -1:
            self.planet_list = self.planet_list[:self.n_planets]

        print('SignalCorrection initialized.')


    def run(self):
        '''
        Run the complete signal correction pipeline.
        
        This method orchestrates the entire preprocessing sequence,
        applying each correction step in order.
        '''

        
        for planet in self.planet_list:

            # Get path to this planet's data
            planet_path = f'{self.input_data_path}/train/{planet}'

            # Load and prep the raw data
            fgs_signal = pd.read_parquet(
                f'{planet_path}/FGS1_signal_0.parquet'
            ).to_numpy().reshape(4, 32, 32)
            
            airs_signal = pd.read_parquet(
                f'{planet_path}/AIRS-CH0_signal_0.parquet'
            ).to_numpy().reshape(4, 32, 356)[:, :, self.cut_inf:self.cut_sup]

            # Load and prep calibration data
            calibration_data = CalibrationData(
                input_data_path=self.input_data_path,
                planet_path=planet_path,
                fgs_signal=fgs_signal,
                cut_inf=self.cut_inf,
                cut_sup=self.cut_sup
            )

            # Step 1: ADC conversion
            if self.adc_conversion:
                airs_signal = self._ADC_convert(airs_signal)
                fgs_signal = self._ADC_convert(fgs_signal)

            # Step 2: Mask hot/dead pixels
            if self.masking:
                airs_signal = self._mask_hot_dead(
                    airs_signal,
                    calibration_data.dead_airs,
                    calibration_data.dark_airs
                )

                fgs_signal = self._mask_hot_dead(
                    fgs_signal,
                    calibration_data.dead_fgs,
                    calibration_data.dark_fgs
                )

            # Step 3: Linearity correction
            if self.linearity_correction:
                airs_signal = self._apply_linear_corr(
                    calibration_data.linear_corr_airs,
                    airs_signal
                )

                fgs_signal = self._apply_linear_corr(
                    calibration_data.linear_corr_fgs,
                    fgs_signal
                )

            # Step 4: Dark current subtraction
            if self.dark_subtraction:
                airs_signal = self._clean_dark(
                    airs_signal,
                    calibration_data.dead_airs,
                    calibration_data.dark_airs,
                    calibration_data.dt_airs
                )

                fgs_signal = self._clean_dark(
                    fgs_signal,
                    calibration_data.dead_fgs,
                    calibration_data.dark_fgs,
                    calibration_data.dt_fgs
                )

            # Step 5: Correlated Double Sampling (CDS)
            if self.cds_subtraction:
                airs_signal = self._get_cds(airs_signal)
                fgs_signal = self._get_cds(fgs_signal) 

            # Step 6: Flat field correction
            if self.flat_field_correction:
                airs_signal = self._correct_flat_field(
                    airs_signal,
                    calibration_data.flat_airs,
                    calibration_data.dead_airs
                )

                fgs_signal = self._correct_flat_field(
                    fgs_signal,
                    calibration_data.flat_fgs,
                    calibration_data.dead_fgs
                )

            # Save the corrected data
            self._save_corrected_data(
                planet,
                airs_signal,
                fgs_signal
            )


    def _ADC_convert(self, signal):
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
        signal /= self.gain    # Apply gain correction
        signal += self.offset  # Apply offset correction

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


    def _get_planet_list(self):
        '''
        Retrieve list of unique planet IDs from input data.
        
        Scans the input data directory to identify all unique
        planet identifiers for processing.
        
        Returns:
            list: List of unique planet IDs
        '''

        planets = list(os.listdir(f'{self.input_data_path}/train'))

        return [planet_path.split('/')[-1] for planet_path in planets]


    def _save_corrected_data(self, planet, airs_signal, fgs_signal):
        '''
        Save corrected data to output directory.
        
        Writes the processed AIRS-CH0 and FGS1 signals to
        parquet files in the specified output path.
        
        Args:
            planet (str): Planet ID
            airs_signal (np.ndarray): Corrected AIRS-CH0 signal
            fgs_signal (np.ndarray): Corrected FGS1 signal
        '''
        
        # File path for hdf5 output
        output_file = (f'{self.output_data_path}/train.h5')

        with h5py.File(output_file, 'a') as hdf:

            # Create groups for this planet if not existing
            planet_group = hdf.require_group(planet)

            # Create datasets for AIRS-CH0 and FGS1 signals
            _ = planet_group.create_dataset('AIRS-CH0_signal', data=airs_signal)
            _ = planet_group.create_dataset('FGS1_signal', data=fgs_signal)

            # Save the corrected signals
            planet_group['AIRS-CH0_signal'][:] = airs_signal
            planet_group['FGS1_signal'][:] = fgs_signal

        return True


class CalibrationData:
    '''
    Class to load and store calibration data for signal correction.
    
    This class reads all necessary calibration files for a given planet
    and stores them as attributes for use in the SignalCorrection pipeline.
    '''

    def __init__(
            self,
            input_data_path: str,
            planet_path: str,
            fgs_signal: np.ndarray,
            cut_inf: int,
            cut_sup: int
        ):
        '''
        Initialize CalibrationData by loading calibration files.
        
        Args:
            planet_path (str): Path to the planet's data directory
            cut_inf (int): Lower wavelength cut index for AIRS-CH0
            cut_sup (int): Upper wavelength cut index for AIRS-CH0
        '''
    
        # Load and prep calibration data
        self.dark_airs = pd.read_parquet(
            f'{planet_path}/AIRS-CH0_calibration_0/dark.parquet'
        ).values.astype(np.float64).reshape((32, 356))[:, cut_inf:cut_sup]
        self.dead_airs = pd.read_parquet(
            f'{planet_path}/AIRS-CH0_calibration_0/dead.parquet'
        ).values.astype(np.float64).reshape((32, 356))[:, cut_inf:cut_sup]

        self.dark_fgs = pd.read_parquet(
            f'{planet_path}/FGS1_calibration_0/dark.parquet'
        ).values.astype(np.float64).reshape((32, 32))
        self.dead_fgs = pd.read_parquet(
            f'{planet_path}/FGS1_calibration_0/dead.parquet'
        ).values.astype(np.float64).reshape((32, 32))

        self.linear_corr_airs = pd.read_parquet(
            f'{planet_path}/AIRS-CH0_calibration_0/linear_corr.parquet'
        ).values.astype(np.float64).reshape((6, 32, 356))[:, :, cut_inf:cut_sup]
        self.linear_corr_fgs = pd.read_parquet(
            f'{planet_path}/FGS1_calibration_0/linear_corr.parquet'
        ).values.astype(np.float64).reshape((6, 32, 32))

        self.flat_airs = pd.read_parquet(
            f'{planet_path}/AIRS-CH0_calibration_0/flat.parquet'
        ).values.astype(np.float64).reshape((32, 356))[:, cut_inf:cut_sup]
        self.flat_fgs = pd.read_parquet(
            f'{planet_path}/FGS1_calibration_0/flat.parquet'
        ).values.astype(np.float64).reshape((32, 32))

        self.axis_info = pd.read_parquet(f'{input_data_path}/axis_info.parquet')

        self.dt_airs = self.axis_info['AIRS-CH0-integration_time'].dropna().values[:4]
        self.dt_airs[1::2] += 0.1 # Why are we adding here - I don't think that is right...

        self.dt_fgs = np.ones(len(fgs_signal)) * 0.1
        self.dt_fgs[1::2] += 0.1 # This one looks more correct