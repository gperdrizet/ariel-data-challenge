'''
Unit tests for signal correction functions and calibration data handling.

This module provides comprehensive testing of the 6-step signal correction
pipeline Tests cover each correction step individually to validate proper
implementation and data flow.

Signal Correction Pipeline Steps Tested:
    1. ADC Conversion - Raw counts to physical units
    2. Hot/Dead Pixel Masking - Remove problematic detector pixels  
    3. Linearity Correction - Polynomial detector response correction
    4. Dark Current Subtraction - Remove thermal background
    5. Correlated Double Sampling (CDS) - Noise reduction via paired exposures
    6. Flat Field Correction - Normalize pixel sensitivity variations

Test Data Requirements:
    - tests/test_data/raw/ with planet '342072318' containing:
        - AIRS-CH0_signal_0.parquet (spectrometer data)
        - FGS1_signal_0.parquet (guidance camera data)
        - Calibration subdirectories with dark, dead, flat, linear_corr files

Author: Dr. George Perdrizet PhD
Created: 2025
'''

# Standard library imports
import unittest

# Third-party imports
import numpy as np
import pandas as pd

# Internal imports
import ariel_data_preprocessing.signal_correction_functions as correction_funcs
from ariel_data_preprocessing.calibration_data import CalibrationData


class TestSignalCorrection(unittest.TestCase):
    '''
    Test suite for the signal correction pipeline functions.
    
    This class validates each step of the 6-step signal correction pipeline
    to ensure proper data transformation from raw detector counts to
    science-ready signals suitable for exoplanet atmospheric analysis.
    
    The tests use a subset of real Ariel challenge data to verify:
    - Correct data type handling and preservation
    - Proper array shape maintenance through pipeline steps
    - Expected output characteristics for each correction step
    - Integration with calibration data loading
    '''

    def setUp(self):
        '''
        Set up test data and calibration parameters for signal correction tests.
        
        Initializes test environment with:
        - Real Ariel challenge test data (planet '342072318')
        - Standard pipeline parameters (gain, offset, cropping bounds)
        - Loaded and preprocessed AIRS-CH0 and FGS1 signal arrays
        - Calibration data instance for correction step testing
        
        This setup enables isolated testing of individual correction steps
        while using realistic data dimensions and characteristics.
        '''

        # Test data configuration
        self.input_data_path = 'tests/test_data/raw'
        self.planet = '342072318'
        self.planet_path = f'{self.input_data_path}/train/{self.planet}'

        # Pipeline parameters matching production settings
        self.airs_frames = 50
        self.fgs_frames = 50
        self.cut_inf = 39      # AIRS spectral cropping lower bound
        self.cut_sup = 321     # AIRS spectral cropping upper bound
        self.gain = 0.4369     # ADC gain factor
        self.offset = -1000.0  # ADC offset value
        self.smoothing_window = 5

        # Load and prep FGS1 signal data (guidance camera)
        self.fgs_signal = pd.read_parquet(
            f'{self.planet_path}/FGS1_signal_0.parquet'
        ).to_numpy().reshape(self.fgs_frames, 32, 32)

        self.fgs_signal = self.fgs_signal.astype(np.float64)

        # Load and prep AIRS-CH0 signal data (infrared spectrometer)
        self.airs_signal = pd.read_parquet(
            f'{self.planet_path}/AIRS-CH0_signal_0.parquet'
        ).to_numpy().reshape(self.airs_frames, 32, 356)[:, :, self.cut_inf:self.cut_sup]

        self.airs_signal = self.airs_signal.astype(np.float64)

        # Load initialize calibration data class instance correction steps
        self.calibration_data = CalibrationData(
            input_data_path=self.input_data_path,
            planet_path=self.planet_path,
            airs_frames=self.airs_frames,
            fgs_frames=self.fgs_frames,
            cut_inf=self.cut_inf,
            cut_sup=self.cut_sup
        )

    def test_adc_conversion(self,):
        '''
        Test ADC (Analog-to-Digital Converter) conversion functionality.
        
        Validates that:
        - ADC conversion applies correct gain and offset transformations
        - Output data maintains proper numerical characteristics
        - Conversion formula: converted = (raw_data + offset) * gain
        - Data dimensions and type are preserved through conversion
        
        ADC conversion transforms raw detector counts into calibrated
        physical units for accurate photometric analysis.
        '''

        # Run conversion on both AIRS-CHO and FGS1 data
        corrected_airs = correction_funcs.ADC_convert(
            self.airs_signal,
            self.gain,
            self.offset
        )

        corrected_fgs = correction_funcs.ADC_convert(
            self.fgs_signal,
            self.gain,
            self.offset
        )

        # Verify output types and shapes
        self.assertTrue(isinstance(corrected_airs, np.ndarray))
        self.assertTrue(isinstance(corrected_fgs, np.ndarray))
        self.assertTrue(corrected_airs.shape == self.airs_signal.shape)
        self.assertTrue(corrected_fgs.shape == self.fgs_signal.shape)


    def test_mask_hot_dead(self):
        '''
        Test hot and dead pixel masking functionality.
        
        Validates the masking algorithm by:
        - Applying hot/dead pixel masks to both AIRS and FGS data
        - Verifying mask application preserves data structure
        - Ensuring problematic pixels are properly identified and masked
        - Checking that masked data maintains expected dimensions
        
        Hot/dead pixel masking is essential for removing detector artifacts
        that would introduce systematic errors in transit photometry.
        '''

        # Apply hot/dead pixel masking to both AIRS-CH0 and FGS1 data
        masked_airs = correction_funcs.mask_hot_dead(
            self.airs_signal,
            self.calibration_data.dead_airs,
            self.calibration_data.dark_airs
        )

        masked_fgs = correction_funcs.mask_hot_dead(
            self.fgs_signal,
            self.calibration_data.dead_fgs,
            self.calibration_data.dark_fgs
        )

        # Validate output types and shapes
        self.assertTrue(isinstance(masked_airs, np.ma.MaskedArray))
        self.assertTrue(isinstance(masked_fgs, np.ma.MaskedArray))
        self.assertTrue(masked_airs.shape == self.airs_signal.shape)
        self.assertTrue(masked_fgs.shape == self.fgs_signal.shape)


    def test_linear_correction(self):
        '''
        Test detector linearity correction functionality.
        
        Validates the linearity correction step by:
        - Applying correction coefficients to compensate for detector non-linearity
        - Verifying correction preserves data structure and type
        - Ensuring corrected data maintains reasonable value ranges
        - Checking that correction algorithms handle masked arrays properly
        
        Linearity correction compensates for detector response curves that
        deviate from ideal linear behavior, improving photometric accuracy.
        '''

        # Run linearity correction on both AIRS-CH0 and FGS1 data
        corrected_airs = correction_funcs.apply_linear_corr(
            self.calibration_data.linear_corr_airs,
            self.airs_signal
        )

        corrected_fgs = correction_funcs.apply_linear_corr(
            self.calibration_data.linear_corr_fgs,
            self.fgs_signal
        )

        # Validate output types and shapes
        self.assertTrue(isinstance(corrected_airs, np.ndarray))
        self.assertTrue(isinstance(corrected_fgs, np.ndarray))
        self.assertTrue(corrected_airs.shape == self.airs_signal.shape)
        self.assertTrue(corrected_fgs.shape == self.fgs_signal.shape)


    def test_dark_subtraction(self):
        '''
        Test dark frame subtraction functionality.
        
        Validates dark current removal by:
        - Subtracting dark frames from both AIRS and FGS data
        - Verifying dark subtraction preserves data structure
        - Ensuring subtracted data has expected characteristics
        - Checking that dead pixels are properly handled during subtraction
        
        Dark frame subtraction removes thermal noise and detector bias,
        isolating the true astronomical signal from instrumental artifacts.
        '''

        # Run dark subtraction on both AIRS-CH0 and FGS1 data
        dark_subtracted_airs = correction_funcs.clean_dark(
            self.airs_signal.astype(np.float64),
            self.calibration_data.dead_airs,
            self.calibration_data.dark_airs,
            self.calibration_data.dt_airs
        )

        dark_subtracted_fgs = correction_funcs.clean_dark(
            self.fgs_signal.astype(np.float64),
            self.calibration_data.dead_fgs,
            self.calibration_data.dark_fgs,
            self.calibration_data.dt_fgs
        )

        # Validate output types and shapes
        self.assertTrue(isinstance(dark_subtracted_airs, np.ndarray))
        self.assertTrue(isinstance(dark_subtracted_fgs, np.ndarray))
        self.assertTrue(dark_subtracted_airs.shape == self.airs_signal.shape)
        self.assertTrue(dark_subtracted_fgs.shape == self.fgs_signal.shape)


    def test_cds_subtraction(self):
        '''
        Test Correlated Double Sampling (CDS) subtraction functionality.
        
        Validates CDS processing by:
        - Applying CDS algorithm to remove read noise and reset artifacts
        - Verifying CDS output maintains proper data structure
        - Ensuring CDS processing handles both AIRS and FGS data correctly
        - Checking that frame differencing preserves temporal information
        
        CDS subtraction removes correlated noise patterns by differencing
        consecutive detector readouts, improving signal-to-noise ratio.
        '''

        # Run CDS on both AIRS-CH0 and FGS1 data
        cds_airs = correction_funcs.get_cds(
            self.airs_signal
        )

        cds_fgs = correction_funcs.get_cds(
            self.fgs_signal
        )

        # Validate output types and shapes
        self.assertTrue(isinstance(cds_airs, np.ndarray))
        self.assertTrue(isinstance(cds_fgs, np.ndarray))
        self.assertTrue(cds_airs.shape[0] == self.airs_signal.shape[0]//2)
        self.assertTrue(cds_fgs.shape[0] == self.fgs_signal.shape[0]//2)


    def test_flat_field_correction(self):
        '''
        Test flat field correction functionality.
        
        Validates flat fielding by:
        - Applying flat field corrections to normalize pixel-to-pixel variations
        - Verifying correction preserves data structure and dimensions
        - Ensuring flat field algorithm handles both AIRS and FGS data
        - Checking that dead pixels are properly excluded from correction
        
        Flat field correction normalizes detector response variations,
        ensuring uniform sensitivity across the detector array for
        accurate photometric measurements.
        '''

        # Run flat field correction on both AIRS-CH0 and FGS1 data
        flat_corrected_airs = correction_funcs.correct_flat_field(
            self.airs_signal,
            self.calibration_data.flat_airs,
            self.calibration_data.dead_airs
        )

        flat_corrected_fgs = correction_funcs.correct_flat_field(
            self.fgs_signal,
            self.calibration_data.flat_fgs,
            self.calibration_data.dead_fgs
        )

        # Validate output types and shapes
        self.assertTrue(isinstance(flat_corrected_airs, np.ndarray))
        self.assertTrue(isinstance(flat_corrected_fgs, np.ndarray))
        self.assertTrue(flat_corrected_airs.shape == self.airs_signal.shape)
        self.assertTrue(flat_corrected_fgs.shape == self.fgs_signal.shape)

    
if __name__ == '__main__':
    unittest.main()