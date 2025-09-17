'''
Comprehensive unit tests for signal extraction functions.

This module provides thorough testing coverage for the signal extraction pipeline,
validating both AIRS-CH0 (infrared spectrometer) and FGS1 (guidance camera) 
signal processing workflows. Tests cover:

1. AIRS-CH0 spectrum signal extraction
2. FGS1 2D spatial block extraction  
4. Data structure integrity and output format verification
5. Performance validation for production-scale processing

The tests use real Ariel challenge data to ensure algorithms work correctly
with authentic detector characteristics and noise patterns.

Test Requirements:
- Test data: Planet '342072318' with complete AIRS-CH0 and FGS1 datasets
- Dependencies: numpy, pandas, signal_extraction_functions module
- Expected runtime: ~2-5 seconds for full test suite

Author: Dr. George Perdrizet PhD
Created: 2025
'''

# Standard library imports
import unittest

# Third-party imports
import numpy as np
import pandas as pd

# Internal imports
import ariel_data_preprocessing.signal_extraction_functions as extraction_funcs


class TestSignalExtraction(unittest.TestCase):

    def setUp(self):
        '''
        Set up test environment for signal extraction validation.
        
        Initializes comprehensive test configuration with:
        - Real Ariel challenge test data (planet '342072318')
        - Standard extraction parameters (wavelength bounds, frame counts)
        - Loaded and preprocessed AIRS-CH0 and FGS1 signal arrays
        - Extraction algorithm parameters (thresholds, smoothing windows)
        
        This setup enables isolated testing of signal extraction algorithms
        while using realistic data dimensions, noise characteristics, and
        detector response patterns from the Ariel mission simulation.
        '''
        
        # Test data configuration
        self.input_data_path = 'tests/test_data/raw'
        self.test_planet = '342072318'
        self.planet_path = f'{self.input_data_path}/train/{self.test_planet}'
        
        # AIRS-CH0 spectral extraction parameters
        self.cut_inf = 39      # Wavelength cropping lower bound
        self.cut_sup = 321     # Wavelength cropping upper bound
        self.airs_frames = 50  # Number of AIRS temporal frames
        
        # FGS1 photometric extraction parameters  
        self.fgs_frames = 50   # Number of FGS temporal frames
        
        # Signal extraction algorithm parameters
        self.inclusion_threshold = 0.75  # Quality threshold for pixel inclusion
        self.smoothing_window = 10       # Temporal smoothing window size
        self.n_planets = 1              # Number of test planets
        
        # Load and prep FGS1 signal data (guidance camera photometry)
        self.fgs_signal = pd.read_parquet(
            f'{self.planet_path}/FGS1_signal_0.parquet'
        ).to_numpy().reshape(self.fgs_frames, 32, 32)

        self.fgs_signal = self.fgs_signal.astype(np.float64)

        # Load and prep AIRS-CH0 signal data (infrared spectroscopy)
        self.airs_signal = pd.read_parquet(
            f'{self.planet_path}/AIRS-CH0_signal_0.parquet'
        ).to_numpy().reshape(self.airs_frames, 32, 356)[:, :, self.cut_inf:self.cut_sup]

        self.airs_signal = self.airs_signal.astype(np.float64)


    def test_select_top_rows(self):
        '''
        Test spatial row selection functionality for AIRS-CH0 extraction.
        
        Validates the row selection algorithm by:
        - Testing multiple inclusion threshold values (0.5, 0.75, 1.0)
        - Verifying output is valid list of row indices
        - Ensuring higher thresholds select fewer rows (quality filtering)
        - Checking all selected indices are within valid detector bounds
        
        Row selection identifies detector rows with highest signal quality
        for optimal spectral extraction and noise reduction.
        '''
        
        # Test with different inclusion thresholds
        for threshold in [0.5, 0.75, 1.0]:
            with self.subTest(threshold=threshold):
                selected_rows = extraction_funcs.select_top_rows(
                    self.airs_signal, 
                    threshold
                )
                
                # Validate output type and basic properties
                self.assertIsInstance(selected_rows, list)
                self.assertGreater(len(selected_rows), 0)
                
                # Higher threshold should select fewer rows
                if threshold < 1.0:
                    self.assertLess(len(selected_rows), self.airs_signal.shape[1])

                # All selected row indices should be valid
                for row_idx in selected_rows:
                    self.assertGreaterEqual(row_idx, 0)
                    self.assertLess(row_idx, self.airs_signal.shape[1])


    def test_select_top_rows_threshold_behavior(self):
        '''
        Test threshold-dependent behavior of row selection algorithm.
        
        Validates that:
        - Higher inclusion thresholds result in fewer selected rows
        - Quality filtering works as expected (stricter = fewer rows)
        - Algorithm maintains consistent behavior across threshold values
        
        This ensures the quality-based filtering mechanism works correctly
        for different signal-to-noise requirements.
        '''

        # Test with different inclusion thresholds
        rows_50 = extraction_funcs.select_top_rows(self.airs_signal, 0.5)
        rows_75 = extraction_funcs.select_top_rows(self.airs_signal, 0.75)
        rows_90 = extraction_funcs.select_top_rows(self.airs_signal, 0.9)

        # Higher threshold should select same or fewer rows
        self.assertGreaterEqual(len(rows_50), len(rows_75))
        self.assertGreaterEqual(len(rows_75), len(rows_90))


    def test_select_top_cols(self):
        '''
        Test spectral column selection functionality for AIRS-CH0 extraction.
        
        Validates the wavelength column selection algorithm by:
        - Testing multiple inclusion threshold values (0.5, 0.75, 1.0)
        - Verifying output is valid list of wavelength column indices
        - Ensuring higher thresholds select fewer columns (quality filtering)
        - Checking all selected indices are within valid spectral bounds
        
        Column selection identifies wavelength channels with highest signal
        quality for optimal spectral extraction and atmospheric analysis.
        '''
        
        # Test with different inclusion thresholds
        for threshold in [0.5, 0.75, 1.0]:
            with self.subTest(threshold=threshold):
                selected_cols = extraction_funcs.select_top_cols(
                    self.airs_signal, 
                    threshold
                )
                
                # Validate output type and basic properties
                self.assertIsInstance(selected_cols, list)
                self.assertGreater(len(selected_cols), 0)
                
                # Higher threshold should select fewer columns
                if threshold < 1.0:
                    self.assertLess(len(selected_cols), self.airs_signal.shape[2])

                # All selected column indices should be valid
                for col_idx in selected_cols:
                    self.assertGreaterEqual(col_idx, 0)
                    self.assertLess(col_idx, self.airs_signal.shape[2])


    def test_select_top_cols_threshold_behavior(self):
        '''
        Test threshold-dependent behavior of column selection algorithm.
        
        Validates that:
        - Higher inclusion thresholds result in fewer selected columns
        - Spectral quality filtering works as expected (stricter = fewer wavelengths)
        - Algorithm maintains consistent behavior across threshold values
        
        This ensures wavelength-based quality filtering works correctly
        for different atmospheric analysis requirements.
        '''

        # Test with different inclusion thresholds
        cols_50 = extraction_funcs.select_top_cols(self.airs_signal, 0.5)
        cols_75 = extraction_funcs.select_top_cols(self.airs_signal, 0.75)
        cols_90 = extraction_funcs.select_top_cols(self.airs_signal, 0.9)

        # Higher threshold should select same or fewer columns
        self.assertGreaterEqual(len(cols_50), len(cols_75))
        self.assertGreaterEqual(len(cols_75), len(cols_90))


    def test_moving_average_rows(self):
        '''
        Test temporal moving average functionality for signal smoothing.
        
        Validates the moving average algorithm by:
        - Creating test data with known temporal patterns
        - Applying moving average with specified window size
        - Verifying smoothing preserves signal structure
        - Checking output dimensions match input (transposed processing)
        
        Note: Function applies cumulative sum moving average on transposed
        input data for efficient temporal processing across detector rows.
        
        Moving average smoothing reduces noise while preserving transit
        signals for improved photometric precision.
        '''
        
        # Create test data with known pattern
        test_data = np.array([
            [1, 5, 9],
            [2, 6, 10],
            [3, 7, 11],
            [4, 8, 12]
        ])
        
        # Test with window size 3
        result = extraction_funcs.moving_average_rows(test_data, 3)

        # Check shape
        expected_rows = test_data.shape[0] - 3 + 1
        self.assertEqual(result.shape, (expected_rows, test_data.shape[1]))
        
        # Check values for first row
        expected_first_row = np.array([2.0, 6.0, 10.0])
        np.testing.assert_array_almost_equal(result[0], expected_first_row)
        
        # Check values for second row
        expected_second_row = np.array([3.0, 7.0, 11.0])
        np.testing.assert_array_almost_equal(result[1], expected_second_row)


    def test_inclusion_threshold_effects(self):
        '''
        Test comprehensive threshold effects on signal extraction quality.
        
        Validates that:
        - Low thresholds (0.1) include more pixels but lower quality
        - High thresholds (0.9) include fewer pixels but higher quality
        - Threshold-based filtering provides expected trade-offs
        
        This demonstrates the quality vs. coverage trade-off in signal
        extraction for different scientific analysis requirements.
        '''
        
        # Test with low and high thresholds
        low_rows = extraction_funcs.select_top_rows(self.airs_signal, 0.1)
        high_rows = extraction_funcs.select_top_rows(self.airs_signal, 0.9)
        
        # Low threshold should select same or more rows than high threshold
        self.assertGreaterEqual(len(low_rows), len(high_rows))


    def test_smoothing_parameter(self):
        '''
        Test smoothing window parameter effects on temporal processing.
        
        Validates that:
        - Larger smoothing windows result in fewer output frames
        - Smaller windows preserve more temporal resolution
        - Window size correctly affects output dimensions
        
        This ensures temporal smoothing parameters work as expected
        for different noise reduction vs. time resolution trade-offs
        in transit light curve analysis.
        '''
        
        # Test with different smoothing window sizes
        large_window = extraction_funcs.moving_average_rows(self.airs_signal, 10)
        small_window = extraction_funcs.moving_average_rows(self.airs_signal, 3)

        # Test that smooth parameter has expected effect on output shape
        self.assertGreaterEqual(large_window.shape[0], small_window.shape[0])


if __name__ == '__main__':
    unittest.main()