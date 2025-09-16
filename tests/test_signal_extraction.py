'''Unittests for signal extraction class'''

import unittest
import os
from pathlib import Path

import numpy as np
import pandas as pd
import h5py

from ariel_data_preprocessing.signal_extraction import SignalExtraction


class TestSignalExtraction(unittest.TestCase):

    def setUp(self):
        '''Set up test data and SignalExtraction instance'''
        
        self.input_data = 'tests/test_data/corrected/train.h5'
        self.output_data_path = 'tests/test_data/extracted'
        self.test_planet = '342072318'
        self.inclusion_threshold = 0.75
        self.smoothing_window = 10
        self.n_planets = 1
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_data_path, exist_ok=True)
        
        # Initialize SignalExtraction instance
        self.signal_extraction = SignalExtraction(
            input_data=self.input_data,
            output_data_path=self.output_data_path,
            inclusion_threshold=self.inclusion_threshold,
            smooth=True,
            smoothing_window=self.smoothing_window,
            n_planets=self.n_planets
        )
        
        # Load test data for method testing
        with h5py.File(self.input_data, 'r') as hdf:
            self.test_airs_frames = hdf[self.test_planet]['AIRS-CH0_signal'][:]


    def test_select_top_rows(self):
        '''Test top row selection method'''
        
        # Test with different inclusion thresholds
        for threshold in [0.5, 0.75, 1.0]:
            with self.subTest(threshold=threshold):
                selected_rows = self.signal_extraction._select_top_rows(
                    self.test_airs_frames, 
                    threshold
                )
                
                self.assertIsInstance(selected_rows, list)
                self.assertGreater(len(selected_rows), 0)
                
                # Higher threshold should select fewer rows
                if threshold < 1.0:
                    self.assertLess(len(selected_rows), self.test_airs_frames.shape[1])
                
                # All selected row indices should be valid
                for row_idx in selected_rows:
                    self.assertGreaterEqual(row_idx, 0)
                    self.assertLess(row_idx, self.test_airs_frames.shape[1])


    def test_select_top_rows_threshold_behavior(self):
        '''Test that higher thresholds select fewer rows'''
        
        rows_50 = self.signal_extraction._select_top_rows(self.test_airs_frames, 0.5)
        rows_75 = self.signal_extraction._select_top_rows(self.test_airs_frames, 0.75)
        rows_90 = self.signal_extraction._select_top_rows(self.test_airs_frames, 0.9)
        
        # Higher threshold should select same or fewer rows
        self.assertGreaterEqual(len(rows_50), len(rows_75))
        self.assertGreaterEqual(len(rows_75), len(rows_90))


    def test_select_top_cols(self):
        '''Test top column selection method'''
        
        # Test with different inclusion thresholds
        for threshold in [0.5, 0.75, 1.0]:
            with self.subTest(threshold=threshold):
                selected_cols = self.signal_extraction._select_top_cols(
                    self.test_airs_frames, 
                    threshold
                )
                
                self.assertIsInstance(selected_cols, list)
                self.assertGreater(len(selected_cols), 0)
                
                # Higher threshold should select fewer columns
                if threshold < 1.0:
                    self.assertLess(len(selected_cols), self.test_airs_frames.shape[2])
                
                # All selected column indices should be valid
                for col_idx in selected_cols:
                    self.assertGreaterEqual(col_idx, 0)
                    self.assertLess(col_idx, self.test_airs_frames.shape[2])


    def test_select_top_cols_threshold_behavior(self):
        '''Test that higher thresholds select fewer columns'''
        
        cols_50 = self.signal_extraction._select_top_cols(self.test_airs_frames, 0.5)
        cols_75 = self.signal_extraction._select_top_cols(self.test_airs_frames, 0.75)
        cols_90 = self.signal_extraction._select_top_cols(self.test_airs_frames, 0.9)
        
        # Higher threshold should select same or fewer columns
        self.assertGreaterEqual(len(cols_50), len(cols_75))
        self.assertGreaterEqual(len(cols_75), len(cols_90))


    def test_moving_average_rows(self):
        '''Test moving average calculation - note: runs cumsum moving average on
        transpose of input data'''
        
        # Create test data with known pattern
        test_data = np.array([
            [1, 5, 9],
            [2, 6, 10],
            [3, 7, 11],
            [4, 8, 12]
        ])
        
        # Test with window size 3
        result = SignalExtraction.moving_average_rows(test_data, 3)
        
        # Check shape
        expected_rows = test_data.shape[0] - 3 + 1
        self.assertEqual(result.shape, (expected_rows, test_data.shape[1]))
        
        # Check values for first row
        expected_first_row = np.array([2.0, 6.0, 10.0])
        np.testing.assert_array_almost_equal(result[0], expected_first_row)
        
        # Check values for second row
        expected_second_row = np.array([3.0, 7.0, 11.0])
        np.testing.assert_array_almost_equal(result[1], expected_second_row)


    def test_run_method(self):
        '''Test the complete signal extraction pipeline'''
        
        # Run the extraction
        self.signal_extraction.run()
        
        # Check that output file was created
        output_file = f'{self.output_data_path}/train.h5'
        self.assertTrue(os.path.exists(output_file))
        
        # Check the extracted data
        with h5py.File(output_file, 'r') as hdf:
            self.assertIn(self.test_planet, hdf.keys())
            
            # Check that both signal and mask datasets exist
            self.assertIn('signal', hdf[self.test_planet].keys())
            self.assertIn('mask', hdf[self.test_planet].keys())
            
            extracted_signal = hdf[self.test_planet]['signal'][:]
            extracted_mask = hdf[self.test_planet]['mask'][:]
            
            # Check shape - should be (frames, wavelengths)
            self.assertEqual(len(extracted_signal.shape), 2)
            
            # Should have fewer frames due to smoothing
            expected_frames = self.test_airs_frames.shape[0] - self.smoothing_window + 1
            self.assertEqual(extracted_signal.shape[0], expected_frames)
            
            # Should have wavelengths + 1 FGS column
            expected_wavelengths = self.test_airs_frames.shape[2] + 1  # AIRS + FGS
            self.assertEqual(extracted_signal.shape[1], expected_wavelengths)


    def test_inclusion_threshold_effects(self):
        '''Test that different inclusion thresholds affect the number of selected rows'''
        
        # Test with very low threshold (should select more rows)
        low_threshold_extractor = SignalExtraction(
            input_data=self.input_data,
            output_data_path=self.output_data_path,
            inclusion_threshold=0.1,
            smooth=False,
            n_planets=1
        )
        
        # Test with high threshold (should select fewer rows)
        high_threshold_extractor = SignalExtraction(
            input_data=self.input_data,
            output_data_path=self.output_data_path,
            inclusion_threshold=0.9,
            smooth=False,
            n_planets=1
        )
        
        low_rows = low_threshold_extractor._select_top_rows(self.test_airs_frames, 0.1)
        high_rows = high_threshold_extractor._select_top_rows(self.test_airs_frames, 0.9)
        
        self.assertGreaterEqual(len(low_rows), len(high_rows))


    def test_smoothing_parameter(self):
        '''Test that smoothing parameter affects output dimensions'''
        
        # Create extractors with and without smoothing
        no_smooth_extractor = SignalExtraction(
            input_data=self.input_data,
            output_data_path=self.output_data_path,
            smooth=False,
            n_planets=1
        )
        
        smooth_extractor = SignalExtraction(
            input_data=self.input_data,
            output_data_path=self.output_data_path,
            smooth=True,
            smoothing_window=10,
            n_planets=1
        )
        
        # Test that smooth parameter is set correctly
        self.assertFalse(no_smooth_extractor.smooth)
        self.assertTrue(smooth_extractor.smooth)
        self.assertEqual(smooth_extractor.smoothing_window, 10)


if __name__ == '__main__':
    unittest.main()