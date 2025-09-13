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
        
        self.input_data_path = 'tests/test_data/corrected'
        self.output_data_path = 'tests/test_data/extracted'
        self.test_planet = '342072318'
        self.inclusion_threshold = 0.75
        self.smoothing_window = 10
        self.n_planets = 1
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_data_path, exist_ok=True)
        
        # Initialize SignalExtraction instance
        self.signal_extraction = SignalExtraction(
            input_data_path=self.input_data_path,
            output_data_path=self.output_data_path,
            inclusion_threshold=self.inclusion_threshold,
            smooth=True,
            smoothing_window=self.smoothing_window,
            n_planets=self.n_planets
        )
        
        # Load test data for method testing
        with h5py.File(f'{self.input_data_path}/train.h5', 'r') as hdf:
            self.test_airs_frames = hdf[self.test_planet]['AIRS-CH0_signal'][:]


    def test_get_planet_list(self):
        '''Test planet list retrieval'''
        planet_list = self.signal_extraction._get_planet_list()
        
        self.assertIsInstance(planet_list, list)
        self.assertGreater(len(planet_list), 0)
        self.assertIn(self.test_planet, planet_list)


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


    def test_moving_average_rows(self):
        '''Test moving average calculation'''
        
        # Create test data with known pattern
        test_data = np.array([
            [1, 2, 3, 4, 5, 6],
            [2, 4, 6, 8, 10, 12]
        ])
        
        # Test with window size 3
        result = SignalExtraction.moving_average_rows(test_data, 3)
        
        # Check shape
        expected_cols = test_data.shape[1] - 3 + 1
        self.assertEqual(result.shape, (test_data.shape[0], expected_cols))
        
        # Check values for first row
        expected_first_row = np.array([2.0, 3.0, 4.0, 5.0])  # (1+2+3)/3, (2+3+4)/3, etc.
        np.testing.assert_array_almost_equal(result[0], expected_first_row)
        
        # Check values for second row
        expected_second_row = np.array([4.0, 6.0, 8.0, 10.0])  # (2+4+6)/3, (4+6+8)/3, etc.
        np.testing.assert_array_almost_equal(result[1], expected_second_row)


    def test_moving_average_edge_cases(self):
        '''Test moving average with edge cases'''
        
        # Test with window size 1 (should return original minus first column padding)
        test_data = np.array([[1, 2, 3, 4]])
        result = SignalExtraction.moving_average_rows(test_data, 1)
        expected = np.array([[1, 2, 3, 4]])
        np.testing.assert_array_equal(result, expected)
        
        # Test with window size equal to array width
        result = SignalExtraction.moving_average_rows(test_data, 4)
        expected = np.array([[2.5]])  # Average of [1,2,3,4]
        np.testing.assert_array_equal(result, expected)


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
            
            extracted_signal = hdf[self.test_planet]['AIRS-CH0_signal'][:]
            
            # Check shape - should be (frames, wavelengths)
            self.assertEqual(len(extracted_signal.shape), 2)
            
            # Should have same number of frames as input
            expected_frames = self.test_airs_frames.shape[0]
            self.assertEqual(extracted_signal.shape[0], expected_frames - self.smoothing_window + 1)
            
            # Should have same number of wavelengths as input
            expected_wavelengths = self.test_airs_frames.shape[2]
            self.assertEqual(extracted_signal.shape[1], expected_wavelengths)


    def test_inclusion_threshold_effects(self):
        '''Test that different inclusion thresholds affect the number of selected rows'''
        
        # Test with very low threshold (should select more rows)
        low_threshold_extractor = SignalExtraction(
            input_data_path=self.input_data_path,
            output_data_path=self.output_data_path,
            inclusion_threshold=0.1,
            smooth=False,
            n_planets=1
        )
        
        # Test with high threshold (should select fewer rows)
        high_threshold_extractor = SignalExtraction(
            input_data_path=self.input_data_path,
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
            input_data_path=self.input_data_path,
            output_data_path=self.output_data_path,
            smooth=False,
            n_planets=1
        )
        
        smooth_extractor = SignalExtraction(
            input_data_path=self.input_data_path,
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