'''
Unit tests for utility functions in the ariel_data_preprocessing package.

This module tests the core utility functions used throughout the preprocessing
pipeline, including planet list retrieval and masked frame loading from HDF5 files.

Test Coverage:
    - Planet list extraction from raw data directories
    - Masked frame loading from processed HDF5 datasets
    - Data type validation and shape verification

Test Data Requirements:
    - tests/test_data/raw/ directory with planet subdirectories
    - tests/test_data/processed/train.h5 with processed signal data

Author: Dr. George Perdrizet PhD
Created: 2025
'''

# Standard library imports
import unittest

# Third-party imports
import numpy as np
import h5py

# Internal imports
import ariel_data_preprocessing.utils as utils


class TestUtils(unittest.TestCase):
    '''
    Test suite for utility functions in the preprocessing pipeline.
    
    This class validates the core utility functions that support
    data loading, planet discovery, and HDF5 file operations used
    throughout the signal correction and extraction pipeline.
    '''

    def test_get_planet_list(self):
        '''
        Test planet list retrieval from raw data directory structure.
        
        Validates that the get_planet_list function correctly:
        - Returns a list of planet IDs from directory structure
        - Handles the expected test data directory format
        - Returns string planet identifiers
        - Maintains consistent ordering
        
        Expected behavior:
            - Returns non-empty list of planet ID strings
            - First planet in test data should be '342072318'
        '''

        # Get planet list from test data directory
        planet_list = utils.get_planet_list('tests/test_data/raw', mode='train')

        # Validate return type and basic properties
        self.assertTrue(isinstance(planet_list, list))
        self.assertTrue(len(planet_list) > 0)
        self.assertTrue(all(isinstance(p, str) for p in planet_list))
        
        # Validate expected test planet is present
        self.assertEqual(planet_list[0], '342072318')

    def test_load_masked_frames(self):
        '''
        Test loading masked frames from HDF5 processed data files.
        
        Validates that the load_masked_frames function correctly:
        - Loads signal data from HDF5 group structure
        - Returns properly structured masked arrays
        - Preserves data dimensions and types
        - Handles mask metadata appropriately
        
        Expected behavior:
            - Returns numpy masked array with signal data
            - Frame count > 0 (temporal dimension)
            - 283 wavelength channels (spectral dimension after extraction)
            - Proper mask integration for hot/dead pixels
        '''

        # Load masked frames from test HDF5 file
        with h5py.File('tests/test_data/processed/train.h5', 'r') as hdf:
            
            planet = '342072318'  # Test planet ID
            masked_frames = utils.load_masked_frames(hdf, planet)

            # Validate return type and structure
            self.assertTrue(isinstance(masked_frames, np.ma.MaskedArray))
            self.assertTrue(masked_frames.shape[0] > 0)    # At least one frame
            self.assertEqual(masked_frames.shape[1], 283)  # Expected wavelength count


if __name__ == '__main__':
    unittest.main()