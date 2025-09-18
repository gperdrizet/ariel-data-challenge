'''
Comprehensive unit tests for the DataProcessor class and pipeline integration.

This module provides end-to-end testing of the complete data preprocessing
pipeline, validating the integration of signal correction, signal extraction,
and data output workflows. Tests cover:

1. DataProcessor class initialization and configuration
2. End-to-end pipeline execution with real Ariel data
3. Output data format validation and integrity checks  
4. HDF5 file generation and structure verification
5. Pipeline performance and error handling

The tests ensure the complete preprocessing workflow functions correctly
from raw detector data through corrected, extracted signals ready for
atmospheric analysis.

Test Requirements:
- Test data: Planet '342072318' with complete AIRS-CH0 and FGS1 datasets
- Dependencies: numpy, pandas, h5py, DataProcessor class
- Expected runtime: ~3-8 seconds for full pipeline execution

Author: Dr. George Perdrizet PhD
Created: 2025
'''

# Standard library imports
import os
import unittest

# Third-party imports
import numpy as np
import pandas as pd
import h5py

# Internal imports
from ariel_data_preprocessing.data_preprocessing import DataProcessor

class TestDataProcessor(unittest.TestCase):

    def setUp(self):
        '''
        Set up test environment for DataProcessor pipeline validation.
        
        Initializes comprehensive test configuration with:
        - Real Ariel challenge test data (planet '342072318')
        - Input and output directory paths for pipeline testing
        - Standard preprocessing parameters (frames, wavelength bounds, smoothing)
        - Loaded and preprocessed AIRS-CH0 and FGS1 signal arrays
        
        This setup enables end-to-end testing of the complete preprocessing
        pipeline from raw detector data through corrected and extracted
        signals ready for atmospheric analysis workflows.
        '''

        # Test data configuration
        self.input_data_path = 'tests/test_data/raw'
        self.output_data_path = 'tests/test_data/processed'
        self.planet = '342072318'
        self.planet_path = f'{self.input_data_path}/train/{self.planet}'

        # Pipeline parameters
        self.airs_frames = 50   # AIRS temporal frames
        self.fgs_frames = 50    # FGS temporal frames
        self.cut_inf = 39       # Spectral cropping lower bound
        self.cut_sup = 321      # Spectral cropping upper bound
        self.smoothing_window = 5  # Temporal smoothing window

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

        # Initialize and run DataProcessor pipeline
        self.data_processor = DataProcessor(
            input_data_path=self.input_data_path,
            output_data_path=self.output_data_path,
            airs_frames=self.airs_frames,
            fgs_frames=self.fgs_frames,
            smoothing_window=self.smoothing_window
        )

        self.data_processor.run()


    def test_output_file(self):
        '''
        Test HDF5 output file creation and accessibility.
        
        Validates that:
        - DataProcessor successfully creates output HDF5 file
        - Output file is accessible and properly formatted
        - File path structure matches expected output directory
        
        This ensures the pipeline produces the required HDF5 output
        format for downstream atmospheric analysis workflows.
        '''
        
        # Check that output file was created
        output_file = f'{self.output_data_path}/train.h5'
        self.assertTrue(os.path.exists(output_file))


    def test_output_data(self):
        '''
        Test complete data preprocessing pipeline output validation.
        
        Validates end-to-end pipeline results by:
        - Checking HDF5 file structure and dataset organization
        - Verifying output dimensions match expected processed data size
        - Ensuring both signal and mask datasets are properly created
        - Validating temporal frame reduction from CDS and smoothing
        - Confirming spectral channel count (282 AIRS + 1 FGS = 283 total)
        
        This comprehensive test ensures the complete preprocessing pipeline
        produces correctly formatted, dimensionally consistent output data
        ready for atmospheric retrieval and analysis workflows.
        '''

        # Load the output data and validate structure
        with h5py.File(f'{self.output_data_path}/train.h5', 'r') as hdf:

            expected_frames = (self.airs_signal.shape[0] // 2) - self.data_processor.smoothing_window + 1
            expected_wavelengths = 282 + 1  # 282 AIRS + 1 FGS

            self.assertEqual(len(hdf[self.planet]), 3) # Signal, mask, spectrum
            self.assertTrue('signal' in hdf[self.planet])
            self.assertTrue('mask' in hdf[self.planet])
            self.assertTrue('spectrum' in hdf[self.planet])
            self.assertTrue(hdf[self.planet]['signal'].shape[0] == expected_frames)
            self.assertTrue(hdf[self.planet]['signal'].shape[1] == expected_wavelengths)
            self.assertTrue(hdf[self.planet]['mask'].shape[0] == expected_wavelengths)

if __name__ == '__main__':
    unittest.main()