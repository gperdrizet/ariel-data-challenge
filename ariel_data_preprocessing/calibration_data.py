'''Calibration data loading and storage class.'''

import numpy as np
import pandas as pd

class CalibrationData:
    '''
    Class to load and store calibration data for signal correction.
    
    This class reads all necessary calibration files for a given planet
    and stores them as attributes for use in the DataProcessor pipeline.
    
    Attributes:
        dark_airs (np.ndarray): AIRS-CH0 dark frame (thermal background)
        dead_airs (np.ndarray): AIRS-CH0 dead pixel mask
        dark_fgs (np.ndarray): FGS1 dark frame (thermal background)
        dead_fgs (np.ndarray): FGS1 dead pixel mask
        linear_corr_airs (np.ndarray): AIRS-CH0 linearity correction coefficients
        linear_corr_fgs (np.ndarray): FGS1 linearity correction coefficients
        flat_airs (np.ndarray): AIRS-CH0 flat field frame
        flat_fgs (np.ndarray): FGS1 flat field frame
        axis_info (pd.DataFrame): Timing and metadata information
        dt_airs (np.ndarray): AIRS-CH0 integration times per frame
        dt_fgs (np.ndarray): FGS1 integration times per frame
    '''

    def __init__(
            self,
            input_data_path: str,
            planet_path: str,
            airs_frames: int,
            fgs_frames: int,
            cut_inf: int,
            cut_sup: int
        ):
        '''
        Initialize CalibrationData by loading calibration files.
        
        Args:
            input_data_path (str): Path to raw data directory containing axis_info.parquet
            planet_path (str): Path to the planet's data directory
            airs_frames (int): Number of AIRS-CH0 frames for timing data
            fgs_frames (int): Number of FGS1 frames for timing data
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

        self.dt_airs = self.axis_info['AIRS-CH0-integration_time'].dropna().values[:airs_frames]
        self.dt_airs[1::2] += 0.1  # Adjust odd frame integration times

        self.dt_fgs = np.ones(fgs_frames) * 0.1
        self.dt_fgs[1::2] += 0.1  # Adjust odd frame integration times