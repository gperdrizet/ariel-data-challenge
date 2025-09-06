'''Unittests for signal correction class'''

import unittest

import numpy as np
import pandas as pd

from ariel_data_preprocessing.signal_correction import SignalCorrection

class TestSignalCorrection(unittest.TestCase):

    def setUp(self):

        cut_inf = 39
        cut_sup = 321
        
        planet = '342072318'
        planet_path = f'tests/test_data/raw/train/{planet}'

        self.gain = 0.4369
        self.offset = -1000.0

        # Load assets
        self.axis_info = pd.read_parquet('tests/test_data/raw/axis_info.parquet')

        self.fgs_signal = pd.read_parquet(f'{planet_path}/FGS1_signal_0.parquet').to_numpy().reshape(4, 32, 32)
        self.airs_signal = pd.read_parquet(f'{planet_path}/AIRS-CH0_signal_0.parquet').to_numpy().reshape(4, 32, 356)[:, :, cut_inf:cut_sup]

        self.dark_airs = pd.read_parquet(f'{planet_path}/AIRS-CH0_calibration_0/dark.parquet').values.astype(np.float64).reshape((32, 356))[:, cut_inf:cut_sup]
        self.dead_airs = pd.read_parquet(f'{planet_path}/AIRS-CH0_calibration_0/dead.parquet').values.astype(np.float64).reshape((32, 356))[:, cut_inf:cut_sup]
        self.dark_fgs = pd.read_parquet(f'{planet_path}/FGS1_calibration_0/dark.parquet').values.astype(np.float64).reshape((32, 32))
        self.dead_fgs = pd.read_parquet(f'{planet_path}/FGS1_calibration_0/dead.parquet').values.astype(np.float64).reshape((32, 32))

        self.linear_corr_airs = pd.read_parquet(f'{planet_path}/AIRS-CH0_calibration_0/linear_corr.parquet').values.astype(np.float64).reshape((6, 32, 356))[:, :, cut_inf:cut_sup]
        self.linear_corr_fgs = pd.read_parquet(f'{planet_path}/FGS1_calibration_0/linear_corr.parquet').values.astype(np.float64).reshape((6, 32, 32))

        self.flat_airs = pd.read_parquet(f'{planet_path}/AIRS-CH0_calibration_0/flat.parquet').values.astype(np.float64).reshape((32, 356))[:, cut_inf:cut_sup]
        self.flat_fgs = pd.read_parquet(f'{planet_path}/FGS1_calibration_0/flat.parquet').values.astype(np.float64).reshape((32, 32))

        self.dt_airs = self.axis_info['AIRS-CH0-integration_time'].dropna().values[:4]
        self.dt_airs[1::2] += 0.1 # Why are we adding here - I don't think that is right...

        self.dt_fgs = np.ones(len(self.fgs_signal)) * 0.1
        self.dt_fgs[1::2] += 0.1 # This one looks more correct

        self.signal_correction = SignalCorrection(
            input_data_path='tests/test_data/raw',
            output_data_path='tests/test_data/corrected',
            gain=self.gain,
            offset=self.offset
        )


    def test_adc_conversion(self):
        '''Test ADC conversion'''

        corrected_airs = self.signal_correction._ADC_convert(
            self.airs_signal,
            self.gain,
            self.offset
        )

        corrected_fgs = self.signal_correction._ADC_convert(
            self.fgs_signal,
            self.gain,
            self.offset
        )

        self.assertTrue(corrected_airs.shape == self.airs_signal.shape)
        self.assertTrue(corrected_fgs.shape == self.fgs_signal.shape)


    def test_mask_hot_dead(self):
        '''Test hot/dead pixel masking'''

        masked_airs = self.signal_correction._mask_hot_dead(
            self.airs_signal,
            self.dead_airs,
            self.dark_airs
        )

        masked_fgs = self.signal_correction._mask_hot_dead(
            self.fgs_signal,
            self.dead_fgs,
            self.dark_fgs
        )

        self.assertTrue(masked_airs.shape == self.airs_signal.shape)
        self.assertTrue(masked_fgs.shape == self.fgs_signal.shape)
        self.assertTrue(isinstance(masked_airs, np.ma.MaskedArray))
        self.assertTrue(isinstance(masked_fgs, np.ma.MaskedArray))


    def test_linear_correction(self):
        '''Test linearity correction'''

        corrected_airs = self.signal_correction._apply_linear_corr(
            self.linear_corr_airs,
            self.airs_signal
        )

        corrected_fgs = self.signal_correction._apply_linear_corr(
            self.linear_corr_fgs,
            self.fgs_signal
        )

        self.assertTrue(corrected_airs.shape == self.airs_signal.shape)
        self.assertTrue(corrected_fgs.shape == self.fgs_signal.shape)
    

    def test_dark_subtraction(self):
        '''Test dark frame subtraction'''

        dark_subtracted_airs = self.signal_correction._clean_dark(
            self.airs_signal.astype(np.float64),
            self.dead_airs,
            self.dark_airs,
            self.dt_airs
        )

        dark_subtracted_fgs = self.signal_correction._clean_dark(
            self.fgs_signal.astype(np.float64),
            self.dead_fgs,
            self.dark_fgs,
            self.dt_fgs
        )

        self.assertTrue(dark_subtracted_airs.shape == self.airs_signal.shape)
        self.assertTrue(dark_subtracted_fgs.shape == self.fgs_signal.shape)


    def test_cds_subtraction(self):
        '''Test CDS subtraction'''

        cds_airs = self.signal_correction._get_cds(
            self.airs_signal
        )

        cds_fgs = self.signal_correction._get_cds(
            self.fgs_signal
        )

        self.assertTrue(cds_airs.shape[0] == self.airs_signal.shape[0]//2)
        self.assertTrue(cds_fgs.shape[0] == self.fgs_signal.shape[0]//2)


    def test_flat_field_correction(self):
        '''Test flat field correction'''

        flat_corrected_airs = self.signal_correction._correct_flat_field(
            self.airs_signal,
            self.flat_airs,
            self.dead_airs
        )

        flat_corrected_fgs = self.signal_correction._correct_flat_field(
            self.fgs_signal,
            self.flat_fgs,
            self.dead_fgs
        )

        self.assertTrue(flat_corrected_airs.shape == self.airs_signal.shape)
        self.assertTrue(flat_corrected_fgs.shape == self.fgs_signal.shape)