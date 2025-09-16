'''Main runner for the signal correction & data extraction pipeline.'''

import ariel_data_preprocessing.signal_correction as sc
import ariel_data_preprocessing.signal_extraction as se

if __name__ == '__main__':

    signal_correction = sc.SignalCorrection(
        input_data_path='data/raw',
        output_data_path='data/signal_corrected',
        output_filename='train.h5',  # Specify the output filename here
        n_cpus=10,
        n_planets=10,
        downsample_fgs=True,
        verbose=True
    )

    signal_correction.run()

    signal_extraction = se.SignalExtraction(
        input_data='data/signal_corrected/train.h5',
        output_data_path='data/extracted',
        output_filename='train.h5',  # Specify the output filename here
        inclusion_threshold=0.9,
        n_planets=10,
    )

    signal_extraction.run()