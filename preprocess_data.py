'''Main runner for the signal correction % data extraction pipeline.'''

import ariel_data_preprocessing.signal_correction as sc

if __name__ == '__main__':
    signal_correction = sc.SignalCorrection(
        input_data_path='data/raw',
        output_data_path='data/signal_corrected',
        output_filename='train.h5',  # Specify the output filename here
        n_cpus=10,
        n_planets=11,
        downsample_fgs=True
    )

    signal_correction.run()