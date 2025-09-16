'''Main runner for the signal correction & extraction pipeline.'''

import ariel_data_preprocessing.signal_correction as sc
import ariel_data_preprocessing.signal_extraction as se
import configuration as config

if __name__ == '__main__':

    print('\nStarting signal correction...')

    signal_correction = sc.SignalCorrection(
        input_data_path=config.RAW_DATA_DIRECTORY,
        output_data_path=config.CORRECTED_SIGNAL_DIRECTORY,
        n_cpus=11,
        n_planets=10,
        downsample_fgs=True,
        verbose=True
    )

    signal_correction.run()

    print('\n\nSignal correction completed, starting signal extraction...')

    signal_extraction = se.SignalExtraction(
        input_data_path=config.CORRECTED_SIGNAL_DIRECTORY,
        output_data_path=config.EXTRACTED_SIGNAL_DIRECTORY,
        inclusion_threshold=0.75,
        n_planets=10,
        verbose=True
    )

    signal_extraction.run()
    print('\n\nSignal extraction completed.')