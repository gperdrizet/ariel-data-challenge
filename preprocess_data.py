'''Main runner for the signal correction & extraction pipeline.'''

from ariel_data_preprocessing.data_preprocessing import DataProcessor
import configuration as config

if __name__ == '__main__':

    print('\nStarting data preprocessing...')

    data_preprocessor = DataProcessor(
        input_data_path=config.RAW_DATA_DIRECTORY,
        output_data_path=config.PROCESSED_DATA_DIRECTORY,
        n_cpus=18,
        n_planets=-1,
        downsample_fgs=True,
        verbose=True
    )

    data_preprocessor.run()

    print('\nData preprocessing complete\n')
