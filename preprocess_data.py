'''Main runner for the signal correction & extraction pipeline.'''

# Standard library imports
import time

# Internal imports
from ariel_data_preprocessing.data_preprocessing import DataProcessor
import configuration as config

if __name__ == '__main__':

    print('\nStarting data preprocessing...')
    start_time = time.time()

    data_preprocessor = DataProcessor(
        input_data_path=config.RAW_DATA_DIRECTORY,
        output_data_path=config.PROCESSED_DATA_DIRECTORY,
        n_cpus=18,
        n_planets=-1,
        downsample_fgs=True,
        verbose=True
    )

    data_preprocessor.run()

    elapsed_time = time.time() - start_time
    print(f'\nData preprocessing complete in {elapsed_time/60:.2f} minutes\n')
