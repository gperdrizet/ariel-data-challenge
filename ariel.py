'''Main runner for the signal correction & extraction pipeline.'''

# Standard library imports
import argparse
import time

# Internal imports
import configuration as config
from ariel_data_preprocessing.data_preprocessing import DataProcessor
from model_training import optimize_cnn
if __name__ == '__main__':

    parser=argparse.ArgumentParser()
    
    parser.add_argument(
        '--task',
        choices=['preprocess_data', 'optimize_cnn'],
        help='task to run'
    )

    args=parser.parse_args()

    if args.task == 'preprocess_data':

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

    if args.task == 'optimize_cnn':

        print('\nStarting CNN hyperparameter optimization...')
        start_time = time.time()

        optimize_cnn.run()

        elapsed_time = time.time() - start_time
        print(f'\nCNN hyperparameter optimization complete in {elapsed_time/(60 * 60):.2f} hours\n')