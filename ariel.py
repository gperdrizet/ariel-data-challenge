'''Main runner for the signal correction & extraction pipeline.'''

# Standard library imports
import argparse
import time
import multiprocessing as mp

# Internal imports
import configuration as config
from ariel_data_preprocessing.data_preprocessing import DataProcessor
from model_training import optimize_cnn
from model_training.functions.utils import clear_tensorboard_logs

mp.set_start_method('spawn', force=True)

if __name__ == '__main__':

    parser=argparse.ArgumentParser()

    parser.add_argument(
        '--task',
        choices=['preprocess_training_data', 'preprocess_testing_data', 'optimize_cnn'],
        help='task to run'
    )

    parser.add_argument(
        '--clear_tensorboard_logs', 
        type=str, 
        default='False', 
        help='Delete Tensorboard logs from previous run (optional)'
    )



    args=parser.parse_args()

    if args.task == 'preprocess_training_data':

        print('\nStarting training data preprocessing...')
        start_time = time.time()

        data_preprocessor = DataProcessor(
            input_data_path=config.RAW_DATA_DIRECTORY,
            output_data_path=config.PROCESSED_DATA_DIRECTORY,
            n_cpus=18,
            n_planets=-1,
            downsample_fgs=True,
            verbose=True,
            mode='train'
        )

        data_preprocessor.run()

        elapsed_time = time.time() - start_time
        print(f'\nData preprocessing complete in {elapsed_time/60:.2f} minutes\n')


    if args.task == 'preprocess_testing_data':

        print('\nStarting testing data preprocessing...')
        start_time = time.time()

        data_preprocessor = DataProcessor(
            input_data_path=config.RAW_DATA_DIRECTORY,
            output_data_path=config.PROCESSED_DATA_DIRECTORY,
            output_filename='test.h5',
            n_cpus=1,
            n_planets=-1,
            downsample_fgs=True,
            verbose=True,
            mode='test'
        )

        data_preprocessor.run()

        elapsed_time = time.time() - start_time
        print(f'\nData preprocessing complete in {elapsed_time/60:.2f} minutes\n')


    if args.task == 'optimize_cnn':

        print('\nStarting CNN hyperparameter optimization...')

        if args.clear_tensorboard_logs.lower() in ['true', '1', 'yes']:
            print('Clearing Tensorboard logs from previous run...')
            clear_tensorboard_logs()

        start_time = time.time()

        with mp.Pool(processes=2) as pool:
            pool.map(optimize_cnn.run, range(2))

        elapsed_time = time.time() - start_time
        print(f'\nCNN hyperparameter optimization complete in {elapsed_time/(60 * 60):.2f} hours\n')
