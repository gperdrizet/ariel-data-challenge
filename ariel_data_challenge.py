'''Main runner function for calorie expenditure project.'''

import argparse

import configuration as config
import functions.data_acquisition as data_acquisition

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        '--task',
        choices=['get_data'],
        help='task to run'
    )

    args=parser.parse_args()

    if args.task == 'ingest_data':
        data_acquisition.download_dataset(config.COMPETITION_NAME, config.RAW_DATA_DIRECTORY)
        data_acquisition.extract_dataset(config.COMPETITION_NAME, config.RAW_DATA_DIRECTORY)