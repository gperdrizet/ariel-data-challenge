'''Optimize CNN model hyperparameters using Optuna and visualize results with Optuna Dashboard'''

# Standard library imports
import random

# Third party imports
import h5py
import optuna

# Local imports
import configuration as config
from model_training.functions.model_definitions import cnn
from model_training.functions.utils import data_loader


def run():
    '''Main function to start Optuna optimization run.'''

    # Load corrected/extracted data for a sample planet
    with h5py.File(f'{config.PROCESSED_DATA_DIRECTORY}/train.h5', 'r') as hdf:
        planet_ids = list(hdf.keys())

    # Split planets into training and validation sets
    random.shuffle(planet_ids)
    training_planet_ids = planet_ids[:len(planet_ids) // 2]
    validation_planet_ids = planet_ids[len(planet_ids) // 2:]

    # Set RDB storage for Optuna
    storage_name=f'postgresql://{config.USER}:{config.PASSWD}@{config.HOST}:{config.PORT}/{config.STUDY_NAME}'
    print(f'\nRDB storage: {storage_name}')