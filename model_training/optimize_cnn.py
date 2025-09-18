'''Optimize CNN model hyperparameters using Optuna and visualize results with Optuna Dashboard'''

# Standard library imports
import random
import shutil
from pathlib import Path

# Third party imports
import h5py
import optuna

# Local imports
import configuration as config
from model_training.functions.utils import training_run


def run():
    '''Main function to start Optuna optimization run.'''

    # Deal with tensorboard log directory
    try:
        shutil.rmtree(config.TENSORBOARD_LOG_DIR)
    except FileNotFoundError:
        pass
    
    Path(config.TENSORBOARD_LOG_DIR).mkdir(parents=True, exist_ok=True)

    # Load corrected/extracted data for a sample planet
    with h5py.File(f'{config.PROCESSED_DATA_DIRECTORY}/train.h5', 'r') as hdf:
        planet_ids = list(hdf.keys())

    # Split planets into training and validation sets
    random.shuffle(planet_ids)
    training_planet_ids = planet_ids[:len(planet_ids) // 2]
    validation_planet_ids = planet_ids[len(planet_ids) // 2:]

    # Set RDB storage for Optuna
    storage_name = f'postgresql://{config.USER}:{config.PASSWD}@{config.HOST}:{config.PORT}/{config.STUDY_NAME}'
    print(f'\nRDB storage: {storage_name}')

    # Define the study
    study = optuna.create_study(
        study_name='cnn_optimization',
        direction='minimize',
        storage=storage_name,
        load_if_exists=False
    )

    study.optimize(
        lambda trial: objective(
            trial,
            training_planet_ids,
            validation_planet_ids
        ),
        n_trials=10000
    )

def objective(trial, training_planet_ids, validation_planet_ids) -> float:
    '''Objective function for Optuna CNN hyperparameter optimization.'''

    rmse = training_run(
        training_planet_ids,
        validation_planet_ids,
        trial.suggest_categorical('sample_size', [50, 100, 200, 500]),
        trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
        trial.suggest_float('l_one', 1e-10, 1e-2, log=True),
        trial.suggest_float('l_two', 1e-10, 1e-2, log=True),
        trial.suggest_categorical('first_filter_set', (8, 16, 32, 64)),
        trial.suggest_categorical('second_filter_set', (16, 32, 64, 128)),
        trial.suggest_categorical('third_filter_set', (32, 64, 128, 256)),
        trial.suggest_categorical('filter_size', (2, 3, 4, 5)),
        trial.suggest_categorical('batch_size', (1, 2, 4, 8, 16, 32, 64, 128)),
        trial.suggest_categorical('steps', (10, 25, 50, 100, 200))
    )
    
    return rmse