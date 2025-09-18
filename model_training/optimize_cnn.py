'''Optimize CNN model hyperparameters using Optuna and visualize results with Optuna Dashboard'''

# Standard library imports
import random

# Third party imports
import h5py
import optuna

# Local imports
import configuration as config
from model_training.functions.utils import training_run


def run(worker_num: int, hyperparams: dict = config.hyperparams):
    '''Main function to start Optuna optimization run.'''

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
        load_if_exists=True
    )

    study.optimize(
        lambda trial: objective(
            trial,
            training_planet_ids,
            validation_planet_ids,
            hyperparams,
            worker_num
        ),
        n_trials=10000
    )

def objective(trial, training_planet_ids, validation_planet_ids, hyperparams, worker_num) -> float:
    '''Objective function for Optuna CNN hyperparameter optimization.'''

    rmse = training_run(
        worker_num,
        training_planet_ids,
        validation_planet_ids,
        trial.suggest_categorical(hyperparams['sample_size']),
        trial.suggest_float('learning_rate', hyperparams['learning_rate'][0], hyperparams['learning_rate'][1], log=True),
        trial.suggest_float('l_one', hyperparams['l_one'][0], hyperparams['l_one'][1], log=True),
        trial.suggest_float('l_two', hyperparams['l_two'][0], hyperparams['l_two'][1], log=True),
        trial.suggest_categorical('first_filter_set', hyperparams['first_filter_set']),
        trial.suggest_categorical('second_filter_set', hyperparams['second_filter_set']),
        trial.suggest_categorical('third_filter_set', hyperparams['third_filter_set']),
        trial.suggest_categorical('filter_size', hyperparams['filter_size']),
        trial.suggest_categorical('batch_size', hyperparams['batch_size']),
        trial.suggest_categorical('steps', hyperparams['steps'])
    )
    
    return rmse