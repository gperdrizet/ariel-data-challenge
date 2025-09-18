'''Optimize CNN model hyperparameters using Optuna and visualize results with Optuna Dashboard'''

# Standard library imports
import datetime
import random
from functools import partial

# Third party imports
import h5py
import optuna
import tensorflow as tf

# Local imports
import configuration as config
from model_training.functions.model_definitions import cnn
from model_training.functions.utils import create_datasets


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
        lambda trial: objective(trial, training_planet_ids, validation_planet_ids),
        n_trials=10000
    )

def objective(trial, training_planet_ids, validation_planet_ids) -> float:
    '''Objective function for Optuna CNN hyperparameter optimization.'''

    # Optimize the sample size used in the data generator
    sample_size = trial.suggest_categorical('sample_size', [50, 100, 200, 500])

    # Create the training and validation datasets
    training_dataset, validation_dataset = create_datasets(
        training_planet_ids,
        validation_planet_ids,
        sample_size=sample_size
    )

    # Optimize the CNN hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [1, 2, 4, 8, 16, 32, 64, 128])
    steps = trial.suggest_categorical('steps', [10, 25, 50, 100, 200])

    l1 = trial.suggest_float('l1', 1e-10, 1e-2, log=True)
    l2 = trial.suggest_float('l2', 1e-10, 1e-2, log=True)
    
    filter_nums = [
        trial.suggest_categorical('filter_num_1', [8, 16, 32]),
        trial.suggest_categorical('filter_num_2', [16, 32, 64]),
        trial.suggest_categorical('filter_num_3', [32, 64, 128]),
        trial.suggest_categorical('filter_num_3', [64, 128, 256]),
    ]
    
    filter_size = trial.suggest_categorical('filter_size', [(2, 2), (3, 3), (4, 4), (5, 5)])


    # Build the CNN model with the suggested hyperparameters
    model = cnn(
        frames=config.FRAMES,
        wavelengths=config.WAVELENGTHS,
        learning_rate=learning_rate,
        l1=l1,
        l2=l2,
        filter_nums=filter_nums,
        filter_size=filter_size
    )

    # Set early stopping callback
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        min_delta=0.000005,
        mode='min',
        verbose=1,
        restore_best_weights=True
    )

    # Set tensorboard callback
    log_dir = config.TENSORBOARD_LOG_DIR + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1
    )

    # Train the model
    model.fit(
        training_dataset.batch(batch_size),
        validation_data=validation_dataset.batch(batch_size),
        epochs=config.EPOCHS,
        steps_per_epoch=steps,
        validation_steps=steps,
        verbose=0,
        callbacks=[early_stopping_callback, tensorboard_callback]
    )

    # Evaluate the model on the validation dataset and return the RMSE
    rmse = model.evaluate(
        validation_dataset.batch(batch_size),
        return_dict=True,
        verbose=0
    )['RMSE']

    print(f'Trial {trial.number}: Validation RMSE={rmse:.5f} with params: '
          f'learning_rate={learning_rate}, l1={l1}, l2={l2}, '
          f'filter_nums={filter_nums}, filter_size={filter_size}')

    return rmse