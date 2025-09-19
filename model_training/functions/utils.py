'''Helper functions for model training'''

# Standard library imports
import datetime
import random
import os
import shutil
from typing import Generator, Tuple
from functools import partial
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Third party imports
import h5py
import numpy as np
import tensorflow as tf

# Local imports
import configuration as config
from model_training.functions.model_definitions import cnn


# Make sure the TensorBoard log directory exists
Path(config.TENSORBOARD_LOG_DIR).mkdir(parents=True, exist_ok=True)

# Set memory growth for GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    
    except RuntimeError as e:
        print(e)


def setup_optuna_run() -> dict:
    '''Function to set up assets for an Optuna hyperparameter optimization run.'''

    # Load corrected/extracted data for a sample planet
    with h5py.File(f'{config.PROCESSED_DATA_DIRECTORY}/train.h5', 'r') as hdf:
        planet_ids = list(hdf.keys())

    # Split planets into training and validation sets
    random.shuffle(planet_ids)
    training_planet_ids = planet_ids[:len(planet_ids) // 2]
    validation_planet_ids = planet_ids[len(planet_ids) // 2:]

    # Set RDB storage for Optuna
    storage_name = f'postgresql://{config.USER}:{config.PASSWD}@{config.HOST}:{config.PORT}/{config.STUDY_NAME}'

    run_assets = {
        'training_planet_ids': training_planet_ids,
        'validation_planet_ids': validation_planet_ids,
        'storage_name': storage_name
    }

    return run_assets


def training_run(
        worker_num: int,
        training_planet_ids: list,
        validation_planet_ids: list,
        sample_size: tuple,
        learning_rate: tuple,
        l1: tuple,
        l2: tuple,
        first_filter_set: tuple,
        second_filter_set: tuple,
        third_filter_set: tuple,
        first_filter_size: tuple,
        second_filter_size: tuple,
        third_filter_size: tuple,
        dense_units: tuple,
        batch_size: tuple,
        steps: tuple,
        epochs: int
) -> float:
    '''Function to run a single training session with fixed hyperparameters.'''

    gpus = tf.config.list_physical_devices('GPU')

    if (worker_num + 1) % 2 == 0:
        tf.config.set_visible_devices(gpus[0], 'GPU')

    else:
        tf.config.set_visible_devices(gpus[1], 'GPU')

    # Create the training and validation datasets
    training_dataset, validation_dataset = create_datasets(
        training_planet_ids,
        validation_planet_ids,
        sample_size=sample_size
    )

    # Build the CNN model with the suggested hyperparameters
    model = cnn(
        samples=sample_size,
        wavelengths=config.WAVELENGTHS,
        learning_rate=learning_rate,
        l1=l1,
        l2=l2,
        filter_nums=[first_filter_set, second_filter_set, third_filter_set],
        filter_sizes=[first_filter_size, second_filter_size, third_filter_size],
        dense_units=dense_units
    )

    # Train the model
    model.fit(
        training_dataset.batch(batch_size),
        validation_data=validation_dataset.batch(batch_size),
        epochs=epochs,
        steps_per_epoch=steps,
        validation_steps=steps,
        verbose=0,
        callbacks=[early_stopping_callback(), tensorboard_callback(worker_num)]
    )

    # Evaluate the model on the validation dataset and return the RMSE
    rmse = model.evaluate(
        validation_dataset.batch(batch_size),
        steps=550, # Evaluate one sample from each validation planet
        return_dict=True,
        verbose=0
    )['RMSE']

    return rmse


def data_loader(planet_ids: list, data_file: str, sample_size: int = 100) -> Generator:
    '''Generator that yields signal, spectrum pairs for training/validation/testing.

    Args:
        planet_ids (list): List of planet IDs to include in the generator.
        data_file (str): Path to the HDF5 file containing the data.
        sample_size (int, optional): Number of frames to draw from each planet. Defaults to 100.
    '''

    with h5py.File(data_file, 'r') as hdf:

        while True:
            np.random.shuffle(planet_ids)
            
            for planet_id in planet_ids:

                signal = hdf[planet_id]['signal'][:]
                spectrum = hdf[planet_id]['spectrum'][:]

                indices = random.sample(range(signal.shape[0]), sample_size)
                sample = signal[sorted(indices), :]

                yield sample, spectrum


def create_datasets(
        training_planet_ids: list,
        validation_planet_ids: list,
        sample_size: int = 100
) -> Tuple[np.ndarray, np.ndarray]:
    '''Creates TensorFlow datasets for training and validation.

    Args:
        training_planet_ids (list): List of planet IDs to include in the training dataset.
        validation_planet_ids (list): List of planet IDs to include in the validation dataset.
        data_file (str): Path to the HDF5 file containing the data.
        sample_size (int, optional): Number of frames to draw from each planet. Defaults to 100.

    Returns:
        tuple: A tuple containing the training and validation TensorFlow datasets.
    '''

    training_data_generator = partial(
        data_loader,
        planet_ids=training_planet_ids,
        data_file=f'{config.PROCESSED_DATA_DIRECTORY}/train.h5',
        sample_size=sample_size
    )

    validation_data_generator = partial(
        data_loader,
        planet_ids=validation_planet_ids,
        data_file=f'{config.PROCESSED_DATA_DIRECTORY}/train.h5',
        sample_size=sample_size
    )

    # Create the training dataset
    training_dataset = tf.data.Dataset.from_generator(
        training_data_generator,
        output_signature=(
            tf.TensorSpec(shape=(sample_size, config.WAVELENGTHS), dtype=tf.float32),
            tf.TensorSpec(shape=(config.WAVELENGTHS,), dtype=tf.float32)
        )
    )

    # Create the validation dataset
    validation_dataset = tf.data.Dataset.from_generator(
        validation_data_generator,
        output_signature=(
            tf.TensorSpec(shape=(sample_size, config.WAVELENGTHS), dtype=tf.float32),
            tf.TensorSpec(shape=(config.WAVELENGTHS,), dtype=tf.float32)
        )
    )

    return training_dataset, validation_dataset


def tensorboard_callback(worker_num: int) -> tf.keras.callbacks.TensorBoard:
    '''Function to create a TensorBoard callback with a unique log directory.'''

    # Set tensorboard callback
    log_dir = config.TENSORBOARD_LOG_DIR + f'{worker_num}-{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1
    )

    return tensorboard_callback


def early_stopping_callback() -> tf.keras.callbacks.EarlyStopping:

    '''Function to create an early stopping callback.'''

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        min_delta=1e-4,
        mode='min',
        verbose=0,
        restore_best_weights=True
    )

    return early_stopping_callback


def clear_tensorboard_logs() -> None:
    '''Function to clear the TensorBoard log directory.'''

    try:
        shutil.rmtree(f'{config.TENSORBOARD_LOG_DIR}')
    except FileNotFoundError:
        pass

    Path(config.TENSORBOARD_LOG_DIR).mkdir(parents=True, exist_ok=True)