'''Helper functions for model training'''

# Standard library imports
import datetime
import random
import shutil
from functools import partial
from pathlib import Path

# Third party imports
import h5py
import numpy as np
import tensorflow as tf

# Local imports
import configuration as config
from model_training.functions.model_definitions import cnn


def data_loader(planet_ids: list, data_file: str, sample_size: int = 100):
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
) -> tuple:
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


def training_run(
        training_planet_ids: list,
        validation_planet_ids: list,
        sample_size: int,
        learning_rate: float,
        l1: float,
        l2: float,
        first_filter_set: int,
        second_filter_set: int,
        third_filter_set: int,
        filter_size: int,
        batch_size: int,
        steps: int
) -> float:
    '''Function to run a single training session with fixed hyperparameters.'''

    # Create the training and validation datasets
    training_dataset, validation_dataset = create_datasets(
        training_planet_ids,
        validation_planet_ids,
        sample_size=sample_size
    )

    # Build the CNN model with the suggested hyperparameters
    model = cnn(
        frames=sample_size,
        wavelengths=config.WAVELENGTHS,
        learning_rate=learning_rate,
        l1=l1,
        l2=l2,
        filter_nums=[first_filter_set, second_filter_set, third_filter_set],
        filter_size=filter_size
    )

    # Train the model
    model.fit(
        training_dataset.batch(batch_size),
        validation_data=validation_dataset.batch(batch_size),
        epochs=config.EPOCHS,
        steps_per_epoch=steps,
        validation_steps=steps,
        verbose=0,
        callbacks=[early_stopping_callback(), tensorboard_callback()]
    )

    # Evaluate the model on the validation dataset and return the RMSE
    rmse = model.evaluate(
        validation_dataset.batch(batch_size),
        return_dict=True,
        verbose=0
    )['RMSE']

    return rmse


def tensorboard_callback():
    '''Function to create a TensorBoard callback with a unique log directory.'''

    # Deal with tensorboard log directory
    try:
        shutil.rmtree(config.TENSORBOARD_LOG_DIR)
    except FileNotFoundError:
        pass
    
    Path(config.TENSORBOARD_LOG_DIR).mkdir(parents=True, exist_ok=True)

    # Set tensorboard callback
    log_dir = config.TENSORBOARD_LOG_DIR + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1
    )

    return tensorboard_callback

def early_stopping_callback():
    '''Function to create an early stopping callback.'''

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        min_delta=0.000005,
        mode='min',
        verbose=1,
        restore_best_weights=True
    )

    return early_stopping_callback