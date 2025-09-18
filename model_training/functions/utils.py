'''Helper functions for model training'''

# Standard library imports
import random
from functools import partial

# Third party imports
import h5py
import numpy as np
import tensorflow as tf

# Local imports
import configuration as config


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