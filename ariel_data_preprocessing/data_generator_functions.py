'''Functions to set up data generators using Tensorflow for training and validation datasets.'''

# Standard library imports
from functools import partial
from pathlib import Path
import pickle
import random

# Third party imports
import h5py
import numpy as np
import tensorflow as tf


def _training_data_loader(planet_ids: list, data_file: str, sample_size: int = 100):
    '''Generator that yields signal - spectrum pairs for training/validation.

    Args:
        planet_ids (list): List of planet IDs to include in the generator.
        data_file (str): Path to the HDF5 file containing the data.
        sample_size (int, optional): Number of frames to draw for each sample. Defaults to 100.
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


def _evaluation_data_loader(
        planet_ids: list,
        data_file: str,
        sample_size: int = 100,
        n_samples: int = 10
):
    '''Generator that yields signal, spectrum pairs for training/validation/testing.

    Args:
        planet_ids (list): List of planet IDs to include in the generator.
        data_file (str): Path to the HDF5 file containing the data.
        sample_size (int, optional): Number of frames to draw from each planet. Defaults to 100.
    '''

    with h5py.File(data_file, 'r') as hdf:

        while True:
            
            for planet_id in planet_ids:

                signal = hdf[planet_id]['signal'][:]

                samples = []
                spectra = []

                for _ in range(n_samples):

                    indices = random.sample(range(signal.shape[0]), sample_size)
                    samples.append(signal[sorted(indices), :])
                    spectra.append(hdf[planet_id]['spectrum'][:])

                yield np.array(samples), np.array(spectra)


def _testing_data_loader(planet_ids: list, data_file: str, sample_size: int = 100, n_samples: int = 10):
    '''Generator that yields signal for prediction on testing data.

    Args:
        planet_ids (list): List of planet IDs to include in the generator.
        data_file (str): Path to the HDF5 file containing the data.
        sample_size (int, optional): Number of frames to draw for each sample. Defaults to 100.
        n_samples (int, optional): Number of samples to draw per planet. Defaults to 10.
    '''

    with h5py.File(data_file, 'r') as hdf:

        while True:
            
            for planet_id in planet_ids:

                signal = hdf[planet_id]['signal'][:]
                samples = []

                for _ in range(n_samples):

                    indices = random.sample(range(signal.shape[0]), sample_size)
                    samples.append(signal[sorted(indices), :])

                yield np.array(samples)


def make_training_datasets(
        data_file: str,
        sample_size: int,
        output_data_path: str = '.',
        n_samples: int = 10,
        wavelengths: int = 283,
        validation: bool = True
) -> tuple:
    
    with h5py.File(data_file, 'r') as hdf:
        planet_ids = list(hdf.keys())

    random.shuffle(planet_ids)

    if validation:

        planet_ids_file = f'{output_data_path}/training_validation_split_planet_ids.pkl'

        if Path(planet_ids_file).exists():

            with open(planet_ids_file, 'rb') as input_file:
                planet_ids = pickle.load(input_file)
                training_planet_ids = planet_ids['training']
                validation_planet_ids = planet_ids['validation']

            print('Loaded existing training/validation split')

        else:
            
            print('Creating new training/validation split')

            random.shuffle(planet_ids)
            training_planet_ids = planet_ids[:len(planet_ids) // 2]
            validation_planet_ids = planet_ids[len(planet_ids) // 2:]

            # Save the training and validation planet IDs
            planet_ids = {
                'training': training_planet_ids,
                'validation': validation_planet_ids
            }

            with open(planet_ids_file, 'wb') as output_file:
                pickle.dump(planet_ids, output_file)

    else:
        training_planet_ids = planet_ids

    training_data_generator = partial(
        _training_data_loader,
        planet_ids=training_planet_ids,
        data_file=data_file,
        sample_size=sample_size
    )

    training_dataset = tf.data.Dataset.from_generator(
        training_data_generator,
        output_signature=(
            tf.TensorSpec(shape=(sample_size, wavelengths), dtype=tf.float64),
            tf.TensorSpec(shape=(wavelengths), dtype=tf.float64)
        )
    )

    validation_dataset = None
    evaluation_dataset = None

    if validation:
        validation_data_generator = partial(
            _training_data_loader,
            planet_ids=validation_planet_ids,
            data_file=data_file,
            sample_size=sample_size
        )

        validation_dataset = tf.data.Dataset.from_generator(
            validation_data_generator,
            output_signature=(
                tf.TensorSpec(shape=(sample_size, wavelengths), dtype=tf.float64),
                tf.TensorSpec(shape=(wavelengths), dtype=tf.float64)
            )
        )

        evaluation_data_generator = partial(
            _evaluation_data_loader,
            planet_ids=validation_planet_ids,
            data_file=data_file,
            sample_size=sample_size,
            n_samples=n_samples
        )

        evaluation_dataset = tf.data.Dataset.from_generator(
            evaluation_data_generator,
            output_signature=(
                tf.TensorSpec(shape=(n_samples, sample_size, wavelengths), dtype=tf.float64),
                tf.TensorSpec(shape=(n_samples, wavelengths), dtype=tf.float64)
            )
        )

    return training_dataset, validation_dataset, evaluation_dataset


def make_testing_dataset(
        data_file: str,
        sample_size: int,
        n_samples: int = 10,
        wavelengths: int = 283
) -> tuple:

    with h5py.File(data_file, 'r') as hdf:
        planet_ids = list(hdf.keys())

    training_data_generator = partial(
        _testing_data_loader,
        planet_ids=planet_ids,
        data_file=data_file,
        sample_size=sample_size,
        n_samples=n_samples
    )

    dataset = tf.data.Dataset.from_generator(
        training_data_generator,
        output_signature=(
            tf.TensorSpec(shape=(n_samples, sample_size, wavelengths), dtype=tf.float64)
        )
    )

    return dataset