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

from ariel_data_preprocessing.transit_differencing import spectral_difference_pairs


def _training_data_loader(
        planet_ids: list,
        data_file: str,
        pairs_per_planet: int = 100
):

    with h5py.File(data_file, 'r') as hdf:

        while True:
            np.random.shuffle(planet_ids)
            
            for planet_id in planet_ids:

                spectrum = hdf[planet_id]['spectrum'][:]
                signal = hdf[planet_id]['signal'][:]
                difference_pairs = spectral_difference_pairs(signal, pairs_per_planet)

                for i in range(pairs_per_planet):
                    yield difference_pairs[i], spectrum


def _evaluation_data_loader(
        planet_ids: list,
        data_file: str,
        pairs_per_planet: int = 100
):

    with h5py.File(data_file, 'r') as hdf:

        while True:
            
            for planet_id in planet_ids:

                signal = hdf[planet_id]['signal'][:]
                difference_pairs = spectral_difference_pairs(signal, pairs_per_planet)
                spectra = [hdf[planet_id]['spectrum'][:]] * pairs_per_planet

                yield np.array(difference_pairs), np.array(spectra)


def _testing_data_loader(
        planet_ids: list,
        data_file: str,
        pairs_per_planet: int = 100
):

    with h5py.File(data_file, 'r') as hdf:

        while True:
            
            for planet_id in planet_ids:

                signal = hdf[planet_id]['signal'][:]
                difference_pairs = spectral_difference_pairs(signal, pairs_per_planet)

                yield np.array(difference_pairs)


def make_training_difference_pairs(
        data_file: str,
        output_data_path: str = '.',
        pairs_per_planet: int = 100,
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
        pairs_per_planet=pairs_per_planet
    )

    training_dataset = tf.data.Dataset.from_generator(
        training_data_generator,
        output_signature=(
            tf.TensorSpec(shape=(wavelengths), dtype=tf.float64),
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
            pairs_per_planet=pairs_per_planet
        )

        validation_dataset = tf.data.Dataset.from_generator(
            validation_data_generator,
            output_signature=(
                tf.TensorSpec(shape=(wavelengths), dtype=tf.float64),
                tf.TensorSpec(shape=(wavelengths), dtype=tf.float64)
            )
        )

        evaluation_data_generator = partial(
            _evaluation_data_loader,
            planet_ids=validation_planet_ids,
            data_file=data_file,
            pairs_per_planet=pairs_per_planet
        )

        evaluation_dataset = tf.data.Dataset.from_generator(
            evaluation_data_generator,
            output_signature=(
                tf.TensorSpec(shape=(pairs_per_planet, wavelengths), dtype=tf.float64),
                tf.TensorSpec(shape=(pairs_per_planet, wavelengths), dtype=tf.float64)
            )
        )

    return training_dataset, validation_dataset, evaluation_dataset


def make_testing_difference_pairs(
        data_file: str,
        pairs_per_planet: int = 100,
        wavelengths: int = 283
) -> tuple:

    with h5py.File(data_file, 'r') as hdf:
        planet_ids = list(hdf.keys())

    training_data_generator = partial(
        _testing_data_loader,
        planet_ids=planet_ids,
        data_file=data_file,
        pairs_per_planet=pairs_per_planet
    )

    dataset = tf.data.Dataset.from_generator(
        training_data_generator,
        output_signature=(
            tf.TensorSpec(shape=(wavelengths), dtype=tf.float64)
        )
    )

    return dataset