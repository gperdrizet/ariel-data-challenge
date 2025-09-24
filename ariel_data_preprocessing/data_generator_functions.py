'''Functions to set up data generators using Tensorflow for training and validation datasets.'''

# Standard library imports
from functools import partial
import pickle
import random

# Third party imports
import h5py
import numpy as np
import tensorflow as tf


def _training_data_loader(
        planet_ids: list,
        data_file: str,
        sample_size: int = 100,
        smoothing_window: int = None,
        standardize_wavelengths: bool = True
):

    with h5py.File(data_file, 'r') as hdf:
        while True:

            np.random.shuffle(planet_ids)
            
            for planet_id in planet_ids:

                spectrum = hdf[planet_id]['spectrum'][:]

                # Select the appropriate smoothed signal
                if smoothing_window is None:
                    signal = hdf[planet_id]['smoothing_none'][:]
                    mask = hdf[planet_id]['smoothing_none_mask'][:]
                    mask = np.tile(mask, (signal.shape[0], 1))
                    signal = np.ma.MaskedArray(signal, mask=mask)

                else:
                    signal = hdf[planet_id][f'smoothing_{smoothing_window}'][:]
                    mask = hdf[planet_id][f'smoothing_{smoothing_window}_mask'][:]
                    mask = np.tile(mask, (signal.shape[0], 1))
                    signal = np.ma.MaskedArray(signal, mask=mask)

                # Standardize each wavelength across frames, if asked
                if standardize_wavelengths:
                    row_means = np.mean(signal, axis=0)
                    row_stds = np.std(signal, axis=0)
                    signal = (signal - row_means[np.newaxis, :]) / row_stds[np.newaxis, :]

                indices = random.sample(range(signal.shape[0]), sample_size)
                sample = signal[sorted(indices), :]

                yield sample, spectrum


def _evaluation_data_loader(
        planet_ids: list,
        data_file: str,
        sample_size: int = 100,
        n_samples: int = 10,
        smoothing_window: int = None,
        standardize_wavelengths: bool = True
):

    with h5py.File(data_file, 'r') as hdf:
        while True:
            for planet_id in planet_ids:

                signal = hdf[planet_id]['signal'][:]

                # Select the appropriate smoothed signal
                if smoothing_window is None:
                    signal = hdf[planet_id]['smoothing_none'][:]
                    mask = hdf[planet_id]['smoothing_none_mask'][:]
                    mask = np.tile(mask, (signal.shape[0], 1))
                    signal = np.ma.MaskedArray(signal, mask=mask)

                else:
                    signal = hdf[planet_id][f'smoothing_{smoothing_window}'][:]
                    mask = hdf[planet_id][f'smoothing_{smoothing_window}_mask'][:]
                    mask = np.tile(mask, (signal.shape[0], 1))
                    signal = np.ma.MaskedArray(signal, mask=mask)

                # Standardize each wavelength across frames
                if standardize_wavelengths:
                    row_means = np.mean(signal, axis=0)
                    row_stds = np.std(signal, axis=0)

                    signal = (signal - row_means[np.newaxis, :]) / row_stds[np.newaxis, :]

                samples = []
                spectra = []

                for _ in range(n_samples):

                    indices = random.sample(range(signal.shape[0]), sample_size)
                    samples.append(signal[sorted(indices), :])
                    spectra.append(hdf[planet_id]['spectrum'][:])

                yield np.array(samples), np.array(spectra)


def _testing_data_loader(
        planet_ids: list,
        data_file: str,
        sample_size: int = 100,
        n_samples: int = 10,
        smoothing_window: int = None,
        standardize_wavelengths: bool = True
):

    with h5py.File(data_file, 'r') as hdf:
        while True:
            for planet_id in planet_ids:

                # Select the appropriate smoothed signal
                if smoothing_window is None:
                    signal = hdf[planet_id]['smoothing_none'][:]
                    mask = hdf[planet_id]['smoothing_none_mask'][:]
                    mask = np.tile(mask, (signal.shape[0], 1))
                    signal = np.ma.MaskedArray(signal, mask=mask)

                else:
                    signal = hdf[planet_id][f'smoothing_{smoothing_window}'][:]
                    mask = hdf[planet_id][f'smoothing_{smoothing_window}_mask'][:]
                    mask = np.tile(mask, (signal.shape[0], 1))
                    signal = np.ma.MaskedArray(signal, mask=mask)

                # Standardize each wavelength across frames
                if standardize_wavelengths:
                    row_means = np.mean(signal, axis=0)
                    row_stds = np.std(signal, axis=0)

                    signal = (signal - row_means[np.newaxis, :]) / row_stds[np.newaxis, :]

                samples = []

                for _ in range(n_samples):

                    indices = random.sample(range(signal.shape[0]), sample_size)
                    samples.append(signal[sorted(indices), :])

                yield np.array(samples)


def make_training_datasets(
        data_file: str,
        sample_size: int,
        n_samples: int = 10,
        wavelengths: int = 283,
        validation: bool = True,
        smoothing_window: int = None,
        standardize_wavelengths: bool = True
) -> tuple:


    if validation:

        # Load the training and validation planet IDs for this dataset
        base_filename = data_file.split('.')[0]
        validation_split_filename = f'{base_filename}_validation_split.pkl'

        with open(validation_split_filename, 'rb') as input_file:
            planet_ids = pickle.load(input_file)
            training_planet_ids = planet_ids['training']
            validation_planet_ids = planet_ids['validation']

    else:

        with h5py.File(data_file, 'r') as hdf:
            planet_ids = list(hdf.keys())
            random.shuffle(planet_ids)
            training_planet_ids = planet_ids

    training_data_generator = partial(
        _training_data_loader,
        planet_ids=training_planet_ids,
        data_file=data_file,
        sample_size=sample_size,
        smoothing_window=smoothing_window,
        standardize_wavelengths=standardize_wavelengths
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
            sample_size=sample_size,
            smoothing_window=smoothing_window,
            standardize_wavelengths=standardize_wavelengths
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
            n_samples=n_samples,
            smoothing_window=smoothing_window,
            standardize_wavelengths=standardize_wavelengths
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
        wavelengths: int = 283,
        smoothing_window: int = None,
        standardize_wavelengths: bool = True
) -> tuple:

    with h5py.File(data_file, 'r') as hdf:
        planet_ids = list(hdf.keys())

    training_data_generator = partial(
        _testing_data_loader,
        planet_ids=planet_ids,
        data_file=data_file,
        sample_size=sample_size,
        n_samples=n_samples,
        smoothing_window=smoothing_window,
        standardize_wavelengths=standardize_wavelengths
    )

    dataset = tf.data.Dataset.from_generator(
        training_data_generator,
        output_signature=(
            tf.TensorSpec(shape=(n_samples, sample_size, wavelengths), dtype=tf.float64)
        )
    )

    return dataset