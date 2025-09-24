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

                    transit_signal = hdf[planet_id]['smoothing_none_transit'][:]
                    non_transit_signal = hdf[planet_id]['smoothing_none_non_transit'][:]

                    transit_mask = hdf[planet_id]['smoothing_none_transit_mask'][:]
                    non_transit_mask = hdf[planet_id]['smoothing_none_non_transit_mask'][:]

                    transit_mask = np.tile(transit_mask, (transit_signal.shape[0], 1))
                    non_transit_mask = np.tile(non_transit_mask, (non_transit_signal.shape[0], 1))

                    transit_signal = np.ma.MaskedArray(transit_signal, mask=transit_mask)
                    non_transit_signal = np.ma.MaskedArray(non_transit_signal, mask=non_transit_mask)

                else:
                    transit_signal = hdf[planet_id][f'smoothing_{smoothing_window}_transit'][:]
                    non_transit_signal = hdf[planet_id][f'smoothing_{smoothing_window}_non_transit'][:]

                    transit_mask = hdf[planet_id][f'smoothing_{smoothing_window}_transit_mask'][:]
                    non_transit_mask = hdf[planet_id][f'smoothing_{smoothing_window}_non_transit_mask'][:]

                    transit_mask = np.tile(transit_mask, (transit_signal.shape[0], 1))
                    non_transit_mask = np.tile(non_transit_mask, (non_transit_signal.shape[0], 1))

                    transit_signal = np.ma.MaskedArray(transit_signal, mask=transit_mask)
                    non_transit_signal = np.ma.MaskedArray(non_transit_signal, mask=non_transit_mask)

                # Standardize each wavelength across frames, if asked
                if standardize_wavelengths:
                    transit_row_means = np.mean(transit_signal, axis=0)
                    transit_row_stds = np.std(transit_signal, axis=0)
                    transit_signal = (transit_signal - transit_row_means[np.newaxis, :]) / transit_row_stds[np.newaxis, :]

                    non_transit_row_means = np.mean(non_transit_signal, axis=0)
                    non_transit_row_stds = np.std(non_transit_signal, axis=0)
                    non_transit_signal = (non_transit_signal - non_transit_row_means[np.newaxis, :]) / non_transit_row_stds[np.newaxis, :]

                # Yield sample size spectral difference pairs from this planet, one at a time
                transit_indices = random.sample(range(transit_signal.shape[0]), sample_size)
                non_transit_indices = random.sample(range(non_transit_signal.shape[0]), sample_size)
                
                for transit_idx, non_transit_idx in zip(transit_indices, non_transit_indices):
                    yield transit_signal[transit_idx] - non_transit_signal[non_transit_idx], spectrum


def _evaluation_data_loader(
        planet_ids: list,
        data_file: str,
        sample_size: int = 100,
        smoothing_window: int = None,
        standardize_wavelengths: bool = True
):

    with h5py.File(data_file, 'r') as hdf:
        while True:
            for planet_id in planet_ids:

                spectrum = hdf[planet_id]['spectrum'][:]

                # Select the appropriate smoothed signal
                if smoothing_window is None:

                    transit_signal = hdf[planet_id]['smoothing_none_transit'][:]
                    non_transit_signal = hdf[planet_id]['smoothing_none_non_transit'][:]

                    transit_mask = hdf[planet_id]['smoothing_none_transit_mask'][:]
                    non_transit_mask = hdf[planet_id]['smoothing_none_non_transit_mask'][:]

                    transit_mask = np.tile(transit_mask, (transit_signal.shape[0], 1))
                    non_transit_mask = np.tile(non_transit_mask, (non_transit_signal.shape[0], 1))

                    transit_signal = np.ma.MaskedArray(transit_signal, mask=transit_mask)
                    non_transit_signal = np.ma.MaskedArray(non_transit_signal, mask=non_transit_mask)

                else:
                    transit_signal = hdf[planet_id][f'smoothing_{smoothing_window}_transit'][:]
                    non_transit_signal = hdf[planet_id][f'smoothing_{smoothing_window}_non_transit'][:]

                    transit_mask = hdf[planet_id][f'smoothing_{smoothing_window}_transit_mask'][:]
                    non_transit_mask = hdf[planet_id][f'smoothing_{smoothing_window}_non_transit_mask'][:]

                    transit_mask = np.tile(transit_mask, (transit_signal.shape[0], 1))
                    non_transit_mask = np.tile(non_transit_mask, (non_transit_signal.shape[0], 1))

                    transit_signal = np.ma.MaskedArray(transit_signal, mask=transit_mask)
                    non_transit_signal = np.ma.MaskedArray(non_transit_signal, mask=non_transit_mask)

                # Standardize each wavelength across frames, if asked
                if standardize_wavelengths:
                    transit_row_means = np.mean(transit_signal, axis=0)
                    transit_row_stds = np.std(transit_signal, axis=0)
                    transit_signal = (transit_signal - transit_row_means[np.newaxis, :]) / transit_row_stds[np.newaxis, :]

                    non_transit_row_means = np.mean(non_transit_signal, axis=0)
                    non_transit_row_stds = np.std(non_transit_signal, axis=0)
                    non_transit_signal = (non_transit_signal - non_transit_row_means[np.newaxis, :]) / non_transit_row_stds[np.newaxis, :]

                # Yield sample size spectral difference pairs from this planet in a batch
                transit_indices = random.sample(range(transit_signal.shape[0]), sample_size)
                non_transit_indices = random.sample(range(non_transit_signal.shape[0]), sample_size)
                
                signals = []
                spectra = []

                for transit_idx, non_transit_idx in zip(transit_indices, non_transit_indices):
                    signals.append(transit_signal[transit_idx] - non_transit_signal[non_transit_idx])
                    spectra.append(spectrum)

                yield np.array(signals), np.array(spectra)


def _testing_data_loader(
        planet_ids: list,
        data_file: str,
        sample_size: int = 100,
        smoothing_window: int = None,
        standardize_wavelengths: bool = True
):

    with h5py.File(data_file, 'r') as hdf:
        while True:
            for planet_id in planet_ids:

                # Select the appropriate smoothed signal
                if smoothing_window is None:

                    transit_signal = hdf[planet_id]['smoothing_none_transit'][:]
                    non_transit_signal = hdf[planet_id]['smoothing_none_non_transit'][:]

                    transit_mask = hdf[planet_id]['smoothing_none_transit_mask'][:]
                    non_transit_mask = hdf[planet_id]['smoothing_none_non_transit_mask'][:]

                    transit_mask = np.tile(transit_mask, (transit_signal.shape[0], 1))
                    non_transit_mask = np.tile(non_transit_mask, (non_transit_signal.shape[0], 1))

                    transit_signal = np.ma.MaskedArray(transit_signal, mask=transit_mask)
                    non_transit_signal = np.ma.MaskedArray(non_transit_signal, mask=non_transit_mask)

                else:
                    transit_signal = hdf[planet_id][f'smoothing_{smoothing_window}_transit'][:]
                    non_transit_signal = hdf[planet_id][f'smoothing_{smoothing_window}_non_transit'][:]

                    transit_mask = hdf[planet_id][f'smoothing_{smoothing_window}_transit_mask'][:]
                    non_transit_mask = hdf[planet_id][f'smoothing_{smoothing_window}_non_transit_mask'][:]

                    transit_mask = np.tile(transit_mask, (transit_signal.shape[0], 1))
                    non_transit_mask = np.tile(non_transit_mask, (non_transit_signal.shape[0], 1))

                    transit_signal = np.ma.MaskedArray(transit_signal, mask=transit_mask)
                    non_transit_signal = np.ma.MaskedArray(non_transit_signal, mask=non_transit_mask)

                # Standardize each wavelength across frames, if asked
                if standardize_wavelengths:
                    transit_row_means = np.mean(transit_signal, axis=0)
                    transit_row_stds = np.std(transit_signal, axis=0)
                    transit_signal = (transit_signal - transit_row_means[np.newaxis, :]) / transit_row_stds[np.newaxis, :]

                    non_transit_row_means = np.mean(non_transit_signal, axis=0)
                    non_transit_row_stds = np.std(non_transit_signal, axis=0)
                    non_transit_signal = (non_transit_signal - non_transit_row_means[np.newaxis, :]) / non_transit_row_stds[np.newaxis, :]

                # Yield sample size spectral difference pairs from this planet in a batch
                transit_indices = random.sample(range(transit_signal.shape[0]), sample_size)
                non_transit_indices = random.sample(range(non_transit_signal.shape[0]), sample_size)
                
                signals = []

                for transit_idx, non_transit_idx in zip(transit_indices, non_transit_indices):
                    signals.append(transit_signal[transit_idx] - non_transit_signal[non_transit_idx])

                yield np.array(signals)


def make_training_datasets(
        data_file: str,
        sample_size: int,
        wavelengths: int = 283,
        validation: bool = True,
        smoothing_window: int = None,
        standardize_wavelengths: bool = False
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
            sample_size=sample_size,
            smoothing_window=smoothing_window,
            standardize_wavelengths=standardize_wavelengths
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
            sample_size=sample_size,
            smoothing_window=smoothing_window,
            standardize_wavelengths=standardize_wavelengths
        )

        evaluation_dataset = tf.data.Dataset.from_generator(
            evaluation_data_generator,
            output_signature=(
                tf.TensorSpec(shape=(sample_size, wavelengths), dtype=tf.float64),
                tf.TensorSpec(shape=(sample_size, wavelengths), dtype=tf.float64)
            )
        )

    return training_dataset, validation_dataset, evaluation_dataset


def make_testing_dataset(
        data_file: str,
        sample_size: int,
        wavelengths: int = 283,
        smoothing_window: int = None,
        standardize_wavelengths: bool = False
) -> tuple:

    with h5py.File(data_file, 'r') as hdf:
        planet_ids = list(hdf.keys())

    training_data_generator = partial(
        _testing_data_loader,
        planet_ids=planet_ids,
        data_file=data_file,
        sample_size=sample_size,
        smoothing_window=smoothing_window,
        standardize_wavelengths=standardize_wavelengths
    )

    dataset = tf.data.Dataset.from_generator(
        training_data_generator,
        output_signature=(
            tf.TensorSpec(shape=(sample_size, wavelengths), dtype=tf.float64)
        )
    )

    return dataset