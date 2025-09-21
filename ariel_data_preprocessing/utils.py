'''Utility/helper functions for Ariel data preprocessing.'''

# Standard library imports
import glob
import os
from pathlib import Path

# Third party imports
import h5py
import numpy as np


def get_planet_list(input_data: str, mode: str) -> list:
    '''
    Retrieve list of unique planet IDs from input data.

    Handles reading raw data from directory structure provided in Kaggle
    zipfile download of the dataset.

    Parameters:
        input_data (str): Path to input data directory
        mode (str): 'train' or 'test' to specify which dataset to use

    Returns:
        list: List of unique planet IDs
    '''

    planets = None
        
    if Path(input_data).is_dir():

        planets = list(os.listdir(f'{input_data}/{mode}'))
        planets = [planet_path.split('/')[-1] for planet_path in planets]

    if planets is None or len(planets) == 0:
        raise ValueError(f'No planet directories found in {input_data}/{mode}.')

    return planets


def load_masked_frames(
        hdf: h5py.File,
        planet: str,
        load_mask: bool = True,
        return_id: bool = False,
        verbose: bool = False
) -> np.ma.MaskedArray:
    '''
    Load the masked frames for a given planet from the HDF5 file.

    Parameters:
        hdf (h5py.File): Open HDF5 file object
        planet (str): Planet ID string, or 'random' for a random planet
        verbose (bool): If True, print available groups in the HDF5 file

    Returns:
        np.ma.MaskedArray: Masked array representing the mask for the planet
    '''

    if verbose:

        print(f'Groups:')

        for id in hdf.keys():
            print(f' {id}')

    if planet == 'random':
        planet = np.random.choice(list(hdf.keys()))

    frames = hdf[planet]['signal'][:]

    if load_mask:
        try:
            mask = hdf[planet]['mask'][:]
            mask = np.tile(mask, (frames.shape[0], 1, 1))
            frames = np.ma.MaskedArray(frames, mask=mask)

        except KeyError:
            if verbose:
                print(f'No mask found for planet {planet}.')
            pass

    if return_id:
        return frames, planet

    return frames