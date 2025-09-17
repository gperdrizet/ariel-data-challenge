'''Utility/helper functions for Ariel data preprocessing.'''

# Standard library imports
import os
from pathlib import Path

# Third party imports
import h5py
import numpy as np
import numpy.ma as ma


def get_planet_list(input_data: str):
    '''
    Retrieve list of unique planet IDs from input data.

    Handles reading raw data from directory structure provided in Kaggle
    zipfile download of the dataset.

    Returns:
        list: List of unique planet IDs
    '''
        
    if Path(input_data).is_dir():

        planets = list(os.listdir(f'{input_data}/train'))
        planets = [planet_path.split('/')[-1] for planet_path in planets]

        if len(planets) == 0:
            raise ValueError('No planet directories found in input data path.')
        
        else:
            return planets
        
    else:
        raise ValueError('Input is not a valid path')


def load_masked_frames(
        hdf: h5py.File,
        planet: str
) -> ma.MaskedArray:
    '''
    Load the masked frames for a given planet from the HDF5 file.

    Parameters:
        hdf (h5py.File): Open HDF5 file object
        planet (str): Planet ID string, or 'random' for a random planet

    Returns:
        np.ma.MaskedArray: Masked array representing the mask for the planet
    '''

    if planet == 'random':
        planet = np.random.choice(list(hdf.keys()))

    frames = hdf[planet]['signal'][:]
    mask = hdf[planet]['mask'][:]
    
    mask = np.tile(mask, (frames.shape[0], 1, 1))
    frames = ma.MaskedArray(frames, mask=mask)

    return frames