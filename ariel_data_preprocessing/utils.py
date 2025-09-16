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

    Handles reading raw data from directory structure or HDF5 file.

    Returns:
        list: List of unique planet IDs
    '''

    if Path(input_data).is_file():
        if Path(input_data).suffix == '.h5':
            with h5py.File(input_data, 'r') as hdf:
                return list(hdf.keys())
            
        else:
            raise ValueError('Input is a file but not an HDF5 file.')
        
    if Path(input_data).is_dir():
        input = input_data
        planets = list(os.listdir(f'{input}/train'))
        planets = [planet_path.split('/')[-1] for planet_path in planets]

        if len(planets) == 0:
            raise ValueError('No planet directories found in input data path.')
        
        else:
            return planets


def load_masked_frames(
        hdf: h5py.File,
        planet: str,
        instrument: str = None
) -> ma.MaskedArray:
    '''
    Load the masked frames for a given planet and instrument from the HDF5 file.

    Parameters:
        hdf (h5py.File): Open HDF5 file object
        planet (str): Planet ID string, or 'random' for a random planet
        instrument (str): Instrument ID string (e.g., 'AIRS-CH0', 'FGS1')

    Returns:
        np.ma.MaskedArray: Masked array representing the mask for the planet
    '''

    if planet == 'random':
        planet = np.random.choice(list(hdf.keys()))

    if len(hdf[planet]) == 4:

        if instrument not in ['AIRS-CH0', 'FGS1']:
            raise ValueError('Instrument must be either "AIRS-CH0" or "FGS1".')
        
        else:
            frames = hdf[planet][f'{instrument}_signal'][:]
            mask = hdf[planet][f'{instrument}_mask'][:]

    elif len(hdf[planet]) == 2:

        frames = hdf[planet]['signal'][:]
        mask = hdf[planet]['mask'][:]

    else:
        raise ValueError('Unexpected number of datasets in planet group.')
    
    mask = np.tile(mask, (frames.shape[0], 1, 1))
    frames = ma.MaskedArray(frames, mask=mask)

    return frames