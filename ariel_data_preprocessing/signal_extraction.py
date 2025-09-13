'''Signal extraction pipeline for Ariel Data Challenge'''

# Standard library imports
import os

# Third party imports
import numpy as np
import h5py


class SignalExtraction:
    def __init__(
            self,
            input_data_path: str,
            output_data_path: str,
            inclusion_threshold: float = 0.75,
            smooth: bool = True,
            smoothing_window: int = 200,
            n_planets: int = -1
    ):
        '''
        Initialize the SignalExtraction class.

        Parameters:
        - input_data_path (str): Path to the input data directory.
        - output_data_path (str): Path to the output data directory.
        '''

        self.input_data_path = input_data_path
        self.output_data_path = output_data_path
        self.inclusion_threshold = inclusion_threshold
        self.smooth = smooth
        self.smoothing_window = smoothing_window
        self.n_planets = n_planets

        # Make sure output directory exists
        os.makedirs(self.output_data_path, exist_ok=True)

        # Remove hdf5 files from previous runs
        filename = (f'{self.output_data_path}/train.h5')
        
        try:
            os.remove(filename)

        except OSError:
            pass

        # Get planet list from input data
        self.planet_list = self._get_planet_list()

        if self.n_planets != -1:
            self.planet_list = self.planet_list[:self.n_planets]


    def run(self):
        '''
        Run the signal extraction pipeline.

        This method processes the input data to extract signals from the AIRS dataset,
        and saves the extracted signals to the specified output path.
        '''

        # Open HDF5 input
        with h5py.File(f'{self.input_data_path}/train.h5', 'r') as hdf:
            for planet in self.planet_list:
                print(f'Processing planet {planet}...')

                # Load AIRS frames
                airs_frames = hdf[planet]['AIRS-CH0_signal'][:]
                print(f'Loaded AIRS frames shape: {airs_frames.shape}')

                # Select top rows based on inclusion threshold
                top_rows = self._select_top_rows(
                    airs_frames,
                    self.inclusion_threshold
                )

                print(f'Selected top rows: {top_rows}')

                # Get the top rows for each frame
                signal_strip = airs_frames[:, top_rows, :]
                print(f'Signal strip shape: {signal_strip.shape}')

                # Sum the selected rows in each frame and transpose
                airs_signal = np.transpose(np.sum(signal_strip, axis=1))

                # Smooth each wavelength across the frames
                if self.smooth:
                    airs_signal = self.moving_average_rows(airs_signal, self.smoothing_window)

                # Transpose the data back to (frames, wavelengths)
                airs_signal = np.transpose(airs_signal)

                # Save the extracted signal to HDF5
                with h5py.File(f'{self.output_data_path}/train.h5', 'a') as out_hdf:

                    planet_group = out_hdf.require_group(planet)

                    planet_group.create_dataset(
                        'AIRS-CH0_signal',
                        data=airs_signal
                    )
    

    def _get_planet_list(self) -> list:
        '''
        Get the list of planets from the input data.

        Returns:
        - list: List of planet IDs.
        '''

        with h5py.File(f'{self.input_data_path}/train.h5', 'r') as hdf:
            planet_list = list(hdf.keys())

        return planet_list


    def _select_top_rows(self, frames: np.ndarray, inclusion_threshold: float) -> list:
        '''
        Select the top N rows from the frame based on some criteria.

        Parameters:
        - frame (np.ndarray): The input data frame.

        Returns:
        - np.ndarray: The selected top N rows.
        '''

        # Sum the first frame's rows
        row_sums = np.sum(frames[0], axis=1)

        # Shift the sums so the minimum is zero
        row_sums -= np.min(row_sums)
        signal_range = np.max(row_sums)
        
        # Determine the threshold for inclusion
        threshold = inclusion_threshold * signal_range

        # Select rows where the sum exceeds the threshold
        selected_rows = np.where(row_sums >= threshold)[0]

        # Return the indices of the selected rows
        return selected_rows.tolist()

    
    @staticmethod
    def moving_average_rows(a, n):
        '''
        Compute the moving average of each row in a 2D array.
        
        Parameters:
        - a (np.ndarray): Input 2D array.
        - n (int): Window size for the moving average.
        
        Returns:
        - np.ndarray: 2D array of the moving averages.
        '''

        # Compute cumulative sum along axis 1 (across columns)
        cumsum_vec = np.cumsum(a, axis=1, dtype=float)

        # Subtract the cumulative sum at the start of the window from the end
        cumsum_vec[:, n:] = cumsum_vec[:, n:] - cumsum_vec[:, :-n]
        
        # Return the average for each window, starting from the (n-1)th element
        return cumsum_vec[:, n - 1:] / n