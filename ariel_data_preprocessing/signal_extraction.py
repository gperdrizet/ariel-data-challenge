'''Signal extraction pipeline for Ariel Data Challenge'''

# Standard library imports
import os

# Third party imports
import numpy as np
import numpy.ma as ma
import h5py

# Internal imports
from ariel_data_preprocessing.utils import get_planet_list, load_masked_frames


class SignalExtraction:
    '''
    Extract clean spectral signals from corrected FGS1 and AIRS-CH0 telescope data.

    This class processes corrected AIRS-CH0 frames to extract 1D spectral time series
    and the FGS1 frames to extract a single value per frame. The extraction recovers
    detector rows (AIRS-CH0) and/or columns (FGS1) with the strongest signals defined
    by a user supplied threshold and applies optional smoothing to reduce noise.

    Example:
        >>> extractor = SignalExtraction(
        ...     input_data='data/corrected',
        ...     output_data_path='data/extracted',
        ...     inclusion_threshold=0.8,
        ...     smooth=True,
        ...     smoothing_window=100
        ... )
        >>> output_file = extractor.run()

    Input Requirements: HDF5 file with corrected FGS1 and AIRS-CH0 data from the signal
    correction pipeline. Input structure:

        train.h5:
        │
        ├── planet_id_1/
        │   ├── AIRS-CH0_signal  # Corrected spectrometer data
        │   ├── AIRS-CH0_mask    # Mask for spectrometer data
        │   ├── FGS1_signal      # Corrected guidance camera data
        │   └── FGS1_mask        # Mask for guidance camera data
        |
        ├── planet_id_2/
        │   ├── AIRS-CH0_signal  # Corrected spectrometer data
        │   ├── AIRS-CH0_mask    # Mask for spectrometer data
        │   ├── FGS1_signal      # Corrected guidance camera data
        │   └── FGS1_mask        # Mask for guidance camera data
        |
        └── ...

    Output: HDF5 file with extracted and combined spectral time series (FGS1 signal as 
    first column). Output Structure:
            
        train.h5
        |
        ├── planet_1/
        │   ├── signal  # Shape: (n_frames, n_wavelengths + 1)
        │   └── mask    # Shape: (n_wavelengths + 1,)
        │
        ├── planet_2/
        │   ├── signal  # Shape: (n_frames, n_wavelengths + 1)
        │   └── mask    # Shape: (n_wavelengths + 1,)
        │
        └── ...
    '''

    def __init__(
            self,
            input_data_path: str = None,
            output_data_path: str = None,
            input_filename: str = 'train.h5',
            output_filename: str = 'train.h5',
            inclusion_threshold: float = 0.75,
            smooth: bool = True,
            smoothing_window: int = 200,
            n_planets: int = -1,
            verbose: bool = False
    ):
        '''
        Initialize the SignalExtraction class with processing parameters.

        Parameters:
            - input_data_path (str): Path to HDF5 input file
            - output_data_path (str): Path to directory for extracted signal output
            - input_filename (str, default=train.h5): Optional custom filename for input HDF5 file
            - output_filename (str, default=train.h5): Optional custom filename for output HDF5 file
            - inclusion_threshold (float, default=0.75): Threshold for selecting spectral rows 
            - smooth (bool, default=True): Apply moving average smoothing per wavelength
            - smoothing_window (int, default=200): Moving average window for smoothing
            - n_planets (int, default=-1): Number of planets to process. -1 = processes all
            - verbose (bool, default=False): If True, print planet progress counter

        Raises:
            ValueError: If input_data or output_data_path are None
        '''

        self.input_data_path = input_data_path
        self.input_filename = input_filename
        self.output_data_path = output_data_path
        self.output_filename = output_filename
        self.inclusion_threshold = inclusion_threshold
        self.smooth = smooth
        self.smoothing_window = smoothing_window
        self.n_planets = n_planets
        self.verbose = verbose

        if input_data_path is None or output_data_path is None:
            raise ValueError("Input data and output data path must be provided.")

        # Make sure output directory exists
        os.makedirs(self.output_data_path, exist_ok=True)

        # Set output filepath
        self.output_filepath = (f'{self.output_data_path}/{self.output_filename}')
        
        # Remove hdf5 output file, if it already exists
        try:
            os.remove(self.output_filepath)

        except OSError:
            pass

        # Set input filepath
        self.input_filepath = f'{self.input_data_path}/{self.input_filename}'
        
        # Get planet list from input data
        self.planet_list = get_planet_list(self.input_filepath)

        if self.n_planets != -1:
            self.planet_list = self.planet_list[:self.n_planets]


    def run(self) -> None:
        '''
        Run the complete signal extraction pipeline.

        This method processes corrected AIRS-CH0 and FGS1 data to extract spectral signals by:
        1. Loading AIRS-CH0 frames from HDF5 input file
        2. Selecting spectral rows with strongest signals based on inclusion threshold
        3. Summing selected rows to create 1D spectrum per frame
        4. Loading FGS1 frames and extracting signal blocks
        5. Combining AIRS-CH0 and FGS1 signals (FGS1 signal inserted as first column)
        6. Applying optional smoothing with moving average
        7. Saving extracted signals to HDF5 output file

        Processing is applied to all planets specified during initialization.

        Parameters:
            None (uses instance attributes set during initialization)
        '''

        planet_counter = 0

        # Open HDF5 input
        with h5py.File(self.input_filepath, 'r') as hdf:
            for planet in self.planet_list:

                # Load AIRS frames & extract signal
                airs_frames = load_masked_frames(hdf, planet, 'AIRS-CH0')
                airs_signal = self._extract_airs_signal(airs_frames)

                # Load FGS frames & extract signal
                fgs_frames = load_masked_frames(hdf, planet, 'FGS1')
                fgs_signal = self._extract_fgs_signal(fgs_frames)

                # Combine the AIRS and FGS signals
                signal = np.insert(airs_signal, 0, fgs_signal, axis=1)

                # Smooth each wavelength across the frames
                if self.smooth:
                    signal = self.moving_average_rows(
                        signal, self.smoothing_window
                    )

                # Save the extracted signal to HDF5
                with h5py.File(self.output_filepath, 'a') as out_hdf:

                    planet_group = out_hdf.require_group(planet)
                    planet_group.create_dataset('signal', data=signal.data)
                    planet_group.create_dataset('mask', data=signal.mask[0])

                if self.verbose:
                    planet_counter += 1
                    print(f'Extracted signal for planet {planet_counter} of {len(self.planet_list)}', end='\r')


    def _extract_airs_signal(self, frames: np.ndarray) -> np.ndarray:
        '''
        Extract 1D spectral signal from 3D AIRS frames.

        This method processes a stack of AIRS frames to extract a clean 1D spectral
        time series by selecting detector rows with the strongest signals based on
        the inclusion threshold. The selected rows are summed for each frame to
        produce the final spectrum.

        Parameters:
            frames (np.ndarray): Input AIRS frames with shape (n_frames, n_rows, n_wavelengths)

        Returns:
            np.ndarray: Extracted 2D spectral signal with shape (n_frames, n_wavelengths)

        Algorithm:
            1. Identify top detector rows using _select_top_rows()
            2. Extract these rows from all frames
            3. Sum the selected rows for each frame
            4. Return the resulting 2D array of shape (n_frames, n_wavelengths)
        '''

        # Select top rows based on inclusion threshold
        top_rows = self._select_top_rows(
            frames,
            self.inclusion_threshold
        )

        # Get the top rows for each frame
        signal_strip = frames[:, top_rows, :]

        # Sum the selected rows in each frame and transpose
        signal = np.sum(signal_strip, axis=1)

        return signal


    def _extract_fgs_signal(self, frames: np.ndarray) -> np.ndarray:
        '''
        Extract 1D signal from 3D FGS frames using 2D block extraction.

        This method processes FGS frames to extract a clean 1D signal time series by
        selecting both detector rows and columns with the strongest signals, creating
        a 2D signal block that is then summed to produce a single value per frame.

        Parameters:
            frames (np.ndarray): Input FGS frames with shape (n_frames, n_rows, n_columns)

        Returns:
            np.ndarray: Extracted 1D signal with shape (n_frames,)

        Algorithm:
            1. Identify top detector rows using _select_top_rows()
            2. Identify top detector columns using _select_top_cols()
            3. Extract the intersection (signal block) from all frames
            4. Sum the signal block for each frame to get single value
            5. Return the resulting 1D array of shape (n_frames,)
        '''

        # Select top rows based on inclusion threshold
        top_rows = self._select_top_rows(
            frames,
            self.inclusion_threshold
        )

        # Select top columns based on inclusion threshold
        top_cols = self._select_top_cols(
            frames,
            self.inclusion_threshold
        )

        # Now index the original array to get the top rows for each frame
        signal_strip = frames[:, top_rows, :]

        # And then the top columns for each frame
        signal_block = signal_strip[:, :, top_cols]

        # Sum the block per frame
        signal = np.sum(signal_block, axis=1)
        signal = np.sum(signal, axis=1)

        return signal


    def _select_top_rows(self, frames: np.ndarray, inclusion_threshold: float) -> list:
        '''
        Select detector pixel rows with strongest signals based on threshold criteria.

        Analyzes the first frame to identify detector rows with the highest signal
        levels, using the inclusion threshold to determine which rows contribute
        significantly to the signal. This focuses extraction on the most
        informative parts of the detector array.

        Parameters:
            frames (np.ndarray): Input AIRS frames with shape (n_frames, n_rows, n_columns)
            inclusion_threshold (float): Threshold value between 0-1 for row selection.
                Higher values select fewer rows with stronger signals.

        Returns:
            list: List of integer row indices that exceed the signal threshold

        Algorithm:
            1. Sum pixel values across wavelengths for each row in first frame
            2. Normalize sums to 0-1 range by subtracting minimum
            3. Calculate threshold as fraction of signal range
            4. Select rows where signal exceeds threshold
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


    def _select_top_cols(self, frames: np.ndarray, inclusion_threshold: float) -> list:
        '''
        Select columns with strongest signal based on threshold criteria.

        Analyzes the first frame to identify detector columns with the highest signal
        levels, using the inclusion threshold to determine which columns contribute
        significantly to the signal. This focuses extraction on the most
        informative parts of the detector array.

        Parameters:
            frames (np.ndarray): Input frames with shape (n_frames, n_rows, n_columns)
            inclusion_threshold (float): Threshold value between 0-1 for column selection.
                Higher values select fewer columns with stronger signals.

        Returns:
            list: List of integer column indices that exceed the signal threshold

        Algorithm:
            1. Sum pixel values across columns for each row in first frame
            2. Normalize sums to 0-1 range by subtracting minimum
            3. Calculate threshold as fraction of signal range
            4. Select columns where signal exceeds threshold
        '''

        # Sum the first frame's columns
        col_sums = np.sum(frames[0], axis=0)

        # Shift the sums so the minimum is zero
        col_sums -= np.min(col_sums)
        signal_range = np.max(col_sums)
        
        # Determine the threshold for inclusion
        threshold = inclusion_threshold * signal_range

        # Select columns where the sum exceeds the threshold
        selected_cols = np.where(col_sums >= threshold)[0]

        # Return the indices of the selected rows
        return selected_cols.tolist()

    
    @staticmethod
    def moving_average_rows(a, n):
        '''
        Compute moving average smoothing for each row in a 2D array.

        Applies a sliding window moving average across the columns (time/wavelength axis)
        of each row independently. This reduces noise while preserving spectral features.
        The output array has fewer columns due to the windowing operation.

        Parameters:
            a (np.ndarray): Input 2D array with shape (n_rows, n_columns)
            n (int): Size of the moving average window. Must be >= 1 and <= n_columns.

        Returns:
            np.ndarray: Smoothed 2D array with shape (n_rows, n_columns - n + 1)

        Algorithm:
            Uses cumulative sum method for efficient O(n_rows * n_columns) computation:
            1. Transpose the array to operate on columns
            2. Calculate cumulative sum along columns
            3. Use sliding window difference to get window sums
            4. Divide by window size to get averages
            5. Transpose back to original orientation

        Example:
            >>> data = np.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
            >>> moving_average_rows(data, 3)
            array([[2., 3., 4.], [7., 8., 9.]])
        '''

        # Transpose the array to operate on rows
        a = np.transpose(a)

        # Compute cumulative sum along axis 1 (across columns)
        cumsum_vec = np.cumsum(a, axis=1, dtype=float)

        # Subtract the cumulative sum at the start of the window from the end
        cumsum_vec[:, n:] = cumsum_vec[:, n:] - cumsum_vec[:, :-n]
        
        # Return the average for each window, starting from the (n-1)th element
        a = cumsum_vec[:, n - 1:] / n

        # Transpose back to original orientation
        a = np.transpose(a)

        return a