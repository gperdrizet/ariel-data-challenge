'''Signal extraction pipeline for Ariel Data Challenge'''

# Standard library imports
import os

# Third party imports
import numpy as np
import numpy.ma as ma
import h5py


class SignalExtraction:
    '''
    Extract clean spectral signals from corrected AIRS-CH0 telescope data.

    This class processes corrected AIRS-CH0 frames to extract 1D spectral time series
    suitable for exoplanet atmospheric analysis. The extraction focuses on detector
    rows with the strongest signals and applies optional smoothing to reduce noise.

    The pipeline transforms 3D AIRS frames (time, detector_rows, wavelengths) into
    clean 2D spectral time series (time, wavelengths) by intelligently selecting
    and combining the most informative detector rows.

    Key Features:
        - Adaptive row selection based on signal strength thresholds
        - Optional moving average smoothing for noise reduction
        - Batch processing of multiple planets
        - HDF5 input/output for efficient large dataset handling
        - Configurable processing parameters

    Typical Workflow:
        1. Load corrected AIRS-CH0 data from signal correction pipeline
        2. Analyze first frame to identify high-signal detector rows
        3. Extract and sum selected rows for each frame
        4. Apply optional smoothing across wavelengths
        5. Save extracted spectra for downstream analysis

    Example:
        >>> extractor = SignalExtraction(
        ...     input_data_path='data/corrected',
        ...     output_data_path='data/extracted',
        ...     inclusion_threshold=0.8,
        ...     smooth=True,
        ...     smoothing_window=100
        ... )
        >>> output_file = extractor.run()

    Input Requirements:
        - HDF5 file with corrected AIRS-CH0 data from signal correction pipeline
        - Data structure: /planet_id/AIRS-CH0_signal with shape (frames, rows, wavelengths)

    Output:
        - HDF5 file with extracted spectral time series
        - Data structure: /planet_id/AIRS-CH0_signal with shape (frames, wavelengths)

    Author: Ariel Data Challenge Team
    Version: 1.0
    '''

    def __init__(
            self,
            input_data_path: str,
            output_data_path: str,
            output_filename: str = None,
            inclusion_threshold: float = 0.75,
            smooth: bool = True,
            smoothing_window: int = 200,
            n_planets: int = -1
    ):
        '''
        Initialize the SignalExtraction class with processing parameters.

        Parameters:
            input_data_path (str): Path to directory containing corrected signal data
            output_data_path (str): Path to directory for extracted signal output
            inclusion_threshold (float, default=0.75): Threshold for selecting spectral rows 
                based on signal strength. Value between 0 and 1, where higher values select fewer
                rows with stronger signals
            smooth (bool, default=True): Whether to apply moving average smoothing to the 
                extracted signals
            smoothing_window (int, default=200): Size of the moving average window for smoothing.
                Only used if smooth=True
            n_planets (int, default=-1): Number of planets to process. If -1, processes all 
                available planets

        Raises:
            ValueError: If input_data_path or output_data_path are None

        Output Structure:
            HDF5 file containing extracted signals organized by planet:
            
                train.h5
                |
                ├── planet_1/
                │   └── AIRS-CH0_signal  # Shape: (n_frames, n_wavelengths)
                ├── planet_2/
                │   └── AIRS-CH0_signal
                └── ...
        '''

        self.input_data_path = input_data_path
        self.output_data_path = output_data_path
        self.output_filename = output_filename
        self.inclusion_threshold = inclusion_threshold
        self.smooth = smooth
        self.smoothing_window = smoothing_window
        self.n_planets = n_planets

        if input_data_path is None or output_data_path is None:
            raise ValueError("Input and output data paths must be provided.")

        # Make sure output directory exists
        os.makedirs(self.output_data_path, exist_ok=True)

        # Set output filename
        if self.output_filename is not None:
            filename = (f'{self.output_data_path}/{self.output_filename}')

        else:
            filename = (f'{self.output_data_path}/train.h5')
        
        # Remove hdf5 file, if it already exists
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
        Run the complete signal extraction pipeline.

        This method processes corrected AIRS-CH0 data to extract spectral signals by:
        1. Loading AIRS frames from HDF5 input file
        2. Selecting spectral rows with strongest signals based on inclusion threshold
        3. Summing selected rows to create 1D spectrum per frame
        4. Applying optional smoothing with moving average
        5. Saving extracted signals to HDF5 output file

        Processing is applied to all planets specified during initialization.

        Parameters:
            None (uses instance attributes set during initialization)

        Returns:
            str: Path to the output HDF5 file containing extracted signals

        Output Structure:
            HDF5 file with groups for each planet:

                train.h5
                |
                ├── planet_1
                |   ├── AIRS-CH0_signal
                │   └── FGS1_signal
                │
                ├── planet_1
                |   ├── AIRS-CH0_signal
                │   └── FGS1_signal
                │
                .
                .
                .
                └── planet_n
                    ├── AIRS-CH0_signal
                    └── FGS1_signal
        '''

        # Open HDF5 input
        with h5py.File(f'{self.input_data_path}/train.h5', 'r') as hdf:
            for planet in self.planet_list:

                # Load AIRS frames
                airs_frames = hdf[planet]['AIRS-CH0_signal'][:]

                # Select top rows based on inclusion threshold
                top_rows = self._select_top_rows(
                    airs_frames,
                    self.inclusion_threshold
                )

                # Get the top rows for each frame
                signal_strip = airs_frames[:, top_rows, :]

                # Sum the selected rows in each frame and transpose
                airs_signal = np.transpose(np.sum(signal_strip, axis=1))

                # Smooth each wavelength across the frames
                if self.smooth:
                    airs_signal = self.moving_average_rows(airs_signal, self.smoothing_window)

                # Transpose the data back to (frames, wavelengths)
                airs_signal = np.transpose(airs_signal)

                # Save the extracted signal to HDF5
                output_file = f'{self.output_data_path}/train.h5'

                with h5py.File(output_file, 'a') as out_hdf:

                    planet_group = out_hdf.require_group(planet)

                    planet_group.create_dataset(
                        'AIRS-CH0_signal',
                        data=airs_signal
                    )

        return output_file

    def _get_planet_list(self) -> list:
        '''
        Retrieve the list of planet IDs from the input HDF5 file.

        Scans the HDF5 input file to identify all available planet groups
        for processing during signal extraction.

        Parameters:
            None

        Returns:
            list: List of planet ID strings found in the input HDF5 file

        Raises:
            IOError: If input HDF5 file cannot be opened or read
        '''

        with h5py.File(f'{self.input_data_path}/train.h5', 'r') as hdf:
            planet_list = list(hdf.keys())

        return planet_list


    def _select_top_rows(self, frames: np.ndarray, inclusion_threshold: float) -> list:
        '''
        Select spectral rows with strongest signals based on threshold criteria.

        Analyzes the first frame to identify detector rows with the highest signal
        levels, using the inclusion threshold to determine which rows contribute
        significantly to the spectrum. This focuses extraction on the most
        informative parts of the detector array.

        Parameters:
            frames (np.ndarray): Input AIRS frames with shape (n_frames, n_rows, n_wavelengths)
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
            1. Calculate cumulative sum along columns
            2. Use sliding window difference to get window sums
            3. Divide by window size to get averages
        '''

        # Compute cumulative sum along axis 1 (across columns)
        cumsum_vec = np.cumsum(a, axis=1, dtype=float)

        # Subtract the cumulative sum at the start of the window from the end
        cumsum_vec[:, n:] = cumsum_vec[:, n:] - cumsum_vec[:, :-n]
        
        # Return the average for each window, starting from the (n-1)th element
        return cumsum_vec[:, n - 1:] / n