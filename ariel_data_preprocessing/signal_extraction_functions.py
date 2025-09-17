'''Functions for FGS1 and AIRS-CH0 signal extraction and
wavelength smoothing'''

# Third party imports
import numpy as np


def extract_airs_signal(frames: np.ndarray, inclusion_threshold: float) -> np.ndarray:
    '''
    Extract 1D spectral signal from 3D AIRS frames.

    This method processes a stack of AIRS frames to extract a clean 1D spectral
    time series by selecting detector rows with the strongest signals based on
    the inclusion threshold. The selected rows are summed for each frame to
    produce the final spectrum.

    Parameters:
        frames (np.ndarray): Input AIRS frames with shape (n_frames, n_rows, n_wavelengths)
        inclusion_threshold (float): Threshold value between 0-1 for row selection.

    Returns:
        np.ndarray: Extracted 2D spectral signal with shape (n_frames, n_wavelengths)

    Algorithm:
        1. Identify top detector rows using _select_top_rows()
        2. Extract these rows from all frames
        3. Sum the selected rows for each frame
        4. Return the resulting 2D array of shape (n_frames, n_wavelengths)
    '''

    # Select top rows based on inclusion threshold
    top_rows = select_top_rows(
        frames,
        inclusion_threshold
    )

    # Get the top rows for each frame
    signal_strip = frames[:, top_rows, :]

    # Sum the selected rows in each frame and transpose
    signal = np.sum(signal_strip, axis=1)

    return signal


def extract_fgs_signal(frames: np.ndarray, inclusion_threshold: float) -> np.ndarray:
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
    top_rows = select_top_rows(
        frames,
        inclusion_threshold
    )

    # Select top columns based on inclusion threshold
    top_cols = select_top_cols(
        frames,
        inclusion_threshold
    )

    # Now index the original array to get the top rows for each frame
    signal_strip = frames[:, top_rows, :]

    # And then the top columns for each frame
    signal_block = signal_strip[:, :, top_cols]

    # Sum the block per frame
    signal = np.sum(signal_block, axis=1)
    signal = np.sum(signal, axis=1)

    return signal


def select_top_rows(frames: np.ndarray, inclusion_threshold: float) -> list:
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


def select_top_cols(frames: np.ndarray, inclusion_threshold: float) -> list:
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