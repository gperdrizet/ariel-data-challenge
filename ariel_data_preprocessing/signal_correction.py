'''Signal correction pipeline for Ariel Data Challenge

This module implements the complete preprocessing pipeline for Ariel telescope data,
including ADC conversion, pixel masking, linearity correction, dark current subtraction,
correlated double sampling (CDS), and flat field correction.
'''

# Standard library imports
import itertools
import os
from multiprocessing import Manager, Process

# Third party imports
import h5py
import numpy as np
import pandas as pd
from astropy.stats import sigma_clip

# Internal imports
from ariel_data_preprocessing.calibration_data import CalibrationData
from ariel_data_preprocessing.utils import get_planet_list


class SignalCorrection:
    '''
    Complete signal correction and calibration pipeline for Ariel telescope data.
    
    This class implements the full 6-step preprocessing pipeline required to transform
    raw Ariel telescope detector outputs into science-ready data suitable for exoplanet
    atmospheric analysis. The pipeline handles both AIRS-CH0 (infrared spectrometer) 
    and FGS1 (guidance camera) data with parallel processing capabilities.
    
    Processing Pipeline:
        1. Analog-to-Digital Conversion (ADC) - Convert raw counts to physical units
        2. Hot/Dead Pixel Masking - Remove problematic detector pixels
        3. Linearity Correction - Account for non-linear detector response
        4. Dark Current Subtraction - Remove thermal background noise
        5. Correlated Double Sampling (CDS) - Reduce read noise via paired exposures
        6. Flat Field Correction - Normalize pixel-to-pixel sensitivity variations
    
    Key Features:
        - Multiprocessing support for parallel planet processing
        - Optional FGS1 downsampling to match AIRS-CH0 cadence
        - Configurable processing steps (can enable/disable individual corrections)
        - Automatic calibration data loading and management
        - HDF5 output for efficient large dataset storage
    
    Performance Optimizations:
        - Process-level parallelization across planets
        - Intelligent FGS downsampling (83% data reduction)
    
    Example:
        >>> corrector = SignalCorrection(
        ...     input_data_path='data/raw',
        ...     output_data_path='data/corrected',
        ...     n_cpus=4,
        ...     downsample_fgs=True,
        ...     n_planets=100
        ... )
        >>> corrector.run()
    
    Input Requirements:
        - Works with Ariel Data Challenge (2025) dataset from Kaggle
        - Raw Ariel telescope data in parquet format
        - Calibration data (dark, dead, flat, linearity correction files)
        - ADC conversion parameters
        - Axis info metadata for timing
    
    Output:
        - HDF5 file with corrected AIRS-CH0 and FGS1 signals and hot/dead pixel masks
        - Organized by planet ID for easy access
        - Reduced data volume (50% reduction from CDS, optional 83% FGS reduction)
        - Science-ready data for downstream analysis
        - Output structure:

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
    '''

    def __init__(
            self,
            input_data_path: str = None,
            output_data_path: str = None,
            adc_conversion: bool = True,
            masking: bool = True,
            linearity_correction: bool = True,
            dark_subtraction: bool = True,
            cds_subtraction: bool = True,
            flat_field_correction: bool = True,
            output_filename: str = None,
            fgs_frames: int = 135000,
            airs_frames: int = 11250,
            cut_inf: int = 39,
            cut_sup: int = 321,
            gain: float = 0.4369,
            offset: float = -1000.0,
            n_cpus: int = 1,
            n_planets: int = -1,
            downsample_fgs: bool = False,
            compress_output: bool = False,
            verbose: bool = False
    ):
        '''
        Initialize the SignalCorrection class with processing parameters.
        
        Parameters:
            - input_data_path (str): Path to directory containing raw Ariel telescope data
            - output_data_path (str): Path to directory for corrected signal output
            - adc_conversion (bool, default=True): Enable analog-to-digital conversion step
            - masking (bool, default=True): Enable hot/dead pixel masking step
            - linearity_correction (bool, default=True): Enable detector linearity correction
            - dark_subtraction (bool, default=True): Enable dark current subtraction
            - cds_subtraction (bool, default=True): Enable correlated double sampling
            - flat_field_correction (bool, default=True): Enable flat field normalization
            - output_filename (str, default=None): Name of output HDF5 file
            - fgs_frames (int, default=135000): Expected number of FGS1 frames per planet
            - airs_frames (int, default=11250): Expected number of AIRS-CH0 frames per planet
            - cut_inf (int, default=39): Lower bound for AIRS spectral cropping
            - cut_sup (int, default=321): Upper bound for AIRS spectral cropping  
            - gain (float, default=0.4369): ADC gain factor from adc_info.csv
            - offset (float, default=-1000.0): ADC offset value from adc_info.csv
            - n_cpus (int, default=1): Number of CPU cores for parallel processing
            - n_planets (int, default=-1): Number of planets to process (-1 for all)
            - downsample_fgs (bool, default=False): Enable FGS1 downsampling to match AIRS cadence
            - compress_output (bool, default=False): Enable compression for HDF5 output datasets
            - verbose (bool, default=False): Enable progress counter output
            
        Raises:
            ValueError: If input_data_path or output_data_path are None
            
        Output Structure:

            HDF5 file containing corrected signals organized by planet:

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
        '''
        
        if input_data_path is None or output_data_path is None:
            raise ValueError("Input and output data paths must be provided.")
        
        self.input_data_path = input_data_path
        self.output_data_path = output_data_path
        self.adc_conversion = adc_conversion
        self.masking = masking
        self.linearity_correction = linearity_correction
        self.dark_subtraction = dark_subtraction
        self.cds_subtraction = cds_subtraction
        self.flat_field_correction = flat_field_correction
        self.output_filename = output_filename
        self.fgs_frames = fgs_frames
        self.airs_frames = airs_frames
        self.gain = gain
        self.offset = offset
        self.cut_inf = cut_inf
        self.cut_sup = cut_sup
        self.n_cpus = n_cpus
        self.n_planets = n_planets
        self.downsample_fgs = downsample_fgs
        self.compress_output = compress_output
        self.verbose = verbose

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
        self.planet_list = get_planet_list(self.input_data_path)

        if self.n_planets != -1:
            self.planet_list = self.planet_list[:self.n_planets]

        # Set downsampling indices for FGS data
        if self.downsample_fgs:
            self.fgs_indices = self._fgs_downsamples()

        # Import tables if needed
        if self.compress_output:
            import tables

    def run(self):
        '''
        Execute the complete signal correction pipeline with multiprocessing.
        
        Orchestrates parallel processing of multiple planets using worker processes:
        1. Sets up multiprocessing manager and communication queues
        2. Spawns worker processes for signal correction (one per CPU core)
        3. Spawns dedicated output process for saving results to HDF5
        4. Distributes planet processing tasks across worker processes
        5. Collects and saves corrected signals from all workers
        
        The pipeline processes planets in parallel while maintaining data integrity
        through proper queue management and process synchronization.
        
        Processing Flow:
            - Input queue: Planet IDs → Worker processes
            - Output queue: Corrected signals → Save process
            - Workers apply full 6-step correction pipeline per planet
            - Save process writes results to HDF5 with proper group structure
        
        Performance:
            - Linear speedup with CPU count (up to 4 cores typically optimal)
            - Memory usage scales with number of worker processes
            - Processing time: ~3-12 hours for 1100 planets (depending on CPU count)
        
        Parameters:
            None (uses instance configuration from __init__)
            
        Returns:
            None (writes output to output_data_path/output_filename)
            
        Side Effects:
            - Creates/overwrites output HDF5 file
            - Spawns and manages multiple worker processes
            - Prints progress information to stdout
        '''

        # Start the multiprocessing manager
        manager = Manager()

        # Takes planed id string and sends to calibration worker
        input_queue = manager.Queue()

        # Takes calibrated data from calibration worker to output worker
        output_queue = manager.Queue()

        # Set up worker process for each CPU
        worker_processes = []

        for _ in range(self.n_cpus):

            worker_processes.append(
                Process(
                    target=self.correct_signal,
                    args=(input_queue, output_queue)
                )
            )

        # Add the planet IDs to the input queue
        for planet in self.planet_list:
            input_queue.put(planet)

        # Add a stop signal for each worker
        for _ in range(self.n_cpus):
            input_queue.put('STOP')

        # Set up an output process to save results
        output_process = Process(
            target=self._save_corrected_data,
            args=(output_queue,)
        )

        # Start all worker processes
        for process in worker_processes:
            process.start()

        # Start the output process
        output_process.start()

        # Join and close all worker processes
        for process in worker_processes:
            process.join()
            process.close()

        # Join and close the output process
        output_process.join()
        output_process.close()


    def correct_signal(self, input_queue, output_queue):
        '''
        Worker process function that applies the complete signal correction pipeline.
        
        This method runs in separate worker processes and continuously processes
        planets from the input queue until receiving a 'STOP' signal. Each planet
        undergoes the full 6-step correction pipeline for both AIRS-CH0 and FGS1 data.
        
        Processing Steps per Planet:
            1. Load raw AIRS-CH0 and FGS1 signal data from parquet files
            2. Apply optional FGS1 downsampling to match AIRS-CH0 cadence
            3. Load calibration data (dark, dead, flat, linearity coefficients)
            4. Execute 6-step correction pipeline:
               - ADC conversion (raw counts → physical units)
               - Hot/dead pixel masking (remove problematic pixels)
               - Linearity correction (polynomial detector response correction)
               - Dark subtraction (remove thermal background)
               - CDS (correlated double sampling for noise reduction)
               - Flat field correction (normalize pixel sensitivity)
            5. Send corrected signals to output queue
        
        Parameters:
            input_queue (multiprocessing.Queue): Queue containing planet IDs to process
            output_queue (multiprocessing.Queue): Queue for sending corrected signals
            
        Queue Protocol:
            - Input: Planet ID strings or 'STOP' termination signal
            - Output: Dictionary with keys: 'planet', 'airs_signal', 'fgs_signal'
            
        Returns:
            bool: True when worker completes (after receiving 'STOP')
            
        Note:
            This method is designed to run in separate processes and handles
            its own error recovery and graceful shutdown.
        '''

        while True:
            planet = input_queue.get()

            if planet == 'STOP':
                result = {
                    'planet': 'STOP',
                    'airs_signal': None,
                    'fgs_signal': None
                }
                output_queue.put(result)
                break

            # Get path to this planet's data
            planet_path = f'{self.input_data_path}/train/{planet}'

            # Load and reshape the FGS1 data
            fgs_signal = pd.read_parquet(
                f'{planet_path}/FGS1_signal_0.parquet'
            ).to_numpy().reshape(self.fgs_frames, 32, 32)

            # Down sample FGS data to match capture cadence of AIRS-CH0
            if self.downsample_fgs:
                fgs_signal = np.take(fgs_signal, self.fgs_indices, axis=0)

            # Convert to float64 from unit16
            fgs_signal = fgs_signal.astype(np.float64)
    
            # Get frame count
            fgs_frames = fgs_signal.shape[0]

            # Load and reshape the AIRS-CH0 data
            airs_signal = pd.read_parquet(
                f'{planet_path}/AIRS-CH0_signal_0.parquet'
            ).to_numpy().reshape(self.airs_frames, 32, 356)[:, :, self.cut_inf:self.cut_sup]

            # Convert to float64 from unit16
            airs_signal = airs_signal.astype(np.float64)

            # Get frame count
            airs_frames = airs_signal.shape[0]

            # Load and prep calibration data
            calibration_data = CalibrationData(
                input_data_path=self.input_data_path,
                planet_path=planet_path,
                fgs_frames=fgs_frames,
                airs_frames=airs_frames,
                cut_inf=self.cut_inf,
                cut_sup=self.cut_sup
            )

            # Step 1: ADC conversion
            if self.adc_conversion:
                airs_signal = self._ADC_convert(airs_signal)
                fgs_signal = self._ADC_convert(fgs_signal)

            # Step 2: Mask hot/dead pixels
            if self.masking:
                airs_signal = self._mask_hot_dead(
                    airs_signal,
                    calibration_data.dead_airs,
                    calibration_data.dark_airs
                )

                fgs_signal = self._mask_hot_dead(
                    fgs_signal,
                    calibration_data.dead_fgs,
                    calibration_data.dark_fgs
                )

            # Step 3: Linearity correction
            if self.linearity_correction:
                airs_signal = self._apply_linear_corr(
                    calibration_data.linear_corr_airs,
                    airs_signal
                )

                fgs_signal = self._apply_linear_corr(
                    calibration_data.linear_corr_fgs,
                    fgs_signal
                )

            # Step 4: Dark current subtraction
            if self.dark_subtraction:
                airs_signal = self._clean_dark(
                    airs_signal,
                    calibration_data.dead_airs,
                    calibration_data.dark_airs,
                    calibration_data.dt_airs
                )

                fgs_signal = self._clean_dark(
                    fgs_signal,
                    calibration_data.dead_fgs,
                    calibration_data.dark_fgs,
                    calibration_data.dt_fgs
                )

            # Step 5: Correlated Double Sampling (CDS)
            if self.cds_subtraction:
                airs_signal = self._get_cds(airs_signal)
                fgs_signal = self._get_cds(fgs_signal) 

            # Step 6: Flat field correction
            if self.flat_field_correction:
                airs_signal = self._correct_flat_field(
                    airs_signal,
                    calibration_data.flat_airs,
                    calibration_data.dead_airs
                )

                fgs_signal = self._correct_flat_field(
                    fgs_signal,
                    calibration_data.flat_fgs,
                    calibration_data.dead_fgs
                )

            result = {
                'planet': planet,
                'airs_signal': airs_signal,
                'fgs_signal': fgs_signal
            }

            output_queue.put(result)

        return True


    def _ADC_convert(self, signal):
        '''
        Step 1: Convert raw detector counts to physical units.
        
        Applies analog-to-digital conversion correction using gain and offset
        values from the adc_info.csv file.
        
        Args:
            signal (np.ndarray): Raw detector signal
            
        Returns:
            np.ndarray: ADC-corrected signal
        '''
        signal = signal.astype(np.float64)
        signal /= self.gain    # Apply gain correction
        signal += self.offset  # Apply offset correction

        return signal


    def _mask_hot_dead(self, signal, dead, dark):
        '''
        Step 2: Mask hot and dead pixels in the detector.
        
        Hot pixels are identified using sigma clipping on dark frames.
        Dead pixels are provided in the calibration data.
        
        Args:
            signal (np.ndarray): Input signal array
            dead (np.ndarray): Dead pixel mask from calibration
            dark (np.ndarray): Dark frame for hot pixel detection
            
        Returns:
            np.ma.MaskedArray: Signal with hot/dead pixels masked
        '''
        # Identify hot pixels using 5-sigma clipping on dark frame
        hot = sigma_clip(
            dark, sigma=5, maxiters=5
        ).mask
        
        # Tile masks to match signal dimensions
        hot = np.tile(hot, (signal.shape[0], 1, 1))
        dead = np.tile(dead, (signal.shape[0], 1, 1))
        
        # Apply masks to signal
        signal = np.ma.masked_where(dead, signal)
        signal = np.ma.masked_where(hot, signal)

        return signal
    

    def _apply_linear_corr(self, linear_corr, signal):
        '''
        Step 3: Apply linearity correction to detector response.
        
        Corrects for non-linear detector response using polynomial
        coefficients from calibration data.
        
        Args:
            linear_corr (np.ndarray): Polynomial coefficients for linearity correction
            signal (np.ndarray): Input signal array
            
        Returns:
            np.ndarray: Linearity-corrected signal
        '''
        # Flip coefficients for correct polynomial order
        linear_corr = np.flip(linear_corr, axis=0)

        axis_one = signal.shape[1]
        axis_two = signal.shape[2]
        
        # Apply polynomial correction pixel by pixel
        for x, y in itertools.product(range(axis_one), range(axis_two)):
            poli = np.poly1d(linear_corr[:, x, y])
            signal[:, x, y] = poli(signal[:, x, y])

        return signal
    

    def _clean_dark(self, signal, dead, dark, dt):
        '''
        Step 4: Subtract dark current from signal.
        
        Removes thermal background scaled by integration time.
        
        Args:
            signal (np.ndarray): Input signal array
            dead (np.ndarray): Dead pixel mask
            dark (np.ndarray): Dark frame
            dt (np.ndarray): Integration time for each frame
            
        Returns:
            np.ndarray: Dark-corrected signal
        '''

        # Mask dead pixels in dark frame
        dark = np.ma.masked_where(dead, dark)
        dark = np.tile(dark, (signal.shape[0], 1, 1))

        # Subtract scaled dark current
        signal -= dark * dt[:, np.newaxis, np.newaxis]

        return signal
    

    def _get_cds(self, signal):
        '''
        Step 5: Apply Correlated Double Sampling (CDS).
        
        Subtracts alternating exposure pairs to remove read noise.
        This reduces the number of frames by half.
        
        Args:
            signal (np.ndarray): Input signal array
            
        Returns:
            np.ndarray: CDS-processed signal (half the input frames)
        '''
        # Subtract even frames from odd frames
        cds = signal[1::2,:,:] - signal[::2,:,:]

        return cds


    def _correct_flat_field(self, signal, flat, dead):
        '''
        Step 6: Apply flat field correction.
        
        Normalizes pixel-to-pixel sensitivity variations using
        flat field calibration data.
        
        Args:
            signal (np.ndarray): Input signal array
            flat (np.ndarray): Flat field frame
            dead (np.ndarray): Dead pixel mask
            
        Returns:
            np.ndarray: Flat field corrected signal
        '''
        # Transpose flat field to match signal orientation
        signal = signal.transpose(0, 2, 1)
        flat = flat.transpose(1, 0)
        dead = dead.transpose(1, 0)
        
        # Mask dead pixels in flat field
        flat = np.ma.masked_where(dead, flat)
        flat = np.tile(flat, (signal.shape[0], 1, 1))
        
        # Apply flat field correction
        signal = signal / flat

        return signal.transpose(0, 2, 1)


    def _fgs_downsamples(self):
        '''
        Generate downsampling indices for FGS signal to match AIRS cadence.
        
        Creates an index array for downsampling FGS1 data from 135,000 frames
        to match the AIRS-CH0 frame rate. Preserves the correlated double sampling
        (CDS) structure by taking frame pairs at regular intervals.
        
        Downsampling Strategy:
            - Take every 24th frame pair (frames n and n+1)
            - Reduces data volume by ~83% (135k → 22.5k frames)
            - Maintains temporal alignment with AIRS-CH0 observations
            - Preserves CDS structure for proper noise reduction
        
        Returns:
            np.ndarray: Sorted array of frame indices to extract from FGS data
            
        Example:
            For n=24, generates indices: [0, 1, 24, 25, 48, 49, ...]
            This creates pairs for CDS while dramatically reducing data volume.
        '''
        n = 24  # Take 2 elements, skip 20
        indices_to_take = np.arange(0, self.fgs_frames, n)  # Start from 0, step by n
        indices_to_take = np.concatenate([  # Add the next index
            indices_to_take,
            indices_to_take + 1
        ])

        indices_to_take = np.sort(indices_to_take).astype(int)

        return indices_to_take
    

    def _save_corrected_data(self, output_queue):
        '''
        Dedicated output process for saving corrected signals to HDF5.
        
        This method runs in a separate process and continuously receives
        corrected signal data from worker processes via the output queue.
        It handles proper HDF5 file creation, group organization, and
        graceful shutdown when all workers complete.
        
        Process Flow:
            1. Listen for corrected signal data from output queue
            2. Create HDF5 groups for each planet
            3. Save AIRS-CH0 and FGS1 signals as datasets
            4. Handle stop signals from worker processes
            5. Terminate when all workers have finished
        
        Parameters:
            output_queue (multiprocessing.Queue): Queue containing corrected signal data
                Expected format: {'planet': str, 'airs_signal': ndarray, 'fgs_signal': ndarray}
                
        Returns:
            bool: True when all data has been saved and workers terminated
            
        Output Format:
            HDF5 file structure:
            ├── planet_id_1/
            │   ├── AIRS-CH0_signal       # Corrected spectrometer data
            │   ├── AIRS-CH0_signal_mask  # Mask for spectrometer data
            │   ├── FGS1_signal           # Corrected guidance camera data
            │   └── FGS1_signal_mask      # Mask for guidance camera data
            |
            ├── planet_id_2/
            │   ├── AIRS-CH0_signal       # Corrected spectrometer data
            │   ├── AIRS-CH0_signal_mask  # Mask for spectrometer data
            │   ├── FGS1_signal           # Corrected guidance camera data
            │   └── FGS1_signal_mask      # Mask for guidance camera data
            |
            └── ...
            
        Error Handling:
            - Catches and reports TypeError exceptions during HDF5 writing
            - Continues processing even if individual planet saves fail
            - Provides diagnostic information for failed save operations
        '''
        
        # File path for hdf5 output
        output_file = (f'{self.output_data_path}/train.h5')

        # Stop signal handler
        stop_count = 0

        # Track progress
        output_count = 0

        while True:
            result = output_queue.get()

            if result['planet'] == 'STOP':
                stop_count += 1

                if stop_count == self.n_cpus:
                    break

            else:
                planet = result['planet']
                airs_signal = result['airs_signal']
                fgs_signal = result['fgs_signal']

                with h5py.File(output_file, 'a') as hdf:

                    try:

                        # Create groups for this planet
                        planet_group = hdf.require_group(planet)

                        # Create datasets for AIRS-CH0 and FGS1 signals
                        if self.compress_output:

                            _ = planet_group.create_dataset(
                                'AIRS-CH0_signal',
                                data=airs_signal.data,
                                compression='gzip',
                                compression_opts=9
                            )

                            _ = planet_group.create_dataset(
                                'AIRS-CH0_mask',
                                data=airs_signal.mask[0],
                                compression='gzip',
                                compression_opts=9
                            )

                            _ = planet_group.create_dataset(
                                'FGS1_signal',
                                data=fgs_signal.data,
                                compression='gzip',
                                compression_opts=9
                            )

                            _ = planet_group.create_dataset(
                                'FGS1_mask',
                                data=fgs_signal.mask[0],
                                compression='gzip',
                                compression_opts=9)
                        else:
                            
                            _ = planet_group.create_dataset('AIRS-CH0_signal',data=airs_signal.data)
                            _ = planet_group.create_dataset('AIRS-CH0_mask',data=airs_signal.mask[0])
                            _ = planet_group.create_dataset('FGS1_signal',data=fgs_signal.data)
                            _ = planet_group.create_dataset('FGS1_mask',data=fgs_signal.mask[0])

                        output_count += 1

                        if self.verbose:
                            print(f'Finished planet {output_count} of {len(self.planet_list)}', end='\r')

                    except TypeError as e:
                        print(f'Error writing data for planet {planet}: {e}')
                        print(f'Workunit was: {result}')

        return True