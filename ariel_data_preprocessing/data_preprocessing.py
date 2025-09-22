'''Signal correction pipeline for Ariel Data Challenge

This module implements the complete preprocessing pipeline for Ariel telescope data,
including ADC conversion, pixel masking, linearity correction, dark current subtraction,
correlated double sampling (CDS), and flat field correction.
'''

# Standard library imports
import os
from multiprocessing import Manager, Process
from random import shuffle

# Third party imports
import h5py
import numpy as np
import pandas as pd

# Internal imports
import ariel_data_preprocessing.signal_correction_functions as correction_funcs
import ariel_data_preprocessing.signal_extraction_functions as extraction_funcs
from ariel_data_preprocessing.calibration_data import CalibrationData
from ariel_data_preprocessing.data_generator_functions import make_training_datasets, make_testing_dataset
from ariel_data_preprocessing.utils import get_planet_list


class DataProcessor:
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
        >>> processor = DataProcessor(
        ...     input_data_path='data/raw',
        ...     output_data_path='data/corrected',
        ...     n_cpus=4,
        ...     downsample_fgs=True,
        ...     n_planets=100
        ... )
        >>> processor.run()
    
    Input Requirements:
        - Works with Ariel Data Challenge (2025) dataset from Kaggle
        - Raw Ariel telescope data in parquet format
        - Calibration data (dark, dead, flat, linearity correction files)
        - ADC conversion parameters
        - Axis info metadata for timing
        - Input structure:

            train/                            # Generated plots and visualizations
            └── 1010375142                    # Planets - 1100 numbered directories
                ├── AIRS-CH0_calibration_0/   # Calibration data
                │   ├── dark.parquet          # Exposure with closed shutter
                │   ├── dead.parquet          # Dead or hot pixels
                │   ├── flat.parquet          # Uniform illuminated surface
                │   ├── linear_corr.parquet   # Correction for nonlinear response
                │   └── read.parquet          # Detector read noise
                │
                ├── FGS1_calibration_0/       # Same set of calibration files
                ├── AIRS-CH0_signal_0.parquet # Image data for observation 0
                └── FGS1_signal_0.parquet     # Image data for observation 0

    
    Output:
        - HDF5 file with corrected AIRS-CH0 and FGS1 signals and hot/dead pixel masks
        - Organized by planet ID for easy access
        - Reduced data volume (50% reduction from CDS, optional 83% FGS reduction)
        - Science-ready data for downstream analysis
        - Output structure:

            train.h5:
            │
            ├── planet_id_1/
            │   ├── signal  # Combined corrected/extracted spectral time series
            │   └── mask    # Dead/hot pixel mask for spectra
            |
            ├── planet_id_2/
            │   ├── signal  
            │   └── mask    
            |
            └── planet_id_n/
    '''

    def __init__(
            self,
            input_data_path: str = None,
            output_data_path: str = None,
            output_filename: str = 'train.h5',
            adc_conversion: bool = True,
            masking: bool = True,
            linearity_correction: bool = True,
            dark_subtraction: bool = True,
            cds_subtraction: bool = True,
            flat_field_correction: bool = True,
            fgs_frames: int = 135000,
            airs_frames: int = 11250,
            cut_inf: int = 39,
            cut_sup: int = 321,
            gain: float = 0.4369,
            offset: float = -1000.0,
            inclusion_threshold: float = 0.75,
            smooth: bool = True,
            smoothing_window: int = 200,
            wavelengths: int = 283,
            n_cpus: int = 1,
            n_planets: int = -1,
            downsample_fgs: bool = False,
            compress_output: bool = False,
            verbose: bool = False,
            mode: str = 'train'
    ):
        '''
        Initialize the DataProcessor class with processing parameters.
        
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
            - smoothing_window (int, default=200): Window size for moving average smoothing
            - inclusion_threshold (float, default=0.75): Pixel inclusion threshold for extraction
            - smooth (bool, default=True): Enable moving average smoothing step
            - wavelengths (int, default=283): Number of wavelengths after AIRS cropping & FGS addition
            - n_cpus (int, default=1): Number of CPU cores for parallel processing
            - n_planets (int, default=-1): Number of planets to process (-1 for all)
            - downsample_fgs (bool, default=False): Enable FGS1 downsampling to match AIRS cadence
            - compress_output (bool, default=False): Enable compression for HDF5 output datasets
            - verbose (bool, default=False): Enable progress counter output
            - mode (str, default='train'): 'train' or 'test' to specify dataset type
            
        Raises:
            ValueError: If input_data_path or output_data_path are None
        '''
        
        # Check required parameters
        if input_data_path is None or output_data_path is None:
            raise ValueError('Input and output data paths must be provided.')
        
        if mode not in ['train', 'test']:
            raise ValueError("Mode must be either 'train' or 'test'.")
        
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
        self.cut_inf = cut_inf
        self.cut_sup = cut_sup
        self.gain = gain
        self.offset = offset
        self.inclusion_threshold = inclusion_threshold
        self.smooth = smooth
        self.smoothing_window = smoothing_window
        self.wavelengths = wavelengths
        self.n_cpus = n_cpus
        self.n_planets = n_planets
        self.downsample_fgs = downsample_fgs
        self.compress_output = compress_output
        self.verbose = verbose
        self.mode = mode

        # Add placeholders for data generators
        if self.mode == 'test':
            self.training = None
            self.validation = None
            self.evaluation = None

        elif mode == 'train':
            self.testing = None

        # Set placeholder for the planet list
        # to be filled in when run() is called
        self.planet_list = None

        # Set output filepath
        self.output_filepath = (f'{self.output_data_path}/{self.output_filename}')


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

        # Make sure output directory exists
        os.makedirs(self.output_data_path, exist_ok=True)

        # Remove output hdf5 file, if it already exists
        try:
            os.remove(self.output_filepath)

        except OSError:
            pass

        # Get planet list from input data
        self.planet_list = get_planet_list(self.input_data_path, mode=self.mode)

        if self.n_planets != -1:
            self.planet_list = self.planet_list[:self.n_planets]

        # Set downsampling indices for FGS data
        if self.downsample_fgs:
            self.fgs_indices = correction_funcs.fgs_downsamples(self.fgs_frames)

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
                    target=self._process_data,
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


    def _process_data(self, input_queue, output_queue):
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

            # Get the next planet ID from the input queue
            planet = input_queue.get()

            # Check for stop signal
            if planet == 'STOP':
                result = {
                    'planet': 'STOP',
                    'signal': None,
                }
                output_queue.put(result)

                break

            # Get path to this planet's data
            planet_path = f'{self.input_data_path}/{self.mode}/{planet}'

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
            ).to_numpy().reshape(
                self.airs_frames, 32, 356
            )[:, :, self.cut_inf:self.cut_sup]

            # Convert to float64 from unit16
            airs_signal = airs_signal.astype(np.float64)

            # Get frame count
            airs_frames = airs_signal.shape[0]

            if airs_frames != fgs_frames:
                raise ValueError(
                    f'Frame count mismatch for planet {planet}'
                )
            
            if airs_frames < self.smoothing_window:
                raise ValueError(
                    f'Not enough frames for smoothing for planet {planet}'
                )

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
                airs_signal = correction_funcs.ADC_convert(airs_signal, self.gain, self.offset)
                fgs_signal = correction_funcs.ADC_convert(fgs_signal, self.gain, self.offset)

            # Step 2: Mask hot/dead pixels
            if self.masking:
                airs_signal = correction_funcs.mask_hot_dead(
                    airs_signal,
                    calibration_data.dead_airs,
                    calibration_data.dark_airs
                )

                fgs_signal = correction_funcs.mask_hot_dead(
                    fgs_signal,
                    calibration_data.dead_fgs,
                    calibration_data.dark_fgs
                )

            # Step 3: Linearity correction
            if self.linearity_correction:
                airs_signal = correction_funcs.apply_linear_corr(
                    calibration_data.linear_corr_airs,
                    airs_signal
                )

                fgs_signal = correction_funcs.apply_linear_corr(
                    calibration_data.linear_corr_fgs,
                    fgs_signal
                )

            # Step 4: Dark current subtraction
            if self.dark_subtraction:
                airs_signal = correction_funcs.clean_dark(
                    airs_signal,
                    calibration_data.dead_airs,
                    calibration_data.dark_airs,
                    calibration_data.dt_airs
                )

                fgs_signal = correction_funcs.clean_dark(
                    fgs_signal,
                    calibration_data.dead_fgs,
                    calibration_data.dark_fgs,
                    calibration_data.dt_fgs
                )

            # Step 5: Correlated Double Sampling (CDS)
            if self.cds_subtraction:
                airs_signal = correction_funcs.get_cds(airs_signal)
                fgs_signal = correction_funcs.get_cds(fgs_signal)

            # Step 6: Flat field correction
            if self.flat_field_correction:
                airs_signal = correction_funcs.correct_flat_field(
                    airs_signal,
                    calibration_data.flat_airs,
                    calibration_data.dead_airs
                )

                fgs_signal = correction_funcs.correct_flat_field(
                    fgs_signal,
                    calibration_data.flat_fgs,
                    calibration_data.dead_fgs
                )

            # Step 7: Extract signal
            airs_signal = extraction_funcs.extract_airs_signal(
                airs_signal,
                inclusion_threshold=self.inclusion_threshold
            )

            fgs_signal = extraction_funcs.extract_fgs_signal(
                fgs_signal,
                inclusion_threshold=self.inclusion_threshold
            )

            # Step 8: Combine AIRS-CH0 and FGS1 signals, handling the masks separately
            signal = np.insert(airs_signal, 0, fgs_signal, axis=1)
            mask = np.insert(airs_signal.mask, 0, fgs_signal.mask, axis=1)
            signal = np.ma.MaskedArray(signal, mask=mask)   
            
            # Step 9: Smooth each wavelength across the frames
            if self.smooth:
                signal = extraction_funcs.moving_average_rows(
                    signal,
                    self.smoothing_window
                )

            # Step 10: Standardize each wavelength across frames
            row_means = np.mean(signal, axis=0)
            row_stds = np.std(signal, axis=0)

            signal = (signal - row_means[np.newaxis, :]) / row_stds[np.newaxis, :]

            # Collect result and submit to output worker
            result = {
                'planet': planet,
                'signal': signal
            }

            output_queue.put(result)

        return True
    

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
                Expected format: {'planet': str, 'signal': ndarray}
                
        Returns:
            bool: True when all data has been saved and workers terminated
            
        Error Handling:
            - Catches and reports TypeError exceptions during HDF5 writing
            - Continues processing even if individual planet saves fail
            - Provides diagnostic information for failed save operations
        '''
        
        # Stop signal handler
        stop_count = 0

        # Track progress
        output_count = 0

        # Load labels, if we have them
        if self.mode == 'train':
            labels = pd.read_csv(
                f'{self.input_data_path}/train.csv',
                index_col='planet_id'
            )

        # Set output compression
        if self.compress_output:
            compression='gzip'
            compression_opts=9

        else:
            compression=None
            compression_opts=None

        while True:

            # Get the next result from the output queue
            result = output_queue.get()

            # Check for stop signals from workers
            if result['planet'] == 'STOP':
                stop_count += 1

                if stop_count == self.n_cpus:
                    break

            else:

                # Unpack workunit
                planet = result['planet']
                signal = result['signal']

                # Get true spectrum for this planet, if we have it
                if self.mode == 'train':
                    true_spectrum = labels.loc[int(planet)].to_numpy(dtype=np.float64)

                with h5py.File(self.output_filepath, 'a') as hdf:

                    try:

                        # Save complete spectrum and mask
                        planet_group = hdf.require_group(planet)

                        _ = planet_group.create_dataset(
                            'signal',
                            data=signal.data,
                            compression=compression,
                            compression_opts=compression_opts
                        )

                        _ = planet_group.create_dataset(
                            'mask',
                            data=signal.mask[0],
                            compression=compression,
                            compression_opts=compression_opts
                        )
                            
                        if self.mode == 'train':
                            _ = planet_group.create_dataset(
                                'spectrum',
                                data=true_spectrum,
                                compression=compression,
                                compression_opts=compression_opts
                            )

                        output_count += 1

                        if self.verbose:
                            print(f'Processed signal for planet {output_count} ' +
                                  f'of {len(self.planet_list)}', end='\r')

                    except TypeError as e:
                        print(f'Error writing data for planet {planet}: {e}')
                        print(f'Workunit was: {result}')

        return True


    def initialize_data_generators(self, sample_size: int = 500, validation: bool = True, n_samples: int = 10):

        if self.mode == 'train':
            self.training, self.validation, self.evaluation = make_training_datasets(
                data_file=self.output_filepath,
                sample_size=sample_size,
                output_data_path=self.output_data_path,
                n_samples=n_samples,
                wavelengths=self.wavelengths,
                validation=validation
            )

        elif self.mode == 'test':
            self.testing = make_testing_dataset(
                data_file=self.output_filepath,
                sample_size=sample_size,
                n_samples=n_samples,
                wavelengths=self.wavelengths
            )

