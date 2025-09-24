'''Helper functions for model training'''

# Standard library imports
import datetime
import os
import shutil
from pathlib import Path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Third party imports
import tensorflow as tf

# Local imports
import configuration as config
from ariel_data_preprocessing.data_generator_functions import make_training_datasets
from ariel_data_preprocessing.difference_pair_generator_functions import make_training_datasets as make_difference_pair_datasets
from model_training.functions.model_definitions import cnn, dnn


# Make sure the TensorBoard log directory exists
Path(config.TENSORBOARD_LOG_DIR).mkdir(parents=True, exist_ok=True)

# Set memory growth for GPUs
gpus = tf.config.experimental.list_physical_devices('GPU')

if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    except RuntimeError as e:
        print(e)


def training_run(
        model_type: str,
        worker_num: int,
        training_data_file: str,
        epochs: int,
        sample_size: int,
        batch_size: int,
        steps: int,
        smoothing_window: int = 200,
        standardize_wavelengths: bool = True,
        **hyperparameters
) -> float:

    '''Function to run a single training session with fixed hyperparameters.'''

    gpus = tf.config.list_physical_devices('GPU')

    # Build the model with the suggested hyperparameters
    if model_type == 'cnn':

        if worker_num == 0:
            tf.config.set_visible_devices(gpus[1], 'GPU')

        elif worker_num == 1:
            tf.config.set_visible_devices(gpus[2], 'GPU')

        # Create the training and validation datasets
        training_dataset, validation_dataset, _ = make_training_datasets(
                data_file=training_data_file,
                sample_size=sample_size,
                smoothing_window=smoothing_window,
                standardize_wavelengths=standardize_wavelengths
        )

        model = cnn(
            samples=sample_size,
            **hyperparameters
        )

        validation_steps = 100 // batch_size  # Evaluate on 100 planets

        # Early stopping conditions
        patience = 20
        min_delta = 0.002

    if model_type == 'dnn':

        if worker_num in [0, 1, 2]:
            tf.config.set_visible_devices(gpus[1], 'GPU')

        elif worker_num in [3, 4, 5]:
            tf.config.set_visible_devices(gpus[2], 'GPU')


        # Create the training and validation datasets
        training_dataset, validation_dataset, _ = make_difference_pair_datasets(
                data_file=training_data_file,
                sample_size=sample_size,
                smoothing_window=smoothing_window,
                standardize_wavelengths=standardize_wavelengths
        )

        model = dnn(
            samples=sample_size,
            **hyperparameters
        )

        validation_steps = 10 * sample_size # Evaluate on 10 planets

        # Early stopping conditions
        patience = 50
        min_delta = 0.005

    # Train the model
    model.fit(
        training_dataset.batch(batch_size),
        validation_data=validation_dataset.batch(batch_size),
        epochs=epochs,
        steps_per_epoch=steps,
        validation_steps=steps,
        verbose=0,
        callbacks=[
            early_stopping_callback(patience, min_delta), 
            tensorboard_callback(worker_num, model_type)
        ]
    )

    # Evaluate the model on the validation dataset and return the RMSE
    rmse = model.evaluate(
        validation_dataset.batch(batch_size),
        steps=validation_steps, # Model specific validation steps
        return_dict=True,
        verbose=0
    )['RMSE']

    return rmse


def tensorboard_callback(worker_num: int, model_type: str) -> tf.keras.callbacks.TensorBoard:
    '''Function to create a TensorBoard callback with a unique log directory.'''

    # Set tensorboard callback
    tensorboard_log_dir = (
        f'{config.TENSORBOARD_LOG_DIR}/{model_type}/{worker_num}' +
        f'-{datetime.datetime.now().strftime("%Y%m%d-%H%M%S")}'
    )

    # Make sure the TensorBoard log directory exists
    Path(f'{config.TENSORBOARD_LOG_DIR}/{model_type}').mkdir(parents=True, exist_ok=True)

    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=tensorboard_log_dir,
        histogram_freq=1
    )

    return tensorboard_callback


def early_stopping_callback(patience: int, min_delta: float) -> tf.keras.callbacks.EarlyStopping:

    '''Function to create an early stopping callback.'''

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor='val_RMSE',
        patience=patience,
        min_delta=min_delta,
        mode='min',
        verbose=0,
        restore_best_weights=True
    )

    return early_stopping_callback


def clear_tensorboard_logs(model_type: str) -> None:
    '''Function to clear the TensorBoard log directory.'''

    try:
        shutil.rmtree(f'{config.TENSORBOARD_LOG_DIR}/{model_type}')
    except FileNotFoundError:
        pass

    Path(f'{config.TENSORBOARD_LOG_DIR}/{model_type}').mkdir(parents=True, exist_ok=True)
