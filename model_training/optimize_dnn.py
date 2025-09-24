'''Full run to optimize CNN hyperparameters using Optuna.'''

# Third party imports
import optuna

# Local imports
import configuration as config
from model_training.functions.training_functions import training_run

RUNS = 1000
TRAINING_DATA_FILE = f'{config.PROCESSED_DATA_DIRECTORY}/train-1100_smoothing-10-20-40-80-160.h5'

def objective(
        trial,
        training_data_file: str,
        worker_num: int
) -> float:
    '''Objective function for Optuna CNN hyperparameter optimization.'''

    rmse = training_run(
        model_type='dnn',
        worker_num=worker_num,
        training_data_file=training_data_file,
        epochs=1000,
        sample_size=100,
        batch_size=1,
        steps=100,
        smoothing_window=trial.suggest_categorical('smoothing_window', [None, 10, 20, 40, 80, 160]),
        standardize_wavelengths=trial.suggest_categorical('standardize_wavelengths', [True, False]),
        learning_rate=trial.suggest_float('learning_rate', 1e-15, 1e-3),
        l1=trial.suggest_float('l_one', 1e-11, 1.0),
        l2=trial.suggest_float('l_two', 1e-11, 1.0),
        hidden_layers=trial.suggest_categorical('hidden_layers', [2, 4, 6, 8, 10]),
        first_layer_dense_units=trial.suggest_int('first_layer_dense_units', 16, 128),
        dense_units_factor=trial.suggest_float('dense_units_factor', 1.1, 3.0),
        beta_one=trial.suggest_float('beta_one', 0.5, 1.0),
        beta_two=trial.suggest_float('beta_two', 0.5, 1.0),
        amsgrad=trial.suggest_categorical('amsgrad', [True, False]),
        weight_decay=trial.suggest_float('weight_decay', 0.0, 0.1),
        use_ema=trial.suggest_categorical('use_ema', [True, False])
    )
    
    return rmse


def run(worker_num: int) -> None:
    '''Main function to start Optuna optimization run.'''

    storage_name = f'postgresql://{config.USER}:{config.PASSWD}@{config.HOST}:{config.PORT}/{config.STUDY_NAME}'

    # Define the study
    study = optuna.create_study(
        study_name='dnn_optimization.2',
        direction='minimize',
        storage=storage_name,
        load_if_exists=True
    )

    study.optimize(
        lambda trial: objective(
            trial=trial,
            training_data_file=TRAINING_DATA_FILE,
            worker_num=worker_num
        ),
        n_trials=RUNS
    )