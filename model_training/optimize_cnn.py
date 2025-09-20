'''Full run to optimize CNN hyperparameters using Optuna.'''

RUNS = 1000

# Third party imports
import optuna

# Local imports
from model_training.functions.utils import setup_optuna_run, training_run


def objective(
        trial,
        training_planet_ids: list, 
        validation_planet_ids: list,
        worker_num: int
) -> float:
    '''Objective function for Optuna CNN hyperparameter optimization.'''

    rmse = training_run(
        model_type='variable_depth_cnn',
        worker_num=worker_num,
        training_planet_ids=training_planet_ids,
        validation_planet_ids=validation_planet_ids,
        epochs=100,
        sample_size=trial.suggest_int('sample_size', 500, 1000, step=1),
        batch_size=trial.suggest_categorical('batch_size', [1, 2, 4]),
        steps=trial.suggest_int('steps', 1, 200, step=1),
        learning_rate=trial.suggest_float('learning_rate', 1e-11, 1e-2),
        l1=trial.suggest_float('l_one', 1e-10, 1e-1),
        l2=trial.suggest_float('l_two', 1e-10, 1e-1),
        cnn_layers=trial.suggest_categorical('cnn_layers', [1, 2, 3]),
        first_filter_set=trial.suggest_int('first_filter_set', 16, 128, step=1),
        second_filter_set=trial.suggest_int('second_filter_set', 16, 64, step=1),
        third_filter_set=trial.suggest_int('third_filter_set', 16, 64, step=1),
        first_filter_size=trial.suggest_int('first_filter_size', 2, 4, step=1),
        second_filter_size=trial.suggest_int('second_filter_size', 2, 4, step=1),
        third_filter_size=trial.suggest_int('third_filter_size', 2, 4, step=1),
        dense_layers=trial.suggest_categorical('dense_layers', [1, 2, 3]),
        first_dense_units=trial.suggest_int('first_dense_units', 16, 64, step=1),
        second_dense_units=trial.suggest_int('second_dense_units', 16, 64, step=1),
        third_dense_units=trial.suggest_int('third_dense_units', 16, 64, step=1),
        beta_one=trial.suggest_float('beta_one', 0.7, 0.999),
        beta_two=trial.suggest_float('beta_two', 0.8, 0.9999),
        amsgrad=trial.suggest_categorical('amsgrad', [True, False]),
        weight_decay=trial.suggest_float('weight_decay', 1e-10, 1e-1),
        use_ema=trial.suggest_categorical('use_ema', [True, False])
    )
    
    return rmse


def run(worker_num: int) -> None:
    '''Main function to start Optuna optimization run.'''

    run_assets = setup_optuna_run()

    # Define the study
    study = optuna.create_study(
        study_name='cnn_optimization',
        direction='minimize',
        storage=run_assets['storage_name'],
        load_if_exists=True
    )

    study.optimize(
        lambda trial: objective(
            trial=trial,
            training_planet_ids=run_assets['training_planet_ids'],
            validation_planet_ids=run_assets['validation_planet_ids'],
            worker_num=worker_num
        ),
        n_trials=RUNS
    )