'''Dummy Optuna run to test how throughput scales with number of GPU 
workers. Only optimizable hyperparameter is learning rate. All others
set to small-ish fixed values.'''

RUNS = 3

# Third party imports
import optuna

# Local imports
from model_training.functions.utils import setup_optuna_run, training_run

optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(
        trial,
        training_planet_ids: list, 
        validation_planet_ids: list,
        worker_num: int
) -> float:
    '''Objective function for Optuna CNN hyperparameter optimization.'''

    rmse = training_run(
        model_tyope='cnn',
        worker_num=worker_num,
        training_planet_ids=training_planet_ids,
        validation_planet_ids=validation_planet_ids,
        epochs=10,
        sample_size=100,
        batch_size=8,
        steps=20,
        learning_rate=trial.suggest_float('learning_rate', 1e-10, 1e-1),
        l1=None,
        l2=None,
        first_filter_set=64,
        second_filter_set=32,
        third_filter_set=16,
        first_filter_size=2,
        second_filter_size=4,
        third_filter_size=6,
        dense_units=32
    )
    
    return rmse


def run(worker_num: int) -> None:
    '''Main function to start Optuna optimization run.'''

    run_assets = setup_optuna_run()

    # Define the study
    study = optuna.create_study(
        study_name='gpu_workers_optimization',
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