'''Run to optimize wavelength smoothing. Other hyperparameters are fixed at optimum values
from previous Optuna run (~225 samples).'''

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
        model_type='variable_smoothing_cnn',
        worker_num=worker_num,
        training_planet_ids=training_planet_ids,
        validation_planet_ids=validation_planet_ids,
        epochs=100,
        sample_size=trial.suggest_int('sample_size', 300, 800, step=1),
        batch_size=4,
        steps=510,
        smoothing_window=trial.suggest_int('smoothing_window', 10, 400, step=1),
        standardize_wavelengths=trial.suggest_categorical('standardize_wavelengths', [True, False]),
        learning_rate=0.0004594031121381584,
        l1=0.2014647418966941,
        l2=0.33915364828871686,
        cnn_layers=4,
        first_filter_set=119,
        second_filter_set=42,
        third_filter_set=24,
        fourth_filter_set=31,
        fifth_filter_set=37,
        first_filter_size=4,
        second_filter_size=4,
        third_filter_size=4,
        fourth_filter_size=6,
        fifth_filter_size=6,
        dense_layers=1,
        first_dense_units=10,
        second_dense_units=12,
        third_dense_units=8,
        beta_one=0.7257265774635964,
        beta_two=0.8537703743059503,
        amsgrad=False,
        weight_decay=0.0669755281989629,
        use_ema=True
    )
    
    return rmse


def run(worker_num: int) -> None:
    '''Main function to start Optuna optimization run.'''

    run_assets = setup_optuna_run()

    # Define the study
    study = optuna.create_study(
        study_name='deeper_cnn_smoothing_optimization',
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