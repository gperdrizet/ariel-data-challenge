'''Model definitions'''

# Standard library imports
import types

# Third party imports
import tensorflow as tf

# Local imports
import configuration as config


def cnn(
        samples: int,
        **hyperparameters
) -> tf.keras.Model:

    '''Builds the convolutional neural network regression model'''

    hyperparameters = types.SimpleNamespace(**hyperparameters)

    # Set-up the L1L2 for the dense layers
    regularizer = tf.keras.regularizers.L1L2(
        l1=hyperparameters.l1,
        l2=hyperparameters.l2
    )

    # Define the model layers in order
    model = tf.keras.Sequential([
        tf.keras.layers.Input((samples,config.WAVELENGTHS,1)),
        tf.keras.layers.Conv2D(
            hyperparameters.first_filter_set,
            hyperparameters.first_filter_size,
            padding='same',
            activation='relu',
        ),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(
            hyperparameters.second_filter_set,
            hyperparameters.second_filter_size,
            padding='same',
            activation='relu',
        ),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(
            hyperparameters.third_filter_set,
            hyperparameters.third_filter_size,
            padding='same',
            activation='relu',
        ),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(
            hyperparameters.dense_units,
            kernel_regularizer=regularizer,
            activation='relu',
        ),
        tf.keras.layers.Dense(config.WAVELENGTHS, activation='linear')
    ])

    # Define the optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=hyperparameters.learning_rate)

    # Compile the model, specifying the type of loss to use during training and any extra
    # metrics to evaluate
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.MeanSquaredError(name='MSE'),
        metrics=[tf.keras.metrics.RootMeanSquaredError(name='RMSE')]
    )

    return model


def variable_depth_cnn(
        samples: int,
        **hyperparameters
) -> tf.keras.Model:

    '''Builds the convolutional neural network regression model'''

    hyperparameters = types.SimpleNamespace(**hyperparameters)

    # Set-up the L1L2 for the dense layers
    regularizer = tf.keras.regularizers.L1L2(
        l1=hyperparameters.l1,
        l2=hyperparameters.l2
    )

    # Define the model layers in order
    layers = [
        tf.keras.layers.Input((samples,config.WAVELENGTHS,1)),
        tf.keras.layers.Conv2D(
            hyperparameters.first_filter_set,
            hyperparameters.first_filter_size,
            padding='same',
            activation='relu',
        ),
        tf.keras.layers.MaxPooling2D()
    ]

    if hyperparameters.cnn_layers > 1:
        layers += [
            tf.keras.layers.Conv2D(
                hyperparameters.second_filter_set,
                hyperparameters.second_filter_size,
                padding='same',
                activation='relu',
            ),
            tf.keras.layers.MaxPooling2D()
        ]

    if hyperparameters.cnn_layers > 2:
        layers += [
            tf.keras.layers.Conv2D(
                hyperparameters.third_filter_set,
                hyperparameters.third_filter_size,
                padding='same',
                activation='relu',
            ),
            tf.keras.layers.MaxPooling2D(),
        ]

    if hyperparameters.cnn_layers > 3:
        layers += [
            tf.keras.layers.Conv2D(
                hyperparameters.fourth_filter_set,
                hyperparameters.fourth_filter_size,
                padding='same',
                activation='relu',
            ),
            tf.keras.layers.MaxPooling2D(),
        ]

    if hyperparameters.cnn_layers > 4:
        layers += [
            tf.keras.layers.Conv2D(
                hyperparameters.fifth_filter_set,
                hyperparameters.fifth_filter_size,
                padding='same',
                activation='relu',
            ),
            tf.keras.layers.MaxPooling2D(),
        ]

    layers += [
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(
            hyperparameters.first_dense_units,
            kernel_regularizer=regularizer,
            activation='relu',
        )
    ]

    if hyperparameters.dense_layers > 1:
        layers += [
            tf.keras.layers.Dense(
                hyperparameters.second_dense_units,
                kernel_regularizer=regularizer,
                activation='relu',
            )
        ]

    if hyperparameters.dense_layers > 2:
        layers += [
            tf.keras.layers.Dense(
                hyperparameters.third_dense_units,
                kernel_regularizer=regularizer,
                activation='relu',
            )
        ]

    layers += [
        tf.keras.layers.Dense(config.WAVELENGTHS, activation='linear')
    ]

    model = tf.keras.Sequential(layers)

    # Define the optimizer
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=hyperparameters.learning_rate,
        beta_1=hyperparameters.beta_one,
        beta_2=hyperparameters.beta_two,
        amsgrad=hyperparameters.amsgrad,
        weight_decay=hyperparameters.weight_decay,
        use_ema=hyperparameters.use_ema
    )

    # Compile the model, specifying the type of loss to use during training and any extra
    # metrics to evaluate
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.MeanSquaredError(name='MSE'),
        metrics=[tf.keras.metrics.RootMeanSquaredError(name='RMSE')]
    )

    return model