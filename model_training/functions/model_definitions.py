'''Model definitions'''

# Third party imports
import tensorflow as tf

# Local imports
import configuration as config

def cnn(
        frames: int=config.FRAMES,
        wavelengths: int=config.WAVELENGTHS,
        learning_rate: float=config.LEARNING_RATE,
        l1: float=config.L1_PENALTY,
        l2: float=config.L2_PENALTY,
        filter_nums=config.FILTER_NUMS,
        filter_size=config.FILTER_SIZE
) -> tf.keras.Model:

    '''Builds the convolutional neural network regression model'''

    # Set-up the L1L2 for the dense layers
    regularizer = tf.keras.regularizers.L1L2(l1=l1, l2=l2)

    # Define the model layers in order
    model = tf.keras.Sequential([
        tf.keras.layers.Input((frames,wavelengths,1)),
        tf.keras.layers.Conv2D(
            filter_nums[0],
            filter_size,
            padding='same',
            activation='relu',
        ),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(
            filter_nums[1],
            filter_size,
            padding='same',
            activation='relu',
        ),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(
            filter_nums[2],
            filter_size,
            padding='same',
            activation='relu',
        ),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(
            128,
            kernel_regularizer=regularizer,
            activation='relu',
        ),
        tf.keras.layers.Dense(wavelengths, activation='relu')
    ])

    # Define the optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    # Compile the model, specifying the type of loss to use during training and any extra
    # metrics to evaluate
    model.compile(
        optimizer=optimizer,
        loss=tf.keras.losses.MeanSquaredError(name='MSE'),
        metrics=[tf.keras.metrics.RootMeanSquaredError(name='RMSE')]
    )

    return model