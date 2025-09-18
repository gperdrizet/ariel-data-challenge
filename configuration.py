'''Globals for ariel data challenge project'''


#############################################################
# Data Stuff ################################################
#############################################################

# Kaggle dataset
COMPETITION_NAME = 'ariel-data-challenge-2025'

# Data paths
DATA_DIRECTORY = './data'
RAW_DATA_DIRECTORY = f'{DATA_DIRECTORY}/raw'
METADATA_DIRECTORY = f'{DATA_DIRECTORY}/metadata'
CORRECTED_DATA_DIRECTORY = f'{DATA_DIRECTORY}/corrected'
PROCESSED_DATA_DIRECTORY = f'{DATA_DIRECTORY}/processed'
EXPERIMENT_RESULTS_DIRECTORY = f'{DATA_DIRECTORY}/experiment_results'
FIGURES_DIRECTORY = './figures'

# Planet to use for demonstration plotting, sample frames etc.
SAMPLE_PLANET = '342072318'

# Number of frames to save for unittesting
SAMPLE_FRAMES = 50


#############################################################
# Figure colors #############################################
#############################################################
from matplotlib import colormaps as cm

COLORMAP = cm.get_cmap('tab20c')
COLORS = COLORMAP.colors

# Set some colors for plotting
BLUE = COLORS[0]
LIGHT_BLUE = COLORS[1]
ORANGE = COLORS[4]
LIGHT_ORANGE = COLORS[5]
GREEN = COLORS[8]
LIGHT_GREEN = COLORS[9]
PURPLE = COLORS[12]
LIGHT_PURPLE = COLORS[13]
GRAY = COLORS[16]
LIGHT_GRAY = COLORS[17]
LIGHTER_GRAY = COLORS[18]
LIGHT_LIGHTER_GRAY = COLORS[19]

TRANSIT_COLOR = ORANGE
SPECTRUM_COLOR = PURPLE

AIRS_HEATMAP_CMAP = 'PuOr_r'
FGS1_HEATMAP_CMAP = 'RdGy'

#############################################################
# Figure export #############################################
#############################################################

STD_FIG_WIDTH = 6
STD_FIG_DPI = 100

#############################################################
# CNN hyperparameters #######################################
#############################################################

NUM_WORKERS = 8
SAMPLES = 100
WAVELENGTHS = 283
EPOCHS = 100
LEARNING_RATE = 0.001
L1_PENALTY = None
L2_PENALTY = None
FILTER_NUMS = [32, 64, 128]
FILTER_SIZE = (3, 3)
BATCH_SIZE = 32
EPOCHS = 20
STEPS = 50
TENSORBOARD_LOG_DIR = 'model_training/logs/'

#############################################################
# CNN hyperparameters distributions for Optuna ##############
#############################################################

hyperparams = {
    'sample_size': (50, 100, 200),
    'learning_rate': (1e-5, 1e-2, 'log'),
    'l_one': (1e-10, 1e-2, 'log'),
    'l_two': (1e-10, 1e-2, 'log'),
    'first_filter_set': (8, 16, 32),
    'second_filter_set': (16, 32, 64),
    'third_filter_set': (32, 64, 128),
    'filter_size': (2, 3, 4),
    'batch_size': (1, 2, 4, 8, 16, 32),
    'steps': (10, 25, 50, 100, 200)
}

#############################################################
# Optuna RDB credentials ####################################
#############################################################
import os

USER = os.environ['POSTGRES_USER']
PASSWD = os.environ['POSTGRES_PASSWD']
HOST = os.environ['POSTGRES_HOST']
PORT = os.environ['POSTGRES_PORT']
STUDY_NAME = 'ariel'