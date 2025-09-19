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
MODELS_DIRECTORY = f'{DATA_DIRECTORY}/models'
FIGURES_DIRECTORY = './figures'
TENSORBOARD_LOG_DIR = f'model_training/logs/'

# Planet to use for demonstration plotting, sample frames etc.
SAMPLE_PLANET = '342072318'

# Number of frames to save for unittesting
SAMPLE_FRAMES = 50

# Number of wavelength channels in the spectra
WAVELENGTHS = 283

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
# Optuna RDB credentials ####################################
#############################################################
import os

USER = os.environ['POSTGRES_USER']
PASSWD = os.environ['POSTGRES_PASSWD']
HOST = os.environ['POSTGRES_HOST']
PORT = os.environ['POSTGRES_PORT']
STUDY_NAME = 'ariel'