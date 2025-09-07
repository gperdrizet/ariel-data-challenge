'''Globals for ariel data challenge project'''

import os

#############################################################
# Data Stuff ################################################
#############################################################

# Kaggle dataset
COMPETITION_NAME = 'ariel-data-challenge-2025'

# Data paths
DATA_DIRECTORY = './data'
RAW_DATA_DIRECTORY = f'{DATA_DIRECTORY}/raw'
METADATA_DIRECTORY = f'{DATA_DIRECTORY}/metadata'

SAMPLE_PLANET = '342072318'

FIGURES_DIRECTORY = './figures'

#############################################################
# Figure export #############################################
#############################################################

STD_FIG_WIDTH = 6
STD_FIG_DPI = 100

#############################################################
# Optuna RDB credentials ####################################
#############################################################

# USER = os.environ['POSTGRES_USER']
# PASSWD = os.environ['POSTGRES_PASSWD']
# HOST = os.environ['POSTGRES_HOST']
# PORT = os.environ['POSTGRES_PORT']
# STUDY_NAME = 'ariel_data'