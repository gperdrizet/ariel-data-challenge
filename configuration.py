'''Globals for ariel data challenge project'''

import os

#############################################################
# Data Stuff ################################################
#############################################################

# Kaggle dataset
COMPETITION_NAME = 'ariel-data-challenge-2025'

# Data paths
DATA_DIRECTORY = './data/fast_scratch'
RAW_DATA_DIRECTORY = f'{DATA_DIRECTORY}/raw_data'
SUBMISSION_DIRECTORY = f'{DATA_DIRECTORY}/submission'

# Data files
TRAINING_DATA_FILE = f'{RAW_DATA_DIRECTORY}/train.csv'
SUBMISSION_DATA_FILE = f'{RAW_DATA_DIRECTORY}/test.csv'

#############################################################
# Optuna RDB credentials ####################################
#############################################################

# USER = os.environ['POSTGRES_USER']
# PASSWD = os.environ['POSTGRES_PASSWD']
# HOST = os.environ['POSTGRES_HOST']
# PORT = os.environ['POSTGRES_PORT']
# STUDY_NAME = 'ariel_data'