'''Functions to get competition data from Kaggle.'''

import kaggle
import zipfile
from pathlib import Path


def download_dataset(competition_name:str, raw_data_directory:str) -> None:
    '''Uses Kaggle API to download competition data.'''

    # Check/make the raw data directory
    Path(raw_data_directory).mkdir(parents=True, exist_ok=True)

    # Create an instance of the API class
    api_instance = kaggle.KaggleApi()
    api_instance.authenticate()

    # Download all competition data files
    api_instance.competition_download_files(
        competition_name,
        path=raw_data_directory
    )


def extract_dataset(file_name:str, raw_data_directory:str) -> None:
    '''Extracts data zip archive.'''

    # Construct archive filepath
    archive_filepath = f'{raw_data_directory}/{file_name}.zip'

    # Open the archive and loop on the contents
    with zipfile.ZipFile(archive_filepath, mode='r') as archive:
        for file in archive.namelist():

            # Extract the file
            archive.extract(file, f'{raw_data_directory}/')