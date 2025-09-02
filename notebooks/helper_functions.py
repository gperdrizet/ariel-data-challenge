'''Helper functions for notebooks.'''

import os
from pathlib import Path


def set_project_root() -> None:
    '''Set the current working directory to project root.'''
    
    current_directory = Path.cwd()
    working_directory = current_directory.parent if current_directory.name == 'notebooks' else current_directory
    os.chdir(working_directory)

    print(f'Working directory: {working_directory}')