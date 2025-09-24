'''WSGI server app for Optuna dashboard. Start with:

gunicorn -b YOUR_LISTEN_IP --workers 2 model_training.functions.optuna_dashboard:application

from the project root directory.

'''

from optuna.storages import RDBStorage
from optuna_dashboard import wsgi

import configuration as config
  
storage_name = f'postgresql://{config.USER}:{config.PASSWD}@{config.HOST}:{config.PORT}/{config.STUDY_NAME}'
storage = RDBStorage(storage_name)
application = wsgi(storage)