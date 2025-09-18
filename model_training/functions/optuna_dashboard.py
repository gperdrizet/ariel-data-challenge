'''WSGI server app for Optuna dashboard'''

from optuna.storages import RDBStorage
from optuna_dashboard import wsgi

import configuration as config
  
storage_name = f'postgresql://{config.USER}:{config.PASSWD}@{config.HOST}:{config.PORT}/{config.STUDY_NAME}'
storage = RDBStorage(storage_name)
application = wsgi(storage)