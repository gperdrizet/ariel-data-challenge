# Ariel Data Challenge

[![Unittest](https://github.com/gperdrizet/ariel-data-challenge/actions/workflows/unittest.yml/badge.svg)](https://github.com/gperdrizet/ariel-data-challenge/actions/workflows/unittest.yml)
[![PyPI release](https://github.com/gperdrizet/ariel-data-challenge/actions/workflows/pypi_release.yml/badge.svg)](https://github.com/gperdrizet/ariel-data-challenge/actions/workflows/pypi_release.yml)
[![pages-build-deployment](https://github.com/gperdrizet/ariel-data-challenge/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/gperdrizet/ariel-data-challenge/actions/workflows/pages/pages-build-deployment)

- [Project progress blog](https://gperdrizet.github.io/ariel-data-challenge/)
- [Signal preprocessing package](https://pypi.org/project/ariel-data-preprocessing/)
- [Kaggle competition page: Ariel Data Challenge 2025](https://www.kaggle.com/competitions/ariel-data-challenge-2025/overview)


## 1. Setup

Assumes the following base system configuration:

- Python: 3.8.10
- GPU: Tesla K80
- Nvidia driver: 470.42.01
- CUDA driver: 11.4
- CUDA runtime: 11.4
- CUDA compute capability: 3.7
- cuDNN 8.1
- GCC 9.4.0


### 1.1. Virtual environment

Create a Python 3.8 virtual environment:

```bash
python3.8 -m venv .venv
```


### 1.2. TensorFlow

Set `LD_LIBRARY_PATH` from `.venv/bin/activate`:

```bash
export LD_LIBRARY_PATH=/path/to/project/directory/.venv/lib/
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/path/to/project/directory/.venv/lib/python3.8/site-packages/tensorrt/
```

Activate the virtual environment, install TensorRT and TensorFlow:

```bash
source .venv/bin/activate
pip install nvidia-tensorrt==7.2.3.4
pip install tensorflow==2.11.0
```

Test TensorFlow with:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

You should see something like:

```bash
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:1', device_type='GPU'), PhysicalDevice(name='/physical_device:GPU:2', device_type='GPU')]
```


### 1.3. LightGBM

Install LightGBM with CUDA support. For the `--config-settings` flag to work pip must be >= 23.1.

```bash
pip install lightgbm --no-binary lightgbm --config-settings=cmake.define.USE_CUDA=ON
```


### 1.4. Other requirements

```bash
pip install -r requirements.txt
```


### 1.5. Optuna RDB storage

Optuna needs an SQL database to store run information. Create a PostgreSQL database called `calories`:

```bash
$ sudo -u postgres_admin createdb <PROJECT_NAME>
$ sudo -u postgres_admin psql <PROJECT_NAME>

psql (17.2 (Ubuntu 17.2-1.pgdg20.04+1), server 16.6 (Ubuntu 16.6-1.pgdg20.04+1))
Type "help" for help.

calories=# ALTER USER postgres_user with encrypted password 'your_password';
ALTER ROLE

postgres=# exit
```

Modify `pg_hba.conf` to allow the machine running Optuna to access the database over the LAN and restart the database. Then, set environment variable for the following, to be read with `os.environ()` in `configuration.py`.

1. `POSTGRES_USER`
2. `POSTGRES_PASSWD`
3. `POSTGRES_HOST`
4. `POSTGRES_PORT`
5. `STUDY_NAME`

Once run data is present, you can start the Optuna dashboard with:

```bash
gunicorn -b YOUR_LISTEN_IP --workers 2 functions.optuna_dashboard:application
```


## 2. Data acquisition

Note: the Kaggle API cannot be used to download this dataset unless you have >265 GB system memory. When calling `competition_download_files()` the python library appears to try and read the whole archive into memory before writing anything to disk. Unfortunately, I only have 128 GB system memory.

Get the data the old fashioned way - manually download the archive by clicking the link in the competition. Then decompress with:

```bash
unzip ariel-data-challenge-2025.zip
```
