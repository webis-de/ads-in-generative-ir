# Setup

## Get geckodriver for Mozilla Firefox in Selenium
1. Download a suitable release from https://github.com/mozilla/geckodriver/releases 
2. Follow the instructions under **Drivers** outlined by the Selenium developers: https://pypi.org/project/selenium

## Create a conda environment
A suitable conda environment can be created from the `.lock`-file as follows:
```
conda create --name ads --file conda-linux-64.lock
conda activate ads
poetry install
```

## Use conda environment as kernel for notebooks
To be able to run the notebooks with the environment, run the following commands:
```
conda activate ads
python -m ipykernel install --user --name=ads
```
Then, in an environment that has jupyter notebooks installed, run `jupyter notebook` and use the ads env as a kernel.

## Update dependencies in conda environment
In order to update the env with additional dependencies, add them to `environment.yml` (and `pyproject.toml`) if they are required to run the library.
Then, perform the following commands:

1. Update the conda lock files based on the `environment.yml`:
```
conda create -p /tmp/bootstrap -c conda-forge mamba conda-lock poetry='1.*' -y
conda activate /tmp/bootstrap
conda-lock -k explicit --conda mamba
conda deactivate
rm -rf /tmp/bootstrap
```
2. Update the conda packages based on the new lock file

```
conda activate ads
mamba update --file conda-linux-64.lock
```
3. Update the poetry packages and poetry.lock
```
poetry update
```