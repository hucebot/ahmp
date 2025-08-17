# Agile Humanoid Motion Planning with Contact Discovery

### Install on your system

#### Create a python venv and install dependencies
- `python3 -m venv .venv`
- `source .venv/bin/activate`
- `python3 -m pip install pin meshcat setuptools cyipopt`
- `deactivate`

### Usage

- `source .venv/bin/activate`
- `export PYTHONPATH=$(pwd)/src`
- `python src/examples/go2_trajopt.py` or `python src/examples/talos_trajopt.py`

### Use with Docker

#### Add coinhsl libraries (optional)
- Extract coinhsl.zip folder in ci/ folder.


#### Create container image
- `cd ci`
- `docker build -t ahmp_cond'

#### Run the container
- `cd [project_folder]`
- `./run_docker.sh'`

