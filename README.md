# AHMP: Agile Humanoid Motion Planning with Contact Discovery

This is the official repository for the experiment code of the paper "AHMP: Agile Humanoid Motion Planning with Contact Sequence Discovery" accepted at the 2025 IEEE-RAS 24th International Conference on Humanoid Robots (Humanoids)".

# Abstract
Planning agile whole-body motions for legged and humanoid robots is a fundamental requirement for enabling dynamic tasks such as running, jumping, and fast reactive maneuvers. In this work, we present AHMP, a multi-contact motion planning framework based on bi-level optimization that integrates a contact sequence discovery technique, using the Mixed-Distribution Cross-Entropy Method (CEM-MD), and an efficient trajectory optimization scheme, which parameterizes the robot’s poses and motions in the tangent space of SE(3). AHMP permits the automatic generation of feasible contact configurations, with associated whole-body dynamic transitions. We validate our approach on a set of challenging agile motion planning tasks for humanoid robots, demonstrating that contact sequence discovery combined with tangent space parameterization leads to highly dynamic motion sequences while remaining computationally efficient.

### Installation instructions to run the paper experiments

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
- `docker build -t ahmp_cond`

#### Run the container
- `cd [project_folder]`
- `./run_docker.sh'`

