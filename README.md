## Constrained Composite Bayesian Optimization

### Overview
This project involves the implementation of constrained composite Bayesian optimization (CCBO) based on Gaussian Process (GP) models using [GPyTorch](http://www.gpytorch.ai/) and [BoTorch](https://botorch.org) for guided experimentation. The code includes:

- `CCBO_benchmark.ipynb` A notebook showing benchmarking CCBO against vanilla BO, constrained BO, and random baseline with a synthetic electrospray problem
- `CCBO_guide_exp.ipynb` An example notebook for using CCBO to guide laboratory experiments, with SOBEL initialization
- `core.py` The core implmentation of CCBO algorithm

### How does it work to guide experiment?
![CCBO Overview](ccbo.png)

### Publication (to-be-updated)
Constrained Composite Bayesian Optimization for Rational Synthesis of Polymeric Micro- and Nanoparticles

## Getting Started
To get started with ccbo, follow these steps:

### Dependencies
To run the notebooks and scripts in this project, you will need the following dependencies:
- `torch = 2.2.2`
- `botorch == 0.10.0`
- `gpytorch == 1.11`
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `joblib`

To install the required dependencies, run the following command:
```bash
conda create -n ccbo python=3.9
conda install botorch==0.10.0 gpytorch==1.11 -c pytorch -c gpytorch -c conda-forge
conda install pandas matplotlib seaborn joblib
```

### clone the repository
```bash
git clone https://github.com/FrankWanger/CCBO.git
```


### License
This project is licensed under the MIT License. See the `LICENSE` file for more details.

### Contact
For any questions or issues, please contact Fanjin Wang at fanjin.wang.20@ucl.ac.uk
