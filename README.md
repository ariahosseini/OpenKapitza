OpenKapitza
==============================
[//]: # (Badges)
[![GitHub Actions Build Status](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/OpenKapitza/workflows/CI/badge.svg)](https://github.com/REPLACE_WITH_OWNER_ACCOUNT/OpenKapitza/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/OpenKapitza/branch/master/graph/badge.svg)](https://codecov.io/gh/REPLACE_WITH_OWNER_ACCOUNT/OpenKapitza/branch/master)


A python package to compute anharmonic heat transfer across inhomogeneous interfaces.
### Introduction

<div align="justify">

Thermal management across interfaces is of increasingly technological importance in variety fields of thermoelectrics, 
nanoelectronics, catalytic cells, optoelectronic devices, 2D electronic devices and nanotheranostics. For example, 
massive interfacial thermal resistance limits the performance and reliability of microelectronic devices because 
shrinking size has led to increased heating. The net thermal conductivity depends on both the scattering/ transmission 
of phonons at interfaces and the ballistic effect between interfaces. Modeling these materials effectively requires 
phonon transport simulation with mesoscale resolution while including atomistic resolution of phonon interactions at 
non-ideal interfaces. OpenKapitza is a NEGFs solver that can be used to model heat transport across inhomogeneous 
interfaces. In practice, OpenKapitza reads the second- and third-order inter-atomic force constants (IFCs) and returns 
the mode-resolved transport coefficients; these coefficients can be used in mesoscale Boltzmann transport solvers to 
accurately simulate the phonon evolution in multi-component structures or to guide experiments on thermal conductivity. 
The development of the OpenKapitza framework started under the 2022 MolSSI Software Fellowship. The workflow includes 
the following main steps: 1) reading the second-order IFCs from molecular dynamics (MD) or density functional theory 
(DFT) solvers, 2) calculating the self-energy terms, the coupled, uncoupled and surface Green's functions, and 3) computing 
the mode-resolved transport coefficients using Caroli formula. These coefficients are combined with the 
Landauer formula to evaluate thermal conductance. The contribution of individual acoustic/optical modes, scattering 
specularity, angular transmission spectrum, and modal transport coefficients (and thus the explicit relation between 
the phonon transmission probability and the phonon dispersion) are computed on the fly. In addition to careful algorithm
developments, the code made use of the hardware-agnostic computational platform JAX to facilitate vectorized linear 
algebra operations on GPUs.

### Coherent Phonon Transport in Bulk Materials 

The transmission function shows the total number of phonon modes available at a given frequency. Although the 
transmission function from either side is identical, a requirement of detailed balance, the transmittance has 
directional dependence.

<p align="center">
<img src="figs/trans_func.png" alt="drawing" width="500px"> 
</p>

### Gradient Boost Regression Model of Interfacial Thermal Conductance


Gradient boosting is a machine learning technique that gives a prediction model in the form of an ensemble of weak 
prediction models. OpenKapitza uses the method to predict interfacial thermal conductance. The model accuracy against 
support vector machine and Gaussian process is shown here

<p align="center">
<img src="figs/ensemble.png" width="500px"> 
</p>



### Copyright

Copyright (c) 2022, Aria Hosseini


#### Acknowledgements
 
Project based on the 
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms) version 1.6.
# OpenKapitza
</div>