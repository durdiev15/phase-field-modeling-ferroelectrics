# Phase Field Simulations of Ferroelectrics

## Overview

This repository contains Python codes for phase field simulations of ferroelectrics. The phase field method is employed to model the evolution of polarization domains in ferroelectric materials. The Fourier spectral method has been used to solve the piezoelectric and polarization equations. For more details, refer to https://doi.org/10.1016/j.commatsci.2022.111928.

## Project Structure

<img src="https://github.com/ddurdiev15/phase-field-modeling-ferroelectrics/blob/main/images/workflow.png" alt="Code Stucture" width="600" height="400">

<!-- ![Sample Image](https://github.com/ddurdiev15/phase-field-modeling-ferroelectrics/blob/main/images/workflow.png) -->

- **main.py**: The main file executed to run the simulation. It contains simulation parameters and imports the `evolve_polarization` module.

- **evolve_polarization.py**: Contains the polarization evolution function. Imports modules for parameters, piezoelectric and strain tensor calculation, energy calculation, Fourier frequency calculation, VTK file writing, Greens function calculation, and piezoelectricity solver.

- **parameters.py**: Stores all material parameters used in the simulation.

- **piezo_strain_tensor.py**: Calculates the piezoelectric and spontaneous strain tensor and its derivative.

- **energy.py**: Calculates bulk energy and its derivative.

- **fourier_frequency.py**: Calculates Fourier frequencies.

- **write_vtk.py**: Provides functionality to write simulation results to VTK files in 3D.

- **greens_function.py**: Calculates the Green's function in Fourier space.

- **solver.py**: Solves the piezoelectric constitutive and balance equations.

## Usage

To run the simulation, execute `main.py` and customize the parameters as needed.

## Result

This is the domain evolution in 3D starting from the initial random polarization leading to the 90° domain wall.
![GIF Example](https://github.com/ddurdiev15/phase-field-modeling-ferroelectrics/blob/main/results/90_DW.gif)
