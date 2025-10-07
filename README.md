# Real-Driving-Data-Driven-Vehicle-Dynamics-and-Powertrain-Analysis-based-on-Automated-ECU-Signal-Identification

## Overview
This presents an automated data analysis tool for the analysis of vehicle dynamics and electric powertrain efficiency. The data is provided by the ECUs and is dervied from real-driving scenarios on open-road tests. The code is structured to facilitate both pipeline execution and fine-tuned experimentation, ensuring adaptability for sensitivity and comparative analysis.

The tool incorporates three main modules in a modular KDD software achrichtecture environment with Driving Resistance Identification, including various vehicle states, Power unit efficiency Analysis, inclduing efficiency maps for multiple gears, and Gear Shift Strategy, inclduing several driving modes.

The design parameters were identified in a Global Sensitivity Analysis for Robustness and Generalizability.

## Main Entry Points
The primary entry points for executing the final algorithms and generating the thesis plots are:

- `roadload_coastdown.ipynb` - Pipeline function call for road load derivation using the coastdown method
- `roadload_constspeed.ipynb` - Pipeline function call for road load derivation using the constant speed method
- `efficiency_map.ipynb` - Pipeline function call for the efficiency map algorithm with finetuned design parameters
- `gear_strategy.ipynb` - Pipeline function call for the gear strategy derivation

Each of these notebooks serves as an interface to interact with their respective pipeline functions. Demo function calls are included in each notebook.

## Grid Search and Log Exploration
Additional notebooks for performing hyperparameter tuning and analyzing results:

- `grid_search.ipynb` - Parent function for running grid searches with example configurations
- `grid_search_log_explorer.ipynb` - Notebook for analyzing the results of grid searches with interactive log exploration

## Simulation of Driving Cycles
- `driving_cycle_simulator.ipynb` - Entry point for the driving cycle simulation pipeline

## Experimental Code
- `interactive_coastdown.py` - Execute this to start a dash dashboard that interactively updates the coastdown plots based on the selected design paramters for quick exploration

## Folder Structure

```
.
├── data                   # Driving data, log outputs, and processed files (e.g., .pkl for gear strategy files, driving cycle velocity profiles)
├── modules                # Custom libraries containing core functions
│   ├── calculator.py       # Analytical calculation functions
│   ├── data_handler.py     # Data loading, handling, conversion, and processing logic
│   ├── parametric_pipelines.py # All pipeline functions, including concurrency optimization and caching
│   ├── plotter.py          # Plot generation functions
│   ├── threadpool_executer.py  # Backend for executing grid searches with robust result retrieval and concurrency handling
└── README.md               # This file
```

## Additional Notes
- Some functions allow setting a `filename_prefix`, which prompts the function to generate plots or result files with the specified prefix.

## Developer
The underlying methodology, signal acquisition, reverse engineering, and previous analysis tools were developed by Nico Rosenberger (Institute for Automotive Technology, Technical University of Munich). 

This tool was implemented by Till Zacher (Institute for Automotive Technology, Technical University of Munich) as part of his [Master's Thesis](https://github.com/tillzacher/EV-Benchmarking-Tool/blob/8ce83c2284a13de1e58962272621db131dda67f7/Zacher%2C%20Till%20-%20Developing%20a%20Framework%20for%20Sensitivity%20and%20Comparative%20Analysis%20in%20Real%20Driving%20Data-Based%20Vehicle%20Parameter%20Extraction.pdf). 

The detailed building of the data processing pipelines, the parameter tuning, and evaluation framework was carried out by Till Zacher.


## Sources

The tool is documented in the following dissertation (Link will be avialable after acceptance):

N. Rosenberger, „Real-Driving Data-Driven Vehicle Dynamics and Powertrain Analysis based on Automated ECU Signal Identification,“ Dissertation, Technical University of Munich, 2025.


