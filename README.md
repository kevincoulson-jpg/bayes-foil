# Airfoil Optimization and Visualization

This project implements airfoil optimization using Bayesian optimization and provides interactive visualization tools for the results.

## Features

- NACA 4-digit airfoil generation
- Interactive Dash-based visualization of optimization results
- Bayesian optimization for airfoil parameters
- CFD simulation integration

## Requirements

- Python 3.x
- Dependencies listed in `requirements.txt`

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To run the interactive visualization:
```bash
python dash_naca_viewer.py
```

## Project Structure

- `dash_naca_viewer.py`: Interactive visualization dashboard
- `gen_airfoil.py`: NACA airfoil generation
- `bayes_optimize.py`: Bayesian optimization implementation
- `cfd_sim.py`: CFD simulation integration
- `optimization_config.py`: Configuration for optimization 