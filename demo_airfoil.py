import subprocess
import os
import pandas as pd
import numpy as np
from io import StringIO

def run_xfoil_simple(naca_profile, reynolds_number, alpha_start, alpha_end, alpha_step):
    output_filename = "polar_output.txt" # Your specified output file

    # XFOIL commands based on your script
    xfoil_commands = f"""
{naca_profile}
OPER
VISC {reynolds_number}
ITER
200
PACC
{output_filename}

ASEQ {alpha_start} {alpha_end} {alpha_step}

QUIT
"""
    xfoil_commands = xfoil_commands.strip()

    # Run XFOIL
    subprocess.run(
        ["xfoil"],
        input=xfoil_commands,
        capture_output=True,
        text=True
    )

    # Parse output
    with open(output_filename, 'r') as f:
        lines = f.readlines()

    # Find the start of the data table
    data_start_line = -1
    for i, line in enumerate(lines):
        if line.strip().startswith('alpha') and 'CL' in line and 'CD' in line:
            data_start_line = i + 1
            break
    if data_start_line == -1:
        raise ValueError("Could not find data table in XFOIL output.")

    # Read the data section using pandas.read_fwf for fixed-width columns
    # Only keep lines that look like data (start with a number or minus sign)
    data_lines = []
    for line in lines[data_start_line:]:
        if line.strip() == '' or not (line.strip()[0].isdigit() or line.strip()[0] == '-'):  # stop at next header or blank
            break
        data_lines.append(line)

    data_str = ''.join(data_lines)
    df = pd.read_fwf(StringIO(data_str),
                     names=['Alpha', 'Cl', 'Cd', 'Cdp', 'Cm', 'Top_Xtr', 'Bot_Xtr'])

    # Remove rows where 'Alpha' is not a number
    df = df[pd.to_numeric(df['Alpha'], errors='coerce').notnull()]

    # Convert columns to float
    for col in ['Alpha', 'Cl', 'Cd', 'Cdp', 'Cm', 'Top_Xtr', 'Bot_Xtr']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Clean up files
    os.remove(output_filename)

    return df

def run_xfoil_single_alpha(naca_profile, reynolds_number, alpha):
    output_filename = "polar_output.txt"
    xfoil_commands = f"""
{naca_profile}
OPER
VISC {reynolds_number}
ITER
200
PACC
{output_filename}

ALFA {alpha}

QUIT
"""
    xfoil_commands = xfoil_commands.strip()

    # Run XFOIL
    subprocess.run(
        ["xfoil"],
        input=xfoil_commands,
        capture_output=True,
        text=True
    )

    # Parse output
    with open(output_filename, 'r') as f:
        lines = f.readlines()

    # Find the start of the data table
    data_start_line = -1
    for i, line in enumerate(lines):
        if line.strip().startswith('alpha') and 'CL' in line and 'CD' in line:
            data_start_line = i + 1
            break
    if data_start_line == -1:
        raise ValueError("Could not find data table in XFOIL output.")

    # Read the data section using pandas.read_fwf for fixed-width columns
    # Only keep lines that look like data (start with a number or minus sign)
    data_lines = []
    for line in lines[data_start_line:]:
        if line.strip() == '' or not (line.strip()[0].isdigit() or line.strip()[0] == '-'):  # stop at next header or blank
            break
        data_lines.append(line)

    data_str = ''.join(data_lines)
    df = pd.read_fwf(StringIO(data_str),
                     names=['Alpha', 'Cl', 'Cd', 'Cdp', 'Cm', 'Top_Xtr', 'Bot_Xtr'])

    # Remove rows where 'Alpha' is not a number
    df = df[pd.to_numeric(df['Alpha'], errors='coerce').notnull()]

    # Convert columns to float
    for col in ['Alpha', 'Cl', 'Cd', 'Cdp', 'Cm', 'Top_Xtr', 'Bot_Xtr']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Clean up files
    os.remove(output_filename)

    # Return only the first valid Cl and Cd
    if not df.empty:
        cl_val = df['Cl'].iloc[0]
        cd_val = df['Cd'].iloc[0]
        return cl_val, cd_val
    else:
        return np.nan, np.nan

# --- Example Usage ---
if __name__ == "__main__":
    naca = "NACA2412"
    re = 3000000
    alpha_start = -5
    alpha_end = 15
    alpha_step = 1

    polar_data = run_xfoil_simple(naca, re, alpha_start, alpha_end, alpha_step)

    print("\n--- Simulation Results (Alpha, Cl, Cd) ---")
    print(polar_data[['Alpha', 'Cl', 'Cd']])

    # Example: Access Cl and Cd for a specific angle
    target_alpha = 5.0
    row = polar_data[np.isclose(polar_data['Alpha'], target_alpha)]

    if not row.empty:
        cl_val = row['Cl'].iloc[0]
        cd_val = row['Cd'].iloc[0]
        print(f"\nFor Alpha = {target_alpha} degrees: Cl = {cl_val:.4f}, Cd = {cd_val:.6f}")

    # Example usage of run_xfoil_single_alpha
    cl_single, cd_single = run_xfoil_single_alpha(naca, re, 5.0)
    print(f"\nSingle Alpha (5.0 deg): Cl = {cl_single:.4f}, Cd = {cd_single:.6f}")