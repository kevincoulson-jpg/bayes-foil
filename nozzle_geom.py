import numpy as np
import matplotlib.pyplot as plt

def create_nozzle_mask_3d_printer_simplified(nx, ny, params):
    """
    Generates a 2D boolean mask for a simplified 3D printer nozzle geometry (chamber, converging, discharge).
    True indicates a fluid cell, False indicates a wall cell.
    Assumes nozzle is symmetric about the x-axis, and we model the upper half.

    Args:
        nx, ny (int): Number of grid points in x and y directions.
        params (dict): Dictionary with nozzle parameters:
            'D_chamber': Diameter of the melting chamber
            'L_chamber': Length of the melting chamber
            'L_converging': Length of the converging section
            'D_discharge': Diameter of the discharge orifice
            'L_discharge': Length of the discharge section
    
    Returns:
        numpy.ndarray: A boolean array of shape (ny, nx) where True is fluid.
        numpy.ndarray: 1D array of x-coordinates of the grid.
        numpy.ndarray: 1D array of y-coordinates of the grid.
    """
    
    # Extract parameters
    D_chamber = params['D_chamber']
    L_chamber = params['L_chamber']
    L_converging = params['L_converging']
    D_discharge = params['D_discharge']
    L_discharge = params['L_discharge']

    # Define x and y coordinates of the full computational domain grid
    # Chamber starts at x=0.0
    x_chamber_start = 0.0
    x_discharge_end = x_chamber_start + L_chamber + L_converging + L_discharge

    # Extend domain slightly beyond the nozzle for outlet BCs
    x_min_domain = x_chamber_start
    x_max_domain = x_discharge_end + 0.005 # Small margin after exit
    
    # Y-range covers the largest diameter, plus a small margin
    y_max_nozzle = D_chamber / 2.0
    y_min_domain = -y_max_nozzle - 0.005
    y_max_domain = y_max_nozzle + 0.005
    
    x = np.linspace(x_min_domain, x_max_domain, nx)
    y = np.linspace(y_min_domain, y_max_domain, ny)
    
    # Initialize mask: Assume all cells are fluid initially
    nozzle_mask = np.ones((ny, nx), dtype=bool)

    # --- Define key points of the upper nozzle contour ---
    # Coordinates (x, y) for the upper profile
    
    # 1. Melting Chamber (cylindrical section)
    y_chamber_top = D_chamber / 2.0
    x_chamber_end = x_chamber_start + L_chamber
    
    # 2. Converging Section (conical)
    x_converging_end = x_chamber_end + L_converging
    y_converging_end_top = D_discharge / 2.0 # At the start of the discharge section
    
    # 3. Discharge Section (small cylindrical exit)
    y_discharge_top = D_discharge / 2.0 # This is the same as y_converging_end_top
    
    # Define the sequence of points for the upper contour (for plotting the ideal shape)
    upper_contour_points = np.array([
        [x_chamber_start, y_chamber_top],    # 0: Start of chamber (upper corner)
        [x_chamber_end, y_chamber_top],      # 1: End of chamber (upper corner)
        [x_converging_end, y_converging_end_top], # 2: End of converging section
        [x_discharge_end, y_discharge_top]  # 3: End of discharge section (nozzle exit)
    ])
    
    # Loop through each grid point to determine if it's a wall or fluid
    for j in range(ny):
        for i in range(nx):
            current_x = x[i]
            current_y = y[j]

            is_fluid = False
            
            # First, check if within the main X bounds of the nozzle body itself
            if current_x < x_chamber_start or current_x > x_discharge_end:
                 # Outside the main nozzle body, but could still be fluid in inlet/outlet extensions
                 # For this simplified case, we assume fluid only exists within the nozzle defined.
                 # If you need flow *before* chamber or *after* discharge for BCs, adjust this.
                 # For now, let's include the outlet extension as fluid.
                 if current_x > x_discharge_end and abs(current_y) <= y_discharge_top:
                     is_fluid = True
                 else:
                     nozzle_mask[j, i] = False
                     continue # Skip to next point if definitely outside

            # Now, check against the nozzle's upper/lower contours
            if current_y < 0: # Lower half
                # Find the y-value of the lower contour at current_x
                y_contour_lower_at_x = 0.0 

                if current_x <= x_chamber_end: # Melting Chamber
                    y_contour_lower_at_x = -y_chamber_top
                elif current_x <= x_converging_end: # Converging Section
                    # Line between (x_chamber_end, -y_chamber_top) and (x_converging_end, -y_converging_end_top)
                    m = (-y_converging_end_top - (-y_chamber_top)) / (x_converging_end - x_chamber_end)
                    b = -y_chamber_top - m * x_chamber_end
                    y_contour_lower_at_x = m * current_x + b
                elif current_x <= x_discharge_end: # Discharge Section
                    y_contour_lower_at_x = -y_discharge_top
                else: # After discharge exit, treat as open fluid below current y_discharge_top
                    y_contour_lower_at_x = -y_discharge_top
                     
                if current_y >= y_contour_lower_at_x:
                    is_fluid = True

            else: # Upper half (current_y >= 0)
                # Find the y-value of the upper contour at current_x
                y_contour_upper_at_x = 0.0

                if current_x <= x_chamber_end: # Melting Chamber
                    y_contour_upper_at_x = y_chamber_top
                elif current_x <= x_converging_end: # Converging Section
                    # Line between (x_chamber_end, y_chamber_top) and (x_converging_end, y_converging_end_top)
                    m = (y_converging_end_top - y_chamber_top) / (x_converging_end - x_chamber_end)
                    b = y_chamber_top - m * x_chamber_end
                    y_contour_upper_at_x = m * current_x + b
                elif current_x <= x_discharge_end: # Discharge Section
                    y_contour_upper_at_x = y_discharge_top
                else: # After discharge exit, treat as open fluid above current y_discharge_top
                    y_contour_upper_at_x = y_discharge_top

                if current_y <= y_contour_upper_at_x:
                    is_fluid = True
            
            nozzle_mask[j, i] = is_fluid

    return nozzle_mask, x, y, upper_contour_points

# --- Example Usage ---
if __name__ == "__main__":
    # Define grid parameters
    nx = 201 
    ny = 101 
    # dx = 0.005 
    # dy = 0.005 

    # Define 3D printer nozzle parameters (example values in meters)
    nozzle_params_3d_printer = {
        'D_chamber': 0.005,    # Melting chamber diameter
        'L_chamber': 0.008,    # Melting chamber length
        'L_converging': 0.004, # Length of conical section
        'D_discharge': 0.0004, # Nozzle orifice diameter (0.4 mm)
        'L_discharge': 0.0008  # Length of the final discharge section (short)
    }

    nozzle_mask, x_grid, y_grid, upper_contour_points = create_nozzle_mask_3d_printer_simplified(nx, ny, nozzle_params_3d_printer)

    # Visualize the mask
    plt.figure(figsize=(12, 6))
    plt.imshow(nozzle_mask, origin='lower', extent=[x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()], cmap='Greys')
    plt.title("Stair-Stepped 3D Printer Nozzle Geometry (Fluid: White, Wall: Black)")
    plt.xlabel("X-coordinate (m)")
    plt.ylabel("Y-coordinate (m)")
    
    # Plot the ideal (continuous) nozzle lines for comparison
    plt.plot(upper_contour_points[:, 0], upper_contour_points[:, 1], 'r--', label='Ideal Nozzle Contour (Upper)')
    
    # Generate and plot the lower contour
    lower_contour_points = np.array([ [p[0], -p[1]] for p in upper_contour_points ])
    plt.plot(lower_contour_points[:, 0], lower_contour_points[:, 1], 'r--', label='Ideal Nozzle Contour (Lower)')
    
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim(x_grid.min(), x_grid.max()) # Ensure full domain is shown
    plt.ylim(y_grid.min(), y_grid.max()) # Ensure full domain is shown
    plt.show()

    print(f"Mask shape: {nozzle_mask.shape}")
    print(f"Fluid cells: {np.sum(nozzle_mask)}")
    print(f"Wall cells: {np.sum(~nozzle_mask)}")