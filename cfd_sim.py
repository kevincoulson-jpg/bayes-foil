import numpy as np
import matplotlib.pyplot as plt
from nozzle_geom import create_nozzle_mask_3d_printer_simplified

def navier_stokes_2d_nozzle_improved(params, nx, ny, dt, nu, rho, nit, max_iterations=10000, tolerance=1e-5,
                                    P_inlet=80.0, P_outlet_gauge=0.0):
    """
    Solves the 2D incompressible Navier-Stokes equations for flow through a 3D printer nozzle.
    Uses a finite difference projection method (similar to CFDPython Step 12).
    Improved boundary condition handling for pressure and velocity.
    Returns:
        tuple: (u, v, p, x_grid, y_grid, nozzle_mask)
    """
    print("Setting up nozzle geometry and grid...")
    nozzle_mask, x_grid, y_grid, upper_contour_points = create_nozzle_mask_3d_printer_simplified(nx, ny, params)
    print(f"Grid shape: {x_grid.shape[0]} x {y_grid.shape[0]}")
    print(f"Fluid cells: {np.sum(nozzle_mask)} | Wall cells: {np.sum(~nozzle_mask)}")

    dx = x_grid[1] - x_grid[0]
    dy = y_grid[1] - y_grid[0]

    u = np.zeros((ny, nx))
    v = np.zeros((ny, nx))
    p = np.zeros((ny, nx))
    b = np.zeros((ny, nx))

    x_chamber_start = 0.0
    x_discharge_end = params['L_chamber'] + params['L_converging'] + params['L_discharge']
    inlet_x_idx = np.argmin(np.abs(x_grid - x_chamber_start))
    outlet_x_idx = np.argmin(np.abs(x_grid - x_discharge_end))

    y_chamber_top = params['D_chamber'] / 2.0
    y_discharge_top = params['D_discharge'] / 2.0
    fluid_inlet_y_indices = np.where(np.abs(y_grid) <= y_chamber_top)[0]
    fluid_outlet_y_indices = np.where(np.abs(y_grid) <= y_discharge_top)[0]

    p[:, :] = P_outlet_gauge
    p[fluid_inlet_y_indices, inlet_x_idx] = P_inlet
    u[:, :] = 0.0
    v[:, :] = 0.0
    u[~nozzle_mask] = 0
    v[~nozzle_mask] = 0

    print("Starting main time-stepping loop...")
    for n in range(max_iterations):
        un = u.copy()
        vn = v.copy()

        b[1:-1, 1:-1] = (rho * (1 / dt * ((u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx) +
                                          (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)) -
                                (u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx) * (u[1:-1, 2:] - u[1:-1, 0:-2]) / (2 * dx) -
                                2 * ((u[2:, 1:-1] - u[0:-2, 1:-1]) / (2 * dy) *
                                     (v[1:-1, 2:] - v[1:-1, 0:-2]) / (2 * dx)) -
                                (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy) * (v[2:, 1:-1] - v[0:-2, 1:-1]) / (2 * dy)))
        b[~nozzle_mask] = 0

        pn = p.copy()
        for _ in range(nit):
            p_iter = pn.copy()
            p[1:-1, 1:-1] = (((p_iter[1:-1, 2:] + p_iter[1:-1, 0:-2]) * dy**2 +
                              (p_iter[2:, 1:-1] + p_iter[0:-2, 1:-1]) * dx**2) /
                             (2 * (dx**2 + dy**2)) -
                             dx**2 * dy**2 / (2 * (dx**2 + dy**2)) * b[1:-1, 1:-1])
            p[fluid_inlet_y_indices, inlet_x_idx] = P_inlet
            p[fluid_outlet_y_indices, outlet_x_idx] = P_outlet_gauge
            p[~nozzle_mask] = P_outlet_gauge
            p[0, :] = P_outlet_gauge
            p[-1, :] = P_outlet_gauge

        u[1:-1, 1:-1] = (un[1:-1, 1:-1] - dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, 0:-2]) +
                         nu * dt * ((u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, 0:-2]) / dx**2 +
                                    (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[0:-2, 1:-1]) / dy**2))
        v[1:-1, 1:-1] = (vn[1:-1, 1:-1] - dt / (2 * rho * dy) * (p[2:, 1:-1] - p[0:-2, 1:-1]) +
                         nu * dt * ((v[1:-1, 2:] - 2 * v[1:-1, 1:-1] + v[1:-1, 0:-2]) / dx**2 +
                                    (v[2:, 1:-1] - 2 * v[1:-1, 1:-1] + v[0:-2, 1:-1]) / dy**2))

        u[~nozzle_mask] = 0
        v[~nozzle_mask] = 0
        v[fluid_inlet_y_indices, inlet_x_idx] = 0
        u[fluid_outlet_y_indices, outlet_x_idx] = u[fluid_outlet_y_indices, outlet_x_idx - 1]
        v[fluid_outlet_y_indices, outlet_x_idx] = v[fluid_outlet_y_indices, outlet_x_idx - 1]
        mid_y_idx = ny // 2
        v[mid_y_idx, :] = 0

        u[~nozzle_mask] = 0
        v[~nozzle_mask] = 0

        if n % 500 == 0:
            diff_u = np.sqrt(np.sum((u - un)**2) / (np.sum(un**2) + 1e-10))
            diff_v = np.sqrt(np.sum((v - vn)**2) / (np.sum(vn**2) + 1e-10))
            max_diff = max(diff_u, diff_v)
            print(f"Iteration {n}: Max velocity diff = {max_diff:.2e}")
            if max_diff < tolerance:
                print(f"Converged after {n} iterations.")
                break
        if n == max_iterations - 1:
            print(f"Reached max iterations ({max_iterations}) without converging.")
    print("Simulation complete.")
    return u, v, p, x_grid, y_grid, nozzle_mask

if __name__ == '__main__':
    nx = 201
    ny = 101
    nozzle_params = {
        'D_chamber': 0.005,
        'L_chamber': 0.008,
        'L_converging': 0.004,
        'D_discharge': 0.0004,
        'L_discharge': 0.0008
    }
    x_chamber_start = 0.0
    x_discharge_end = nozzle_params['L_chamber'] + nozzle_params['L_converging'] + nozzle_params['L_discharge']
    x_max_domain = x_discharge_end + 0.005
    y_max_nozzle = nozzle_params['D_chamber'] / 2.0
    y_min_domain = -y_max_nozzle - 0.003
    y_max_domain = y_max_nozzle + 0.003
    dx = (x_max_domain - x_chamber_start) / (nx - 1)
    dy = (y_max_domain - y_min_domain) / (ny - 1)
    nu = 1e-3
    rho = 1000
    dt = 5e-7
    nit = 100
    max_time_steps = 50000
    convergence_tolerance = 1e-6
    P_inlet_val = 80.0
    P_outlet_gauge_val = 0.0
    print("Starting Navier-Stokes simulation with improved BCs...")
    u, v, p, x_grid, y_grid, nozzle_mask = navier_stokes_2d_nozzle_improved(
        nozzle_params, nx, ny, dt, nu, rho, nit, max_time_steps, convergence_tolerance,
        P_inlet=P_inlet_val, P_outlet_gauge=P_outlet_gauge_val
    )
    print("Simulation finished.")
    _mask, _x, _y, ideal_contour_pts = create_nozzle_mask_3d_printer_simplified(nx, ny, nozzle_params)
    lower_contour_pts = np.array([[pt[0], -pt[1]] for pt in ideal_contour_pts])
    plt.figure(figsize=(10, 5))
    plt.imshow(p, origin='lower', extent=[x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()], cmap='jet', alpha=0.9)
    plt.colorbar(label='Pressure (Pa)')
    plt.title("Pressure Field")
    plt.xlabel("X-coordinate (m)")
    plt.ylabel("Y-coordinate (m)")
    plt.plot(ideal_contour_pts[:, 0], ideal_contour_pts[:, 1], 'k--', linewidth=1.5)
    plt.plot(lower_contour_pts[:, 0], lower_contour_pts[:, 1], 'k--', linewidth=1.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()
    vel_mag = np.sqrt(u**2 + v**2)
    plt.figure(figsize=(10, 5))
    plt.imshow(vel_mag, origin='lower', extent=[x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()], cmap='viridis', alpha=0.9)
    plt.colorbar(label='Velocity Magnitude (m/s)')
    plt.title("Velocity Magnitude Field")
    plt.xlabel("X-coordinate (m)")
    plt.ylabel("Y-coordinate (m)")
    plt.plot(ideal_contour_pts[:, 0], ideal_contour_pts[:, 1], 'k--', linewidth=1.5)
    plt.plot(lower_contour_pts[:, 0], lower_contour_pts[:, 1], 'k--', linewidth=1.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()
    plt.figure(figsize=(10, 5))
    plt.imshow(vel_mag, origin='lower', extent=[x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()], cmap='viridis', alpha=0.5)
    plt.colorbar(label='Velocity Magnitude (m/s)')
    plt.title("Velocity Vectors")
    plt.xlabel("X-coordinate (m)")
    plt.ylabel("Y-coordinate (m)")
    skip = 4
    Xg, Yg = np.meshgrid(x_grid, y_grid)
    X_plot = Xg[nozzle_mask][::skip]
    Y_plot = Yg[nozzle_mask][::skip]
    U_plot = u[nozzle_mask][::skip]
    V_plot = v[nozzle_mask][::skip]
    plt.quiver(X_plot, Y_plot, U_plot, V_plot, color='white', scale=1.0)
    plt.plot(ideal_contour_pts[:, 0], ideal_contour_pts[:, 1], 'k--', linewidth=1.5)
    plt.plot(lower_contour_pts[:, 0], lower_contour_pts[:, 1], 'k--', linewidth=1.5)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.show()
    if np.sum(nozzle_mask[:, outlet_x_idx]) > 0:
        u_outlet_fluid = u[nozzle_mask[:, outlet_x_idx], outlet_x_idx]
        avg_u_outlet = np.mean(u_outlet_fluid)
        print(f"\nAverage X-velocity at outlet: {avg_u_outlet:.4f} m/s")
    else:
        print("\nNo fluid cells found at the outlet boundary.")