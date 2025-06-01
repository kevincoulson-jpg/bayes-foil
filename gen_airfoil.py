import numpy as np

def generate_naca_4digit_airfoil(m, p, t, num_points=100, chord=1.0):
    """
    Generates the (x, y) coordinates for a NACA 4-digit airfoil.

    Parameters:
    m (float): Maximum camber as a fraction of the chord (e.g., 0.02 for 2%).
               Corresponds to the 1st digit M in NACA MPXX.
    p (float): Position of maximum camber as a fraction of the chord (e.g., 0.4 for 40%).
               Corresponds to the 2nd digit P in NACA MPXX.
    t (float): Maximum thickness as a fraction of the chord (e.g., 0.12 for 12%).
               Corresponds to the 3rd & 4th digits XX in NACA MPXX.
    num_points (int): Number of points along the chord for both upper and lower surfaces.
                      Points will be denser towards the leading and trailing edges.
    chord (float): The chord length of the airfoil (default is 1.0).

    Returns:
    tuple: (xu, yu, xl, yl)
           xu, yu (np.array): x and y coordinates of the upper surface.
           xl, yl (np.array): x and y coordinates of the lower surface.
    """

    # Generate x-coordinates (cosine spacing for better distribution)
    # This places more points near leading and trailing edges
    x = np.cos(np.linspace(0, np.pi, num_points)) * -0.5 + 0.5
    x = x * chord # Scale by chord length

    # Initialize arrays for y_c, dy_c_dx, and y_t
    yc = np.zeros_like(x)
    dyc_dx = np.zeros_like(x)
    yt = np.zeros_like(x)

    # --- Calculate Mean Camber Line (yc) and its derivative (dyc_dx) ---
    if m == 0 or p == 0:
        yc[:] = 0.0
        dyc_dx[:] = 0.0
    else:
        for i, xi in enumerate(x):
            if 0 <= xi <= p * chord:
                yc[i] = (m / p**2) * (2 * p * xi - xi**2)
                dyc_dx[i] = (2 * m / p**2) * (p - xi)
            elif p * chord < xi <= chord:
                yc[i] = (m / (1 - p)**2) * ((1 - 2 * p) * chord + 2 * p * xi - xi**2)
                dyc_dx[i] = (2 * m / (1 - p)**2) * (p - xi) # Still (p - x) even for second part

    # --- Calculate Thickness Distribution (yt) ---
    # Rescale x to be between 0 and 1 for the thickness formula
    x_norm = x / chord

    yt = 5 * t * (0.2969 * np.sqrt(x_norm) - 0.1260 * x_norm - 0.3516 * x_norm**2 + \
                  0.2843 * x_norm**3 - 0.1015 * x_norm**4)

    # --- Calculate Upper and Lower Surface Coordinates ---
    theta = np.arctan(dyc_dx)

    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)

    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)

    return xu, yu, xl, yl

# --- Example Usage and Visualization ---
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    # Define NACA parameters for a NACA 2412 airfoil (as percentages, convert to fractions)
    naca_m = 2 / 100.0   # Max camber 2%
    naca_p = 4 / 10.0    # Max camber at 40% chord
    naca_xx = 12 / 100.0 # Max thickness 12%

    # Generate airfoil coordinates
    xu, yu, xl, yl = generate_naca_4digit_airfoil(naca_m, naca_p, naca_xx, num_points=150)

    # Plotting the airfoil
    plt.figure(figsize=(10, 3))
    plt.plot(xu, yu, label='Upper Surface')
    plt.plot(xl, yl, label='Lower Surface')
    plt.plot(xu, yu, 'b.', markersize=2) # Plot points
    plt.plot(xl, yl, 'r.', markersize=2) # Plot points
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f'NACA {int(naca_m*100)}{int(naca_p*10)}{int(naca_xx*100)} Airfoil')
    plt.xlabel('X/C')
    plt.ylabel('Y/C')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()

    # Test with a symmetric airfoil (NACA 0012)
    naca_m_sym = 0 / 100.0
    naca_p_sym = 0 / 10.0
    naca_xx_sym = 12 / 100.0
    xu_sym, yu_sym, xl_sym, yl_sym = generate_naca_4digit_airfoil(naca_m_sym, naca_p_sym, naca_xx_sym, num_points=150)

    plt.figure(figsize=(10, 3))
    plt.plot(xu_sym, yu_sym, label='Upper Surface')
    plt.plot(xl_sym, yl_sym, label='Lower Surface')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.title(f'NACA {int(naca_m_sym*100)}{int(naca_p_sym*10)}{int(naca_xx_sym*100)} Airfoil (Symmetric)')
    plt.xlabel('X/C')
    plt.ylabel('Y/C')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.show()