import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
import pandas as pd

class FDMMeltingModel:
    """
    Implementation of the Osswald et al. thin layer melting model for FDM nozzles.
    Based on "Fused filament fabrication melting model" (Additive Manufacturing, 2018)
    """
    
    def __init__(self, material_props, nozzle_geometry):
        """
        Initialize the melting model with material properties and nozzle geometry.
        
        Parameters:
        material_props: dict with material properties
        nozzle_geometry: dict with nozzle dimensions
        """
        self.mat = material_props
        self.geo = nozzle_geometry
        
        # Convert angle to radians
        self.alpha_rad = np.radians(self.geo['alpha_deg'])
        
    def power_law_viscosity(self, shear_rate, temperature, params):
        """
        Calculate viscosity using power-law model with temperature dependence.
        η = m₀ * exp(-a*(T_ref - T)) * γ̇^(n-1)
        """
        T_ref = params.get('T_ref', 250 + 273.15)  # Reference temperature in K
        exp_term = np.exp(-params['a'] * (T_ref - temperature))
        return params['m0'] * exp_term * (shear_rate ** (params['n'] - 1))
    
    def calculate_average_shear_rate_film(self, U_sz, delta):
        """
        Calculate average shear rate in the melt film (Eq. 24 in paper).
        """
        R0_star = self.geo['R0'] / np.cos(self.alpha_rad)
        Ri_star = self.geo['Ri'] / np.cos(self.alpha_rad)
        
        numerator = (R0_star**2 - Ri_star**2) * (R0_star + 2*Ri_star)
        denominator = (R0_star - Ri_star)
        
        shear_rate = (U_sz * np.cos(self.alpha_rad) / delta) * (numerator / denominator)
        return shear_rate
    
    def calculate_average_shear_rate_capillary(self, pressure_i):
        """
        Calculate average shear rate in the capillary (Eq. 27 in paper).
        """
        return (pressure_i / (3 * self.mat['mu_c'])) * (self.geo['Ri'] / self.geo['Lc'])
    
    def calculate_melt_film_thickness(self, U_sz, T_h, T_0=298.15):
        """
        Calculate melt film thickness using Stefan condition (Eq. 31).
        δ = k_m(T_h - T_m) / [ρ_s * U_sz * (λ + C_s(T_m - T_0))]
        """
        T_m = self.mat['T_m']
        numerator = self.mat['k_m'] * (T_h - T_m)
        denominator = self.mat['rho_s'] * U_sz * (self.mat['lambda'] + 
                                                 self.mat['C_s'] * (T_m - T_0))
        return numerator / denominator
    
    def calculate_pressure_capillary(self, U_sz):
        """
        Calculate pressure at capillary entrance using Hagen-Poiseuille (Eq. 21).
        """
        Q = np.pi * self.geo['R0']**2 * U_sz
        return (8 * self.mat['mu_c'] * Q * self.geo['Lc']) / (np.pi * self.geo['Ri']**4)
    
    def force_balance_equation(self, U_sz, F_z, T_h, T_0=298.15):
        """
        Main force balance equation (Eq. 29) rearranged to solve for U_sz.
        Returns the residual for root finding.
        """
        # Calculate melt film thickness
        delta = self.calculate_melt_film_thickness(U_sz, T_h, T_0)
        
        # Calculate average shear rates
        gamma_dot_film = self.calculate_average_shear_rate_film(U_sz, delta)
        
        # Calculate viscosities
        mu_f = self.power_law_viscosity(gamma_dot_film, T_h, self.mat['power_law'])
        
        # Geometric terms
        R0, Ri = self.geo['R0'], self.geo['Ri']
        Lc = self.geo['Lc']
        alpha = self.alpha_rad
        
        # Density ratio
        rho_hat = self.mat['rho_s'] / self.mat['rho_m']
        
        # First term (melt film contribution)
        geo_factor = (Ri/R0)**2 + 3/4 + np.log(R0/Ri) - 1/4 * (Ri/R0)**4
        term1 = (6 * np.pi * mu_f * rho_hat * U_sz * R0**3) / (np.cos(alpha) * delta**3) * geo_factor
        
        # Second term (capillary contribution)  
        term2 = (8 * np.pi * self.mat['mu_c'] * rho_hat * U_sz * Lc * R0**2) / Ri**4
        
        # Total predicted force
        F_predicted = term1 + term2
        
        return F_predicted - F_z
    
    def solve_for_velocity(self, F_z, T_h, T_0=298.15, initial_guess=1e-3):
        """
        Solve for filament velocity given applied force and temperatures.
        """
        try:
            U_sz = fsolve(lambda U: self.force_balance_equation(U[0], F_z, T_h, T_0), 
                         [initial_guess])[0]
            if U_sz > 0:
                return U_sz
            else:
                return None
        except:
            return None
    
    def run_simulation(self, forces, temperatures, T_0=298.15):
        """
        Run simulation for multiple force and temperature combinations.
        """
        results = []
        
        for T_h in temperatures:
            for F_z in forces:
                U_sz = self.solve_for_velocity(F_z, T_h, T_0)
                if U_sz is not None:
                    delta = self.calculate_melt_film_thickness(U_sz, T_h, T_0)
                    results.append({
                        'Force_N': F_z,
                        'Temperature_C': T_h - 273.15,
                        'Velocity_mm_s': U_sz * 1000,  # Convert to mm/s
                        'Film_thickness_um': delta * 1e6  # Convert to micrometers
                    })
        
        return pd.DataFrame(results)

def create_abs_material():
    """Create ABS material properties from Table 1 in the paper."""
    return {
        'rho_s': 1060,      # kg/m³
        'rho_m': 945,       # kg/m³
        'C_s': 1470,        # J/kg/K
        'k_m': 0.33,        # W/m/K
        'lambda': 0,        # J/kg (no heat of fusion for ABS)
        'T_m': 373.15,      # Assumed melting temp in K
        'mu_c': 1000,       # Approximate capillary viscosity (Pa·s)
        'power_law': {
            'm0': 6434,     # Pa·s^n
            'n': 0.378,     # Power law index
            'a': 0.0185,    # Temperature dependence (1/K)
            'T_ref': 523.15 # Reference temperature (K)
        }
    }

def create_nozzle_geometry():
    """Create typical FDM nozzle geometry."""
    return {
        'R0': 1.42e-3,      # Filament radius (m) - 2.84mm diameter
        'Ri': 0.5e-3,       # Capillary radius (m) - 1mm diameter
        'Lc': 5e-3,         # Capillary length (m)
        'alpha_deg': 30     # Nozzle tip angle (degrees)
    }

def plot_results(df):
    """Create plots comparing simulation results with experimental data."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # Experimental data from Table 2
    exp_data_240 = {
        'forces': [21.6, 36.3, 40.4, 64.7, 98.1],
        'velocities': [0.705, 0.833, 0.872, 1.008, 1.180],
        'std_dev': [0.018, 0.020, 0.016, 0.022, 0.011]
    }
    
    exp_data_250 = {
        'forces': [21.6, 36.3, 40.4, 64.7, 98.1],
        'velocities': [0.805, 1.000, 1.030, 1.223, 1.403],
        'std_dev': [0.018, 0.029, 0.027, 0.064, 0.084]
    }
    
    # Plot 1: Velocity vs Force comparison
    sim_240 = df[df['Temperature_C'] == 240]
    sim_250 = df[df['Temperature_C'] == 250]
    
    ax1.errorbar(exp_data_240['forces'], exp_data_240['velocities'], 
                yerr=exp_data_240['std_dev'], fmt='ro', label='Exp 240°C', capsize=5)
    ax1.errorbar(exp_data_250['forces'], exp_data_250['velocities'], 
                yerr=exp_data_250['std_dev'], fmt='bo', label='Exp 250°C', capsize=5)
    
    if not sim_240.empty:
        ax1.plot(sim_240['Force_N'], sim_240['Velocity_mm_s'], 'r-', 
                label='Sim 240°C', linewidth=2)
    if not sim_250.empty:
        ax1.plot(sim_250['Force_N'], sim_250['Velocity_mm_s'], 'b-', 
                label='Sim 250°C', linewidth=2)
    
    ax1.set_xlabel('Applied Force (N)')
    ax1.set_ylabel('Filament Velocity (mm/s)')
    ax1.set_title('Velocity vs Force Comparison')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Melt film thickness vs Force
    for temp in df['Temperature_C'].unique():
        temp_data = df[df['Temperature_C'] == temp]
        ax2.plot(temp_data['Force_N'], temp_data['Film_thickness_um'], 
                'o-', label=f'{temp}°C', linewidth=2)
    
    ax2.set_xlabel('Applied Force (N)')
    ax2.set_ylabel('Melt Film Thickness (μm)')
    ax2.set_title('Melt Film Thickness vs Force')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Effect of temperature
    forces_subset = [25, 50, 75, 100]
    for force in forces_subset:
        force_data = df[df['Force_N'].isin([force-2.5, force+2.5])]  # Approximate matching
        if not force_data.empty:
            force_data = force_data.groupby('Temperature_C').mean().reset_index()
            ax3.plot(force_data['Temperature_C'], force_data['Velocity_mm_s'], 
                    'o-', label=f'{force}N', linewidth=2)
    
    ax3.set_xlabel('Temperature (°C)')
    ax3.set_ylabel('Filament Velocity (mm/s)')
    ax3.set_title('Effect of Temperature on Velocity')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Velocity vs Film thickness relationship
    ax4.scatter(df['Film_thickness_um'], df['Velocity_mm_s'], 
               c=df['Force_N'], cmap='viridis', alpha=0.7)
    colorbar = plt.colorbar(ax4.collections[0], ax=ax4)
    colorbar.set_label('Applied Force (N)')
    ax4.set_xlabel('Melt Film Thickness (μm)')
    ax4.set_ylabel('Filament Velocity (mm/s)')
    ax4.set_title('Velocity vs Film Thickness')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def main():
    """Main simulation function."""
    print("FDM Thin Layer Melting Model Simulation")
    print("Based on Osswald et al. (2018)")
    print("-" * 50)
    
    # Create material and geometry
    abs_material = create_abs_material()
    nozzle_geo = create_nozzle_geometry()
    
    # Initialize model
    model = FDMMeltingModel(abs_material, nozzle_geo)
    
    # Define simulation parameters
    forces = np.array([21.6, 36.3, 40.4, 64.7, 98.1])  # From experimental data
    temperatures = np.array([240, 250]) + 273.15  # Convert to Kelvin
    
    # Run simulation
    print("Running simulation...")
    results = model.run_simulation(forces, temperatures)
    
    # Display results
    print("\nSimulation Results:")
    print(results.round(3))
    
    # Calculate some statistics
    print(f"\nMelt film thickness range: {results['Film_thickness_um'].min():.1f} - {results['Film_thickness_um'].max():.1f} μm")
    print(f"Velocity range: {results['Velocity_mm_s'].min():.3f} - {results['Velocity_mm_s'].max():.3f} mm/s")
    
    # Create plots
    plot_results(results)
    
    return model, results

if __name__ == "__main__":
    model, results = main()