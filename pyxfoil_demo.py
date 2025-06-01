from xfoil import XFoil
from xfoil.model import Airfoil
import numpy as np
import matplotlib.pyplot as plt

# Define airfoil and analysis parameters
naca_code = '0012'
reynolds = 1e6
alpha_start = -20
alpha_end = 20
alpha_step = 0.5

# Create XFoil instance
xf = XFoil()
xf.Re = reynolds
xf.max_iter = 40

# Generate airfoil coordinates using xfoil's Airfoil class
airfoil = Airfoil.naca(naca_code)

# Sweep angle of attack
alphas = np.arange(alpha_start, alpha_end + alpha_step, alpha_step)
cl = []
cd = []
cm = []
for a in alphas:
    res = xf.a(airfoil, a)
    if res is not None:
        cl.append(res['CL'])
        cd.append(res['CD'])
        cm.append(res['CM'])
    else:
        cl.append(np.nan)
        cd.append(np.nan)
        cm.append(np.nan)

# Plot lift coefficient vs. angle of attack
plt.plot(alphas, cl)
plt.xlabel('Angle of Attack (deg)')
plt.ylabel('Lift Coefficient')
plt.title(f'Lift Coefficient vs. Angle of Attack (NACA {naca_code}, Re={int(reynolds):.0e})')
plt.grid(True, linestyle='--', alpha=0.6)
plt.show() 