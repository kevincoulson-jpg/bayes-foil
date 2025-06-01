import numpy as np
from flask import Flask, jsonify
from flask_cors import CORS
import json

app = Flask(__name__)
CORS(app)

def generate_naca_points(m, p, t, num_points=100):
    """Generate points for a NACA 4-digit airfoil."""
    x = np.linspace(0, 1, num_points)
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
    
    if m == 0:
        yc = np.zeros_like(x)
        dyc_dx = np.zeros_like(x)
    else:
        yc = np.where(x < p,
                     m * (2 * p * x - x**2) / p**2,
                     m * (1 - 2 * p + 2 * p * x - x**2) / (1 - p)**2)
        dyc_dx = np.where(x < p,
                         2 * m * (p - x) / p**2,
                         2 * m * (p - x) / (1 - p)**2)
    
    theta = np.arctan(dyc_dx)
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)
    
    # Combine upper and lower surfaces
    points = np.vstack((np.column_stack((xu, yu)), np.column_stack((xl[::-1], yl[::-1]))))
    return points.tolist()

def load_optimization_history():
    """Load optimization history from the saved results file."""
    try:
        data = np.load('optimization_results.npz')
        history = []
        
        # Get iteration data
        X = data['iter_X']  # Contains [m, p, t, alpha] for each iteration
        Y = data['iter_Y'].flatten()  # Cl/Cd values
        
        for i in range(len(X)):
            m, p, t, _ = X[i]  # Unpack parameters, ignore alpha
            points = generate_naca_points(m, p, t)
            history.append({
                'iteration': i,
                'parameters': {
                    'm': float(m),
                    'p': float(p),
                    't': float(t)
                },
                'points': points,
                'objective_value': float(Y[i])  # Cl/Cd value
            })
        
        return history
    except Exception as e:
        print(f"Error loading optimization results: {e}")
        return []

@app.route('/api/optimization-history')
def get_optimization_history():
    history = load_optimization_history()
    return jsonify(history)

if __name__ == '__main__':
    app.run(debug=True, port=5000) 