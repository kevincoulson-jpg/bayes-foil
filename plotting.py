import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import base64
from io import BytesIO
from gen_airfoil import generate_naca_4digit_airfoil

def load_optimization_results(filename='optimization_results.npz'):
    """Load optimization results from saved file."""
    data = np.load(filename)
    return {
        'init': {
            'X': data['init_X'],
            'Y': data['init_Y']
        },
        'iter': {
            'X': data['iter_X'],
            'Y': data['iter_Y']
        },
        'best_params': data['best_params'],
        'best_value': data['best_value']
    }

def naca_profile_base64(m, p, t):
    xu, yu, xl, yl = generate_naca_4digit_airfoil(m, p, t, num_points=100)
    fig, ax = plt.subplots(figsize=(2, 1))
    ax.plot(xu, yu, 'k-')
    ax.plot(xl, yl, 'k-')
    ax.axis('off')
    ax.set_aspect('equal')
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    return img_base64

def plot_optimization_history(results):
    """Animated scatter plot of Cl/Cd and parameter evolution over optimization iterations, with NACA hover images."""
    clcd = results['iter']['Y'].flatten()
    X = results['iter']['X']
    iterations = np.arange(len(clcd))
    variables = ['m', 'p', 't', 'alpha']
    colors = ['blue', 'green', 'red', 'orange']
    max_iter = iterations[-1]

    # Precompute base64 images for all points
    naca_imgs = [naca_profile_base64(m, p, t) for m, p, t, _ in X]

    # Build HTML hover text for each point
    hover_texts = [
        f"Iteration: {ix}<br>Cl/Cd: {clcd_val:.3f}<br>"
        f"<img src='data:image/png;base64,{img}' width='120' />"
        for ix, clcd_val, img in zip(iterations, clcd, naca_imgs)
    ]

    # --- Frames for Cl/Cd animation ---
    frames = []
    for i in range(1, len(clcd)+1):
        # Top plot: Cl/Cd vs iteration
        clcd_trace = go.Scatter(
            x=iterations[:i],
            y=clcd[:i],
            mode='markers+lines',
            marker=dict(
                color=iterations[:i],
                colorscale='Reds',
                size=10,
                cmin=0,
                cmax=max_iter,
                colorbar=dict(title='Iteration', orientation='v', x=1.02, y=0.8, xanchor='left', len=0.5)
            ),
            line=dict(color='pink'),
            showlegend=False,
            xaxis='x1', yaxis='y1',
            text=hover_texts[:i],
            hoverinfo='text'
        )
        # Bottom plot: parameter evolution
        param_traces = []
        for j, (var, color) in enumerate(zip(variables, colors)):
            param_traces.append(go.Scatter(
                x=iterations[:i],
                y=X[:i, j],
                mode='lines+markers',
                name=var,
                line=dict(color=color),
                marker=dict(color=color),
                showlegend=(i == len(clcd)),  # Only show legend on last frame
                xaxis='x2', yaxis='y2'
            ))
        frames.append(go.Frame(
            data=[clcd_trace] + param_traces,
            name=f"step={i}"
        ))

    # --- Initial traces ---
    clcd_trace = go.Scatter(
        x=iterations[:1],
        y=clcd[:1],
        mode='markers+lines',
        marker=dict(
            color=iterations[:1],
            colorscale='Reds',
            size=10,
            cmin=0,
            cmax=max_iter,
            colorbar=dict(title='Iteration', orientation='v', x=1.02, y=0.8, xanchor='left', len=0.5)
        ),
        line=dict(color='pink'),
        showlegend=False,
        xaxis='x1', yaxis='y1',
        text=hover_texts[:1],
        hoverinfo='text'
    )
    param_traces = []
    for j, (var, color) in enumerate(zip(variables, colors)):
        param_traces.append(go.Scatter(
            x=iterations[:1],
            y=X[:1, j],
            mode='lines+markers',
            name=var,
            line=dict(color=color),
            marker=dict(color=color),
            showlegend=True,
            xaxis='x2', yaxis='y2'
        ))

    # --- Layout with two subplots (vertical) ---
    layout = go.Layout(
        title="Optimization Progress: Cl/Cd and Parameter Evolution",
        height=900,
        width=1000,
        template='plotly_white',
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            y=1.15,
            x=1.05,
            xanchor="right",
            yanchor="top",
            buttons=[
                dict(label="Play",
                     method="animate",
                     args=[None, {"frame": {"duration": 100, "redraw": True}}]),
                dict(label="Pause",
                     method="animate",
                     args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate", "transition": {"duration": 0}}])
            ]
        )],
        sliders=[{
            "steps": [
                {
                    "args": [[f"step={i+1}"], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                    "label": str(i+1),
                    "method": "animate"
                } for i in range(len(clcd))
            ],
            "transition": {"duration": 0},
            "x": 0.1,
            "y": -0.08,
            "currentvalue": {"prefix": "step="},
            "len": 0.8
        }],
        xaxis=dict(domain=[0, 1], anchor='y1', title='Iteration'),
        yaxis=dict(domain=[0.6, 1], anchor='x1', title='Cl/Cd'),
        xaxis2=dict(domain=[0, 1], anchor='y2', title='Iteration'),
        yaxis2=dict(domain=[0, 0.4], anchor='x2', title='Parameter Value'),
        legend=dict(x=0, y=0.38, xanchor='left', yanchor='top')
    )

    fig = go.Figure(
        data=[clcd_trace] + param_traces,
        layout=layout,
        frames=frames
    )

    fig.write_html('optimization_history.html')

def plot_parameter_space(results):
    """Plot 3D parameter space with Cl/Cd as color and colorbar at the bottom."""
    fig = go.Figure(data=[go.Scatter3d(
        x=results['iter']['X'][:, 0],  # m
        y=results['iter']['X'][:, 1],  # p
        z=results['iter']['X'][:, 2],  # t
        mode='markers',
        marker=dict(
            size=8,
            color=results['iter']['Y'].flatten(),
            colorscale='Reds',
            opacity=0.8,
            showscale=True,
            colorbar=dict(title='Cl/Cd', orientation='h', x=0.5, y=-0.15, xanchor='center', len=0.7)
        ),
        hovertemplate="m: %{x:.4f}<br>p: %{y:.4f}<br>t: %{z:.4f}<br>Cl/Cd: %{marker.color:.4f}<extra></extra>"
    )])

    # Add best point
    best_idx = np.argmax(results['iter']['Y'])
    fig.add_trace(go.Scatter3d(
        x=[results['iter']['X'][best_idx, 0]],
        y=[results['iter']['X'][best_idx, 1]],
        z=[results['iter']['X'][best_idx, 2]],
        mode='markers',
        marker=dict(
            size=12,
            color='green',
            symbol='diamond'
        ),
        name='Best Point'
    ))

    # Update layout
    fig.update_layout(
        title='Parameter Space (colored by Cl/Cd)',
        scene=dict(
            xaxis_title='m',
            yaxis_title='p',
            zaxis_title='t'
        ),
        width=1000,
        height=800,
        template='plotly_white',
        legend=dict(x=0, y=1, xanchor='left', yanchor='top')
    )

    # Save as HTML
    fig.write_html('parameter_space.html')

if __name__ == "__main__":
    # Load and plot results
    results = load_optimization_results()
    plot_optimization_history(results)
    plot_parameter_space(results) 