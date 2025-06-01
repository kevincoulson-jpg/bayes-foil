import numpy as np
import plotly.graph_objs as go
from dash import Dash, dcc, html, Output, Input, State
from gen_airfoil import generate_naca_4digit_airfoil

# Load your optimization results

def load_optimization_results(filename='optimization_results.npz'):
    data = np.load(filename)
    return {
        'X': data['iter_X'],
        'Y': data['iter_Y'].flatten()
    }

def naca_profile_fig(m, p, t):
    xu, yu, xl, yl = generate_naca_4digit_airfoil(m, p, t, num_points=100)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xu, y=yu, mode='lines', name='Upper', line=dict(color='black')))
    fig.add_trace(go.Scatter(x=xl, y=yl, mode='lines', name='Lower', line=dict(color='black')))
    fig.update_layout(
        title=f'NACA {int(m*100)}{int(p*10)}{int(t*100):02d}',
        xaxis=dict(title='x', scaleanchor='y', scaleratio=1),
        yaxis=dict(title='y'),
        showlegend=False,
        margin=dict(l=20, r=20, t=40, b=20),
        height=300, width=400
    )
    return fig

results = load_optimization_results()
X = results['X']
Y = results['Y']
iterations = np.arange(len(Y))

# Dash app
app = Dash(__name__)

app.layout = html.Div([
    html.Div([
        dcc.Graph(id='clcd-plot', style={'display': 'inline-block'}),
        dcc.Graph(id='naca-profile', style={'display': 'inline-block'})
    ]),
    html.Div([
        dcc.Slider(
            id='iteration-slider',
            min=0,
            max=len(Y)-1,
            value=0,
            marks={i: str(i) for i in range(0, len(Y), max(1, len(Y)//10))},
            step=1,
            updatemode='drag',
            tooltip={"placement": "bottom", "always_visible": False}
        ),
        html.Button('Play', id='play-button', n_clicks=0, style={'marginLeft': '20px'}),
        html.Button('Pause', id='pause-button', n_clicks=0, style={'marginLeft': '10px'}),
        dcc.Interval(id='interval', interval=200, n_intervals=0, disabled=True),
        dcc.Store(id='is-playing', data=False)
    ], style={'width': '80%', 'margin': 'auto', 'paddingTop': '30px'})
])

@app.callback(
    Output('interval', 'disabled'),
    Output('is-playing', 'data'),
    Input('play-button', 'n_clicks'),
    Input('pause-button', 'n_clicks'),
    State('is-playing', 'data')
)
def play_pause(play_clicks, pause_clicks, is_playing):
    ctx = Dash.callback_context
    if not ctx.triggered:
        return True, False
    button_id = ctx.triggered[0]['prop_id'].split('.')[0]
    if button_id == 'play-button':
        return False, True
    elif button_id == 'pause-button':
        return True, False
    return True, False

@app.callback(
    Output('iteration-slider', 'value'),
    Input('interval', 'n_intervals'),
    State('iteration-slider', 'value'),
    State('is-playing', 'data')
)
def update_slider(n_intervals, slider_value, is_playing):
    if not is_playing:
        return slider_value
    next_value = slider_value + 1
    if next_value >= len(Y):
        return 0  # Loop back to start
    return next_value

@app.callback(
    Output('clcd-plot', 'figure'),
    Output('naca-profile', 'figure'),
    Input('iteration-slider', 'value')
)
def update_figures(iter_idx):
    # Cl/Cd plot with current point highlighted
    highlight_marker = dict(color='lime', size=18, symbol='star')
    base_marker = dict(
        color=iterations,
        colorscale='Reds',
        size=10,
        cmin=0,
        cmax=iterations[-1],
        colorbar=dict(title='Iteration')
    )
    clcd_fig = go.Figure(
        data=[
            go.Scatter(
                x=iterations,
                y=Y,
                mode='markers+lines',
                marker=base_marker,
                hovertemplate="Iteration: %{x}<br>Cl/Cd: %{y:.3f}<extra></extra>",
                showlegend=False
            ),
            go.Scatter(
                x=[iterations[iter_idx]],
                y=[Y[iter_idx]],
                mode='markers',
                marker=highlight_marker,
                showlegend=False,
                hoverinfo='skip'
            )
        ],
        layout=go.Layout(
            title="Cl/Cd vs Iteration (Current highlighted)",
            xaxis=dict(title='Iteration'),
            yaxis=dict(title='Cl/Cd'),
            height=400, width=600
        )
    )
    m, p, t, alpha = X[iter_idx]
    naca_fig = naca_profile_fig(m, p, t)
    naca_fig.update_layout(title=f'NACA {int(m*100)}{int(p*10)}{int(t*100):02d} | alpha={alpha:.2f}')
    return clcd_fig, naca_fig

if __name__ == '__main__':
    app.run(debug=True) 