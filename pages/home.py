import numpy as np
from dash import html, dcc, Input, Output, callback
import dash_vtk
import plotly.graph_objects as go
import dash_bootstrap_components as dbc

from superquadrics.superquadrics import SuperQuadrics
from utils.helper_functions import convert_to_polydata, create_point_cloud, create_superquadric
from superquadrics.representation_2d import create_plot
from utils.losses_calculation import calculate_iou, calculate_chamfer_distance

# Initial mesh state using the SuperQuadrics class
initial_mesh_state = create_superquadric([1.0, 1.0, 1.0], [0.9, 0.9])


layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Superquadric Deformation and Evaluation"), width=12)
    ]),

    dbc.Row([
        dbc.Col([
            html.Label('Scale (a1, a2, a3):'),
            dcc.Input(id='scale-a1', type='number', value=1.0, step=0.1, min=0.1, max=3.0),
            dcc.Input(id='scale-a2', type='number', value=1.0, step=0.1, min=0.1, max=3.0),
            dcc.Input(id='scale-a3', type='number', value=1.0, step=0.1, min=0.1, max=3.0),
            html.Br(),
            html.Label('Exponent-e1:'),
            dcc.Slider(id='exponent-e1', min=0.1, max=2.0, step=0.1, value=0.9, marks={i: str(i) for i in np.arange(0.1, 2.1, 0.5)}),
            html.Br(),
            html.Label('Exponent-e2:'),
            dcc.Slider(id='exponent-e2', min=0.1, max=2.0, step=0.1, value=0.9, marks={i: str(i) for i in np.arange(0.1, 2.1, 0.5)}),
            html.Br(),
            html.Label('Tapering (ty, tz):'),
            dcc.Input(id='taper-ty', type='number', value=0.0, step=0.1, min=-1.0, max=1.0),
            dcc.Input(id='taper-tz', type='number', value=0.0, step=0.1, min=-1.0, max=1.0),
            html.Br(),
            html.Label('Deformation Type:'),
            dcc.Dropdown(
                id='deformation-type',
                options=[
                    {'label': 'Linear Tapering', 'value': 'linear'},
                    {'label': 'Exponential Tapering', 'value': 'exponential'},
                    {'label': 'Twisting', 'value': 'twisting'},
                    {'label': 'Global Bending', 'value': 'global_bending'},
                    {'label': 'Circular Bending', 'value': 'circular_bending'},
                    {'label': 'Parabolic Bending', 'value': 'parabolic_bending'},
                    {'label': 'Gaussian Weighted Bending', 'value': 'gaussian_weighted_bending'},
                    {'label': 'Corner Bending', 'value': 'corner_bending'}
                ],
                value='linear'
            )
        ], xs=12, lg=4)
    ], style={'padding': 20}),

    dbc.Row([
        dbc.Col([
            html.H2("Original 3D Superquadric"),
            dash_vtk.View([
                dash_vtk.GeometryRepresentation([
                    dash_vtk.PolyData(id='vtk-mesh', points=initial_mesh_state[0], polys=initial_mesh_state[1])
                ])
            ], style={"width": "100%", "height": "400px"})
        ], xs=12, lg=6),

        dbc.Col([
            dcc.Graph(id='magnitude-plot')
        ], xs=12, lg=6)
    ]),

    dbc.Row([
        dbc.Col([
            html.H2("Deformed 3D Superquadric"),
            dash_vtk.View([
                dash_vtk.GeometryRepresentation([
                    dash_vtk.PolyData(id='vtk-mesh-deformed')
                ])
            ], style={"width": "100%", "height": "400px"})
        ], xs=12, lg=6),

        dbc.Col([
            dcc.Graph(id='magnitude-plot-deformed')
        ], xs=12, lg=6)
    ])
])

@callback(
    [
        Output('vtk-mesh', 'points'),
        Output('vtk-mesh', 'polys'),
        Output('magnitude-plot', 'figure'),
        Output('vtk-mesh-deformed', 'points'),
        Output('vtk-mesh-deformed', 'polys'),
        Output('magnitude-plot-deformed', 'figure'),
    ],
    [
        Input('scale-a1', 'value'),
        Input('scale-a2', 'value'),
        Input('scale-a3', 'value'),
        Input('exponent-e1', 'value'),
        Input('exponent-e2', 'value'),
        Input('taper-ty', 'value'),
        Input('taper-tz', 'value')
    ]
)
def update_superquadric(a1, a2, a3, e1, e2, ty, tz):
    size = [a1, a2, a3]
    shape = [e1, e2]
    superquadric = SuperQuadrics(size, shape)
    
    # Original superquadric
    points, polys = convert_to_polydata(superquadric)
    
    # Magnitude plot for original superquadric
    magnitudes = np.sqrt(superquadric.x**2 + superquadric.y**2 + superquadric.z**2)
    fig = create_plot(magnitudes, title="Magnitude Plot", withgrid=False, withlabels=True)
    
    # Apply tapering
    if ty != 0.0 or tz != 0.0:
        x_t, y_t, z_t = superquadric.apply_tapering(ty, tz)
    else:
        x_t, y_t, z_t = superquadric.x, superquadric.y, superquadric.z

    # Convert deformed superquadric to polydata
    deformed_points = np.vstack((x_t.flatten(), y_t.flatten(), z_t.flatten())).T
    points_deformed, polys_deformed = convert_to_polydata(SuperQuadrics(size, shape))

    # Magnitude plot for deformed superquadric
    magnitudes_deformed = np.sqrt(x_t**2 + y_t**2 + z_t**2)
    fig_deformed = create_plot(magnitudes_deformed, title="Deformed Magnitude Plot", withgrid=False, withlabels=True)
    
    return (points, polys, fig, deformed_points.flatten().tolist(), polys_deformed, fig_deformed)
