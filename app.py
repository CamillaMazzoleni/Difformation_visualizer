import numpy as np
from dash import Dash, html, dcc, Input, Output
import dash_vtk
import plotly.graph_objects as go
from sklearn.metrics import pairwise_distances

# Define SuperQuadrics and helper functions
def fexp(x, e):
    return np.sign(x) * np.abs(x) ** e

class SuperQuadrics:
    def __init__(self, size, shape, resolution=64):
        self.a1, self.a2, self.a3 = size
        self.e1, self.e2 = shape
        self.N = resolution
        self.x, self.y, self.z, self.eta, self.omega = self.sample_equal_distance_on_sq()

    def sq_surface(self, eta, omega):
        x = self.a1 * fexp(np.cos(eta), self.e1) * fexp(np.cos(omega), self.e2)
        y = self.a2 * fexp(np.cos(eta), self.e1) * fexp(np.sin(omega), self.e2)
        z = self.a3 * fexp(np.sin(eta), self.e1)
        return x, y, z

    def sample_equal_distance_on_sq(self):
        eta = np.linspace(-np.pi / 2, np.pi / 2, self.N)
        omega = np.linspace(-np.pi, np.pi, self.N)
        eta, omega = np.meshgrid(eta, omega)
        x, y, z = self.sq_surface(eta, omega)
        return x, y, z, eta, omega

    def apply_global_linear_tapering(self, ty, tz):
        Y = (((ty / self.a1) * self.x) + 1) * self.y
        Z = (((tz / self.a1) * self.x) + 1) * self.z
        return self.x, Y, Z

    def apply_tapering(self, ty, tz, method="linear"):
        if method == "linear":
            x, Y, Z = self.apply_global_linear_tapering(ty, tz)
            return x, Y, Z

# Helper functions
def convert_to_polydata(superquadric):
    points = np.vstack((superquadric.x.flatten(), superquadric.y.flatten(), superquadric.z.flatten())).T
    num_points = superquadric.x.shape[0]
    
    polys = []
    for i in range(num_points - 1):
        for j in range(num_points - 1):
            p1 = i * num_points + j
            p2 = p1 + 1
            p3 = p1 + num_points
            p4 = p3 + 1
            polys.extend([4, p1, p2, p4, p3])  # A quad made of 4 points
    
    return points.flatten().tolist(), polys

def create_point_cloud(num_points=1000, shape="sphere"):
    if shape == "sphere":
        phi = np.random.uniform(0, np.pi, num_points)
        theta = np.random.uniform(0, 2 * np.pi, num_points)
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
    return np.vstack((x, y, z)).T

def create_superquadric(size, shape):
    superquadric = SuperQuadrics(size=size, shape=shape)
    return convert_to_polydata(superquadric)

def create_plot(magnitudes, title=None, withgrid=False, withlabels=False):
    fig = go.Figure()
    norm_magnitudes = (magnitudes - magnitudes.min()) / (magnitudes.max() - magnitudes.min())
    
    fig.add_trace(go.Heatmap(
        z=norm_magnitudes,
        colorscale='Viridis',
        zmin=0,
        zmax=1,
        x=np.linspace(-180, 180, norm_magnitudes.shape[1]),
        y=np.linspace(-90, 90, norm_magnitudes.shape[0])
    ))

    if title is not None:
        fig.update_layout(title=title)

    if not withgrid:
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)

    if withlabels:
        fig.update_xaxes(title_text="β (degrees)")
        fig.update_yaxes(title_text="α (degrees)")

    return fig

def calculate_iou(point_cloud, superquadric_points):
    pc_min, pc_max = point_cloud.min(axis=0), point_cloud.max(axis=0)
    sq_min, sq_max = superquadric_points.min(axis=0), superquadric_points.max(axis=0)
    
    inter_min = np.maximum(pc_min, sq_min)
    inter_max = np.minimum(pc_max, sq_max)
    
    if np.any(inter_min >= inter_max):
        return 0.0
    
    inter_volume = np.prod(inter_max - inter_min)
    pc_volume = np.prod(pc_max - pc_min)
    sq_volume = np.prod(sq_max - sq_min)
    
    union_volume = pc_volume + sq_volume - inter_volume
    iou = inter_volume / union_volume
    return iou

def calculate_chamfer_distance(pc1, pc2):
    dists_pc1_to_pc2 = pairwise_distances(pc1, pc2).min(axis=1)
    dists_pc2_to_pc1 = pairwise_distances(pc2, pc1).min(axis=1)
    return np.mean(dists_pc1_to_pc2) + np.mean(dists_pc2_to_pc1)

# Initial mesh state using the SuperQuadrics class
initial_mesh_state = create_superquadric([1.0, 1.0, 1.0], [0.9, 0.9])
initial_point_cloud = create_point_cloud()

# Dash setup
app = Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H1("Superquadric Deformation and Evaluation"),
    html.Div([
        html.Div([
            html.H2("Point Cloud"),
            dcc.Graph(id='point-cloud', figure={
                'data': [
                    go.Scatter3d(
                        x=initial_point_cloud[:, 0], y=initial_point_cloud[:, 1], z=initial_point_cloud[:, 2],
                        mode='markers'
                    )
                ],
                'layout': go.Layout(title='Point Cloud', scene=dict(aspectmode='data'))
            })
        ], style={"display": "inline-block", "width": "30%", "vertical-align": "top"}),

        html.Div([
            html.H2("Fitted Superquadric"),
            dash_vtk.View([
                dash_vtk.GeometryRepresentation([
                    dash_vtk.PolyData(id='vtk-mesh-fitted')
                ])
            ], style={"width": "100%", "height": "400px"})
        ], style={"display": "inline-block", "width": "30%", "vertical-align": "top"}),

        html.Div([
            html.H2("Metrics"),
            html.Div(id='metrics')
        ], style={"display": "inline-block", "width": "30%", "vertical-align": "top",  "padding-left": 20})
    ]),
    html.Div([
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
    ], style={'padding': 20}),
    html.Div([
        html.Div([
            html.H2("Original 3D Superquadric"),
            dash_vtk.View([
                dash_vtk.GeometryRepresentation([
                    dash_vtk.PolyData(id='vtk-mesh', points=initial_mesh_state[0], polys=initial_mesh_state[1])
                ])
            ], style={"width": "100%", "height": "400px"})
        ], style={"display": "inline-block", "width": "48%", "vertical-align": "top"}),

        html.Div([
            dcc.Graph(id='magnitude-plot')
        ], style={"display": "inline-block", "width": "48%", "vertical-align": "top"})
    ]),
    html.Div([
        html.Div([
            html.H2("Deformed 3D Superquadric"),
            dash_vtk.View([
                dash_vtk.GeometryRepresentation([
                    dash_vtk.PolyData(id='vtk-mesh-deformed')
                ])
            ], style={"width": "100%", "height": "400px"})
        ], style={"display": "inline-block", "width": "48%", "vertical-align": "top"}),

        html.Div([
            dcc.Graph(id='magnitude-plot-deformed')
        ], style={"display": "inline-block", "width": "48%", "vertical-align": "top"})
    ])
])

@app.callback(
    Output('point-cloud', 'figure'),
    Output('vtk-mesh-fitted', 'points'),
    Output('vtk-mesh-fitted', 'polys'),
    Output('metrics', 'children'),
    Output('vtk-mesh', 'points'),
    Output('vtk-mesh', 'polys'),
    Output('magnitude-plot', 'figure'),
    Output('vtk-mesh-deformed', 'points'),
    Output('vtk-mesh-deformed', 'polys'),
    Output('magnitude-plot-deformed', 'figure'),
    Input('scale-a1', 'value'),
    Input('scale-a2', 'value'),
    Input('scale-a3', 'value'),
    Input('exponent-e1', 'value'),
    Input('exponent-e2', 'value'),
    Input('taper-ty', 'value'),
    Input('taper-tz', 'value')
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
    
    # Calculate metrics
    iou = calculate_iou(initial_point_cloud, deformed_points)
    chamfer_loss = calculate_chamfer_distance(initial_point_cloud, deformed_points)
    metrics_text = [
        html.P(f"Intersection over Union (IoU): {iou:.4f}"),
        html.P(f"Chamfer Loss: {chamfer_loss:.4f}")
    ]
    
    # Magnitude plot for deformed superquadric
    magnitudes_deformed = np.sqrt(x_t**2 + y_t**2 + z_t**2)
    fig_deformed = create_plot(magnitudes_deformed, title="Deformed Magnitude Plot", withgrid=False, withlabels=True)
    
    # Point cloud figure
    point_cloud_figure = {
        'data': [
            go.Scatter3d(
                x=initial_point_cloud[:, 0], y=initial_point_cloud[:, 1], z=initial_point_cloud[:, 2],
                mode='markers'
            )
        ],
        'layout': go.Layout(title='Point Cloud', scene=dict(aspectmode='data'))
    }
    
    return (point_cloud_figure,
            deformed_points.flatten().tolist(), polys_deformed,
            metrics_text,
            points, polys,
            fig,
            deformed_points.flatten().tolist(), polys_deformed,
            fig_deformed)

if __name__ == "__main__":
    app.run(debug=True)

