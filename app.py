import numpy as np

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

def create_superquadric(size, shape):
    superquadric = SuperQuadrics(size=size, shape=shape)
    return convert_to_polydata(superquadric)

from dash import Dash, html, dcc, Input, Output
import dash_vtk
import numpy as np
import plotly.graph_objects as go

# Initial mesh state using the SuperQuadrics class
initial_mesh_state = create_superquadric([1.0, 1.0, 1.0], [0.9, 0.9])

# Function to create plot using Plotly
def create_plot(magnitudes, title=None, withgrid=False, withlabels=False):
    fig = go.Figure()

    # Normalize magnitudes to the range [0, 1]
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

# Dash setup
app = Dash(__name__)
server = app.server

app.layout = html.Div([
    html.H1("Superquadric Deformation and Magnitude Plot"),
    html.Div([
        html.Div([
            html.H2("3D Superquadric"),
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
    ], style={'padding': 20})
])

@app.callback(
    Output('vtk-mesh', 'points'),
    Output('vtk-mesh', 'polys'),
    Output('magnitude-plot', 'figure'),
    Input('scale-a1', 'value'),
    Input('scale-a2', 'value'),
    Input('scale-a3', 'value'),
    Input('exponent-e1', 'value'),
    Input('exponent-e2', 'value')
)
def update_superquadric(a1, a2, a3, e1, e2):
    size = [a1, a2, a3]
    shape = [e1, e2]
    superquadric = SuperQuadrics(size, shape)   

    points, polys = create_superquadric(size, shape)
    
    # Create a sample magnitude array for demonstration purposes
    magnitudes = np.sqrt(superquadric.x**2 + superquadric.y**2 + superquadric.z**2)
    fig = create_plot(magnitudes, title="Magnitude Plot", withgrid=False, withlabels=True)
    
    return points, polys, fig

if __name__ == "__main__":
    app.run(debug=True)
