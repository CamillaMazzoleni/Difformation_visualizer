from dash import Dash, dcc, html, Input, Output
import dash_bootstrap_components as dbc
from pages import home, page1, page2

app = Dash(__name__,
           title="Simple Multi-page Dash App",
           external_stylesheets=[dbc.themes.BOOTSTRAP])

server = app.server

index_layout = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

app.validation_layout = html.Div([
    index_layout,
    home.layout,
    page1.layout,
    page2.layout
])

app.layout = index_layout

@app.callback(
    Output('page-content', 'children'),
    Input('url', 'pathname')
)
def display_page(pathname):
    if pathname == '/' or pathname == '/home':
        return home.layout
    elif pathname == '/page1':
        return page1.layout
    elif pathname == '/page2':
        return page2.layout
    else:
        return "404 Page Not Found"

if __name__ == '__main__':
    app.run_server(debug=True)

