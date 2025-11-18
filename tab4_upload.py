import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
from dash import callback_context
import pandas as pd
import io
import base64
import plotly.express as px
import plotly.io as pio
from dash import dcc, html, Input, Output, State, dash_table
import dash_bootstrap_components as dbc

class FileUploadApp:
    def __init__(self, app):
        self.app = app
        self.app.layout = self.get_layout()
        self.register_callbacks()

    def get_layout(self):
        return html.Div([
            html.H2("\ud83d\udcc2 Upload CSV File", style={"textAlign": "center"}),
            dcc.Upload(
                id="upload-data",
                children=html.Button("\ud83d\udcc4 Upload CSV File", className="btn-upload"),
                multiple=False
            ),
            html.Div(id="upload-status", style={"marginTop": "10px", "color": "green"}),
            html.H3("Uploaded File Preview (First 5 Rows):"),
            dash_table.DataTable(id="uploaded-table", style_table={"overflowX": "auto"}),
            html.Hr(),
            html.H3("\ud83d\udcca Bacterial Colony Counts"),
            html.Label("Select X-axis Column:"),
            dcc.Dropdown(id="x-axis-dropdown", placeholder="Select X-axis column"),
            html.Label("Select Y-axis Column(s):"),
            dcc.Dropdown(id="y-axis-dropdown", placeholder="Select Y-axis column(s)", multi=True),
            html.Label("Select Graph Type:"),
            dcc.Dropdown(
                id="graph-type-dropdown",
                options=[
                    {"label": "Box Plot", "value": "box"},
                    {"label": "Bar Chart", "value": "bar"}
                ],
                value="box",  # Default graph type
                clearable=False
            ),
            html.Label("Select Hover Data:"),
            dcc.Dropdown(id="hover-data-dropdown", placeholder="Select columns to display in hover", multi=True),
            html.Button("Generate Graph", id="generate-graph-button", className="btn-generate"),
            dcc.Graph(id="user-graph")
        ])

    def register_callbacks(self):
        @self.app.callback(
            [Output("upload-status", "children"),
             Output("uploaded-table", "data"),
             Output("uploaded-table", "columns"),
             Output("x-axis-dropdown", "options"),
             Output("y-axis-dropdown", "options"),
             Output("hover-data-dropdown", "options")],
            [Input("upload-data", "contents")],
            [State("upload-data", "filename")]
        )
        def handle_upload(contents, filename):
            if contents is None:
                return "No file uploaded yet.", [], [], [], [], []
            try:
                content_type, content_string = contents.split(",")
                decoded = io.StringIO(io.BytesIO(base64.b64decode(content_string)).read().decode("utf-8"))
                df = pd.read_csv(decoded, sep=";", na_values=[""], index_col=False)
                options = [{"label": col, "value": col} for col in df.columns]
                return (
                    f"\u2705 {filename} uploaded successfully!",
                    df.head(5).to_dict("records"),
                    [{"name": col, "id": col} for col in df.columns],
                    options,
                    options,
                    options
                )
            except Exception as e:
                return f"\u274c Error processing file: {str(e)}", [], [], [], [], []

        @self.app.callback(
            Output("user-graph", "figure"),
            [Input("generate-graph-button", "n_clicks")],
            [State("x-axis-dropdown", "value"),
             State("y-axis-dropdown", "value"),
             State("graph-type-dropdown", "value"),
             State("hover-data-dropdown", "value"),
             State("upload-data", "contents")]
        )
        def generate_graph(n_clicks, x_axis, y_axis, graph_type, hover_columns, contents):
            if n_clicks is None or contents is None or x_axis is None or y_axis is None:
                return dash.no_update
            try:
                content_type, content_string = contents.split(",")
                decoded = io.StringIO(io.BytesIO(base64.b64decode(content_string)).read().decode("utf-8"))
                df = pd.read_csv(decoded, sep=";", na_values=[""], index_col=False)

                hover_data = hover_columns if hover_columns else None

                if graph_type == "box":
                    fig = px.box(df, x=x_axis, y=y_axis, points="all", hover_data=hover_data)
                else:  # Bar chart as fallback
                    fig = px.bar(df, x=x_axis, y=y_axis, hover_data=hover_data)

                # Set axis labels dynamically
                fig.update_layout(
                    xaxis_title=x_axis,
                    yaxis_title=", ".join(y_axis) if isinstance(y_axis, list) else y_axis,
                    title=f"{graph_type.capitalize()} of {', '.join(y_axis) if isinstance(y_axis, list) else y_axis} vs {x_axis}"
                )

                return fig
            except Exception as e:
                return dash.no_update


