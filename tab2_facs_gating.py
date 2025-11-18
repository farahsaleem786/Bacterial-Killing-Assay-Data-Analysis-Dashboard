import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import dash
from dash import dcc, html, dash_table
import plotly.graph_objects as go
from dash.dependencies import Input, Output

class FACSAnalysis:
    def __init__(self, xml_file):
        self.xml_file = xml_file

        # Define namespaces
        self.namespaces = {
            "gating": "http://www.isac-net.org/std/Gating-ML/v2.0/gating",
            "datatypes": "http://www.isac-net.org/std/Gating-ML/v2.0/datatypes",
        }

        # Extract available specimens
        self.all_specimens = self.extract_all_specimens()

    def extract_all_specimens(self):
        """Extract all specimen names from the XML file."""
        tree = ET.parse(self.xml_file)
        root = tree.getroot()
        return [sample.get("name") for sample in root.findall('.//SampleNode')]

    def extract_specimen_data(self, specimen_name):
        """Extract population data for a given specimen."""
        tree = ET.parse(self.xml_file)
        root = tree.getroot()

        for sample in root.findall('.//SampleNode'):
            if sample.get("name") == specimen_name:
                populations = []
                for pop in sample.findall('.//Population'):
                    pop_data = {
                        "name": pop.get("name"),
                        "count": pop.get("count"),
                        "gating": [],
                        "statistics": []
                    }

                    # Extract gating information
                    for gate in pop.findall('.//Gate', self.namespaces):
                        rect_gate = gate.find('gating:RectangleGate', self.namespaces)
                        if rect_gate is not None:
                            dimensions = rect_gate.findall('gating:dimension', self.namespaces)

                            if len(dimensions) == 2:
                                x_dim = dimensions[0]
                                y_dim = dimensions[1]

                                x_min = float(x_dim.get('{http://www.isac-net.org/std/Gating-ML/v2.0/gating}min', 0))
                                x_max = float(x_dim.get('{http://www.isac-net.org/std/Gating-ML/v2.0/gating}max', 100))
                                x_name = x_dim.find('datatypes:fcs-dimension', self.namespaces).get('{http://www.isac-net.org/std/Gating-ML/v2.0/datatypes}name', "X")

                                y_min = float(y_dim.get('{http://www.isac-net.org/std/Gating-ML/v2.0/gating}min', 0))
                                y_max = float(y_dim.get('{http://www.isac-net.org/std/Gating-ML/v2.0/gating}max', 100))
                                y_name = y_dim.find('datatypes:fcs-dimension', self.namespaces).get('{http://www.isac-net.org/std/Gating-ML/v2.0/datatypes}name', "Y")

                                gating_info = {
                                    "type": "RectangleGate",
                                    "x_dimension": {"name": x_name, "min": x_min, "max": x_max},
                                    "y_dimension": {"name": y_name, "min": y_min, "max": y_max}
                                }
                                pop_data["gating"].append(gating_info)

                    # Extract statistics
                    for stat in pop.findall('.//Statistic'):
                        pop_data["statistics"].append({
                            "Statistic": stat.get("name", "Unknown"),
                            "Value": stat.get("value", "N/A")
                        })

                    populations.append(pop_data)

                return {specimen_name: {"populations": populations}}

        return {specimen_name: "Not Found"}

    def layout(self):
        """Return the layout for FACS analysis tab."""
        return html.Div([
            #html.H1("📊 FACS Gating Strategies"),

            html.Label("Select a Specimen:"),
            dcc.Dropdown(
                id="specimen-dropdown",
                options=[{"label": spec, "value": spec} for spec in self.all_specimens],
                placeholder="Select a specimen"
            ),

            html.Label("Select a Population:", style={"margin-top": "20px"}),
            dcc.Dropdown(
                id="population-dropdown",
                placeholder="Select a population",
                disabled=True
            ),

            dcc.Graph(id="gating-graph", style={"margin-top": "20px"}),

            html.H3("Statistical Data"),
            dash_table.DataTable(
                id="statistics-table",
                columns=[
                    {"name": "Statistic", "id": "Statistic"},
                    {"name": "Value", "id": "Value"}
                ],
                style_table={'overflowX': 'auto'},
                style_cell={'textAlign': 'left'},
            )
        ])

    def register_callbacks(self, app):
        """Register callbacks for the Dash app."""
        @app.callback(
            [Output("population-dropdown", "options"),
             Output("population-dropdown", "disabled")],
            [Input("specimen-dropdown", "value")]
        )
        def update_population_dropdown(selected_specimen):
            """Update population dropdown when a specimen is selected."""
            if selected_specimen:
                specimen_data = self.extract_specimen_data(selected_specimen)
                populations = specimen_data.get(selected_specimen, {}).get("populations", [])

                if populations:
                    return [{"label": pop["name"], "value": pop["name"]} for pop in populations], False
            return [], True

        @app.callback(
            [Output("gating-graph", "figure"),
             Output("statistics-table", "data")],
            [Input("specimen-dropdown", "value"),
             Input("population-dropdown", "value")]
        )
        def display_population_data(selected_specimen, selected_population):
            """Display population data as a graph and table when selected."""
            fig = go.Figure()
            fig.add_trace(go.Scatter(mode="markers", marker=dict(size=5, opacity=0.5, color="blue")))

            if selected_specimen and selected_population:
                specimen_data = self.extract_specimen_data(selected_specimen)
                populations = specimen_data.get(selected_specimen, {}).get("populations", [])

                for pop in populations:
                    if pop["name"] == selected_population:
                        statistics_data = pop.get("statistics", [])
                        gating_info = pop.get("gating", [])

                        if gating_info:
                            gating = gating_info[0]
                            x_dim = gating["x_dimension"]
                            y_dim = gating["y_dimension"]

                            fig.add_trace(go.Scatter(
                                x=[x_dim["min"], x_dim["max"], x_dim["max"], x_dim["min"], x_dim["min"]],
                                y=[y_dim["min"], y_dim["min"], y_dim["max"], y_dim["max"], y_dim["min"]],
                                mode="lines",
                                line=dict(color="red", width=2),
                                name=f"Gating: {x_dim['name']} vs {y_dim['name']}"
                            ))

                            fig.update_layout(
                                title=f"Gating for {selected_population}",
                                xaxis_title=x_dim["name"],
                                yaxis_title=y_dim["name"],
                                template="plotly_white"
                            )

                        return fig, statistics_data

            return fig, []

