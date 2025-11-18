import xml.etree.ElementTree as ET
import pandas as pd
import plotly.express as px
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

class MFIAnalysis:
    def __init__(self, xml_file):
        self.xml_file = xml_file

        # Define namespaces
        self.namespaces = {
            "gating": "http://www.isac-net.org/std/Gating-ML/v2.0/gating",
            "datatypes": "http://www.isac-net.org/std/Gating-ML/v2.0/datatypes",
        }

        # Extract all available specimens
        self.all_specimens = self.extract_all_specimens()

    def extract_all_specimens(self):
        """Extract all specimen names from the XML file."""
        try:
            tree = ET.parse(self.xml_file)
            root = tree.getroot()
            return [sample.get("name") for sample in root.findall('.//SampleNode')]
        except Exception as e:

            return []

    def extract_mfi_data(self, specimen_names):
        """
        Extract MFI data for selected specimens.
        Filters data for Macrophages only.
        """
        tree = ET.parse(self.xml_file)
        root = tree.getroot()
        mfi_data = []

        for sample in root.findall('.//SampleNode'):
            if sample.get("name") in specimen_names:
                for pop in sample.findall('.//Population'):
                    pop_name = pop.get("name")

                    # Filter for Macrophages only
                    if "macrophage" in pop_name.lower():
                        for stat in pop.findall('.//Statistic'):
                            if stat.get("name") == "Median":  # Median Fluorescence Intensity
                                marker = stat.get("id")  # Marker name (e.g., FITC-A, PE-A)
                                mfi_value = float(stat.get("value", 0))
                                mfi_data.append({
                                    "Specimen": sample.get("name"),
                                    "Population": pop_name,
                                    "Marker": marker,
                                    "MFI": mfi_value
                                })

        return mfi_data

    def create_mfi_bar_plot(self, mfi_data):
        """Create a bar plot for MFI values."""
        if not mfi_data:
            return px.bar(title="No MFI Data Available")

        df = pd.DataFrame(mfi_data)

        fig = px.bar(df,
                     x="Specimen",
                     y="MFI",
                     color="Marker",
                     barmode="group",
                     title="Macrophage Mean Fluorescence Intensity (MFI)",
                     labels={"MFI": "MFI (Median)", "Specimen": "Specimen"},
                     text="MFI")

        fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
        fig.update_layout(xaxis={'categoryorder': 'total descending'})
        return fig

    def get_layout(self):
        """Return layout for MFI analysis tab."""
        return html.Div([
            html.H1("📊 MFI Analysis (Macrophages)"),

            html.Label("Select Specimens:"),
            dcc.Dropdown(
                id="specimen-dropdown",
                options=[{"label": spec, "value": spec} for spec in self.all_specimens],
                placeholder="Select specimens",
                multi=True
            ),

            dcc.Graph(id="mfi-bar-plot", style={"margin-top": "20px"})
        ])

    def register_callbacks(self, app):
        """Register callbacks for the Dash app."""
        @app.callback(
            Output("mfi-bar-plot", "figure"),
            [Input("specimen-dropdown", "value")]
        )
        def update_mfi_bar_plot(selected_specimens):
            """Update MFI bar plot when specimens are selected."""
            if selected_specimens:
                mfi_data = self.extract_mfi_data(selected_specimens)
                return self.create_mfi_bar_plot(mfi_data)
            return px.bar(title="Select specimens to view MFI data")
