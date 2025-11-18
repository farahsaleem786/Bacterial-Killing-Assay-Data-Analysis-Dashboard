import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import pandas as pd
import dash_bootstrap_components as dbc

# Import the classes from tab1_bacterial_killing
from tabs.tab1_bacterial_killing import (
    BacterialKillingAssay, TriplicateDataDistribution, KillingEfficiencyAnalysis,
    KillingEfficiencyWithBaseline, KillingEfficiencyAnalysisForSpecificGroups, MeanOfAll
)
from tabs.tab2_facs_gating import FACSAnalysis
from tabs.tab3_mfi_bar_plots import MFIAnalysis
from tabs.tab4_upload import FileUploadApp

# Initialize Dash app with Bootstrap theme
app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server  # Needed for file downloads
app.title = "Bacterial Killing Assay & FACS Data Analysis"

# Load data with error handling
try:
    data = pd.read_csv("data/MasterTable_MdMUsed.csv", sep=";", na_values=[""])
except Exception as e:
    data = pd.DataFrame(columns=['Date', 'Probe', 'Sample', 'Condition', 'Killing Efficiency'])


# Initialize the individual classes
bacterial_killing_assay_instance = BacterialKillingAssay(data)
triplicate_data_distribution_instance = TriplicateDataDistribution(data)
killing_efficiency_analysis_instance = KillingEfficiencyAnalysis(data)
killing_efficiency_with_baseline = KillingEfficiencyWithBaseline("data/MasterTable_MdMUsed.csv")
killing_efficiency_analysis_for_specific_groups = KillingEfficiencyAnalysisForSpecificGroups("data/MasterTable_MdMUsed.csv")
mean_of_all = MeanOfAll("data/MasterTable_MdMUsed.csv")

# Initialize the FACS analysis class
facs_analysis = FACSAnalysis("data/20-Jan-2025.wsp")
MFI_analysis = MFIAnalysis("data/20-Jan-2025.wsp")
###########################
File_Upload_App=FileUploadApp(app)
app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H1("Bacterial Killing Assay & FACS Data Analysis", className="text-center my-4"))),

    # Tabs for different sections
    dbc.Row(dbc.Col(
        dcc.Tabs(
            id="tabs",
            value="tab-1",  # Default selected tab
            children=[
                dcc.Tab(label="🔬 Bacterial Killing Assay", value="tab-1"),
                dcc.Tab(label="📊 FACS Gating Strategies", value="tab-2"),
                dcc.Tab(label="📉 MFI Bar Plots", value="tab-3"),
                dcc.Tab(label="📂 Upload Data", value="tab-4"),
            ],
            className="custom-tabs"
        )
    )),

    # Div to display content based on selected tab
    dbc.Row(dbc.Col(html.Div(id="tab-content", className="tab-content my-4"))),
    dbc.Row(dbc.Col(html.Div(id="error-message", style={'color': 'red', 'fontWeight': 'bold'})))
])

# Register callbacks using the class methods
bacterial_killing_assay_instance.register_callbacks(app)
triplicate_data_distribution_instance.register_callbacks(app)
killing_efficiency_analysis_instance.register_callbacks(app)
killing_efficiency_with_baseline.register_callbacks(app)
killing_efficiency_analysis_for_specific_groups.register_callbacks(app)
mean_of_all.register_callbacks(app)
facs_analysis.register_callbacks(app)
MFI_analysis.register_callbacks(app)

# Callback to render tabs based on the selected tab value
@app.callback(
    Output('tab-content', 'children'),
    [Input('tabs', 'value')]
)
def render_tab(tab_name):
    try:
        if tab_name == 'tab-1':
            # Layout for Tab 1 with a sidebar
            return dbc.Row([
                dbc.Col([
                    html.H4("Select Analysis", className="mb-3", style={"color": "white", "fontWeight": "bold"}),
                    dbc.RadioItems(
                        id="sidebar-radio",
                        options=[
                            {"label": "Bacterial Killing Assay", "value": "bacterial-killing"},
                            {"label": "Triplicate Data Distribution", "value": "triplicate-distribution"},
                            {"label": "Killing Efficiency Analysis", "value": "killing-efficiency"},
                            {"label": "Killing Efficiency with Baseline", "value": "killing-efficiency-baseline"},
                            {"label": "Killing Efficiency for Specific Groups", "value": "killing-efficiency-groups"},
                            {"label": "Mean of All", "value": "mean-of-all"},
                        ],
                        value="bacterial-killing",  # Default selected option
                        labelStyle={"color": "white", "display": "block", "marginBottom": "10px"},
                        inputClassName="radio-input",
                    ),
                ], width=3, style={"backgroundColor": "#001f3f", "padding": "20px", "height": "100vh"}),  # Navy Blue (#001f3f)

                # Main Content Column
                dbc.Col([
                    html.Div(id="bacterial-killing", children=bacterial_killing_assay_instance.get_layout()),
                    html.Div(id="triplicate-distribution", children=triplicate_data_distribution_instance.get_layout(), style={"display": "none"}),
                    html.Div(id="killing-efficiency", children=killing_efficiency_analysis_instance.get_layout(), style={"display": "none"}),
                    html.Div(id="killing-efficiency-baseline", children=killing_efficiency_with_baseline.get_layout(), style={"display": "none"}),
                    html.Div(id="killing-efficiency-groups", children=killing_efficiency_analysis_for_specific_groups.get_layout(), style={"display": "none"}),
                    html.Div(id="mean-of-all", children=mean_of_all.get_layout(), style={"display": "none"}),
                ], width=9, className="main-content"),
            ], style={"height": "100vh"})  # Ensure the row takes full height
        elif tab_name == 'tab-2':
            return facs_analysis.layout()
        elif tab_name == 'tab-3':
            return MFI_analysis.get_layout()
        elif tab_name == 'tab-4':
             return File_Upload_App.get_layout()
        else:
            return html.Div([html.H2("Invalid Tab")])
    except Exception as e:
        print("Error in render_tab:", str(e))
        return html.Div(["An error occurred while rendering the tab."])

# Callback to handle sidebar radio button selection
@app.callback(
    [Output(f"{section}", "style") for section in [
        "bacterial-killing", "triplicate-distribution", "killing-efficiency",
        "killing-efficiency-baseline", "killing-efficiency-groups", "mean-of-all"
    ]],
    [Input("sidebar-radio", "value")]
)
def update_content(selected_value):
    # Hide all sections initially
    styles = [{"display": "none"}] * 6

    # Show the selected section
    if selected_value == "bacterial-killing":
        styles[0] = {"display": "block"}
    elif selected_value == "triplicate-distribution":
        styles[1] = {"display": "block"}
    elif selected_value == "killing-efficiency":
        styles[2] = {"display": "block"}
    elif selected_value == "killing-efficiency-baseline":
        styles[3] = {"display": "block"}
    elif selected_value == "killing-efficiency-groups":
        styles[4] = {"display": "block"}
    elif selected_value == "mean-of-all":
        styles[5] = {"display": "block"}

    return styles

# Run the server
if __name__ == '__main__':
    app.run_server(debug=True)