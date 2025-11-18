import dash
from dash import dcc, html, dash_table, Input, Output, State
import pandas as pd
import plotly.express as px
import plotly.io as pio
import numpy as np
import matplotlib.pyplot as plt

# Load data with error handling
# try:
#     data = pd.read_csv("data/MasterTable_MdMUsed.csv", sep=";", na_values=[""])
# except Exception as e:
#
#     data = pd.DataFrame(columns=['Date', 'Probe', 'Sample', 'Condition', 'Killing Efficiency'])

# ==================================================
# Class 1: Bacterial Killing Assaykea-efficiency-chart
# ==================================================
class BacterialKillingAssay:
    def __init__(self, data):
        self.data = data
        self.data_melted = self.data.melt(id_vars=['Date', 'Probe', 'Sample'], var_name='Condition', value_name='Bacterial Colony Count')

    def get_layout(self):
        # Default values for dropdowns
        default_strains = self.data['Sample'].unique().tolist()[:1]  # First strain as default
        default_conditions = self.data_melted['Condition'].unique().tolist()[:1]  # First condition as default

        return html.Div([
            html.H2("🧫 Bacterial Killing Assay: Average Colony Counts by Strain", style={"textAlign": "center", "marginBottom": "20px"}),
            dcc.Dropdown(
                id='bka-strain-dropdown',  # Unique ID
                options=[{'label': strain, 'value': strain} for strain in self.data['Sample'].unique()] + [{'label': 'All', 'value': 'All'}],
                value=default_strains,  # Set default value
                multi=True,
                searchable=True,
                placeholder="Select Strain..."
            ),
            dcc.Dropdown(
                id='bka-condition-dropdown',  # Unique ID
                options=[{'label': cond, 'value': cond} for cond in self.data_melted['Condition'].unique()] + [{'label': 'All', 'value': 'All'}],
                value=default_conditions,  # Set default value
                multi=True,
                searchable=True,
                placeholder="Select Condition..."
            ),
            dcc.RadioItems(
                id='bka-plot-type',  # Unique ID
                options=[
                    {'label': 'Box Plot', 'value': 'box'},
                    {'label': 'Violin Plot', 'value': 'violin'}
                ],
                value='box',  # Default plot type
                inline=True,
                style={"margin": "10px 0"}
            ),
            dcc.Graph(id='bka-killing-plot'),  # Unique ID
            html.P(
                "This plot shows the average bacterial colony counts for selected strains and conditions. "
                "Use the dropdowns to filter by strain and condition. The box plot displays the distribution "
                "of colony counts, while the violin plot provides a smoothed distribution with a box plot inside.",
                style={"marginTop": "10px", "fontSize": "14px", "color": "#666"}
            ),
            html.Div([
                html.Button("📥 Download Graph as PDF", id="bka-download-graph-btn", n_clicks=0, className="btn-download"),
                dcc.Download(id="bka-download-graph"),
                html.Button("📊 Download Summary Table", id="bka-download-csv-btn", className="btn-download"),
                dcc.Download(id="bka-download-csv"),
            ], className="button-container"),
            dash_table.DataTable(
                id='bka-summary-table',  # Unique ID
                columns=[
                    {'name': 'Strain', 'id': 'Sample'},
                    {'name': 'Condition', 'id': 'Condition'},
                    {'name': 'Mean', 'id': 'Mean', 'type': 'numeric'},
                    {'name': 'Median', 'id': 'Median', 'type': 'numeric'},
                    {'name': 'Std', 'id': 'Std', 'type': 'numeric'},
                    {'name': 'Min', 'id': 'Min', 'type': 'numeric'},
                    {'name': 'Max', 'id': 'Max', 'type': 'numeric'},
                    {'name': 'IQR', 'id': 'IQR', 'type': 'numeric'},
                ],
                style_table={'overflowX': 'auto'},
                style_header={'backgroundColor': '#000', 'color': 'white', 'textAlign': 'center'},
                style_cell={'textAlign': 'center'}
            ),
        ])

    def register_callbacks(self, app):
        @app.callback(
            [Output('bka-killing-plot', 'figure'),
             Output('bka-summary-table', 'data'),
             Output('error-message', 'children')],
            [Input('bka-strain-dropdown', 'value'),
             Input('bka-condition-dropdown', 'value'),
             Input('bka-plot-type', 'value')]
        )
        def update_graph(selected_strains, selected_conditions, plot_type):
            try:
                if 'All' in selected_strains:
                    selected_strains = self.data['Sample'].unique().tolist()
                if 'All' in selected_conditions:
                    selected_conditions = self.data_melted['Condition'].unique().tolist()
                if not selected_strains or not selected_conditions:
                    return {}, [], "Please select at least one strain and one condition."

                filtered_data = self.data_melted[(self.data_melted['Sample'].isin(selected_strains)) & (self.data_melted['Condition'].isin(selected_conditions))]
                if filtered_data.empty:
                    return {}, [], "No data available for the selected strains and conditions."

                fig = px.box(filtered_data, x='Condition', y='Bacterial Colony Count', color='Sample') if plot_type == 'box' else \
                    px.violin(filtered_data, x='Condition', y='Bacterial Colony Count', color='Sample', box=True)

                summary_df = filtered_data.groupby(['Sample', 'Condition'])['Bacterial Colony Count'].agg(
                    ['mean', 'median', 'std', 'min', 'max']).reset_index()
                summary_df['IQR'] = filtered_data.groupby(['Sample', 'Condition'])['Bacterial Colony Count'].apply(
                    lambda x: x.quantile(0.75) - x.quantile(0.25)).reset_index(drop=True)

                summary_df.columns = ['Sample', 'Condition', 'Mean', 'Median', 'Std', 'Min', 'Max', 'IQR']
                summary_df = summary_df.round(2)

                return fig, summary_df.to_dict('records'), ""
            except Exception as e:

                return {}, [], f"An error occurred: {str(e)}"

        @app.callback(
            Output("bka-download-graph", "data"),
            [Input("bka-download-graph-btn", "n_clicks")],
            [State("bka-strain-dropdown", "value"),
             State("bka-condition-dropdown", "value"),
             State("bka-plot-type", "value")],
            prevent_initial_call=True
        )
        def download_graph(n_clicks, selected_strains, selected_conditions, plot_type):
            if n_clicks > 0:
                fig, _, _ = update_graph(selected_strains, selected_conditions, plot_type)
                if fig:
                    pdf_path = "graph.pdf"
                    pio.write_image(fig, pdf_path, format="pdf")
                    return dcc.send_file(pdf_path)
            return dash.no_update

        @app.callback(
            Output("bka-download-csv", "data"),
            [Input("bka-download-csv-btn", "n_clicks")],
            [State("bka-strain-dropdown", "value"),
             State("bka-condition-dropdown", "value"),
             State("bka-plot-type", "value")],
            prevent_initial_call=True
        )
        def download_csv(n_clicks, selected_strains, selected_conditions, plot_type):
            if n_clicks > 0:
                _, summary_df, _ = update_graph(selected_strains, selected_conditions, plot_type)
                if summary_df:
                    csv_path = "summary_table.csv"
                    pd.DataFrame(summary_df).to_csv(csv_path, index=False)
                    return dcc.send_file(csv_path)
            return dash.no_update


# ==================================================
# Class 2: Triplicate Data Distribution
# ==================================================
class TriplicateDataDistribution:
    def __init__(self, data):
        self.data = data
        self.df_melted, self.condition_groups = self.preprocess_data(data)

    def preprocess_data(self, df):
        condition_groups = {
            "U": ["U_1", "U_2", "U_3"],
            "D_250": ["D_250_1", "D_250_2", "D_250_3"],
            "L_250": ["L_250_1", "L_250_2", "L_250_3"],
            "NaBu_500": ["NaBu_500_1", "NaBu_500_2", "NaBu_500_3"],
            "GLPG+D_250": ["GLPG+D_250_1", "GLPG+D_250_2", "GLPG+D_250_3"],
            "6OAU_10": ["6OAU_10_1", "6OAU_10_2", "6OAU_10_3"],
            "LPS_100ng": ["LPS_100ng_1", "LPS_100ng_2", "LPS_100ng_3"],
            "D_250+LPS100ng": ["D_250+LPS100ng_1", "D_250+LPS100ng_2", "D_250+LPS100ng_3"],
            "L_250+LPS100ng": ["L_250+LPS100ng_1", "L_250+LPS100ng_2", "L_250+LPS100ng_3"],
            "Baf_20ng": ["Baf_20ng_1", "Baf_20ng_2", "Baf_20ng_3"],
            "Rap_100ng": ["Rap_100ng_1", "Rap_100ng_2", "Rap_100ng_3"],
            "D_100": ["D_100_1", "D_100_2", "D_100_3"],
            "L_50": ["L_50_1", "L_50_2", "L_50_3"],
            "NaBu_50": ["NaBu_50_1", "NaBu_50_2", "NaBu_50_3"],
            "GLPG+D100": ["GLPG+D100_1", "GLPG+D100_2", "GLPG+D100_3"],
            "D_100+LPS_100ng": ["D_100+LPS_100ng_1", "D_100+LPS_100ng_2", "D_100+LPS_100ng_3"],
            "L_100+LPS_100ng": ["L_100+LPS_100ng_1", "L_100+LPS_100ng_2", "L_100+LPS_100ng_3"]
        }

        df_filtered = df[["Sample", "Probe", "Date"] + [col for sublist in condition_groups.values() for col in sublist]]
        df_melted = df_filtered.melt(id_vars=["Sample", "Probe", "Date"], value_vars=[col for sublist in condition_groups.values() for col in sublist])

        condition_mapping = {}
        for group, triplicates in condition_groups.items():
            for triplicate in triplicates:
                condition_mapping[triplicate] = group
        df_melted['condition'] = df_melted['variable'].map(condition_mapping)

        df_melted['combined_selection'] = df_melted['Date'].astype(str) + ";" + df_melted['Probe'].astype(str) + ";" + df_melted['Sample'].str.split("_").str[0]

        return df_melted, condition_groups

    def get_layout(self):
        unique_combinations = self.df_melted['combined_selection'].unique()
        return html.Div([
            html.H2("📊 Triplicate Data Distribution Across Experimental Conditions",
                    style={"textAlign": "center", "marginBottom": "20px", "marginTop": "50px"}),
            html.Label("Select Date, Probe, and Strain:"),
            dcc.Dropdown(
                id="tdd-dropdown-combination",  # Unique ID
                options=[{"label": combination, "value": combination} for combination in unique_combinations],
                value=unique_combinations[:1],
                multi=True
            ),
            dcc.Dropdown(
                id="tdd-box-plot-dropdown",  # Unique ID
                options=[{"label": condition, "value": condition} for condition in self.condition_groups.keys()],
                value=list(self.condition_groups.keys()),
                multi=True,
                placeholder="Select conditions for Box Plot...",
                style={"marginBottom": "20px"}
            ),
            html.Div(id="tdd-box-plots", children=[]),  # Unique ID
            html.P(
                "This section displays the distribution of triplicate data for each experimental condition. "
                "Each box plot represents the spread of values across three replicates. Use the dropdowns "
                "to select specific combinations of date, probe, and strain.",
                style={"marginTop": "10px", "fontSize": "14px", "color": "#666"}
            ),
        ])

    def register_callbacks(self, app):
        @app.callback(
            Output("tdd-box-plots", "children"),
            [Input("tdd-box-plot-dropdown", "value"),
             Input("tdd-dropdown-combination", "value")]
        )
        def update_box_plots(selected_conditions, selected_combinations):
            filtered_df = self.df_melted[self.df_melted['combined_selection'].isin(selected_combinations)]
            filtered_df = filtered_df[filtered_df['condition'].isin(selected_conditions)]

            figures = []
            for combination in selected_combinations:
                df_combination = filtered_df[filtered_df['combined_selection'] == combination]
                fig = px.box(df_combination,
                             x="condition", y="value", color="condition",
                             title=f"Triplicate Data Distribution for {combination}",
                             labels={"value": " ", "condition": "Condition"})
                figures.append(dcc.Graph(figure=fig))

            return figures


# ==================================================
# Class 3: Killing Efficiency Analysis
# ==================================================
class KillingEfficiencyAnalysis:
    def __init__(self, data):
        self.data = data
        self.melted_data, self.treatment_conditions = self.preprocess_data(data)
        self.color_map = self.generate_color_map(self.melted_data)

    def preprocess_data(self, data):
        uncontrolled_conditions = [col for col in data.columns[3:] if col.startswith("U_")]
        treatment_conditions = [col for col in data.columns[3:] if col not in uncontrolled_conditions]

        if uncontrolled_conditions:
            data["Control_Mean"] = data[uncontrolled_conditions].mean(axis=1)
            for col in treatment_conditions:
                data[col] = ((1 - (data[col] / data["Control_Mean"])) * 100).round(2)
            data.drop(columns=["Control_Mean"], inplace=True)

        melted_data = data[["Sample", "Probe", "Date"] + treatment_conditions].melt(id_vars=["Sample", "Probe", "Date"])
        melted_data = melted_data.rename(columns={"variable": "Condition", "value": "BKE%"})
        return melted_data, treatment_conditions

    # def generate_color_map(self, melted_data):
    #     melted_data = melted_data.copy()
    #     melted_data.loc[:, "base_condition"] = melted_data["Condition"].str.replace(r"_\d+$", "", regex=True)
    #     unique_base_conditions = melted_data["base_condition"].unique()
    #     cmap = plt.get_cmap("tab10") if len(unique_base_conditions) <= 10 else plt.get_cmap("tab20")
    #     color_map = {base_condition: px.colors.label_rgb((np.array(cmap(i / len(unique_base_conditions))) * 255).astype(int))
    #                  for i, base_condition in enumerate(unique_base_conditions)}
    #     return color_map
    def generate_color_map(self, melted_data):

        melted_data = melted_data.copy()

        # Extract base conditions by removing the last underscore and digit
        melted_data["base_condition"] = melted_data["Condition"].apply(lambda x: "_".join(x.split("_")[:-1]))

        unique_base_conditions = melted_data["base_condition"].unique()

        # Choose an appropriate colormap
        cmap = plt.get_cmap("tab10") if len(unique_base_conditions) <= 10 else plt.get_cmap("tab20")

        # Assign colors to base conditions
        color_map = {
            base: px.colors.label_rgb((np.array(cmap(i / len(unique_base_conditions))) * 255).astype(int))
            for i, base in enumerate(unique_base_conditions)
        }

        # Map colors to conditions
        melted_data["color"] = melted_data["base_condition"].map(color_map)
        full_color_map = dict(zip(melted_data["Condition"], melted_data["color"]))

        return full_color_map


    def get_layout(self):
        return html.Div([
            html.H2("🦠 Killing Efficiency Analysis Using Unstimulated Control As Baseline", style={"textAlign": "center", "marginTop": "50px", "marginBottom": "20px"}),
            dcc.Dropdown(
                id="kea-condition-dropdown",  # Unique ID
                options=[{"label": "All", "value": "All"}] + [{"label": condition, "value": condition} for condition in self.treatment_conditions],
                value="All",
                multi=True,
                placeholder="Select conditions to compare...",
                searchable=True,
                style={"marginBottom": "20px"}
            ),
            dcc.Graph(id="kea-efficiency-chart"),  # Unique ID
            html.P(
                "This bar chart compares the killing efficiency (%) of different conditions while keeping unstimulated control as baseline. "
                "Killing efficiency is calculated as the percentage reduction in bacterial colonies across every experiment"
                "compared to the control. Use the dropdown to filter by specific conditions.",
                style={"marginTop": "10px", "fontSize": "14px", "color": "#666"}
            ),

            dash_table.DataTable(
                id="kea-data-table",  # Unique ID
                columns=[{"name": col, "id": col} for col in ["Sample", "Probe", "Date", "Condition", "BKE%"]],
                style_table={'height': '350px', 'overflowY': 'auto'},
                style_header={'backgroundColor': '#000', 'color': 'white', 'textAlign': 'center'},
                page_size=10
            ),
        ])

    def register_callbacks(self, app):
        @app.callback(
            [Output("kea-efficiency-chart", "figure"),
             Output("kea-data-table", "data")],
            [Input("kea-condition-dropdown", "value")]
        )
        def update_chart_and_table(selected_conditions):
            if not selected_conditions:
                return px.bar(title="Unstimulated vs Stimulated: Killing Efficiency"), self.melted_data.to_dict('records')

            if "All" in selected_conditions:
                selected_conditions = self.treatment_conditions
            else:
                selected_conditions = [cond for cond in selected_conditions if cond != "All"]

            # Check if filtered_data is empty
            filtered_data = self.melted_data[self.melted_data["Condition"].isin(selected_conditions)].copy()




            if filtered_data.empty:
                return px.bar(title="No Data Available for Selected Conditions"), []


            # Ensure all conditions in filtered_data have a color mapping
            missing_colors = set(filtered_data["Condition"].unique()) - set(self.color_map.keys())
            if missing_colors:

                # Assign a default color (e.g., black) to missing conditions
                for condition in missing_colors:
                    self.color_map[condition] = "#000000"  # Black as default color

            bar_fig = px.bar(filtered_data,
                             x="Sample", y="BKE%", color="Condition",
                             title="Unstimulated vs Stimulated: Killing Efficiency",
                             labels={"BKE%": "Killing Efficiency (%)", "Condition": "Condition"},
                             color_discrete_map=self.color_map,
                             hover_data=["Probe", "Date", "BKE%", "Condition"])

            return bar_fig, filtered_data.to_dict('records')

# ==================================================
# Class 4: Killing Efficiency Analysis- Combining all triplicates
# ==================================================
class KillingEfficiencyWithBaseline:
    def __init__(self, data_path):
        # Load data
        self.data = pd.read_csv(data_path, sep=";", na_values=[""])

        # Convert Control (U_1, U_2, U_3) to numeric and calculate its mean
        self.data[['U_1', 'U_2', 'U_3']] = self.data[['U_1', 'U_2', 'U_3']].apply(pd.to_numeric, errors='coerce')
        self.data["U"] = self.data[['U_1', 'U_2', 'U_3']].mean(axis=1)  # Control Mean

        # Treatment groups
        self.treatment_groups = [
            "D_250", "L_250", "NaBu_500", "GLPG+D_250", "6OAU_10", "LPS_100ng",
            "D_250+LPS100ng", "L_250+LPS100ng", "Baf_20ng", "Rap_100ng",
            "D_100", "L_50", "NaBu_50", "GLPG+D100", "D_100+LPS_100ng", "L_100+LPS_100ng"
        ]

        # Compute mean for each treatment
        self.treatment_means = {treatment: self.data[[f"{treatment}_1", f"{treatment}_2", f"{treatment}_3"]].mean(axis=1) for treatment in self.treatment_groups}
        self.treatments_mean = pd.DataFrame(self.treatment_means)

        # Calculate Killing Efficiency (%)
        self.killing_efficiency = (1 - (self.treatments_mean.div(self.data["U"], axis=0))) * 100

        # Round all values to 2 decimal places
        self.killing_efficiency = self.killing_efficiency.round(2)

        # Concatenate Date, Probe, and Sample into a single ID column
        self.killing_efficiency['ID'] = self.data['Date'].astype(str) + " | " + self.data['Probe'].astype(str) + " | " + self.data['Sample']

    def get_layout(self):
        return html.Div([
            html.H2("🦠 Killing Efficiency Using Treatment/Control Triplicates Mean Per Experiment", style={'text-align': 'center', 'margin-bottom': '20px'}),

            html.Label("Select ID (Date | Probe | Sample):", style={'font-weight': 'bold', 'margin-bottom': '5px'}),
            html.Br(),  # Adds space between label and dropdown

            dcc.Dropdown(
                id='id-dropdown',
                options=[{'label': i, 'value': i} for i in self.killing_efficiency['ID'].unique()],
                value=self.killing_efficiency['ID'].unique()[0],  # Default value
                style={'width': '100%', 'margin-bottom': '20px'},  # Adjust width and add spacing
                searchable=True,
            ),

            dcc.Graph(id='killing-efficiency-graph', style={'margin-top': '20px'}),
            html.P(
                "This horizontal bar chart shows the killing efficiency (%) for each treatment, "
                "calculated by combining all triplicates. The efficiency is derived from "
                "the Treatment/Control triplicates mean per experiment. Select a specific ID (Date | Probe | Sample) "
                "to view the results for that experiment.",
                style={"marginTop": "10px", "fontSize": "14px", "color": "#666"}
            ),

            # Add DataTable for displaying all efficiency data
            html.H3("Efficiency Data", style={'margin-top': '40px', 'margin-bottom': '10px'}),
            dash_table.DataTable(
                id='efficiency-table',
                columns=[{"name": i, "id": i} for i in ['ID'] + [col for col in self.killing_efficiency.columns if col != 'ID']],  # Ensure ID is first
                data=self.killing_efficiency[['ID'] + [col for col in self.killing_efficiency.columns if col != 'ID']].to_dict('records'),  # Reorder data
                style_table={'overflowX': 'auto', 'margin-top': '20px'},
                style_cell={'textAlign': 'left', 'padding': '5px'},
                style_header={'backgroundColor': '#000', 'color': 'white', 'textAlign': 'center'},
                page_size=15  # Show 15 rows per page
            )
        ], style={'padding': '20px'})  # Adds padding around the layout

    def register_callbacks(self, app):
        @app.callback(
            Output('killing-efficiency-graph', 'figure'),
            [Input('id-dropdown', 'value')]
        )
        def update_graph(selected_id):
            # Filter data based on the selected ID
            filtered_data = self.killing_efficiency[self.killing_efficiency['ID'] == selected_id]

            if filtered_data.empty:
                return px.bar(title="No Data Available")

            # Melt data for Plotly (convert wide to long format)
            melted_data = filtered_data.drop(columns=['ID']).melt(var_name='Treatment', value_name='Efficiency')

            # Plot using px.bar() with horizontal orientation
            fig = px.bar(
                melted_data,
                y='Treatment',  # Treatments on y-axis
                x='Efficiency',  # Efficiency values on x-axis
                labels={'Efficiency': 'Killing Efficiency (%)'},
                text='Efficiency',
                color='Treatment',  # Color bars by treatment type
                color_discrete_sequence=px.colors.qualitative.Set3,  # Set color palette
                orientation='h'  # Make bars horizontal
            )

            # Update traces and layout
            fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
            fig.update_layout(
                yaxis_title="Treatment",  # Treatments on y-axis
                xaxis_title="Killing Efficiency (%)",  # Efficiency on x-axis
                template="plotly_white",  # Clean white background
                showlegend=False  # Hide legend for cleaner look
            )

            return fig


# ==================================================
# Class 5: Killing Efficiency Analysis- Combining specific triplicates against their treatments
# ==================================================


class KillingEfficiencyAnalysisForSpecificGroups:
    def __init__(self, data_path):
        self.df = pd.read_csv(data_path, sep=";")
        self.group_pairs = self.define_group_pairs()
        self.process_data()
        # self.app = dash.Dash(__name__)
        # self.app.layout = self.get_layout()
        # self.register_callbacks()

    def define_group_pairs(self):
        return {
            "D_250": (["D_250_1", "D_250_2", "D_250_3"], ["GLPG+D_250_1", "GLPG+D_250_2", "GLPG+D_250_3"]),
            "L_250": (["L_250_1", "L_250_2", "L_250_3"], ["L_250+LPS100ng_1", "L_250+LPS100ng_2", "L_250+LPS100ng_3"]),
            "LPS_100ng": (["LPS_100ng_1", "LPS_100ng_2", "LPS_100ng_3"],
                          ["D_100+LPS_100ng_1", "D_100+LPS_100ng_2", "D_100+LPS_100ng_3",
                           "L_100+LPS_100ng_1", "L_100+LPS_100ng_2", "L_100+LPS_100ng_3"]),
            "D_100": (["D_100_1", "D_100_2", "D_100_3"],
                      ["GLPG+D100_1", "GLPG+D100_2", "GLPG+D100_3",
                       "D_100+LPS_100ng_1", "D_100+LPS_100ng_2", "D_100+LPS_100ng_3"])
        }

    def process_data(self):
        for group, (baseline_cols, treatment_cols) in self.group_pairs.items():
            baseline_cols = [col for col in baseline_cols if col in self.df.columns]
            treatment_cols = [col for col in treatment_cols if col in self.df.columns]

            if baseline_cols:
                self.df[f"{group}_Baseline"] = self.df[baseline_cols].mean(axis=1)
            if treatment_cols:
                self.df[f"{group}_Treatment"] = self.df[treatment_cols].mean(axis=1)

            if f"{group}_Baseline" in self.df.columns and f"{group}_Treatment" in self.df.columns:
                self.df[f"{group}_KillingEfficiency"] = ((self.df[f"{group}_Baseline"] - self.df[f"{group}_Treatment"]) / self.df[f"{group}_Baseline"]) * 100

        self.df["Experiment"] = self.df["Date"].astype(str) + " | " + self.df["Probe"].astype(str) + " | " + self.df["Sample"].astype(str)

    def get_layout(self):
        return html.Div([
            html.H2("Killing Efficiency 🔬: Baseline Groups vs. Their Stimulated Counterparts", style={'textAlign': 'center'}),
            html.Label("Select Experiment:"),
            dcc.Dropdown(
                id="experiment-dropdown",
                options=[{"label": exp, "value": exp} for exp in self.df["Experiment"].unique()],
                value=self.df["Experiment"].unique()[0],
                clearable=False
            ),
            dcc.Graph(id="bar-plot"),
            html.P(
                "This bar chart compares the killing efficiency (%) of different conditions while keeping unstimulated control as baseline. "
                "Killing efficiency is calculated as the percentage reduction in bacterial colonies across every experiment"
                "compared to the control. Use the dropdown to filter by specific conditions.",
                style={"marginTop": "10px", "fontSize": "14px", "color": "#666"}
            )
        ])

    def register_callbacks(self,app):
        @app.callback(
            Output("bar-plot", "figure"),
            Input("experiment-dropdown", "value")
        )
        def update_bar_chart(selected_experiment):
            filtered_df = self.df[self.df["Experiment"] == selected_experiment]
            plot_data = []
            for group, (_, treatment_cols) in self.group_pairs.items():
                if f"{group}_KillingEfficiency" in filtered_df.columns:
                    plot_data.append({
                        "Baseline Group": group,
                        "Killing Efficiency": filtered_df[f"{group}_KillingEfficiency"].values[0],
                        "Treatment Groups": ", ".join(treatment_cols)
                    })

            plot_df = pd.DataFrame(plot_data)

            if plot_df.empty:
                return px.bar(title="No data available for this experiment")

            fig = px.bar(
                plot_df,
                x="Baseline Group",
                y="Killing Efficiency",
                title=f"Killing Efficiency for {selected_experiment}",
                labels={"Killing Efficiency": "Killing Efficiency (%)"},
                color="Baseline Group",
                hover_data=["Treatment Groups"],
                height=500
            )

            return fig



# ==================================================
# Class 5:  Mean of All - Killing Efficiency Analysis
# ==================================================

class MeanOfAll:
    def __init__(self, data_path="data/MasterTable_MdMUsed.csv"):
        self.data_path = data_path
        self.df_triplicate_mean = self.process_data()
        self.killing_efficiency_df = self.compute_killing_efficiency()

    def process_data(self):
        """Load and preprocess the data to compute triplicate means."""
        df = pd.read_csv(self.data_path, sep=';')
        df = df.drop(columns=['Date', 'Probe'])

        df_mean = df.groupby('Sample').mean(numeric_only=True).reset_index()

        triplicate_groups = [
            "U", "D_250", "L_250", "NaBu_500", "GLPG+D_250", "6OAU_10", "LPS_100ng",
            "D_250+LPS100ng", "L_250+LPS100ng", "Baf_20ng", "Rap_100ng", "D_100",
            "L_50", "NaBu_50", "GLPG+D100", "D_100+LPS_100ng", "L_100+LPS_100ng"
        ]

        df_triplicate_mean = df_mean[['Sample']].copy()

        for group in triplicate_groups:
            cols = [f"{group}_1", f"{group}_2", f"{group}_3"]
            df_triplicate_mean[group] = df_mean[cols].mean(axis=1)

        return df_triplicate_mean

    def compute_killing_efficiency(self):
        """Compute killing efficiency using 'U' as the baseline."""
        df_triplicate_mean = self.df_triplicate_mean
        killing_efficiency_df = df_triplicate_mean[['Sample']].copy()

        for group in df_triplicate_mean.columns[1:]:  # Skip 'Sample' column
            if group != "U":  # Skip baseline
                killing_efficiency_df[f"{group}_killing_eff"] = ((1 - (df_triplicate_mean[group] / df_triplicate_mean['U'])) * 100).round(2)

        return killing_efficiency_df

    def get_layout(self):
        """Define the Dash app layout."""
        return html.Div([
            html.H2("📉 Killing Efficiency Across Stimuli: Mean of Triplicates & Samples",style={'textAlign': 'center'}),

            # Multi-select dropdown for samples
            dcc.Dropdown(
                id="sample-dropdown",
                options=[{"label": sample, "value": sample} for sample in self.killing_efficiency_df["Sample"]],
                value=[self.killing_efficiency_df["Sample"].iloc[0]],  # Default to first sample
                multi=True,
                clearable=False
            ),

            # Bar chart for killing efficiency
            dcc.Graph(id="killing-efficiency-bar-chart"),
            html.P(
                "This bar chart displays the killing efficiency of various treatments relative to the untreated control (U). "
                "Each bar represents the percentage reduction in viability compared to the baseline. "
                "Use the dropdown to select multiple samples and compare their responses to different treatments.",
                style={"font-size": "16px", "margin-top": "10px"},
            )
        ])

    def register_callbacks(self,app):
        """Register callback functions to update the graph dynamically."""
        @app.callback(
            Output("killing-efficiency-bar-chart", "figure"),
            [Input("sample-dropdown", "value")]
        )
        def update_chart(selected_samples):
            """Update the bar chart based on selected samples."""
            selected_data = self.killing_efficiency_df[self.killing_efficiency_df["Sample"].isin(selected_samples)]
            selected_data = selected_data.melt(id_vars=["Sample"], var_name="Treatment", value_name="Killing Efficiency")

            # Clean treatment names
            selected_data["Treatment"] = selected_data["Treatment"].str.replace("_killing_eff", "")

            # Create bar chart
            fig = px.bar(
                selected_data,
                x="Treatment",
                y="Killing Efficiency",
                color="Sample",
                title=f"Killing Efficiency for Selected Samples",
                labels={"Treatment": "Treatment", "Killing Efficiency": "Killing Efficiency (%)"},
                text_auto=True,
                barmode="group"
            )

            fig.update_layout(xaxis_tickangle=-45, xaxis_title="Treatment", yaxis_title="Killing Efficiency (%)")
            return fig