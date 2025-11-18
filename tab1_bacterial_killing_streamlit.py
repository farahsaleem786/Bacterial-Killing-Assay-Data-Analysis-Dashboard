import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt

# Load data with error handling
try:
    data = pd.read_csv("data/MasterTable_MdMUsed.csv", sep=";", na_values=[""])
except Exception as e:
    st.error(f"Error loading data: {str(e)}")
    data = pd.DataFrame(columns=['Date', 'Probe', 'Sample', 'Condition', 'Killing Efficiency'])

# ==================================================
# Class 1: Bacterial Killing Assay
# ==================================================
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.io as pio

class BacterialKillingAssay:
    def __init__(self, data):
        self.data = data
        self.data_melted = self.data.melt(
            id_vars=['Date', 'Probe', 'Sample'],
            var_name='Condition',
            value_name='Bacterial Colony Count'
        )

    def render(self):
        st.title("🧫 Bacterial Killing Assay: Average Colony Counts by Strain")

        # Dropdowns for strain and condition
        strains = self.data['Sample'].unique().tolist()
        selected_strains = st.multiselect(
            "Select Strain",
            options=strains + ['All'],
            default=strains[:1]
        )

        conditions = self.data_melted['Condition'].unique().tolist()
        selected_conditions = st.multiselect(
            "Select Condition",
            options=conditions + ['All'],
            default=conditions[:1]
        )

        # Plot type selection
        plot_type = st.radio(
            "Select Plot Type",
            options=['Box Plot', 'Violin Plot'],
            index=0
        )

        # Filter data based on selections
        if 'All' in selected_strains:
            selected_strains = strains
        if 'All' in selected_conditions:
            selected_conditions = conditions

        filtered_data = self.data_melted[
            (self.data_melted['Sample'].isin(selected_strains)) &
            (self.data_melted['Condition'].isin(selected_conditions))
            ]

        if not filtered_data.empty:
            # Create plot
            if plot_type == 'Box Plot':
                fig = px.box(
                    filtered_data,
                    x='Condition',
                    y='Bacterial Colony Count',
                    color='Sample',
                    title="Bacterial Colony Counts by Strain and Condition"
                )
            else:
                fig = px.violin(
                    filtered_data,
                    x='Condition',
                    y='Bacterial Colony Count',
                    color='Sample',
                    box=True,
                    title="Bacterial Colony Counts by Strain and Condition"
                )

            # Display plot
            st.plotly_chart(fig)

            # Summary table
            summary_df = filtered_data.groupby(['Sample', 'Condition'])['Bacterial Colony Count'].agg(
                ['mean', 'median', 'std', 'min', 'max']
            ).reset_index()
            summary_df['IQR'] = filtered_data.groupby(['Sample', 'Condition'])['Bacterial Colony Count'].apply(
                lambda x: x.quantile(0.75) - x.quantile(0.25))
            summary_df.columns = ['Sample', 'Condition', 'Mean', 'Median', 'Std', 'Min', 'Max', 'IQR']
            summary_df = summary_df.round(2)

            st.write("Summary Table:")
            st.dataframe(summary_df)

            # Download buttons
            if st.button("📥 Download Graph as PDF"):
                pdf_path = "graph.pdf"
            pio.write_image(fig, pdf_path, format="pdf")
            st.success(f"Graph saved to {pdf_path}")

            if st.button("📊 Download Summary Table as CSV"):
                csv_path = "summary_table.csv"
                summary_df.to_csv(csv_path, index=False)
                st.success(f"Table saved to {csv_path}")
            else:
                st.warning("No data available for the selected strains and conditions.")


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

    def render(self):
        st.title("📊 Triplicate Data Distribution Across Experimental Conditions")

        # Dropdowns for combination and conditions
        unique_combinations = self.df_melted['combined_selection'].unique()
        selected_combinations = st.multiselect(
            "Select Date, Probe, and Strain",
            options=unique_combinations,
            default=unique_combinations[:1]
        )

        selected_conditions = st.multiselect(
            "Select Conditions for Box Plot",
            options=list(self.condition_groups.keys()),
            default=list(self.condition_groups.keys())
        )

        # Filter data and display plots
        if selected_combinations and selected_conditions:
            filtered_df = self.df_melted[
                (self.df_melted['combined_selection'].isin(selected_combinations)) &
                (self.df_melted['condition'].isin(selected_conditions))
                ]

            for combination in selected_combinations:
                df_combination = filtered_df[filtered_df['combined_selection'] == combination]
                fig = px.box(
                    df_combination,
                    x="condition",
                    y="value",
                    color="condition",
                    title=f"Triplicate Data Distribution for {combination}",
                    labels={"value": " ", "condition": "Condition"}
                )
                st.plotly_chart(fig)


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

    def generate_color_map(self, melted_data):
        melted_data = melted_data.copy()
        melted_data.loc[:, "base_condition"] = melted_data["Condition"].str.replace(r"_\d+$", "", regex=True)
        unique_base_conditions = melted_data["base_condition"].unique()
        cmap = plt.get_cmap("tab10") if len(unique_base_conditions) <= 10 else plt.get_cmap("tab20")
        color_map = {base_condition: px.colors.label_rgb((np.array(cmap(i / len(unique_base_conditions))) * 255).astype(int))
                     for i, base_condition in enumerate(unique_base_conditions)}
        return color_map

    def render(self):
        st.title("🦠 Killing Efficiency Analysis Using Unstimulated Control As Baseline")

        # Dropdown for conditions
        selected_conditions = st.multiselect(
            "Select Conditions to Compare",
            options=["All"] + self.treatment_conditions,
            default="All"
        )

        if "All" in selected_conditions:
            selected_conditions = self.treatment_conditions

        # Filter data and display chart
        filtered_data = self.melted_data[self.melted_data["Condition"].isin(selected_conditions)]
        if not filtered_data.empty:
            bar_fig = px.bar(
                filtered_data,
                x="Sample",
                y="BKE%",
                color="Condition",
                title="Unstimulated vs Stimulated: Killing Efficiency",
                labels={"BKE%": "Killing Efficiency (%)", "Condition": "Condition"},
                color_discrete_map=self.color_map,
                hover_data=["Probe", "Date", "BKE%", "Condition"]
            )
            st.plotly_chart(bar_fig)

            # Display data table
            st.write("Data Table:")
            st.dataframe(filtered_data)
        else:
            st.warning("No data available for the selected conditions.")
