# FACS & Bacterial Killing Assay Data Visualization

## Project Overview
This project provides an interactive Python-based tool to visualize and analyze FACS data and Bacterial Killing Assay results.  
FACS data gating and MFI calculations are performed in FlowJo, while the visualization leverages **Plotly** for interactive exploration. The tool also generates intuitive plots and tables for bacterial killing assay experiments.

## Features
- **FACS Data Visualization & Gating**
  - Overlay gating strategies from FlowJo workspace files (XML) onto interactive Plotly plots.
  - Display polygon and threshold-based gating visually for each sample.
  - View MFI (Mean Fluorescence Intensity) per gated population.

- **Bacterial Killing Assay Analysis**
  - Interactive boxplots and bar plots for bacterial killing data.
  - Summary statistics tables for quick insights.
  - Supports multiple manually filtered datasets.

## Technologies & Libraries
- Python 3.x  
- [Plotly](https://plotly.com/python/) – Interactive plotting  
- [Pandas](https://pandas.pydata.org/) – Data manipulation  
- [ElementTree](https://docs.python.org/3/library/xml.etree.elementtree.html) – Parsing FlowJo workspace XML  
- Optional: [Dash](https://dash.plotly.com/) for web-based interactive dashboards  


