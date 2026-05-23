# Flood Risk Predictor

Analyzed flood vulnerability across **3,200+ US counties** using geospatial
data and machine learning to identify high-risk communities.

## Key Results
- Mapped FEMA National Risk Index scores across all US counties
- Built a Random Forest classifier to predict high-risk counties
- Integrated social vulnerability scores and precipitation normals
- Generated interactive choropleth maps

## Tech Stack
Python · GeoPandas · scikit-learn · Plotly · Folium · Streamlit

## Data Sources
- [FEMA National Risk Index](https://hazards.fema.gov/nri/)
- [Census TIGER Shapefiles](https://www.census.gov/geographies/mapping-files)
- [NOAA Climate Normals](https://www.ncei.noaa.gov)

## How to Run
1. Download data files (links above) into `data/`
2. pip install -r requirements.txt
3. python main.py
