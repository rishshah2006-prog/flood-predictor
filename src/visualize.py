import folium
import os
import json
import pandas as pd


def create_map(gdf, output):

    print("Generating interactive flood risk map")

    gdf_map = gdf.copy()
    gdf_map = gdf_map.to_crs("EPSG:4326")

    for col in ['flood_risk_score', 'IFLD_RISKS', 'CFLD_RISKS', 'SOVI_SCORE']:
        gdf_map[col] = pd.to_numeric(gdf_map[col], errors='coerce').fillna(0)

    gdf_map['flood_risk_score'] = gdf_map['flood_risk_score'].round(4)
    gdf_map['POPULATION'] = gdf_map['POPULATION'].astype(int)

    gdf_map['risk_label'] = gdf_map['flood_risk_score'].apply(
        lambda x: f"🔴 HIGH ({x:.3f})" if x >= 0.6 else (
                  f"🟠 MEDIUM ({x:.3f})" if x >= 0.4 else
                  f"🟢 LOW ({x:.3f})")
    )

    gdf_map['population_fmt'] = gdf_map['POPULATION'].apply(lambda x: f"{x:,}")

    m = folium.Map(
        location=[39.5, -98.35],
        zoom_start=4,
        tiles='CartoDB positron'
    )

    geojson_data = json.loads(gdf_map[['GEOID', 'COUNTY', 'STATE',
                                        'flood_risk_score', 'at_risk',
                                        'POPULATION', 'IFLD_RISKS',
                                        'CFLD_RISKS', 'SOVI_SCORE',
                                        'risk_label', 'population_fmt',
                                        'geometry']].to_json())

    folium.Choropleth(
        geo_data=geojson_data,
        data=gdf_map,
        columns=['GEOID', 'flood_risk_score'],
        key_on='feature.properties.GEOID',
        fill_color='YlOrRd',
        fill_opacity=0.7,
        line_opacity=0.1,
        legend_name='Flood Risk Score (0 = low, 1 = high)',
        nan_fill_color='lightgray'
    ).add_to(m)

    tooltip = folium.GeoJsonTooltip(
        fields=['COUNTY', 'STATE', 'risk_label', 'population_fmt'],
        aliases=['County:', 'State:', 'Risk Level:', 'Population:'],
        localize=True,
        sticky=True,
        style="""
            background-color: white;
            border: 1px solid #ccc;
            border-radius: 6px;
            padding: 8px 12px;
            font-size: 13px;
            font-family: Arial, sans-serif;
            box-shadow: 2px 2px 6px rgba(0,0,0,0.2);
        """
    )

    folium.GeoJson(
        geojson_data,
        style_function=lambda x: {
            'fillOpacity': 0,
            'weight': 0
        },
        tooltip=tooltip
    ).add_to(m)

    os.makedirs(os.path.dirname(output), exist_ok=True)
    m.save(output)
    print(f"Map saved to {output}")
    print("Open that file in your browser to view the map!")


if __name__ == "__main__":
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.data_loader import load_data
    from src.feature_engineering import build_risk_score

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    gdf = load_data(
        shapefile=os.path.join(base, "data/tl_2022_us_county.shp"),
        nri_csv=os.path.join(base, "data/NRI_Table_Counties.csv"),
        precip_csv=os.path.join(base, "data/normals-monthly-precipitation.csv")
    )

    gdf = build_risk_score(gdf)

    create_map(
        gdf,
        output=os.path.join(base, "outputs/flood_risk_map.html")
    )
