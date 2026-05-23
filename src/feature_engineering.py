import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def build_risk_score(gdf):

    print("Building risk score...")

    features = ['IFLD_RISKS', 'CFLD_RISKS', 'SOVI_SCORE', 'RESL_SCORE']

    for col in features:
        gdf[col] = pd.to_numeric(gdf[col], errors='coerce').fillna(0)

    scaler = MinMaxScaler()
    gdf_scaled = gdf.copy()
    gdf_scaled[features] = scaler.fit_transform(gdf[features])

    # RESL_SCORE is resilience — higher resilience means LOWER risk, so we invert it
    gdf_scaled['flood_risk_score'] = (
        0.40 * gdf_scaled['IFLD_RISKS'] +
        0.25 * gdf_scaled['CFLD_RISKS'] +
        0.20 * gdf_scaled['SOVI_SCORE'] +
        0.15 * (1 - gdf_scaled['RESL_SCORE'])
    )

    # Label counties as high risk (1) or low risk (0)
    threshold = gdf_scaled['flood_risk_score'].quantile(0.75)
    gdf_scaled['at_risk'] = (gdf_scaled['flood_risk_score'] >= threshold).astype(int)

    at_risk_count = gdf_scaled['at_risk'].sum()
    total = len(gdf_scaled)

    print(f"Total counties analyzed: {total:,}")
    print(f"Counties identified at high risk: {at_risk_count:,}")
    print(f"Population at risk: {gdf_scaled[gdf_scaled['at_risk']==1]['POPULATION'].sum():,.0f}")
    print(f"Top 5 highest risk counties:")
    top5 = gdf_scaled.nlargest(5, 'flood_risk_score')[['COUNTY', 'STATE', 'flood_risk_score']]
    print(top5.to_string())

    return gdf_scaled


if __name__ == "__main__":
    import os
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from src.data_loader import load_data

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    gdf = load_data(
        shapefile=os.path.join(base, "data/tl_2022_us_county.shp"),
        nri_csv=os.path.join(base, "data/NRI_Table_Counties.csv"),
        precip_csv=os.path.join(base, "data/normals-monthly-precipitation.csv")
    )

    gdf = build_risk_score(gdf)
    print(gdf[['COUNTY', 'STATE', 'flood_risk_score', 'at_risk']].head(10))