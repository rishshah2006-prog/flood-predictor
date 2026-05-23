import pandas as pd
import geopandas as gpd
import os

def load_data(shapefile, nri_csv, precip_csv):

    print("Loading county shapefile...")
    counties = gpd.read_file(shapefile)
    counties = counties[['GEOID', 'NAME', 'STATEFP', 'geometry']]
    counties = counties.to_crs("EPSG:4326")

    print("Loading FEMA NRI data...")
    nri = pd.read_csv(nri_csv, encoding='latin1', low_memory=False)
    print(nri.columns.tolist())

    nri = nri[[
    'STCOFIPS',
    'COUNTY',
    'STATE',
    'RISK_SCORE',
    'RISK_RATNG',
    'IFLD_RISKS',    # inland flood risk
    'CFLD_RISKS',    # coastal flood risk
    'SOVI_SCORE',    # social vulnerability
    'RESL_SCORE',    # community resilience
    'POPULATION',
    'AREA'
]].copy()

    nri['STCOFIPS'] = nri['STCOFIPS'].astype(str).str.zfill(5)

    print("Merging counties with FEMA data...")
    merged = counties.merge(nri, left_on='GEOID', right_on='STCOFIPS', how='inner')

    print("Loading NOAA precipitation data...")
    precip = pd.read_csv(precip_csv, encoding='latin1', low_memory=False)

    print("Precipitation columns:", precip.columns.tolist())

    if 'MLY-PRCP-NORMAL' in precip.columns:
        precip_grouped = precip.groupby('STATION').agg(
            annual_precip=('MLY-PRCP-NORMAL', 'sum'),
            peak_month_precip=('MLY-PRCP-NORMAL', 'max'),
            lat=('LATITUDE', 'first'),
            lon=('LONGITUDE', 'first')
        ).reset_index()

        precip_gdf = gpd.GeoDataFrame(
            precip_grouped,
            geometry=gpd.points_from_xy(precip_grouped['lon'], precip_grouped['lat']),
            crs="EPSG:4326"
        )

        precip_by_county = gpd.sjoin(precip_gdf, merged[['GEOID', 'geometry']], how='left', predicate='within')
        precip_avg = precip_by_county.groupby('GEOID').agg(
            annual_precip=('annual_precip', 'mean'),
            peak_month_precip=('peak_month_precip', 'mean')
        ).reset_index()

        merged = merged.merge(precip_avg, on='GEOID', how='left')
    else:
        print("Could not find MLY-PRCP-NORMAL column — skipping precipitation merge")
        merged['annual_precip'] = 0
        merged['peak_month_precip'] = 0

    merged = merged.fillna(0)

    print(f"Data loaded successfully — {len(merged):,} counties")
    print(f"Columns: {merged.columns.tolist()}")

    return merged


if __name__ == "__main__":
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    gdf = load_data(
    shapefile=os.path.join(base, "data/tl_2022_us_county.shp"),
    nri_csv=os.path.join(base, "data/NRI_Table_Counties.csv"),
    precip_csv=os.path.join(base, "data/normals-monthly-precipitation.csv")
)

    print(gdf.head())