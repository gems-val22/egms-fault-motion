
import numpy as np
import pandas as pd
import geopandas as gpd

from pathlib import Path
from shapely.geometry import Point
from pyproj import Transformer


def load_world_data(path = 'data/world/Water_Bodies.shp', target_crs = 'EPSG:4326'):

    world = gpd.read_file(path)
    world.set_crs(epsg=4326, inplace=True)
    world = world.to_crs(target_crs)

    return world


def load_faults_gdf(kml_path = None, target_crs = None):

    if kml_path == None:
        kml_path = 'data/sam_mitchell_faults.kml'
        print("DATA BY SAM MITCHELL (2026)")

    if not Path(kml_path).exists():
        raise FileNotFoundError(f"File not found: {kml_path}")

    if target_crs == None:
        target_crs = 'EPSG:4326'
        print("DEFAULT CRS: EPSG:4326")
    
    gdf = gpd.read_file(kml_path, driver='KML')
    gdf = gdf.to_crs(target_crs)

    return gdf


def load_faults_as_dict(kml_path = None):

    gdf = load_faults_gdf(kml_path)
    fault_dict = dict(zip(gdf.Name, gdf.geometry))

    return fault_dict

    
def load_egms_df(egms_path, target_crs = None):

    if not Path(egms_path).exists():
        raise FileNotFoundError(f"File not found: {egms_path}")
    
    if target_crs == None:
        target_crs = 'EPSG:4326'
        print("DEFAULT CRS: EPSG:4326")
    
    df = pd.read_csv(egms_path)

    # Adjust to target coordinate system
    geometry = [Point(xy) for xy in zip(df.easting, df.northing)]
    gdf_3035 = gpd.GeoDataFrame(df, geometry=geometry, crs='EPSG:3035') # EGMS data is in EPSG: 3035
    gdf_adj = gdf_3035.to_crs(target_crs) 

    gdf_adj['longitude'], gdf_adj['latitude'] = gdf_adj.geometry.x, gdf_adj.geometry.y 
    gdf_adj.drop(columns=['geometry', 'easting', 'northing'], inplace=True)

    return gdf_adj


def load_coseismic_displacement_data(path, target_crs = None):

    if not Path(path).exists():
        raise FileNotFoundError(f"File not found: {path}")
    
    if target_crs == None:
        target_crs = 'EPSG:4326'
        print("DEFAULT CRS: EPSG:4326")

    # load coseismic slip data
    pisia_coseismic = pd.read_csv(path, header = 3)
    pisia_coseismic = pisia_coseismic[['Locality ID', 'Eastings (m)', 'Northings (m)', 'Coseismic Throw (cm)']] 
    pisia_coseismic.reset_index(inplace=True, drop=True)
    pisia_coseismic["Coseismic Slip (cm)"] = pd.to_numeric(pisia_coseismic["Coseismic Throw (cm)"], errors="coerce")

    # convert coordinates
    transformer = Transformer.from_crs("EPSG:32634", target_crs)
    pisia_coseismic["latitude"], pisia_coseismic["longitude"] = transformer.transform(
        pisia_coseismic["Eastings (m)"].values,
        pisia_coseismic["Northings (m)"].values)

    # average slip and throw for each locality
    df_dict = {}

    for i,loc in enumerate(pisia_coseismic['Locality ID'].unique()):
        loc_df = pisia_coseismic[pisia_coseismic['Locality ID'] == loc]
        slip = np.nanmean(loc_df['Coseismic Slip (cm)'])
        throw = np.nanmean(loc_df['Coseismic Throw (cm)'])
        df_dict[i] = [loc_df['Locality ID'].iloc[0], loc_df['latitude'].iloc[0], loc_df['longitude'].iloc[0], slip, throw]

    # convert to dataframe
    pisia_coseismic_avg = pd.DataFrame(df_dict).T
    pisia_coseismic_avg.columns = ['Locality ID', 'Latitude', 'Longitude', 'Mean Coseismic Slip (cm)', 'Mean Coseismic Throw (cm)']

    return pisia_coseismic_avg

    