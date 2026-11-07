'''
Module containing functions helping work with coordinates. 
'''

import numpy as np


def degrees_to_radians(angle):
    """
    Converts an angle from degrees to radians.

    Parameters:
    -----------
    angle : float
        The angle in degrees.

    Returns:
    --------
    float
        The angle in radians.
    """
    return angle * np.pi / 180


def km_to_lat(delta_km):
    """
    Converts a distance in kilometers to degrees of latitude.

    This function assumes that one degree of latitude is approximately equal to 111.32 kilometers.

    Parameters:
    -----------
    delta_km : float
        The distance in kilometers.

    Returns:
    --------
    float
        The equivalent change in latitude, in degrees.
    """
    
    return delta_km / 111.32


def km_to_lon(delta_km, lat):
    """
    Converts a distance in kilometers to degrees of longitude, based on a given latitude.

    Since the distance covered by one degree of longitude varies with latitude, this function adjusts the conversion 
    by taking the cosine of the latitude into account.
    This function assumes that one degree of latitude is approximately equal to 111.32 kilometers.

    Parameters:
    -----------
    delta_km : float
        The distance in kilometers.
    lat : float
        The latitude (in degrees) at which the conversion is being made.

    Returns:
    --------
    float
        The equivalent change in longitude, in degrees.
    """
    
    return delta_km / (111.32 * np.cos(np.radians(lat)))


def lat_to_km(delta_lat):
    """
    Converts a change in latitude (in degrees) to a distance in kilometers.

    This function assumes that one degree of latitude is approximately equal to 111.32 kilometers.

    Parameters:
    -----------
    delta_lat : float
        The change in latitude, in degrees.

    Returns:
    --------
    float
        The equivalent distance in kilometers.
    """
    
    return delta_lat * 111.32


def lon_to_km(delta_lon, lat):
    """
    Converts a change in longitude (in degrees) to a distance in kilometers, based on latitude.

    Since the distance covered by one degree of longitude varies with latitude, this function adjusts the conversion 
    by taking the cosine of the latitude into account.
    This function assumes that one degree of latitude is approximately equal to 111.32 kilometers.

    Parameters:
    -----------
    delta_lon : float
        The change in longitude, in degrees.
    lat : float
        The latitude (in degrees) at which the conversion is being made.

    Returns:
    --------
    float
        The equivalent distance in kilometers.
    """
    
    return delta_lon * 111.32 * np.cos(degrees_to_radians(lat))


def euclidean_distance(x1, y1, x2, y2):
    '''
    Calculates the euclidean distance (km) between two points (x1,y1) and (x2,y2) (coordinates). 
    
    Parameters: 
    --------
    x1, y1: float
        Longitude and latitude of point 1.
    x2, y2: float
        Longitude and latitude of point 2.
    
    Returns: 
    --------
    distance: float
        The euclidean distance between the two points in km.
    '''
    
    avg_lat = np.average([y1, y2])
    
    delta_x = lon_to_km(x2 - x1, avg_lat)
    delta_y = lat_to_km(y2 - y1)
    distance = np.sqrt(delta_x**2 + delta_y**2)
    
    return distance