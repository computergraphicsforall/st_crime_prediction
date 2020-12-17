import math
import numpy as np
import plotly.graph_objects as go
import json

def bw_intervals(mts, long, lat):
    """
    Function to obtain the intervals in latitude and longitude for crime prediction

    :param mts: bandwidth expected for intervals. This value correspond to the a circle diameter in meters units
    :param long: longitude value of centroid
    :param lat: latitude value of centroid
    :return: two float arrays with the intervals in longitude and latitude for the bandwidth
    """

    """
    Details: 
        1 degree of latitude to equals 111.31 kms in average
        1 degree of longitude depend of latitude value because earth shape
        To calculate a proximal degrees in longitude that would to sum for the bandwidth, we need know the latitude value: 
            1 degree of longitude correspond to  40075 kms * cos(latitude en radians) / 360
    """
    cearth = 40075  # earth circumference in kms
    onedeg_km_lat = cearth / 360  # 1 degree to kms latitude
    mt_km = mts / 1000  # mts to kms
    rad = mt_km * 0.5  # radius for the bandwidth

    onedeg_km_lon = cearth * (math.cos(math.radians(lat)) / 360)

    lat_min = lat - (rad / onedeg_km_lat)
    lat_max = lat + (rad / onedeg_km_lat)

    long_min = long - (rad / onedeg_km_lon)
    long_max = long + (rad / onedeg_km_lon)

    return [long_min, long_max], [lat_min, lat_max]


def distance_points(lat1, lon1, lat2, lon2):
    """
    Function to find the distance between two points in latitude and longitude values

    :param lat1: latitude value of point 1
    :param lon1: longitude value of point 1
    :param lat2: latitude value of point 2
    :param lon2: longitude value of point 2
    :return: the distance in meters between the points
    """

    r = 6378.137  # Radius of earth in KM
    dlat = lat2 * math.pi / 180 - lat1 * math.pi / 180
    dlon = lon2 * math.pi / 180 - lon1 * math.pi / 180
    a = math.sin(dlat / 2) * math.sin(dlat / 2) + math.cos(lat1 * math.pi / 180) * math.cos(
        lat2 * math.pi / 180) * math.sin(dlon / 2) * math.sin(dlon / 2)
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    d = r * c
    return d * 1000  # meters


def get_degrees(mts, lat, t):
    """
    Function to convert meters in degrees

    :param mts: value in meters to convert
    :param lat: reference latitude value
    :param t: parameter to specify whether the degrees are in longitude or latitude. get "lon" for longitude and "lat" for latitude
    :return: degrees in latitude or longitude
    """

    cearth = 40075  # earth circumference in kms
    onedeg_km_lat = cearth / 360  # 1 degree to kms latitude
    mt_km = mts / 1000  # mts to kms

    onedeg_km_lon = cearth * (math.cos(math.radians(lat)) / 360)
    if t == 'lat':
        deg = mt_km / onedeg_km_lat
    if t == 'lon':
        deg = mt_km / onedeg_km_lon
    return deg


def grid_forecast(data, bandwith=500, c_points=False, token_mapbox=None, figure=False):
    x, y, xp, yp = None, None, None, None
    # constants
    add_mts = 2 / 111.32
    min_lat = 4.38981655133336

    max_lat = data['LATITUD'].max()
    min_lon = data['LONGITUD'].min()
    max_lon = data['LONGITUD'].max() + add_mts

    # calculate the distance between the min and max
    dis_lat = distance_points(min_lat, min_lon, max_lat, min_lon)
    dis_lon = distance_points(min_lat, min_lon, min_lat, max_lon)

    # calculate in degrees the distance between points
    step_x = get_degrees(bandwith, min_lat, 'lat')
    step_y = get_degrees(bandwith, min_lat, 'lon')

    # generate the x and y values to the grid
    x = np.arange(min_lon, max_lon, step_x)
    y = np.arange(min_lat, max_lat, step_y)
    x1, y1 = np.meshgrid(x, y)

    if c_points:
        min_xp = min_lon + (step_x / 2)
        min_yp = min_lat + (step_y / 2)
        max_xp = max_lon - (step_x / 2)
        max_yp = max_lat - (step_x / 2)

        xp = np.arange(min_xp, max_xp, step_x)
        yp = np.arange(min_yp, max_yp, step_y)
        x2, y2 = np.meshgrid(xp, yp)

    if figure:
        with open('../data/lcl.geojson') as f:
        lcl = json.load(f)
        if token_mapbox is None:
            token_mapbox = 'pk.eyJ1IjoibWJhcnJlcm9wIiwiYSI6ImNrN28zZjczbTA0ZWwzaXF3aDAxcHl1dGkifQ.DlJqaPP1eC7PF_y-bbWjeg'
        fig = go.Figure()
        for j in range(len(y)):
            fig.add_trace(go.Scattermapbox(mode="markers+lines", lon=x1[0],
                                           lat=y1[j], line={'color': 'rgba(255, 255, 255, 0.71)'},
                                           marker={'size': 10, 'color': 'rgba(203, 255, 92, 0.71)'},
                                           showlegend=False))
        for i in range(len(x)):
            fig.add_trace(go.Scattermapbox(mode="markers+lines",
                                           lon=x1.T[i], lat=y1.T[0],
                                           line={'color': 'rgba(255, 255, 255, 0.71)'},
                                           marker={'size': 10, 'color': 'rgba(203, 255, 92, 0.71)'},
                                           showlegend=False))
        if c_points:
            for j in range(len(yp)):
                fig.add_trace(go.Scattermapbox(mode="markers+lines",
                                               lon=x2[0], lat=y2[j],
                                               marker={'size': 10, 'color': 'red'},
                                               showlegend=False))
            for i in range(len(xp)):
                fig.add_trace(go.Scattermapbox(mode="markers+lines",
                                               lon=x2.T[i], lat=y2.T[0],
                                               marker={'size': 10, 'color': 'red'},
                                               showlegend=False))

            fig.update_layout(mapbox={'style': "dark", 'accesstoken': token_mapbox,
                                      'center': {'lon': -74.081749, 'lat': 4.6097102},
                                      'zoom': 7, 'layers': [{'source': lcl, 'type': "fill", 'below': "traces",
                                                             'color': "rgba(92, 214, 255, 0.5)"}]},
                              margin={'l': 0, 'r': 0, 'b': 0, 't': 0})

        fig.show()
    return (x, y), (xp, yp)
