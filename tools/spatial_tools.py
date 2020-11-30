import math


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
    a = math.sin(dlat / 2) * math.sin(dlat / 2) + math.cos(lat1 * math.pi / 180) * math.cos(lat2 * math.pi / 180) * math.sin(dlon / 2) * math.sin(dlon / 2)
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

    cearth = 40075                  # earth circumference in kms
    onedeg_km_lat = cearth / 360    # 1 degree to kms latitude
    mt_km = mts / 1000              # mts to kms

    onedeg_km_lon = cearth * (math.cos(math.radians(lat))/360)
    if t == 'lat':
        deg = mt_km / onedeg_km_lat
    if t == 'lon':
        deg = mt_km / onedeg_km_lon
    return deg
