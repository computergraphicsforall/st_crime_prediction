import pandas as pd
import numpy as np
from pandas.tseries.offsets import MonthEnd
from forecast_time_series import forecast_plots as fplots
from statsmodels.tsa.seasonal import seasonal_decompose
from forecast_time_series import forecast_statistics as fstats
from stldecompose import decompose, forecast
from stldecompose.forecast_funcs import (naive, drift, mean, seasonal_naive)
from data_io import space_time_series as series

"""
Global vars
Start and end date of nuse dataset. You always need specify both dates by the init and final year
"""
fi_dataset, ff_dataset = '2014', '2020'


def montly_forecast(fid, ffd, fie, ffe, dfn, bwt, lon, lat, n_months=None, graphics=False, period=None):
    """
    Function to forecast in months

    :param fid: data start date
    :param ffd: data end date
    :param fie: training start date
    :param ffe: training end date
    :param dfn: filtered NUSE dataframe
    :param bwt: bandwidth in meters. This value corresponds to the diameter of the area over which the prediction will be made
    :param lon: value in longitude of the center point on which the prediction is calculated
    :param lat: value in latitude of the center point on which the prediction is calculated
    :param n_months: number of months to predict
    :param graphics: inclusion of graphs in the results True to return graphs, false to not return graphs
    :param period: number of periods for decomposition time series
    :return: statistical and graphical results of prediction
    """
    stats_fcast, fig = None, None
    fli = pd.to_datetime(fi_dataset)  # fecha límite inicial
    flf = pd.to_datetime(ff_dataset)  # fecha límite final
    fid = pd.to_datetime(fid)
    ffd = pd.to_datetime(ffd)
    fie = pd.to_datetime(fie)
    ffe = pd.to_datetime(ffe)
    fie1, ffe1 = None, None
    pred = False

    if (fid >= fli and ffd < flf) and (fie >= fid and ffe <= ffd):
        pred = True

    if pred:

        if fid.day != 1:
            dias = pd.to_datetime(fid).day
            fid = pd.to_datetime(fid) - pd.DateOffset(days=dias - 1)

        if ffd.day != 1:
            dias = pd.to_datetime(ffd).day
            ffd = pd.to_datetime(ffd) - pd.DateOffset(days=dias - 1)

        if not fie.is_month_end:
            fie1 = pd.to_datetime(fie) + MonthEnd(1)

        if not ffe.is_month_end:
            ffe1 = pd.to_datetime(ffe) + MonthEnd(1)

        if n_months is None:
            n_months = 12

        # forecast period
        fip1 = ffe1 + pd.DateOffset(months=1)
        ffp1 = ffe1 + pd.DateOffset(months=n_months)
        # get data
        d_men = series.st_series(fi_dataset, ff_dataset, dfn, 0, 0, 0, bwt, lon, lat)
        d_dmt = d_men[fid:ffd]  # full data with the incidents between the time and the space (bandwidth)
        d_dme = d_men[fie1:ffe1]  # data trainining between dates

        if period is None:
            period = 12

        dfdm = pd.DataFrame(d_dme['EVENTS'])  # dataframe to time series decomposition
        decm = decompose(dfdm, period=period)  # time series decompositon

        ptm = forecast(decm, steps=n_months, fc_func=drift)
        pem = forecast(decm, steps=n_months, fc_func=drift, seasonal=True)
        index = pd.date_range(fip1, ffp1, freq='M')
        pem.index = index
        ptm.index = index
        # calculate the statistics to forecast
        stats_fcast = fstats.get_statistics(ffd, fip1, ffp1, fie1, ptm, pem, d_dmt)
        stats_fcast['fcast'] = np.round(pem, 2)
        if graphics:
            fig = fplots.forecast_graphs(d_dmt, d_dme, ptm, pem, decm, ffe1, fip1, 'Monthly ' + str(bwt), False)

    return stats_fcast, fig


# parámetros predicción a para días (máximo 7)
# fid: fecha inicial del total de conjunto de datos
# ffd: fecha final del conjunto de datos
# fie: fecha inicial del cojunto de datos de entrenamiento
# ffe: fecha final del conjunto de datos de entrenamiento
# cdias: días adicionales a partir del ndia
# dfn: dataframe de nuse filtrado
# bwt: ancho de banda en metros
# lon: coordenada longitudinal del punto de interés
# lat: coordenada latitudinal del punto de interés
def daily_forecast(fid, ffd, fie, ffe, dfn, bwt, lon, lat, n_days=1, graphics=False, period=None):
    """

    :param fid:
    :param ffd:
    :param fie:
    :param ffe:
    :param dfn:
    :param bwt:
    :param lon:
    :param lat:
    :param n_days:
    :param graphics:
    :param period:
    :return:
    """
    stats_fcast, fig = None, None
    fli = pd.to_datetime(fi_dataset)  # fecha límite inicial
    flf = pd.to_datetime(ff_dataset)  # fecha límite final
    fid = pd.to_datetime(fid)
    ffd = pd.to_datetime(ffd)
    fie = pd.to_datetime(fie)
    ffe = pd.to_datetime(ffe)
    pred = False

    if (fid >= fli and ffd < flf) and (fie >= fid and ffe <= ffd) and (n_days >= 0):
        pred = True

    if pred:
        # ajustando las fechas para los datos
        if pd.to_datetime(fid).dayofweek != 0:
            dias = pd.to_datetime(fid).dayofweek
            fid = pd.to_datetime(fid) - pd.DateOffset(days=dias)

        if pd.to_datetime(ffd).dayofweek != 0:
            dias = pd.to_datetime(ffd).dayofweek
            ffd = pd.to_datetime(ffd) - pd.DateOffset(days=dias)

        # forcast period
        fip1 = pd.to_datetime(ffe) + pd.DateOffset(days=1)
        ffp1 = pd.to_datetime(fip1) + pd.DateOffset(days=n_days - 1)
        # get data
        d_dc = series.st_series(fi_dataset, ff_dataset, dfn, 2, 0, 0, bwt, lon, lat)
        d_dct = d_dc[fid:ffd]  # full data with the incidents between the time and the space (bandwidth)
        d_dce = d_dc[fie:ffe]  # data trainining between dates

        if period is None:
            period = 28
        dfdd = pd.DataFrame(d_dce['EVENTS'])  # dataframe to time series decomposition
        decd = decompose(dfdd, period=period)  # time series decompositon

        # forcast
        ptd = forecast(decd, steps=n_days, fc_func=drift)
        ped = forecast(decd, steps=n_days, fc_func=drift, seasonal=True)
        # calculate the statistics to forecast
        print(fid, ffd, fip1, ffp1, fie)

        stats_fcast = fstats.get_statistics(ffd, fip1, ffp1, fie, ptd, ped, d_dct)
        stats_fcast['fcast'] = np.round(ped, 2)
        if graphics:
            fig = fplots.forecast_graphs(d_dct, d_dce, ptd, ped, decd, ffe, fip1, 'Daily ' + str(bwt), False)

        return stats_fcast, fig
