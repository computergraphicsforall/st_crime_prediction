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
    fli = pd.to_datetime(fi_dataset)
    flf = pd.to_datetime(ff_dataset)
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
            dias = fid.day
            fid = fid - pd.DateOffset(days=dias - 1)

        if ffd.day != 1:
            dias = ffd.day
            ffd = ffd - pd.DateOffset(days=dias - 1)

        if not fie.is_month_end:
            fie1 = fie + MonthEnd(1)

        if not ffe.is_month_end:
            ffe1 = ffe + MonthEnd(1)

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


def weekly_forecast(fid, ffd, fie, ffe, dfn, bwt, lon, lat, n_weeks=1, graphics=False, period=None):
    """
    Function to forecast in weeks

    :param fid: data start date
    :param ffd: data end date
    :param fie: training start date
    :param ffe: training end date
    :param dfn: filtered NUSE dataframe
    :param bwt: bandwidth in meters. This value corresponds to the diameter of the area over which the prediction will be made
    :param lon: value in longitude of the center point on which the prediction is calculated
    :param lat: value in latitude of the center point on which the prediction is calculated
    :param n_weeks: number of weeks to predict
    :param graphics: inclusion of graphs in the results True to return graphs, false to not return graphs
    :param period: number of periods for decomposition time series
    :return: statistical and graphical results of prediction
    """
    stats_fcast, fig = None, None
    fli = pd.to_datetime(fi_dataset)
    flf = pd.to_datetime(ff_dataset)
    fid = pd.to_datetime(fid)
    ffd = pd.to_datetime(ffd)
    fie = pd.to_datetime(fie)
    ffe = pd.to_datetime(ffe)
    pred = False

    if (fid >= fli and ffd < flf) and (fie >= fid and ffe <= ffd):
        pred = True

    if pred:

        if fid.dayofweek != 0:
            dias = fid.dayofweek
            fid = fid - pd.DateOffset(days=dias)

        if ffd.dayofweek != 0:
            dias = ffd.dayofweek
            ffd = ffd - pd.DateOffset(days=dias)

        if fie.dayofweek != 0:
            dias = fie.dayofweek
            fie = fie - pd.DateOffset(days=dias)

        if ffe.dayofweek != 0:
            dias = ffe.dayofweek
            ffe = ffe - pd.DateOffset(days=dias)

        d_sem = series.st_series(fi_dataset, ff_dataset, dfn, 1, 0, 0, bwt, lon, lat)
        # forecast period
        fips = ffe + pd.DateOffset(weeks=1)
        ffps = fips + pd.DateOffset(weeks=n_weeks - 1)
        # get data
        d_tots = d_sem[fid:ffd]  # full data with the incidents between the time and the space (bandwidth)
        d_ents = d_sem[fie:ffe]  # data trainining between dates

        if period is None:
            period = 52

        dfds = pd.DataFrame(d_ents['EVENTS'])  # dataframe to time series decomposition
        decs = decompose(dfds, period=period)  # time series decompositon

        # predicción tendencia y estacional para las semanas
        pts = forecast(decs, steps=n_weeks, fc_func=drift)
        pes = forecast(decs, steps=n_weeks, fc_func=drift, seasonal=True)

        # calculate the statistics to forecast
        stats_fcast = fstats.get_statistics(ffd, fips, ffps, fie, pts, pes, d_tots)
        stats_fcast['fcast'] = np.round(pes, 2)

        if graphics:
            fig = fplots.forecast_graphs(d_tots, d_ents, pts, pes, decs, ffe, fips, 'Weekly ' + str(bwt), False)

    return stats_fcast, fig


def daily_forecast(fid, ffd, fie, ffe, dfn, bwt, lon, lat, n_days=1, graphics=False, period=None):
    """
    Function to forecast in days

    :param fid: data start date
    :param ffd: data end date
    :param fie: training start date
    :param ffe: training end date
    :param dfn: filtered NUSE dataframe
    :param bwt: bandwidth in meters. This value corresponds to the diameter of the area over which the prediction will be made
    :param lon: value in longitude of the center point on which the prediction is calculated
    :param lat: value in latitude of the center point on which the prediction is calculated
    :param n_days: number of days to predict
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
        stats_fcast = fstats.get_statistics(ffd, fip1, ffp1, fie, ptd, ped, d_dct)
        stats_fcast['fcast'] = np.round(ped, 2)
        if graphics:
            fig = fplots.forecast_graphs(d_dct, d_dce, ptd, ped, decd, ffe, fip1, 'Daily ' + str(bwt), False)

    return stats_fcast, fig


def next_day_continous_forecast(x_points, y_points, sdt, edt, data, period=None):
    stats_fcast, fig = list(), None
    pred = False
    matrix_fcast = np.zeros((len(y_points), len(x_points)))
    matrix_obser = np.zeros((len(y_points), len(x_points)))

    fli = pd.to_datetime('2014')  # fecha límite inicial
    flf = pd.to_datetime('2020')  # fecha límite final
    fid = pd.to_datetime(data.index[0])
    ffd = pd.to_datetime(data.index[-1])
    fie = pd.to_datetime(sdt)
    ffe = pd.to_datetime(edt)

    if (fid >= fli and ffd < flf) and (fie >= fid and ffe <= ffd):
        pred = True

    if pred:

        if fid.dayofweek != 0:
            dias = fid.dayofweek
            fid = fid - pd.DateOffset(days=dias)

        if pd.to_datetime(ffd).dayofweek != 0:
            dias = ffd.dayofweek
            ffd = ffd - pd.DateOffset(days=dias)

        # forecast period
        fip1 = pd.to_datetime(ffe) + pd.DateOffset(days=1)
        ffp1 = pd.to_datetime(fip1) + pd.DateOffset(days=1 - 1)
        fip1_str = fip1.strftime("%Y-%m-%d")

        if period is None:
            period = 28

        cols = data.columns
        # data trainining between dates
        d_dce = data[fie:ffe]

        for i in range(len(cols)):

            cx = cols[i][0]
            cy = cols[i][1]

            decd = decompose(d_dce[cx, cy], period=period)  # seasonal decompose
            ptd = forecast(decd, steps=1, fc_func=drift)  # trend forecast
            ped = forecast(decd, steps=1, fc_func=drift, seasonal=True)  # seasonal forecast

            if float(ped['drift+seasonal']) > 0:
                matrix_fcast[cy][cx] = float(ped['drift+seasonal'])

            matrix_obser[cy][cx] = data.loc[fip1_str][cx, cy]
        return matrix_fcast, matrix_obser


def montly_forecast_for_point(data, start_training, end_training, bandwidth, n_months=1, graphics=False, period=None):
    """
    Function to forecast in months

    :param data: NUSE dataframe transform
    :param start_training: start date for training
    :param end_training: end date for training
    :param bandwidth: bandwidth in meters. This value corresponds to the diameter of the area over which the prediction will be made
    :param n_months: number of months to predict
    :param graphics: inclusion of graphs in the results True to return graphs, false to not return graphs
    :param period: number of periods for decomposition time series
    :return: statistical and graphical results of prediction
    """
    stats_fcast, fig = None, None
    fli = pd.to_datetime(fi_dataset)
    flf = pd.to_datetime(ff_dataset)
    fid = pd.to_datetime(data.index[0])
    ffd = pd.to_datetime(data.index[-1])
    fie = pd.to_datetime(start_training)
    ffe = pd.to_datetime(end_training)
    fie1, ffe1 = None, None
    pred = False

    if (fid >= fli and ffd < flf) and (fie >= fid and ffe <= ffd):
        pred = True

    if pred:

        if fid.day != 1:
            dias = fid.day
            fid = fid - pd.DateOffset(days=dias - 1)

        if ffd.day != 1:
            dias = ffd.day
            ffd = ffd - pd.DateOffset(days=dias - 1)

        if not fie.is_month_end:
            fie1 = fie + MonthEnd(1)

        if not ffe.is_month_end:
            ffe1 = ffe + MonthEnd(1)

        # forecast period
        fip1 = ffe1 + pd.DateOffset(months=1)
        ffp1 = ffe1 + pd.DateOffset(months=n_months)

        # get data
        d_dmt = data[fid:ffd]  # full data with the incidents between the time and the space (bandwidth)
        d_dme = data[fie1:ffe1]  # data trainining between dates

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
            fig = fplots.forecast_graphs(d_dmt, d_dme, ptm, pem, decm, ffe1, fip1, 'Monthly ' + str(bandwidth), False)

    return stats_fcast, fig


def weekly_forecast_for_point(data, start_training, end_training, bandwidth, n_weeks=1, graphics=False, period=None):
    """
    Function to forecast in weeks

    :param data: NUSE dataframe transform
    :param start_training: start date for training
    :param end_training: end date for training
    :param bandwidth: bandwidth in meters. This value corresponds to the diameter of the area over which the prediction will be made
    :param n_weeks: number of weeks to predict
    :param graphics: inclusion of graphs in the results True to return graphs, false to not return graphs
    :param period: number of periods for decomposition time series
    :return: statistical and graphical results of prediction
    """
    stats_fcast, fig = None, None
    fid = pd.to_datetime(data.index[0])
    ffd = pd.to_datetime(data.index[-1])
    fie = pd.to_datetime(start_training)
    ffe = pd.to_datetime(end_training)

    if True:

        if fid.dayofweek != 0:
            dias = fid.dayofweek
            fid = fid - pd.DateOffset(days=dias)

        if ffd.dayofweek != 0:
            dias = ffd.dayofweek
            ffd = ffd - pd.DateOffset(days=dias)

        if fie.dayofweek != 0:
            dias = fie.dayofweek
            fie = fie - pd.DateOffset(days=dias)

        if ffe.dayofweek != 0:
            dias = ffe.dayofweek
            ffe = ffe - pd.DateOffset(days=dias)

        # forecast period
        fips = ffe + pd.DateOffset(weeks=1)
        ffps = ffe + pd.DateOffset(weeks=n_weeks)
        # get data
        d_tots = data[fid:ffd]  # full data with the incidents between the time and the space (bandwidth)
        d_ents = data[fie:ffe]  # data trainining between dates

        if period is None:
            period = 52

        dfds = pd.DataFrame(d_ents['EVENTS'])  # dataframe to time series decomposition
        decs = decompose(dfds, period=period)  # time series decompositon

        # forecast
        pts = forecast(decs, steps=n_weeks, fc_func=drift)
        pes = forecast(decs, steps=n_weeks, fc_func=drift, seasonal=True)

        # calculate the statistics to forecast
        stats_fcast = fstats.get_statistics(ffd, fips, ffps, fie, pts, pes, d_tots)
        stats_fcast['fcast'] = np.round(pes, 2)

        if graphics:
            fig = fplots.forecast_graphs(d_tots, d_ents, pts, pes, decs, ffe, fips, 'Weekly ' + str(bandwidth), False)

    return stats_fcast, fig


# medium term
def daily_forecast_for_point(data, start_training, end_training, bandwidth, n_days=1, graphics=False, period=None):
    """
    Function to forecast in days

    :param data: NUSE dataframe transform
    :param start_training: start date for training
    :param end_training: end date for training
    :param bandwidth: bandwidth in meters. This value corresponds to the diameter of the area over which the prediction will be made
    :param n_days: number of days to predict
    :param graphics: inclusion of graphs in the results True to return graphs, false to not return graphs
    :param period: number of periods for decomposition time series
    :return: statistical and graphical results of prediction
    """
    stats_fcast, fig = None, None
    fli = pd.to_datetime(fi_dataset)  # fecha límite inicial
    flf = pd.to_datetime(ff_dataset)  # fecha límite final
    fid = pd.to_datetime(data.index[0])
    ffd = pd.to_datetime(data.index[-1])
    fie = pd.to_datetime(start_training)
    ffe = pd.to_datetime(end_training)
    pred = False

    if (fid >= fli and ffd < flf) and (fie >= fid and ffe <= ffd) and (n_days >= 0):
        pred = True

    if pred:
        # date adjust
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
        d_dct = data[fid:ffd]  # full data with the incidents between the time and the space (bandwidth)
        d_dce = data[fie:ffe]  # data trainining between dates

        if period is None:
            period = 28
        dfdd = pd.DataFrame(d_dce['EVENTS'])  # dataframe to time series decomposition
        decd = decompose(dfdd, period=period)  # time series decompositon

        # forcast
        ptd = forecast(decd, steps=n_days, fc_func=drift)
        ped = forecast(decd, steps=n_days, fc_func=drift, seasonal=True)

        # calculate the statistics to forecast
        stats_fcast = fstats.get_statistics(ffd, fip1, ffp1, fie, ptd, ped, d_dct)
        stats_fcast['fcast'] = np.round(ped, 2)
        if graphics:
            fig = fplots.forecast_graphs(d_dct, d_dce, ptd, ped, decd, ffe, fip1, 'Daily ' + str(bandwidth), False)

    return stats_fcast, fig


# medium term
def weekday_daily_forecast_for_point(data, start_training, end_training, bandwidth, n_days=7, graphics=False,
                                     period=None):
    """
    Function to forecast weeks with a specific days composition

    :param data: NUSE dataframe transform
    :param start_training: start date for training
    :param end_training: end date for training
    :param bandwidth: bandwidth in meters. This value corresponds to the diameter of the area over which the prediction will be made
    :param n_days: number of days to predict
    :param graphics: inclusion of graphs in the results True to return graphs, false to not return graphs
    :param period: number of periods for decomposition time series
    :return: statistical and graphical results of prediction
    """
    stats_fcast, fig = None, None
    fli = pd.to_datetime(fi_dataset)  # fecha límite inicial
    flf = pd.to_datetime(ff_dataset)  # fecha límite final
    fid = pd.to_datetime(data.FECHA.min())
    ffd = pd.to_datetime(data.FECHA.max())
    fid = pd.to_datetime(str(fid.year) + '-' + str(fid.month) + '-' + str(fid.day))
    ffd = pd.to_datetime(str(ffd.year) + '-' + str(ffd.month) + '-' + str(ffd.day))
    fie = pd.to_datetime(start_training)
    ffe = pd.to_datetime(end_training)
    pred = False

    if (fid >= fli and ffd < flf) and (fie >= fid and ffe <= ffd) and (n_days >= 7) and (n_days % 7 == 0):
        pred = True
    else:
        print(
            'Check the traning dates and the number of days. For this function the number of days must be a multiple of seven.')

    if pred:
        # date adjust
        if fid.dayofweek != 0:
            dias = fid.dayofweek
            fid = fid - pd.DateOffset(days=dias)

        if ffd.dayofweek != 0:
            dias = ffd.dayofweek
            ffd = ffd - pd.DateOffset(days=dias)

        dpt = pd.DataFrame()  # df of accumulated trend forecast
        dpe = pd.DataFrame()  # df of accumulated seasonal forecast
        dtt = pd.DataFrame()  # df of the total data
        dte = pd.DataFrame()  # df of the training data
        dec = pd.DataFrame()  # df of accumulated decomposition

        fiede = fie
        ffede = ffe
        # forcast period
        fip1 = pd.to_datetime(ffede) + pd.DateOffset(days=1)
        ffp1 = pd.to_datetime(fip1) + pd.DateOffset(days=n_days - 1)

        # start day of the week
        di = fip1.dayofweek
        # weeks forecast
        week_fc = int(n_days / 7)

        if period is None:
            period = 52

        # get data and forecast
        for i in range(7):

            dias = ffd.dayofweek
            ffd = ffd + pd.DateOffset(days=abs(i - dias))

            d_de = series.time_series_by_point(data=data, t_serie='weekday', start_data_date=fid, end_data_date=ffd,
                                               weekday=i)

            d_totde = d_de[fid:ffd]      # dataframe with total data of events specific day
            d_entde = d_de[fiede:ffede]  # dataframe with training data on a specific day

            # df decomposition of the time series for a specific day
            # (Mon, Tue, Wed, Thu, Fri, Sat, Sun)
            dfdde = pd.DataFrame(d_entde['EVENTS'])  # dataframe to time series decomposition
            decde = decompose(dfdde, period=period)  # time series decompositon

            # forecast
            ptde = forecast(decde, steps=week_fc, fc_func=drift)
            pede = forecast(decde, steps=week_fc, fc_func=drift, seasonal=True)

            # accumulating the predictions
            dpt = dpt.append(ptde)
            dpe = dpe.append(pede)
            dtt = dtt.append(d_totde)
            dte = dte.append(d_entde)

            dect = pd.DataFrame(
                {'seasonal': decde.seasonal.EVENTS, 'resid': decde.resid.EVENTS, 'trend': decde.trend.EVENTS},
                index=dfdde.index)
            dec = dec.append(dect)

            fiede = pd.to_datetime(fiede) + pd.DateOffset(days=1)
            ffede = pd.to_datetime(ffede) + pd.DateOffset(days=1)
            di = di + 1
            if di > 6:
                di = 0

        # sort the forecast (1-Mon, 1-Tue, 1-Wed, 1-Thu, ..., 2-Mon, 2-Tue....)
        dpt = dpt.sort_index()
        dpe = dpe.sort_index()
        dtt = dtt.sort_index()
        dte = dte.sort_index()
        dec = dec.sort_index()

        # calculate the statistics to forecast
        stats_fcast = fstats.get_statistics(ffd, fip1, ffp1, fie, dpt, dpe, dtt)
        stats_fcast['fcast'] = np.round(dpe, 2)
        if graphics:
            fig = fplots.forecast_graphs(dtt, dte, dpt, dpe, dec, ffe, fip1, 'Weekday Daily ' + str(bandwidth), True)

    return stats_fcast, fig


def daily_turny_forecast_for_point(data, start_training, end_training, bandwidth, n_days=1, graphics=False, period=None):
    """
    Function to forecast days in mode tourny (early morning, morning, afternoon and night)

    :param data: NUSE dataframe transform
    :param start_training: start date for training
    :param end_training: end date for training
    :param bandwidth: bandwidth in meters. This value corresponds to the diameter of the area over which the prediction will be made
    :param n_days: number of days to predict
    :param graphics: inclusion of graphs in the results True to return graphs, false to not return graphs
    :param period: number of periods for decomposition time series
    :return: statistical and graphical results of prediction
    """
    stats_fcast, fig = None, None
    fli = pd.to_datetime(fi_dataset)  # fecha límite inicial
    flf = pd.to_datetime(ff_dataset)  # fecha límite final
    fid = pd.to_datetime(data.index[0])
    ffd = pd.to_datetime(data.index[-1])
    fie = pd.to_datetime(start_training)
    ffe = pd.to_datetime(end_training)
    pred = False

    if (fid >= fli and ffd < flf) and (fie >= fid and ffe <= ffd) and (n_days > 0):
        pred = True

    if pred:
        # date adjust
        if fid.dayofweek != 0:
            dias = fid.dayofweek
            fid = fid - pd.DateOffset(days=dias)

        if ffd.dayofweek != 0:
            dias = ffd.dayofweek
            ffd = ffd - pd.DateOffset(days=dias)

        fie2 = fie
        ffe2 = ffe + pd.DateOffset(hours=18)

        # forcast period
        fip2 = ffe2 + pd.DateOffset(days=1) - pd.DateOffset(hours=18)
        ffp2 = fip2 + pd.DateOffset(days=n_days-1) + pd.DateOffset(hours=18)

        if period is None:
            period = 28

        d_jct = data[fid:ffd]     # dataframe with total data of events in turnys
        d_jce = data[fie2:ffe2]   # dataframe with training data of events in turnys

        dfdj = pd.DataFrame(d_jce['EVENTS'])  # dataframe to time series decomposition in turnys in a complete day
        decj = decompose(dfdj, period=period)     # time series decomposition

        ptj = forecast(decj, steps=(4*n_days), fc_func=drift)
        pej = forecast(decj, steps=(4*n_days), fc_func=drift, seasonal=True)

        stats_fcast = fstats.get_statistics(ffd, fip2, ffp2, fie2, ptj, pej, d_jct)
        stats_fcast['fcast'] = np.round(pej, 2)
        if graphics:
            fig = fplots.forecast_graphs(d_jct, d_jce, ptj, pej, decj, ffe2, fip2, 'Turny ' + str(bandwidth), False)

    return stats_fcast, fig


def weekday_one_turny_forecast_for_point(data, start_training, end_training, bandwidth, n_days=1, graphics=False,
                                         period=None):
    """
    Function to predict days based on the composition of time series where only the specific day and shift is used to predict the next day in that same turn
    until completing the four turns of the day. This composition is extended to the number of days requested.
    The result of the composition is the prediction for the number of days requested using only the specific day and specific turn until completing
    the 4 existing shifts (early morning, morning, afternoon and night) for each requested day.


    :param data: NUSE dataframe transform
    :param start_training: start date for training
    :param end_training: end date for training
    :param bandwidth: bandwidth in meters. This value corresponds to the diameter of the area over which the prediction will be made
    :param n_days: number of days to predict
    :param graphics: inclusion of graphs in the results True to return graphs, false to not return graphs
    :param period: number of periods for decomposition time series
    :return: statistical and graphical results of prediction
    """
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    stats_fcast, fig = None, None
    fli = pd.to_datetime(fi_dataset)  # fecha límite inicial
    flf = pd.to_datetime(ff_dataset)  # fecha límite final
    fid = pd.to_datetime(data.FECHA.min())
    ffd = pd.to_datetime(data.FECHA.max())
    fid = pd.to_datetime(str(fid.year) + '-' + str(fid.month) + '-' + str(fid.day))
    ffd = pd.to_datetime(str(ffd.year) + '-' + str(ffd.month) + '-' + str(ffd.day))
    fie = pd.to_datetime(start_training)
    ffe = pd.to_datetime(end_training)
    pred = False

    if (fid >= fli and ffd < flf) and (fie >= fid and ffe <= ffd) and (n_days >= 1):
        pred = True
    else:
        print(
            'Check the traning dates and the number of days. For this function the min number of days is one.')

    if pred:
        # date adjust
        if fid.dayofweek != 0:
            dias = fid.dayofweek
            fid = fid - pd.DateOffset(days=dias)

        if ffd.dayofweek != 0:
            dias = ffd.dayofweek
            ffd = ffd - pd.DateOffset(days=dias)

        de = ffe.dayofweek
        fie7 = fie

        if fie7.dayofweek != de:
            dias = fie7.dayofweek
            fie7 = fie7 - pd.DateOffset(days=dias)
            fie7 = fie7 + pd.DateOffset(days=de) + pd.DateOffset(days=1)
        else:
            fie7 = fie7 + pd.DateOffset(days=1)

        ffe7 = ffe + pd.DateOffset(days=1, weeks=-1) + pd.DateOffset(hours=18)
        fip7 = ffe + pd.DateOffset(days=1)
        ffp7 = fip7 + pd.DateOffset(days=n_days-1) + pd.DateOffset(hours=18)

        fied = fie7
        ffed = ffe7 - pd.DateOffset(hours=18)

        if period is None:
            period = 52

        ddt = pd.DataFrame()  # df of total data
        dpt = pd.DataFrame()  # df of accumulated trend forecast
        dpe = pd.DataFrame()  # df of accumulated seasonal forecast
        dte = pd.DataFrame()  # df of accumulated traning data
        dec = pd.DataFrame()  # df of accumulated decomposition data

        ndia = ffed.dayofweek
        lst_data = {}         # dict of total data training for specific day
        if n_days > 7:
            arr_data = 7
        else:
            arr_data = n_days

        for i in range(arr_data):
            if ndia > 6:
                ndia = 0
            lst_data['data_'+str(ndia)] = series.time_series_by_point(data=data, t_serie='weekday-full-turny',
                                                                      start_data_date=fid, end_data_date=ffd,
                                                                      weekday=ndia)
            ndia = ndia + 1

        ndia = ffed.dayofweek
        for d in range(n_days):

            if ndia > 6:
                ndia = 0
            d_dtjc = lst_data['data_'+str(ndia)]
            d_dtjce = d_dtjc[fie7 + pd.DateOffset(days=d):ffe7 + pd.DateOffset(days=d)]   # data training of the day
            ddt = ddt.append(d_dtjc)

            for i in range(4):
                # adjustment of the index to the focal day to be able to predict that day in the turn
                fiej = fied + pd.DateOffset(hours=6*i)
                ffej = ffed + pd.DateOffset(hours=6*i)
                index = pd.date_range(fiej, ffej, freq='W-'+days[ndia].upper())

                # filtering the data by turn
                dj = d_dtjce[d_dtjce['JORN'] == i]
                dje = dj[fiej:ffej]

                # updating the index by turn and specific day
                dje.index = index

                dfdj = pd.DataFrame(dje['EVENTS'])      # dataframe to time series decomposition
                decj = decompose(dfdj, period=period)   # time series decompositon

                # forecast
                ptj = forecast(decj, steps=1, fc_func=drift)
                pej = forecast(decj, steps=1, fc_func=drift, seasonal=True)

                # accumulating the predictions
                dte = dte.append(dje)
                dpt = dpt.append(ptj)
                dpe = dpe.append(pej)
                dect = pd.DataFrame({'seasonal': decj.seasonal.EVENTS, 'resid': decj.resid.EVENTS, 'trend': decj.trend.EVENTS}, index=dfdj.index)
                dec = dec.append(dect)

            ndia = ndia + 1
            fied = pd.to_datetime(fied) + pd.DateOffset(days=1)
            ffed = pd.to_datetime(ffed) + pd.DateOffset(days=1)

        # sorting
        dpt = dpt.sort_index()
        dpe = dpe.sort_index()
        dte = dte.sort_index()
        dec = dec.sort_index()
        ddt = ddt.sort_index()

        # calculate the statistics to forecast
        stats_fcast = fstats.get_statistics(ffd, fip7, ffp7, fie7, dpt, dpe, ddt)
        stats_fcast['fcast'] = np.round(dpe, 2)
        if graphics:
            fig = fplots.forecast_graphs(ddt, dte, dpt, dpe, dec, ffe7 + pd.DateOffset(days=n_days-1), fip7, 'Weekday 1-Turny ' + str(bandwidth), True)

    return stats_fcast, fig


def weekday_four_turny_forecast_for_point(data, start_training, end_training, bandwidth, n_days=7, graphics=False,
                                          period=None):

    """
    Function to predict days based on the composition of time series where all the days in a specific turn are used to predict the following days in that same turn.
    The result of the composition is the prediction for the number of days requested in the four existing turns (early morning, morning, afternoon and night)

    Example:

    Data training                                                   Forecast
    Day_1_turn_0, Day_2_turn_0, Day_3_turn_0, Day_N_turn_0 ----->   Day_N+1_turn_0 to n_days
    Day_1_turn_1, Day_2_turn_1, Day_3_turn_1, Day_N_turn_1 ----->   Day_N+1_turn_2 to n_days
    Day_1_turn_2, Day_2_turn_2, Day_3_turn_2, Day_N_turn_2 ----->   Day_N+1_turn_2 to n_days
    Day_1_turn_3, Day_2_turn_3, Day_3_turn_3, Day_N_turn_3 ----->   Day_N+1_turn_3 to n_days

    :param data: NUSE dataframe transform
    :param start_training: start date for training
    :param end_training: end date for training
    :param bandwidth: bandwidth in meters. This value corresponds to the diameter of the area over which the prediction will be made
    :param n_days: number of days to predict
    :param graphics: inclusion of graphs in the results True to return graphs, false to not return graphs
    :param period: number of periods for decomposition time series
    :return: statistical and graphical results of prediction
    """

    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    stats_fcast, fig = None, None
    fli = pd.to_datetime(fi_dataset)  # fecha límite inicial
    flf = pd.to_datetime(ff_dataset)  # fecha límite final
    fid = pd.to_datetime(data.FECHA.min())
    ffd = pd.to_datetime(data.FECHA.max())
    fid = pd.to_datetime(str(fid.year) + '-' + str(fid.month) + '-' + str(fid.day))
    ffd = pd.to_datetime(str(ffd.year) + '-' + str(ffd.month) + '-' + str(ffd.day))
    fie = pd.to_datetime(start_training)
    ffe = pd.to_datetime(end_training)
    pred = False

    if (fid >= fli and ffd < flf) and (fie >= fid and ffe <= ffd) and (n_days >= 1):
        pred = True
    else:
        print(
            'Check the traning dates and the number of days. For this function the min number of days is one.')

    if pred:
        # ajustando las fechas para los datos
        if fid.dayofweek != 0:
            dias = fid.dayofweek
            fid = fid - pd.DateOffset(days=dias)

        if ffd.dayofweek != 0:
            dias = ffd.dayofweek
            ffd = ffd - pd.DateOffset(days=dias)

        fie8 = fie
        fied = fie8
        ffe8 = ffe + pd.DateOffset(hours=18)
        ffed = ffe8 - pd.DateOffset(hours=18)

        fip8 = ffed + pd.DateOffset(days=1)
        ffp8 = fip8 + pd.DateOffset(days=n_days-1) + pd.DateOffset(hours=18)

        ddt = pd.DataFrame()  # df of total data
        dpt = pd.DataFrame()  # df of accumulated trend forecast
        dpe = pd.DataFrame()  # df of accumulated seasonal forecast
        dte = pd.DataFrame()  # df of accumulated traning data
        dec = pd.DataFrame()  # df of accumulated decomposition data

        if period is None:
            period = 28

        for i in range(4):

            fii = fied + pd.DateOffset(hours=6*i)
            fff = ffed + pd.DateOffset(hours=6*i)

            d_dtdje = series.time_series_by_point(data=data, t_serie='fulldays-one-turny', start_data_date=fid,
                                                  end_data_date=ffd, period_day=i)
            d_ddjet = d_dtdje[fid:ffd]
            d_ddjee = d_dtdje[fii:fff]

            dfdje = pd.DataFrame(d_ddjee['EVENTS'])  # dataframe to time series decomposition
            decje = decompose(dfdje, period=period)      # time series decomposition

            # forecast
            ptdj = forecast(decje, steps=n_days, fc_func=drift)
            pedj = forecast(decje, steps=n_days, fc_func=drift, seasonal=True)
            # accumulating the predictions
            dte = dte.append(d_ddjee)
            dpt = dpt.append(ptdj)
            dpe = dpe.append(pedj)
            ddt = ddt.append(d_ddjet)
            dect = pd.DataFrame({'seasonal': decje.seasonal.EVENTS, 'resid': decje.resid.EVENTS, 'trend': decje.trend.EVENTS}, index=dfdje.index)
            dec = dec.append(dect)

        # sorting
        dpt = dpt.sort_index()
        dpe = dpe.sort_index()
        dte = dte.sort_index()
        dec = dec.sort_index()
        ddt = ddt.sort_index()

        # calculate the statistics to forecast
        stats_fcast = fstats.get_statistics(ffd, fip8, ffp8, fie8, dpt, dpe, ddt)
        stats_fcast['fcast'] = np.round(dpe, 2)
        if graphics:
            fig = fplots.forecast_graphs(ddt, dte, dpt, dpe, dec, ffe8, fip8, 'Weekday 4-Turny ' + str(bandwidth), True)

        return stats_fcast, fig
