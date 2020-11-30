import pandas as pd
import numpy as np
from tools import spatial_tools as st


def data_crime_transform(df):
    """
    Function to filter and transform the original data to forecast

    :param df: dataframe with crime data of NUSE
    :return: returns a filtered and transformed dataframe
    """

    # data filter and sorted values
    dfo = df[['LONGITUD', 'LATITUD', 'LOCALIDAD', 'COD_UPZ', 'UPZ', 'FECHA', 'MES', 'ANIO']].sort_values(by=['COD_UPZ'])
    dfo = dfo[((dfo['COD_UPZ'] != 'ND') & (dfo['COD_UPZ'] != 'UPZ999'))]
    dfo.reset_index(inplace=True)
    dfo = dfo.rename(columns={'index': 'INDEX_TIME'})

    # transforming the data to manipulation
    anio = pd.to_datetime('2013-12-30')
    dfo['JORN'] = pd.to_datetime(dfo['FECHA']).dt.hour // 6
    dfo['DIA'] = pd.to_datetime(dfo['FECHA']).dt.day
    dfo['PMES'] = 1 + (pd.to_datetime(dfo['FECHA']).dt.dayofyear // 30)
    dfo['DANIO'] = pd.to_datetime(dfo['FECHA']).dt.dayofyear
    dfo['PSEM'] = 1 + ((pd.to_datetime(dfo['FECHA']) - anio) // pd.Timedelta(days=7))
    dfo['DSEM'] = pd.to_datetime(dfo['FECHA']).dt.dayofweek
    dfo['ESDIA'] = [0 if (x == 0 or x == 3) else 1 for x in dfo['JORN']]
    dfo['NSEM'] = pd.to_datetime(dfo['FECHA']).dt.week
    dfo['FSEM'] = [0 if x < 4 else 1 for x in dfo['DSEM']]

    return dfo


def st_series(fi, ff, df, tp, ds, jd, mts, plg, plt):
    """
     Function to genereate differents type of spatio-temporal time series by a time window in a specific point in longitude and latitude

    :param fi: init date of dataset (NUSE)
    :param ff: final date of dataset (NUSE)
    :param df: filtered dataframe (NUSE)
    :param tp: type of time series (view the documentation of function data_series)
    :param ds: day of week if you have a type of time series by options 4 and 8. If you haven't a day of week you can set 0 as parameter (view the documentation of function data_series)
    :param jd: turn of the day if you have a type of time series by options 5 and 9. If haven't a turn of the day you can set 0 as parameter (view the documentation of function data_series)
    :param mts: meters to bandwidth to use an filter the data for t
    :param plg:
    :param plt:
    :return:
    """
    dffs = pd.DataFrame()
    # finding bandwitdh intervals
    bwlong, bwlat = st.bw_intervals(mts, plg, plt)
    dff = df[((df['LATITUD'] >= bwlat[0]) & (df['LATITUD'] <= bwlat[1])) &
             ((df['LONGITUD'] >= bwlong[0]) & (df['LONGITUD'] <= bwlong[1]))]
    if not dff.empty:
        dffs = data_series(fi, ff, dff, tp, ds, jd)
    return dffs


def data_series(fi, ff, df, tp, ds, jd):
    """
    Function to generate the data to differents scales and dynamics. This function can generate the follow type of time series;

    Type of time series. (tp)
        It can be monthly = 0, weekly = 1, daily = 2, weekday = 3, day (on one day of the week) = 4,
        day (every day) = 5, weekdays (Monday to Thursday) = 6, weekends (Friday to Sunday) = 7,
        specific day of the week with all its days = 8, every day in a specific week = 9

    :param fi: init date of dataset (NUSE)
    :param ff: final date of dataset (NUSE)
    :param df: filtered dataframe (NUSE). This dataframe must be structured under the function of data_crime_transform
    :param tp: type of time series
    :param ds: day of week if you have a type of time series by options 4 and 8. If you haven't a day of week you can set 0 as parameter
    :param jd: turn of the day if you have a type of time series by options 5 and 9. If haven't a turn of the day you can set 0 as parameter
    :return: dataframe with de data by the type of time series selected
    """
    dfi, dfo = None, None
    index, serie = list(), list()
    days = ['W-MON', 'W-TUE', 'W-WED', 'W-THU', 'W-FRI', 'W-SAT', 'W-SUN']
    if 0 <= tp <= 9:

        if tp == 0:
            index = pd.date_range(start=pd.to_datetime(fi), end=pd.to_datetime(ff), freq='M')
            serie = ['ANIO', 'MES']
            data = {'ANIO': pd.to_datetime(index).year.tolist(),
                    'MES': pd.to_datetime(index).month.tolist(),
                    'EVENTS': np.zeros(len(index), dtype=int)}
            dfo = pd.DataFrame(index=index, data=data)

            dfi = pd.DataFrame(df.groupby(serie)['UPZ'].count()).reset_index()
            dfi['FECHA'] = pd.to_datetime(dfi['ANIO'].astype(str) +
                                          '-' + dfi['MES'].astype(str))

            for i in range(len(dfo)):
                d = dfi[(dfi['FECHA'].dt.year == dfo.index[i].year) & (dfi['FECHA'].dt.month == dfo.index[i].month)]
                if not d.empty:
                    dfo['EVENTS'].iloc[i] = int(d['UPZ'])
            return dfo

        if tp == 1:

            index = pd.date_range(start=fi, end=ff, freq='W',
                                  closed="left").strftime('%Y-%m-%d')
            indexb = pd.date_range(start=fi, end=ff, freq='W',
                                   closed="left").strftime('%Y-%m-%d')
            indexb = (pd.to_datetime(index) + pd.offsets.Week(weekday=0)).shift(-1, freq='W-MON')

            serie = ['PSEM']
            data = {'ANIO': pd.to_datetime(indexb).year.tolist(),
                    'NSEM': np.arange(1, len(indexb) + 1),
                    'EVENTS': np.zeros(len(indexb), dtype=int)}

            dfo = pd.DataFrame(index=indexb, data=data)
            dfi = pd.DataFrame(df.groupby(serie)['UPZ'].count()).reset_index()
            dfi = dfi.sort_values('PSEM')

            for i in range(len(dfo)):

                d = dfi[dfi['PSEM'] == dfo['NSEM'].iloc[i]]
                if not d.empty:
                    dfo['EVENTS'].iloc[i] = int(d['UPZ'])

            return dfo

        if tp == 2:
            index = pd.date_range(start=fi, end=ff, freq='D',
                                  closed="left").to_list()
            serie = ['ANIO', 'MES', 'DIA']
            data = {'ANIO': pd.to_datetime(index).year.tolist(),
                    'MES': pd.to_datetime(index).month.tolist(),
                    'DIA': pd.to_datetime(index).day.tolist(),
                    'EVENTS': np.zeros(len(index), dtype=int)}
            dfo = pd.DataFrame(index=index, data=data)
            dfi = pd.DataFrame(df.groupby(serie)['UPZ'].count()).reset_index()
            dfi['FECHA'] = pd.to_datetime(dfi['ANIO'].astype(str) +
                                          '-' + dfi['MES'].astype(str) +
                                          '-' + dfi['DIA'].astype(str))
            for i in range(len(dfo)):
                d = dfi[(dfi['ANIO'] == dfo['ANIO'].iloc[i]) &
                        (dfi['MES'] == dfo['MES'].iloc[i]) &
                        (dfi['DIA'] == dfo['DIA'].iloc[i])]
                if not d.empty:
                    dfo['EVENTS'].iloc[i] = int(d['UPZ'])
            return dfo

        if tp == 3 and ds is not None:
            if 0 <= ds <= 6:
                index = pd.date_range(start=fi, end=ff, freq=days[ds],
                                      closed="left")
                serie = ['ANIO', 'MES', 'NSEM', 'DSEM']
                data = {'ANIO': pd.to_datetime(index).year.tolist(),
                        'MES': pd.to_datetime(index).month.tolist(),
                        'NSEM': pd.to_datetime(index).week.tolist(),
                        'DSEM': pd.to_datetime(index).dayofweek.tolist(),
                        'EVENTS': np.zeros(len(index), dtype=int)}
                dfo = pd.DataFrame(index=index, data=data)
                dfi = pd.DataFrame(df.groupby(serie)['UPZ'].count()).reset_index()
                dfi = dfi[dfi['DSEM'] == ds]

                for i in range(len(dfo)):
                    d = dfi[(dfi['ANIO'] == (dfo['ANIO'].iloc[i])) &
                            (dfi['MES'] == dfo['MES'].iloc[i]) &
                            (dfi['NSEM'] == dfo['NSEM'].iloc[i]) &
                            (dfi['DSEM'] == dfo['DSEM'].iloc[i])]

                    if not d.empty:
                        dfo['EVENTS'].iloc[i] = int(d['UPZ'])
                return dfo

        if tp == 4 and ds is not None and jd is not None:

            if (0 <= ds <= 6) and (0 <= jd <= 3):
                index = pd.date_range(start=fi, end=ff, freq=days[ds],
                                      closed="left")
                index = index + pd.DateOffset(hours=jd * 6)
                serie = ['ANIO', 'MES', 'NSEM', 'DSEM', 'JORN']
                data = {'ANIO': pd.to_datetime(index).year.tolist(),
                        'MES': pd.to_datetime(index).month.tolist(),
                        'NSEM': pd.to_datetime(index).week.tolist(),
                        'DSEM': pd.to_datetime(index).dayofweek.tolist(),
                        'JORN': np.full(len(index), jd, dtype=int),
                        'EVENTS': np.zeros(len(index), dtype=int)}
                dfo = pd.DataFrame(index=index, data=data)
                # generando agrupamiento
                dfi = pd.DataFrame(df.groupby(serie)['UPZ'].count()).reset_index()
                dfi = dfi[(dfi['DSEM'] == ds) & (dfi['JORN'] == jd)]

                for i in range(len(dfo)):
                    d = dfi[(dfi['ANIO'] == (dfo['ANIO'].iloc[i])) &
                            (dfi['MES'] == dfo['MES'].iloc[i]) &
                            (dfi['NSEM'] == dfo['NSEM'].iloc[i]) &
                            (dfi['DSEM'] == dfo['DSEM'].iloc[i]) &
                            (dfi['JORN'] == dfo['JORN'].iloc[i])]

                    if not d.empty:
                        dfo['EVENTS'].iloc[i] = int(d['UPZ'])
                return dfo

        if tp == 5 and jd is not None:

            if 0 <= jd <= 3:
                index = pd.date_range(start=fi, end=ff, freq='6H',
                                      closed="left")
                serie = ['ANIO', 'MES', 'NSEM', 'DSEM', 'JORN']
                # preparando dataframe de salida
                data = {'ANIO': pd.to_datetime(index).year.tolist(),
                        'MES': pd.to_datetime(index).month.tolist(),
                        'NSEM': pd.to_datetime(index).week.tolist(),
                        'DSEM': pd.to_datetime(index).dayofweek.tolist(),
                        'JORN': pd.to_datetime(index).hour // 6,
                        'EVENTS': np.zeros(len(index), dtype=int)}
                dfo = pd.DataFrame(index=index, data=data)
                dfi = pd.DataFrame(df.groupby(serie)['UPZ'].count()).reset_index()

                for i in range(len(dfo)):
                    d = dfi[(dfi['ANIO'] == (dfo['ANIO'].iloc[i])) &
                            (dfi['MES'] == dfo['MES'].iloc[i]) &
                            (dfi['NSEM'] == dfo['NSEM'].iloc[i]) &
                            (dfi['DSEM'] == dfo['DSEM'].iloc[i]) &
                            (dfi['JORN'] == dfo['JORN'].iloc[i])]

                    if not d.empty:
                        dfo['EVENTS'].iloc[i] = int(d['UPZ'])
                return dfo

        if tp == 6:

            index = pd.date_range(start=fi, end=ff, freq='W-THU',
                                  closed="left").strftime('%Y-%m-%d').to_list()

            serie = ['ANIO', 'NSEM']
            data = {'ANIO': pd.to_datetime(index).year.tolist(),
                    'NSEM': pd.to_datetime(index).week.tolist(),
                    'EVENTS': np.zeros(len(index), dtype=int)}
            dfo = pd.DataFrame(index=index, data=data)

            dfi = df[df['DSEM'] <= 3]
            dfi = pd.DataFrame(dfi.groupby(serie)['UPZ'].count()).reset_index()
            for i in range(len(dfi)):
                if dfo['ANIO'].iloc[i] == dfi['ANIO'].iloc[i] and dfo['NSEM'].iloc[i] == dfi['NSEM'].iloc[i]:
                    dfo['EVENTS'].iloc[i] = dfi['UPZ'].iloc[i]
            return dfo

        if tp == 7:

            index = pd.date_range(start=fi, end=ff, freq='D',
                                  closed="left").strftime('%Y-%m-%d')
            index = index[pd.to_datetime(index).dayofweek == 6]
            serie = ['ANIO', 'NSEM']
            data = {'ANIO': pd.to_datetime(index).year.tolist(),
                    'NSEM': pd.to_datetime(index).week.tolist(),
                    'EVENTS': np.zeros(len(index), dtype=int)}
            dfo = pd.DataFrame(index=index.to_list(), data=data)
            dfi = df[(df['DSEM'] > 3) & (df['DSEM'] <= 6)]
            dfi = pd.DataFrame(dfi.groupby(serie)['UPZ'].count()).reset_index()

            for i in range(len(dfi)):
                if dfo['ANIO'].iloc[i] == dfi['ANIO'].iloc[i] and dfo['NSEM'].iloc[i] == dfi['NSEM'].iloc[i]:
                    dfo['EVENTS'].iloc[i] = dfi['UPZ'].iloc[i]
            return dfo

        if tp == 8 and ds is not None and jd is not None:

            if 0 <= ds <= 6:

                index = pd.date_range(start=fi, end=ff, freq='6H',
                                      closed="left").strftime('%Y-%m-%d %H:%M:%S')

                index = index[pd.to_datetime(index).dayofweek == ds]
                index = pd.to_datetime(index)
                serie = ['ANIO', 'MES', 'NSEM', 'DSEM', 'JORN']
                data = {'ANIO': pd.to_datetime(index).year.tolist(),
                        'MES': pd.to_datetime(index).month.tolist(),
                        'NSEM': pd.to_datetime(index).week.tolist(),
                        'DSEM': pd.to_datetime(index).dayofweek.tolist(),
                        'JORN': pd.to_datetime(index).hour // 6,
                        'EVENTS': np.zeros(len(index), dtype=int)}
                dfo = pd.DataFrame(index=index, data=data)
                dfi = pd.DataFrame(df.groupby(serie)['UPZ'].count()).reset_index()
                dfi = dfi[(dfi['DSEM'] == ds)]

                for i in range(len(dfo)):
                    d = dfi[(dfi['ANIO'] == (dfo['ANIO'].iloc[i])) &
                            (dfi['MES'] == dfo['MES'].iloc[i]) &
                            (dfi['NSEM'] == dfo['NSEM'].iloc[i]) &
                            (dfi['DSEM'] == dfo['DSEM'].iloc[i]) &
                            (dfi['JORN'] == dfo['JORN'].iloc[i])]

                    if not d.empty:
                        dfo['EVENTS'].iloc[i] = int(d['UPZ'])
                return dfo

        if tp == 9 and jd is not None:

            if 0 <= jd <= 3:

                index = pd.date_range(start=fi, end=ff, freq='D',
                                      closed="left")
                index = index + pd.DateOffset(hours=jd * 6)
                serie = ['ANIO', 'MES', 'NSEM', 'DSEM', 'JORN']
                data = {'ANIO': pd.to_datetime(index).year.tolist(),
                        'MES': pd.to_datetime(index).month.tolist(),
                        'NSEM': pd.to_datetime(index).week.tolist(),
                        'DSEM': pd.to_datetime(index).dayofweek.tolist(),
                        'JORN': pd.to_datetime(index).hour // 6,
                        'EVENTS': np.zeros(len(index), dtype=int)}
                dfo = pd.DataFrame(index=index, data=data)
                dfi = pd.DataFrame(df.groupby(serie)['UPZ'].count()).reset_index()
                dfi = dfi[(dfi['JORN'] == jd)]

                for i in range(len(dfo)):
                    d = dfi[(dfi['ANIO'] == (dfo['ANIO'].iloc[i])) &
                            (dfi['MES'] == dfo['MES'].iloc[i]) &
                            (dfi['NSEM'] == dfo['NSEM'].iloc[i]) &
                            (dfi['DSEM'] == dfo['DSEM'].iloc[i]) &
                            (dfi['JORN'] == dfo['JORN'].iloc[i])]

                    if not d.empty:
                        dfo['EVENTS'].iloc[i] = int(d['UPZ'])
                return dfo
