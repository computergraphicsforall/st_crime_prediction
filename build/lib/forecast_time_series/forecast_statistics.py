import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import math


# parámetros para la función de estadísticos de predicción
# fid: fecha inicial del conjunto de datos
# ffd: fecha final del conjunto de datos
# fip: fecha inicial de predicción
# ffp: fecha final de predicción
# dpt: resultados predicción en tendencia
# dpe: resultados predicción estacional
# dt: conjunto de datos originales entre la fecha inicial y final
# de: datos de entrenamiento
# dec: datos de la descomposición
# c: composición de datos: True para cuando haya composición. De lo contrario False
def get_statistics(ffd, fip, ffp, fie, dpt, dpe, dt):
    calc = False
    if ffp <= ffd:
        fi = fip
        ff = ffp
        calc = True
    elif fip < ffd < ffp:
        fi = fip
        ff = ffd
        calc = True
    if calc:

        if len(dpt) and len(dpe) > 1:
            r = stats.pearsonr(dt['EVENTS'][fi:ff], dpe.iloc[:, 0])
            r = np.round(r[0], 2)
        else:
            r = None

        mobs = dt['EVENTS'][fie:ffp].mean()
        stdobs = dt['EVENTS'][fie:ffp].std()
        mfcast = dpe.iloc[:, 0].mean()
        stdfcast = dpe.iloc[:, 0].std()

        obsn = (dt['EVENTS'][fi:ff] - mobs) / stdobs
        pdrn = (dpe.iloc[:, 0] - mobs) / stdobs
        rmsn = math.sqrt((np.sum((obsn - pdrn) ** 2)) / len(pdrn))
        m_error_fcast = (abs(dpe.iloc[:, 0] - dt['EVENTS'][fi:ff])).mean()
        std_error_fcast = (abs(dpe.iloc[:, 0] - dt['EVENTS'][fi:ff])).std()
        per_error_fcast = (m_error_fcast / mobs) * 100

        results = dict(mean_observations=np.round(mobs, 2),
                       std_obsertions=np.round(stdobs, 2),
                       mean_forecast=np.round(mfcast, 2),
                       std_forecast=np.round(stdfcast, 2),
                       rmsn=np.round(rmsn, 2),
                       pearson_cor=r,
                       mean_error_fcast=np.round(m_error_fcast, 2),
                       std_error_fcast=np.round(std_error_fcast, 2),
                       interval_error=str(np.round(mobs, 2)) + ' +- ' + str(np.round(std_error_fcast, 2)),
                       percent_error_fcast=np.round(per_error_fcast, 2)
                       )

    return results


def pai(fcast, percentage):
    pai_data = np.sort(fcast.ravel())
    step = math.floor((len(pai_data.ravel()) * percentage) / 100)

    A = len(pai_data)
    N = pai_data.sum()

    pai_array = list()
    area_per = list()
    init_pai = True
    a = 0
    while init_pai:

        a = a + step

        if a < A:
            n = pai_data[0:a].sum()
            pai_box = ((n / N) * 100) / ((a / A) * 100)
            pai_array.append(pai_box)
            area_per.append((a / A) * 100)
        else:
            a = A
            n = pai_data[0:].sum()
            pai_box = ((n / N) * 100) / ((a / A) * 100)
            init_pai = False
            pai_array.append(pai_box)
            area_per.append((a / A) * 100)

    return pai_array, area_per


def results_forecast_comparison(mesh_data, fcast_stkde, fcast_tseries, obs, bandwidth):
    fcast_stkde = np.array(fcast_stkde)
    fcast_tseries = np.array(fcast_tseries)
    obs = np.array(obs)

    # plot
    fig1, ax1 = plt.subplots(1, 2, figsize=(25, 10))

    ocp = ax1[0].pcolor(*mesh_data, fcast_stkde)
    ts = ax1[1].pcolor(*mesh_data, fcast_tseries)

    cb1 = plt.colorbar(ocp, ax=ax1[0])
    cb2 = plt.colorbar(ts, ax=ax1[1])
    cb1.set_label("Relative risk")
    cb2.set_label("Incidents forecast")
    ax1[0].set_title('Open cp STKDE forecast')
    ax1[1].set_title('Time series forecast')

    # normalized observations and forecast for open cp and time series
    norm_matrix_obser = obs / obs.max()
    norm_matrix_fcast_ocp = fcast_stkde / fcast_stkde.max()
    norm_matrix_fcast_ts = fcast_tseries / fcast_tseries.max()

    # difference bewtween methods forecasting
    dif_ocp = norm_matrix_obser.ravel() - norm_matrix_fcast_ocp.ravel()
    dif_ts = norm_matrix_obser.ravel() - norm_matrix_fcast_ts.ravel()
    fig2 = plt.figure(figsize=(15, 5))
    plt.title('Difference STKDE - Time series')
    plt.xlabel('Box grid')
    plt.ylabel('Difference')
    plt.plot(dif_ocp, color='green', label='ST method')
    plt.plot(dif_ts, color='red', label='ST KDE')
    plt.legend()
    plt.grid()
    plt.show()

    pai_tseries, area_tseries = pai(fcast=norm_matrix_fcast_ts, percentage=10)
    pai_stkde, area_stkde = pai(fcast=norm_matrix_fcast_ocp, percentage=10)

    fig2 = plt.figure(figsize=(15, 5))
    plt.plot(pai_stkde, area_stkde, label='STKDE PAI')
    plt.plot(pai_tseries, area_tseries, label='Time series PAI')
    plt.title('PAI STKDE - Time series')
    plt.xlabel('Hit rate')
    plt.ylabel('Area percentage')
    plt.legend()
    plt.grid()
    plt.show()

    RMSE_TS = np.sqrt(
        np.sum((norm_matrix_obser.ravel() - norm_matrix_fcast_ts.ravel()) ** 2) * (1 / len(norm_matrix_obser.ravel())))
    RMSE_STKDE = np.sqrt(
        np.sum((norm_matrix_obser.ravel() - norm_matrix_fcast_ocp.ravel()) ** 2) * (1 / len(norm_matrix_obser.ravel())))
    print('RMSE_STKDE-%s: %f ' % (str(bandwidth), RMSE_STKDE), 'RMSE_TS-%s: %f ' % (str(bandwidth), RMSE_TS))
