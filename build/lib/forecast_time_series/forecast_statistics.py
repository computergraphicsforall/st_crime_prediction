import numpy as np
from scipy import stats
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
                       rmsn=rmsn,
                       pearson_cor=np.round(r[0], 2),
                       mean_error_fcast=np.round(m_error_fcast, 2),
                       std_error_fcast=np.round(std_error_fcast, 2),
                       interval_error=str(np.round(mobs, 2)) + ' +- ' + str(np.round(std_error_fcast, 2)),
                       percent_error_fcast=np.round(per_error_fcast, 2)
                       )

    return results
