import utilsPY as utils
import numpy as np
import pandas as pd

def cal2_SPEI(
        pre: np.ndarray,
        scale: int,
        type: str,
):
    culpre = utils.scale_values_spei(pre, scale, type)
    f_alpa, f_beta, f_P0 = utils.fit_gev_para(culpre)
    rawSPEI = utils.caculate_SPEI_gev(culpre, f_alpa, f_beta, f_P0)
    SPEI = np.clip(rawSPEI, -3, 3).flatten()
    return SPEI

def cal_SPI(
            pre: np.ndarray,
            scale: int,
            type: str,
    ):
    culpre = utils.scale_values(pre, scale, type)
    f_alpa, f_beta, f_P0 = utils.fit_gamma_para(culpre)
    rawSPI = utils.caculate_gamma(culpre, f_alpa, f_beta, f_P0)
    SPI = np.clip(rawSPI, -3, 3).flatten()
    return SPI

#Daily SPEI
inputfile='prepet031_394.csv'
data=pd.read_csv(inputfile)
culpre = utils.scale_values_spei(np.array(data['pre-pet']), 30, 'daily')
f_alpa, f_beta, f_P0 = utils.fit_gev_para(culpre)
rawSPEI = utils.caculate_SPEI_gev(culpre, f_alpa, f_beta, f_P0)
SPEI = np.clip(rawSPEI, -3, 3).flatten()
outSPEI=pd.DataFrame({'Date':data['date'],'SPEI':SPEI})
outSPEI.to_csv('SPEI031_394.csv')

#Daily SPI
inputfile='pre031_394.csv'
data=pd.read_csv(inputfile)
culpre = utils.scale_values(np.array(data['pre']), 30, 'daily')
f_alpa, f_beta, f_P0 = utils.fit_gamma_para(culpre)
rawSPI = utils.caculate_gamma(culpre, f_alpa, f_beta, f_P0)
SPI = np.clip(rawSPI, -3, 3).flatten()
outSPI=pd.DataFrame({'Date':data['date'],'SPI':SPI})
outSPI.to_csv('SPI031_394.csv')
print('down')
