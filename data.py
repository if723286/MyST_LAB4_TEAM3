
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: Proyecto Final (Analisis Técnico)                                                          -- #
# -- script: functions.py : python script with general functions                                         -- #
# -- author: Equipo 3 MICROESTRUCTURA Y SISTEMAS DE TRADING PRIM. 2023                                   -- #
# -- license: THE LICENSE TYPE AS STATED IN THE REPOSITORY                                               -- #
# -- repository: https://github.com/if723286/MyST_LAB5_TEAM3                                             -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""
import pandas as pd
import functions as fc
import warnings
warnings.filterwarnings("ignore")

#prices = fc.f_import_mt5(["EURUSD","USDMXN"])

eurusd = pd.read_csv("files/EURUSD.csv")
eurusd['time'] = eurusd['time'].astype('datetime64[ns]')
#le baje el tiempo porque de esta forma si funciona, pero si lo quiero correr el año colapsa mi compu
eurusd_train = eurusd[(eurusd['time'] >= "2020-01-01") & (eurusd['time'] < "2020-01-30")]
eurusd_test = eurusd[(eurusd['time'] >= "2021-01-01") & (eurusd['time'] < "2022-01-01")]

usdmxn = pd.read_csv("files/USDMXN.csv")
usdmxn['time'] = usdmxn['time'].astype('datetime64[ns]')
usdmxn_train = usdmxn[(usdmxn['time'] >= "2020-01-01") & (usdmxn['time'] < "2021-01-01")]
usdmxn_test = usdmxn[(usdmxn['time'] >= "2021-01-01") & (usdmxn['time'] < "2022-01-01")]

#print(fc.automated_trading(eurusd_train, window_length=100, k=3, rsi_window=1, volume=100000, stop_loss=0.5, take_profit=0.5))
print(fc.optimize_parameters(eurusd_train,max_volume=100000, max_stop_loss=0.5, max_take_profit=0.5))