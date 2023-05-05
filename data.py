
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

#prices = fc.f_import_mt5(["EURUSD","USDMXN"])

eurusd = pd.read_csv("files/EURUSD.csv")
eurusd['time'] = eurusd['time'].astype('datetime64')
#le baje el tiempo porque de esta forma si funciona, pero si lo quiero correr el año colapsa mi compu
eurusd_train = eurusd[(eurusd['time'] >= "2020-01-01") & (eurusd['time'] < "2020-01-10")]
eurusd_test = eurusd[(eurusd['time'] >= "2021-01-01") & (eurusd['time'] < "2022-01-01")]

usdmxn = pd.read_csv("files/USDMXN.csv")
usdmxn['time'] = usdmxn['time'].astype('datetime64')
usdmxn_train = usdmxn[(usdmxn['time'] >= "2020-01-01") & (usdmxn['time'] < "2021-01-01")]
usdmxn_test = usdmxn[(usdmxn['time'] >= "2021-01-01") & (usdmxn['time'] < "2022-01-01")]

fc.automated_trading(eurusd_train, window_length=20, k=2, rsi_window=25, volume=1000, stop_loss=0.02, take_profit=0.03)