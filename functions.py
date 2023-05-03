
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: Proyecto Final (Analisis Técnico)                                                          -- #
# -- script: functions.py : python script with general functions                                         -- #
# -- author: Equipo 3 MICROESTRUCTURA Y SISTEMAS DE TRADING PRIM. 2023                                   -- #
# -- license: THE LICENSE TYPE AS STATED IN THE REPOSITORY                                               -- #
# -- repository: https://github.com/if723286/MyST_LAB5_TEAM3                                             -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import pytz
from datetime import datetime

from ta import *
import plotly.graph_objs as go
from ta.volatility import BollingerBands

"""
# This function was used to initially download
# all the prices from mt5 to then be used
# in future labs and different computers
# To run again remove comment and import MetaTrader as mt5.
"""
 
def f_import_mt5(list_tickers):
    '''
    Imports prices from MetaTrader5
    given a list of tickers.
    Parameters
    ---------
    list_tickers : list
        list of tickers
    Returns
    --------
    mt5_rates : dict
        dictionary of DataFrames
        with prices of tickers
    '''
    mt5.initialize()
    timezone = pytz.timezone("Etc/UTC")
    utc_from = datetime(2023,5,1,tzinfo=timezone)
    mt5_rates = {}
    for i in list_tickers:
        rates=(mt5.copy_rates_from(i,mt5.TIMEFRAME_M15,utc_from,60000))
        rates=pd.DataFrame(rates)
        rates['time'] = pd.to_datetime(rates['time'], unit='s')
        #rates.to_csv('')
        mt5_rates[i]=rates
        mt5_rates[i].to_csv('files/'+i+'.csv')
    mt5.shutdown()
    return mt5_rates


##Bollinger Bands 

def bollinger(data, window_length=20, k=2): # k = cantidad de desviaciones o anchura de BB
    bb = BollingerBands(close=data['price'], window=window_length, window_dev=k)
    
    # Dataframe
    bb_df = pd.DataFrame()
    bb_df['middle'] = bb.bollinger_mavg()
    bb_df['upper'] = bb.bollinger_hband()
    bb_df['lower'] = bb.bollinger_lband()
    bb_df['upper_signal'] = bb.bollinger_hband_indicator()
    bb_df['lower_signal'] = bb.bollinger_lband_indicator()
    
    # gráfico BB con linea de precio 
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data['price'], name='Price', line=dict(color='#yellow', width=2), connectgaps=True))
    fig.add_trace(go.Scatter(x=data.index, y=bb_df['upper'], name='Upper Bollinger Band', line=dict(color='#navy', width=2), connectgaps=True))
    fig.add_trace(go.Scatter(x=data.index, y=bb_df['middle'], name='Middle Bollinger Band', line=dict(color='#lightblue', width=2), connectgaps=True))
    fig.add_trace(go.Scatter(x=data.index, y=bb_df['lower'], name='Lower Bollinger Band', line=dict(color='#navy', width=2), connectgaps=True,))
    fig.update_layout(title='Bollinger Bands', yaxis_title='Price', xaxis_title='Date')
    
    return bb_df, fig

## RSI

def rsi(data, window_length=25):
    rsi = ta.momentum.RSIIndicator(data['price'], window=window_length)
    
    # Dataframe
    rsi_df = pd.DataFrame()
    rsi_df['rsi'] = rsi.rsi()
    rsi_df['rsi_upper'] = 80 #para mejores entrys (normal 70)
    rsi_df['rsi_lower'] = 20 # para mejores entrys (normal 30)
    
    # gráfico RSI
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=rsi_df['rsi'], name='RSI', line=dict(color='#purple', width=2), connectgaps=True))
    fig.add_trace(go.Scatter(x=data.index, y=rsi_df['rsi_upper'], name='Overbought', line=dict(color='yellow', width=2), connectgaps=True))
    fig.add_trace(go.Scatter(x=data.index, y=rsi_df['rsi_lower'], name='Oversold', line=dict(color='#yellow', width=2), connectgaps=True))
    fig.update_layout(title='Relative Strength Index', yaxis_title='RSI', xaxis_title='Date')
    
    return rsi_df, fig