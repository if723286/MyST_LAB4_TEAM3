
"""
# -- --------------------------------------------------------------------------------------------------- -- #
# -- project: Proyecto Final (Analisis Técnico)                                                          -- #
# -- script: functions.py : python script with general functions                                         -- #
# -- author: Equipo 3 MICROESTRUCTURA Y SISTEMAS DE TRADING PRIM. 2023                                   -- #
# -- license: THE LICENSE TYPE AS STATED IN THE REPOSITORY                                               -- #
# -- repository: https://github.com/if723286/MyST_LAB5_TEAM3                                             -- #
# -- --------------------------------------------------------------------------------------------------- -- #
"""

#import MetaTrader5 as mt5
import numpy as np
import pandas as pd
import pytz
from datetime import datetime

from ta import *
import plotly.graph_objs as go
from ta.volatility import BollingerBands
from ta.momentum import RSIIndicator
import warnings
import pyswarms as ps
warnings.filterwarnings("ignore")

"""
# This function was used to initially download
# all the prices from mt5 to then be used
# in future labs and different computers
# To run again remove comment and import MetaTrader as mt5.

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
        rates=(mt5.copy_rates_from(i,mt5.TIMEFRAME_M15,utc_from,85000))
        rates=pd.DataFrame(rates)
        rates['time'] = pd.to_datetime(rates['time'], unit='s')
        #rates.to_csv('')
        mt5_rates[i]=rates
        mt5_rates[i].to_csv('files/'+i+'.csv')
    mt5.shutdown()
    return mt5_rates
"""


##Bollinger Bands 

import plotly.graph_objs as go
import ta
import pandas as pd
import numpy as np

def bollinger(data, window_length=20, k=2):
    bb = ta.volatility.BollingerBands(close=data['close'], window=window_length, window_dev=k)
    
    # Dataframe
    bb_df = data
    bb_df['middle'] = bb.bollinger_mavg()
    bb_df['upper'] = bb.bollinger_hband()
    bb_df['lower'] = bb.bollinger_lband()
    bb_df['upper_signal'] = bb.bollinger_hband_indicator()
    bb_df['lower_signal'] = bb.bollinger_lband_indicator()
    
    # gráfico BB con linea de precio 
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['time'], y=data['close'], name='Close', line=dict(color='yellow', width=2), connectgaps=True))
    fig.add_trace(go.Scatter(x=data['time'], y=bb_df['upper'], name='Upper Bollinger Band', line=dict(color='navy', width=2), connectgaps=True))
    fig.add_trace(go.Scatter(x=data['time'], y=bb_df['middle'], name='Middle Bollinger Band', line=dict(color='lightblue', width=2), connectgaps=True))
    fig.add_trace(go.Scatter(x=data['time'], y=bb_df['lower'], name='Lower Bollinger Band', line=dict(color='navy', width=2), connectgaps=True,))
    fig.update_layout(title='Bollinger Bands', yaxis_title='Close', xaxis_title='Date')
    
    return bb_df, fig


## RSI

def rsi(data, window_length=25):
    rsi = RSIIndicator(data['close'], window=window_length)
    
    # Dataframe
    rsi_df = data
    rsi_df['rsi'] = rsi.rsi()
    rsi_df['rsi_upper'] = 80 #para mejores entrys (normal 70)
    rsi_df['rsi_lower'] = 20 # para mejores entrys (normal 30)
    
    # gráfico RSI
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['time'], y=rsi_df['rsi'], name='RSI', line=dict(color='purple', width=2), connectgaps=True))
    fig.add_trace(go.Scatter(x=data['time'], y=rsi_df['rsi_upper'], name='Overbought', line=dict(color='yellow', width=2), connectgaps=True))
    fig.add_trace(go.Scatter(x=data['time'], y=rsi_df['rsi_lower'], name='Oversold', line=dict(color='yellow', width=2), connectgaps=True))
    fig.update_layout(title='Relative Strength Index', yaxis_title='RSI', xaxis_title='Date')
    
    return rsi_df, fig

def signal(data, window_length=20, k=2, rsi_window=25):
    bb_df, _ = bollinger(data, window_length, k)
    rsi_df, _ = rsi(data, rsi_window)
    
    bb_signal = bb_df['upper_signal'] - bb_df['lower_signal']
    rsi_signal = rsi_df['rsi'] - 50
    
    if bb_signal.iloc[-1] == 1 and rsi_signal.iloc[-1] > 0:
        return "Buy"
    elif bb_signal.iloc[-1] == -1 and rsi_signal.iloc[-1] < 0:
        return "Sell"
    elif rsi_signal.iloc[-1] > 0:
        return "Buy"
    else:
        return "Sell"
    

def automated_trading(data, window_length=15, k=3, rsi_window=1, volume=1000, stop_loss=0.02, take_profit=0.03):
    data.index = pd.to_datetime(data.time)
    data = data.groupby(pd.Grouper(freq='w')).last()
    data = data.dropna()
    data
    positions = []
    balance = 100000
    max_loss = 1000
    total_positions = pd.DataFrame(columns=['type', 'open_price', 'amount', 'value', 'stop_loss', 'take_profit', 'pnl', 'balance'])

    # Iterar sobre los datos
    for i in range(len(data)):
        # Obtener la señal
        sig = signal(data.iloc[:i+1], window_length, k, rsi_window)
        
        # Si la señal es de compra
        if sig == "Buy":
            # Si no hay una posición abierta, abrir una nueva
            if not positions:
                # Calcular el precio de compra
                price = data.iloc[i]['close']
                # Calcular el volumen de compra
                amount = volume
                # Calcular el valor total de la posición
                value = price * amount
                # Actualizar el balance
                balance -= value
                # Añadir la posición a la lista de posiciones
                positions.append({'type': 'long', 'open_price': price, 'amount': amount, 'value': value, 'stop_loss': price * (1 - stop_loss), 'take_profit': price * (1 + take_profit)})
                total_positions = pd.concat([total_positions, pd.DataFrame({'type': 'long', 'open_price': price, 'amount': amount, 'value': value, 'stop_loss': price * (1 - stop_loss), 'take_profit': price * (1 + take_profit),'pnl':0,'balance':abs(balance)}, index=[i])])
                
        # Si la señal es de venta
        elif sig == "Sell":
            # Si hay una posición abierta, cerrarla
            if positions:
                # Calcular el precio de venta
                price = data.iloc[i]['close']
                # Calcular el volumen de venta
                amount = volume
                # Calcular el valor total de la posición
                value = price * amount
                # Actualizar el balance
                balance += value
                # Calcular la ganancia o pérdida de la posición
                pnl = (price - positions[0]['open_price']) * amount
                # Añadir la ganancia o pérdida a la lista de posiciones
                positions[0]['pnl'] = pnl
                total_positions = pd.concat([total_positions, pd.DataFrame({'type': 'short', 'open_price': price, 'amount': amount, 'value': value, 'stop_loss': price * (1 - stop_loss), 'take_profit': price * (1 + take_profit),'pnl':pnl,'balance':balance}, index=[i])])
                # Eliminar la posición de la lista de posiciones
                positions.pop(0)
        
        # Si hay una posición abierta, comprobar si se alcanza el stop loss o el take profit
        if positions:
            # Calcular el precio actual
            price = data.iloc[i]['close']
            # Comprobar si se alcanza el stop loss
            if price <= positions[0]['stop_loss'] or abs(positions[0]['open_price'] - price) >= max_loss:
                # Calcular el volumen de venta
                amount = positions[0]['amount']
                # Calcular el valor total de la posición
                value = price * amount
                # Actualizar el balance
                balance += value
                # Calcular la pérdida de la posición
                pnl = (price - positions[0]['open_price']) * amount
                # Añadir la pérdida a la lista de posiciones
                positions[0]['pnl'] = pnl
                total_positions = pd.concat([total_positions, pd.DataFrame({'open_price': price, 'amount': amount, 'value': value, 'stop_loss': price * (1 - stop_loss), 'take_profit': price * (1 + take_profit),'pnl':pnl,'balance':balance}, index=[i])])
                # Eliminar la posición de la lista de posiciones
                positions.pop(0)
            # Comprobar si se alcanza el take profit
            elif price >= positions[0]['take_profit']:
                # Calcular el volumen de venta
                amount = positions[0]['amount']
                # Calcular el valor total de la posición
                value = price * amount
                # Actualizar el balance
                balance += value
                # Calcular la ganancia o pérdida de la posición
                pnl = (price - positions[0]['open_price']) * amount
                # Añadir la ganancia o pérdida a la lista de posiciones
                positions[0]['pnl'] = pnl
                total_positions = pd.concat([total_positions, pd.DataFrame({'open_price': price, 'amount': amount, 'value': value, 'stop_loss': price * (1 - stop_loss), 'take_profit': price * (1 + take_profit),'pnl':pnl,'balance':balance}, index=[i])])
                # Eliminar la posición de la lista de posiciones
                positions.pop(0)
        
        # Si hay una posición abierta, comprobar si se alcanza el stop loss o el take profit
        if positions:
            # Calcular el precio actual
            price = data.iloc[i]['close']
            # Comprobar si se alcanza el stop loss
            if price <= positions[0]['stop_loss'] or abs(positions[0]['open_price'] - price) >= max_loss:
                # Calcular el volumen de venta
                amount = positions[0]['amount']
                # Calcular el valor total de la posición
                value = price * amount
                # Actualizar el balance
                balance += value
                # Calcular la pérdida de la posición
                pnl = (price - positions[0]['open_price']) * amount
                # Añadir la pérdida a la lista de posiciones
                positions[0]['pnl'] = pnl
                total_positions = pd.concat([total_positions, pd.DataFrame({'open_price': price, 'amount': positions[0]['amount'], 'value': value, 'stop_loss': stop_loss, 'take_profit': take_profit, 'pnl': pnl, 'balance': balance}, index=[i])])
                # Eliminar la posición de la lista de posiciones
                positions.pop(0)
                
            # Comprobar si se alcanza el take profit
            elif price >= positions[0]['take_profit']:
                # Calcular el volumen de venta
                amount = positions[0]['amount']
                # Calcular el valor total de la posición
                value = price * amount
                # Actualizar el balance
                balance += value
                # Calcular la ganancia de la posición
                pnl = (price - positions[0]['open_price']) * amount
                # Añadir la ganancia a la
                positions[0]['pnl'] = pnl
                total_positions = pd.concat([total_positions, pd.DataFrame({'open_price': price, 'amount': positions[0]['amount'], 'value': value, 'stop_loss': stop_loss, 'take_profit': take_profit, 'pnl': 0, 'balance': balance}, index=[i])])
                # Eliminar la posición de la lista de posiciones
                positions.pop(0)

    # Cerrar todas las posiciones que queden abiertas al final de los datos
    while positions:
        # Calcular el precio actual
        price = data.iloc[-1]['close']
        # Calcular el volumen de venta
        amount = positions[0]['amount']
        # Calcular el valor total de la posición
        value = price * amount
        # Actualizar el balance
        balance += value
        # Calcular la ganancia o pérdida de la posición
        pnl = (price - positions[0]['open_price']) * amount
        # Añadir la ganancia o pérdida a la lista de posiciones
        positions[0]['pnl'] = pnl
        total_positions = pd.concat([total_positions, pd.DataFrame({'open_price': price, 'amount': positions[0]['amount'], 'value': value, 'stop_loss': stop_loss, 'take_profit': take_profit, 'pnl': 0, 'balance': balance}, index=[i])])
        # Eliminar la posición de la lista de posiciones
        positions.pop(0)
    
    #total_positions.dropna(inplace = True)
    # Plot RSI and Bollinger Bands
    _, bb_fig = bollinger(data, window_length, k)
    _, rsi_fig = rsi(data, rsi_window)
    #bb_fig.show()
    #rsi_fig.show()
    return balance, positions, total_positions

def optimize_parameters(data, max_volume, max_stop_loss, max_take_profit):
    best_volume = 0
    best_stop_loss = 0
    best_take_profit = 0
    best_balance = 0
    results = []
    
    for volume in np.linspace(100, max_volume, num=10):
        for stop_loss in np.linspace(0.01, max_stop_loss, num=10):
            for take_profit in np.linspace(0.01, max_take_profit, num=10):
                balance, _, _, = automated_trading(data, volume=volume, stop_loss=stop_loss, take_profit=take_profit)
                if balance > best_balance:
                    best_volume = volume
                    best_stop_loss = stop_loss
                    best_take_profit = take_profit
                    best_balance = balance
                
                results.append([volume, stop_loss, take_profit, balance])
    
    results_df = pd.DataFrame(results, columns=['volume', 'stop_loss', 'take_profit', 'balance'])
    
    results_df.to_csv('files/parameter_results.csv', index=False)
    
    return best_volume, best_stop_loss, best_take_profit, best_balance

def optimize_parameters_pso(data, max_volume=10000, max_stop_loss=0.5, max_take_profit=0.5, num_particles=10, num_iterations=50):
    # Define the bounds of the parameters
    bounds = (np.array([100, 0, 0]), np.array([max_volume, max_stop_loss, max_take_profit]))

    # Define the function to be minimized
    def objective_function(params):
        params = np.reshape(params, (-1, 3))
        volumes, stop_losses, take_profits = params[:, 0], params[:, 1], params[:, 2]
        balances = np.array([automated_trading(data, volume=v, stop_loss=s, take_profit=t)[0] for v, s, t in zip(volumes, stop_losses, take_profits)])
        return -balances

    # Set up the optimizer with bounds
    optimizer = ps.single.GlobalBestPSO(n_particles=num_particles, dimensions=3, options={'c1': 0.5, 'c2': 0.3, 'w': 0.9}, bounds=bounds)

    # Run the optimization
    best_params, best_balance = optimizer.optimize(objective_function, iters=num_iterations)

    # Check if best_params is a scalar value
    if np.isscalar(best_params):
        return abs(best_params), abs(-best_balance)
    else:
        best_volume, best_stop_loss, best_take_profit = best_params
        return best_volume, best_stop_loss, best_take_profit, -best_balance[0]


def mad(datos,rf):
    datos["rendimiento"] = (datos["balance"] / datos["balance"].shift(1) - 1) * 100

    # Calcular las dos medias de atribución de desempeño y el índice de Sharpe
    media_aritmetica = datos["rendimiento"].mean()
    media_geometrica = (datos["rendimiento"] + 100).prod() ** (1 / len(datos)) - 100
    sharpe_ratio = (datos["rendimiento"].mean() - rf) / datos["rendimiento"].std()

    medidas = {
        "Media Aritmética": [media_aritmetica],
        "Media Geométrica": [media_geometrica],
        "Radio de Sharpe": [sharpe_ratio]
    }
    df_medidas = pd.DataFrame(medidas)

    return df_medidas