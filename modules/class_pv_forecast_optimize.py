import numpy as np
import pandas as pd
import mariadb, math
from datetime import datetime, timedelta
from sklearn.metrics import mean_squared_error
import random
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from class_pv_forecast import PVForecast  # Importiere die PVForecast Klasse

backcast_days = 14
forecast_hours = 0




def plot_results(df_initial, df_final, df_real):
    plt.figure(figsize=(14, 7))
    bar_width = 0.2
    indices = np.arange(len(df_real.index))
    
    plt.bar(indices - bar_width, df_real['solarallpower'], width=bar_width, label='Real Measured Solar Power', color='blue')
    plt.bar(indices, df_initial['ac_power'], width=bar_width, label='Initial Forecasted AC Power', color='orange')
    plt.bar(indices + bar_width, df_final['ac_power'], width=bar_width, label='Optimized Forecasted AC Power', color='red')
    
    plt.xticks(indices, df_real.index.strftime('%Y-%m-%d %H:%M'), rotation=45, ha='right')
    plt.xlabel('Time')
    plt.ylabel('Power (W)')
    plt.title('Comparison of Real Measured, Initial, and Optimized Forecasted AC Power')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
def plot_dataframe(df, column_name, title='Power Plot', ylabel='Power (W)'):
    # Überprüfen, ob die Spalte im DataFrame vorhanden ist
    if column_name not in df.columns:
        print(f"Column '{column_name}' not found in the dataframe.")
        return

    # Erstellen des Plots
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df[column_name], marker='o', linestyle='-')

    # Einstellen der Beschriftungen und Titel
    plt.xlabel('Time')
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=45, ha='right')  # Dreht die x-Achsen-Beschriftungen für bessere Lesbarkeit
    plt.grid(True)
    plt.tight_layout()
    plt.show()



class SolarDataFetcher:
    def __init__(self, config):
        self.config = config
        self.conn = None
        self.cursor = None
        self.connect_db()
        
    def connect_db(self):
        self.conn = mariadb.connect(**self.config)
        self.cursor = self.conn.cursor()

    def fetch_data(self, days):
        end_date = datetime.now() 
        start_date = end_date - timedelta(days=days)

        query = f"""
        SELECT
          DATE_FORMAT(timestamp, '%Y-%m-%d %H:00:00') AS time,
          topic AS metric,
          AVG(data) AS solarpower
        FROM data
        WHERE
          timestamp BETWEEN '{start_date.strftime('%Y-%m-%d %H:%M:%S')}' AND '{end_date.strftime('%Y-%m-%d %H:%M:%S')}' AND 
          (topic = 'solarallpower')
        GROUP BY time, metric
        ORDER BY time
        """
        self.cursor.execute(query)
        result = self.cursor.fetchall()

        df = pd.DataFrame(result, columns=['time', 'metric', 'solarpower'])
        df['time'] = pd.to_datetime(df['time'])
        return df.pivot(index='time', columns='metric', values='solarpower')

    def close(self):
        if self.conn:
            self.cursor.close()
            self.conn.close()

def calculate_mse(params, config, base_url_template, days=backcast_days):
    azimuth1, azimuth2, azimuth3, azimuth4 = params[0:4]
    tilt1, tilt2, tilt3, tilt4 = params[4:8]
    h1 = ','.join(map(str, map(int, params[8:12])))
    h2 = ','.join(map(str, map(int, params[12:16])))
    h3 = ','.join(map(str, map(int, params[16:20])))
    h4 = ','.join(map(str, map(int, params[20:24])))
    inverter_efficiency = params[24]

    forecast_url = base_url_template.format(
        azimuth1=azimuth1, azimuth2=azimuth2, azimuth3=azimuth3, azimuth4=azimuth4,
        tilt1=tilt1, tilt2=tilt2, tilt3=tilt3, tilt4=tilt4,
        h1=h1, h2=h2, h3=h3, h4=h4,
        inverterEfficiency=inverter_efficiency
    )
    #print(forecast_url)
    
    fetcher = SolarDataFetcher(config=config)
    df_solarallpower = fetcher.fetch_data(days=days)
    fetcher.close()

    # Nutze die PVForecast Klasse für die Vorhersage
    forecast = PVForecast(prediction_hours=forecast_hours, url=forecast_url)
    df_forecast = forecast.get_forecast_dataframe()
    
    pd.set_option('display.max_rows', None)
   # print(df_forecast )
    
    df_forecast['date_time'] = pd.to_datetime(df_forecast['date_time']).dt.tz_localize(None)
    #print(df_forecast )
    
    #print(df_forecast['date_time'])
    df_forecast.set_index('date_time', inplace=True)

    common_index = df_solarallpower.index.intersection(df_forecast.index)
    df_solarallpower_common = df_solarallpower.loc[common_index]
    df_forecast_common = df_forecast.loc[common_index]

    #print(df_solarallpower)
    #print(df_forecast)

    df_comparison = pd.DataFrame({
        'solarallpower': df_solarallpower_common['solarallpower'],
        'ac_power': df_forecast_common['ac_power']
    })
    df_comparison['difference'] = df_comparison['solarallpower'] - df_comparison['ac_power']

    mse = mean_squared_error(df_comparison['solarallpower'], df_comparison['ac_power'])
    
    return mse, df_forecast_common, df_forecast

config = {
    'user': 'soc',
    'password': 'Rayoflight123!',
    'host': '192.168.1.135',
    'database': 'sensor'
}

initial_params = np.array([
    -6, -100, -31, 9,
    5.,  5., 64., 46.,
    30, 50, 0, 0,
    30, 50, 0, 0,
    60, 50, 50, 40,
    45, 50, 40, 0,
    0.8
])

base_url_template = (
    "https://api.akkudoktor.net/forecast?lat=50.8588&lon=7.3747&power=5000&azimuth={azimuth1}&tilt={tilt1}&powerInvertor=10000&horizont={h1}&"
    "power=4800&azimuth={azimuth2}&tilt={tilt2}&powerInvertor=10000&horizont={h2}&"
    "power=1400&azimuth={azimuth3}&tilt={tilt3}&powerInvertor=2000&horizont={h3}&"
    "power=1600&azimuth={azimuth4}&tilt={tilt4}&powerInvertor=1400&horizont={h4}&"
    "past_days="+str(backcast_days)+"&cellCoEff=-0.36&inverterEfficiency={inverterEfficiency}&albedo=0.25&timezone=Europe%2FBerlin&hourly=relativehumidity_2m%2Cwindspeed_10m"
)


initial_mse, df_initial_forecast_common,df_initial_forecast= calculate_mse(initial_params, config, base_url_template)
#print(f"Initial MSE with provided parameters: {initial_mse}")

fetcher = SolarDataFetcher(config=config)
df_real = fetcher.fetch_data(days=backcast_days)
fetcher.close()

common_index_initial = df_real.index.intersection(df_initial_forecast_common.index)
df_real_common_initial = df_real.loc[common_index_initial]
df_initial_common_initial = df_initial_forecast_common.loc[common_index_initial]


# start_date = pd.to_datetime(datetime.now())
# end_date = start_date + pd.Timedelta(days=1)
# daily_data = df_initial_forecast.loc[start_date:end_date]

#print(daily_data)
#plot_dataframe(daily_data,"ac_power")








# Callback function to print progress
iteration = 0
best_mse = float('inf')
best_horizons = None

# Optimize horizons using Differential Evolution
config = {
    'user': 'soc',
    'password': 'Rayoflight123!',
    'host': '192.168.1.135',
    'database': 'sensor'
}




# URL-Vorlage definieren
base_url_template = (
    "https://api.akkudoktor.net/forecast?lat=50.8588&lon=7.3747&power=5000&azimuth={azimuth1}&tilt={tilt1}&powerInvertor=10000&horizont={h1}&"
    "power=4800&azimuth={azimuth2}&tilt={tilt2}&powerInvertor=10000&horizont={h2}&"
    "power=1400&azimuth={azimuth3}&tilt={tilt3}&powerInvertor=2000&horizont={h3}&"
    "power=1600&azimuth={azimuth4}&tilt={tilt4}&powerInvertor=1400&horizont={h4}&"
    "past_days="+str(backcast_days)+"&cellCoEff=-0.36&inverterEfficiency={inverterEfficiency}&albedo=0.25&timezone=Europe%2FBerlin&hourly=relativehumidity_2m%2Cwindspeed_10m"
)

# Berechnung des MSE für die initialen Parameter
initial_mse,_,_ = calculate_mse(initial_params, config, base_url_template)
print(f"Initial MSE with provided parameters: {initial_mse}")


# Funktion zur Optimierung eines einzelnen Parameters in diskreten Schritten
def optimize_single_param(params, idx, step_size, config, base_url_template, lower_bound, upper_bound):
    current_value = params[idx]
    best_value = current_value
    best_mse,_ ,_= calculate_mse(params, config, base_url_template)
    print("\n\n\n")
    print("Start_BestMSE:",best_mse," Best_value:",best_value)
    print(np.arange(lower_bound,upper_bound,step_size))
    for step in np.arange(lower_bound,upper_bound,step_size):
        new_value = step 
        #if lower_bound <= new_value <= upper_bound:
        params[idx] = step
        current_mse,_,_ = calculate_mse(params, config, base_url_template)
        print(f"Testing value {new_value} for parameter {idx} gave MSE={current_mse}", flush=True)
        if current_mse < best_mse:
            best_mse = current_mse
            best_value = new_value

    print("End MSE::",best_mse," End: Best_value:",best_value)
    params[idx] = best_value
    return params, best_mse

#Optimierung der Parameter in der Reihenfolge: inverter efficiency, azimuth, tilt, horizon
optimized_params = initial_params.copy()

# Grenzen für azimuth und tilt und inverter efficiency
azimuth_bounds = [(-20, 0), (-100, -80), (-50, -30), (-5, 15)]
tilt_bounds = [(5, 9), (5, 9), (55, 65), (43, 47)]
inverter_efficiency_bounds = (0.8, 0.85)



# Optimierung der azimuth Werte
# for idx in range(4):
    # lower_bound, upper_bound = azimuth_bounds[idx]
    # optimized_params, single_mse = optimize_single_param(optimized_params, idx, 1, config, base_url_template, lower_bound, upper_bound)
    # print(optimized_params," ",single_mse)

# # Optimierung der tilt Werte
# for idx in range(4, 8):
    # lower_bound, upper_bound = tilt_bounds[idx - 4]
    # optimized_params, single_mse = optimize_single_param(optimized_params, idx, 1, config, base_url_template, lower_bound, upper_bound)
    # print(optimized_params," ",single_mse)

# Optimierung der Horizonte in 10er Schritten
step_size = 10
for idx in range(8, 24):
    optimized_params, single_mse = optimize_single_param(optimized_params, idx, step_size, config, base_url_template, 0, 100)
    print(optimized_params," ",single_mse)

# Optimierung der inverter efficiency Werte
optimized_params, single_mse = optimize_single_param(optimized_params, 24, 0.01, config, base_url_template, *inverter_efficiency_bounds)
print(optimized_params," ",single_mse)    
    
# Finales Ergebnis
azimuth1, azimuth2, azimuth3, azimuth4 = optimized_params[0:4]
tilt1, tilt2, tilt3, tilt4 = optimized_params[4:8]
h1 = ','.join(map(str, map(int, optimized_params[8:12])))
h2 = ','.join(map(str, map(int, optimized_params[12:16])))
h3 = ','.join(map(str, map(int, optimized_params[16:20])))
h4 = ','.join(map(str, map(int, optimized_params[20:24])))
inverter_efficiency = optimized_params[24]

print(f"Final best Parameters: Azimuths=[{azimuth1}, {azimuth2}, {azimuth3}, {azimuth4}], Tilts=[{tilt1}, {tilt2}, {tilt3}, {tilt4}], Horizons: H1={h1}, H2={h2}, H3={h3}, H4={h4}, Inverter Efficiency={inverter_efficiency} with final MSE: {single_mse}")

# Fetch initial forecast data
initial_forecast_url = base_url_template.format(
    azimuth1=initial_params[0], azimuth2=initial_params[1], azimuth3=initial_params[2], azimuth4=initial_params[3],
    tilt1=initial_params[4], tilt2=initial_params[5], tilt3=initial_params[6], tilt4=initial_params[7],
    h1=','.join(map(str, map(int, initial_params[8:12]))),
    h2=','.join(map(str, map(int, initial_params[12:16]))),
    h3=','.join(map(str, map(int, initial_params[16:20]))),
    h4=','.join(map(str, map(int, initial_params[20:24]))),
    inverterEfficiency=initial_params[24]
)

initial_forecast = PVForecast(prediction_hours=0, url=initial_forecast_url)
df_initial_forecast = initial_forecast.get_forecast_dataframe()
df_initial_forecast['date_time'] = pd.to_datetime(df_initial_forecast['date_time']) #.dt.tz_convert(None)
df_initial_forecast.set_index('date_time', inplace=True)
print(df_initial_forecast)

# Fetch optimized forecast data
optimized_forecast_url = base_url_template.format(
    azimuth1=optimized_params[0], azimuth2=optimized_params[1], azimuth3=optimized_params[2], azimuth4=optimized_params[3],
    tilt1=optimized_params[4], tilt2=optimized_params[5], tilt3=optimized_params[6], tilt4=optimized_params[7],
    h1=','.join(map(str, map(int, optimized_params[8:12]))),
    h2=','.join(map(str, map(int, optimized_params[12:16]))),
    h3=','.join(map(str, map(int, optimized_params[16:20]))),
    h4=','.join(map(str, map(int, optimized_params[20:24]))),
    inverterEfficiency=optimized_params[24]
)

print(optimized_forecast_url)

optimized_forecast = PVForecast(prediction_hours=0, url=optimized_forecast_url)
df_optimized_forecast = optimized_forecast.get_forecast_dataframe()
df_optimized_forecast['date_time'] = pd.to_datetime(df_optimized_forecast['date_time']) #.dt.tz_convert(None)
df_optimized_forecast.set_index('date_time', inplace=True)

# Fetch historical data
fetcher = SolarDataFetcher(config=config)
df_real = fetcher.fetch_data(days=backcast_days)
fetcher.close()

# Reduce to common timestamps
common_index = df_real.index.intersection(df_initial_forecast.index).intersection(df_optimized_forecast.index)
df_real_common = df_real.loc[common_index]
df_initial_common = df_initial_forecast.loc[common_index]
df_optimized_common = df_optimized_forecast.loc[common_index]

# Plot results
plot_results(df_initial_common, df_optimized_common, df_real_common)
