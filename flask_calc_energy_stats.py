import mariadb
from flask import Flask, jsonify, g,request
import pandas as pd
import numpy as np
from config import *
from datetime import datetime, timedelta


# Flask-Server erstellen
app = Flask(__name__)


class LastEstimator:
    def __init__(self):
        self.conn_params = db_config
        self.conn = mariadb.connect(**self.conn_params)

    def fetch_data(self, start_date, end_date):
        queries = {
            "Stromzaehler": f"SELECT DATE_FORMAT(timestamp, '%Y-%m-%d %H:00:00') as timestamp, AVG(data) AS Stromzaehler FROM sensor_stromzaehler WHERE topic = 'stromzaehler leistung' AND timestamp BETWEEN '{start_date}' AND '{end_date}' GROUP BY 1 ORDER BY timestamp ASC",
            "PV": f"SELECT DATE_FORMAT(timestamp, '%Y-%m-%d %H:00:00') as timestamp, AVG(data) AS PV FROM data WHERE topic = 'solarallpower' AND timestamp BETWEEN '{start_date}' AND '{end_date}' GROUP BY 1 ORDER BY timestamp ASC",
            "Batterie_Strom_PIP": f"SELECT DATE_FORMAT(timestamp, '%Y-%m-%d %H:00:00') as timestamp, AVG(data) AS Batterie_Strom_PIP FROM pip WHERE topic = 'battery_current' AND timestamp BETWEEN '{start_date}' AND '{end_date}' GROUP BY 1 ORDER BY timestamp ASC",
            "Batterie_Volt_PIP": f"SELECT DATE_FORMAT(timestamp, '%Y-%m-%d %H:00:00') as timestamp, AVG(data) AS Batterie_Volt_PIP FROM pip WHERE topic = 'battery_voltage' AND timestamp BETWEEN '{start_date}' AND '{end_date}' GROUP BY 1 ORDER BY timestamp ASC",
            "Stromzaehler_Raus": f"SELECT DATE_FORMAT(timestamp, '%Y-%m-%d %H:00:00') as timestamp, AVG(data) AS Stromzaehler_Raus FROM sensor_stromzaehler WHERE topic = 'stromzaehler leistung raus' AND timestamp BETWEEN '{start_date}' AND '{end_date}' GROUP BY 1 ORDER BY timestamp ASC",
            "Wallbox": f"SELECT DATE_FORMAT(timestamp, '%Y-%m-%d %H:00:00') as timestamp, AVG(data) AS Wallbox_Leistung FROM wallbox WHERE topic = 'power_total' AND timestamp BETWEEN '{start_date}' AND '{end_date}' GROUP BY 1 ORDER BY timestamp ASC",

        }


        dataframes = {}
        for key, query in queries.items():
            dataframes[key] = pd.read_sql(query, self.conn)
        
        return dataframes

    def get_last(self, start_date, end_date):
        dataframes = self.fetch_data(start_date, end_date)
        last_df = self.calculate_last(dataframes)
        
        # Last in der Datenbank speichern
        self.store_last_in_db(last_df)
        
        return last_df


    def calculate_last(self, dataframes):
        # Batterie_Leistung = Batterie_Strom_PIP * Batterie_Volt_PIP
        dataframes["Batterie_Leistung"] = dataframes["Batterie_Strom_PIP"].merge(dataframes["Batterie_Volt_PIP"], on="timestamp", how="outer")
        dataframes["Batterie_Leistung"]["Batterie_Leistung"] = dataframes["Batterie_Leistung"]["Batterie_Strom_PIP"] * dataframes["Batterie_Leistung"]["Batterie_Volt_PIP"]

        # Stromzaehler_Saldo = Stromzaehler - Stromzaehler_Raus
        dataframes["Stromzaehler_Saldo"] = dataframes["Stromzaehler"].merge(dataframes["Stromzaehler_Raus"], on="timestamp", how="outer")
        dataframes["Stromzaehler_Saldo"]["Stromzaehler_Saldo"] = dataframes["Stromzaehler_Saldo"]["Stromzaehler"] - dataframes["Stromzaehler_Saldo"]["Stromzaehler_Raus"]

        # Stromzaehler_Saldo - Batterie_Leistung
        dataframes["Netzleistung"] = dataframes["Stromzaehler_Saldo"].merge(dataframes["Batterie_Leistung"], on="timestamp", how="outer")
        dataframes["Netzleistung"]["Netzleistung"] = dataframes["Netzleistung"]["Stromzaehler_Saldo"] - dataframes["Netzleistung"]["Batterie_Leistung"]

        # Füge die Wallbox-Leistung hinzu
        dataframes["Netzleistung"] = dataframes["Netzleistung"].merge(dataframes["Wallbox"], on="timestamp", how="left")
        dataframes["Netzleistung"]["Wallbox_Leistung"] = dataframes["Netzleistung"]["Wallbox_Leistung"].fillna(0)  # Fülle fehlende Werte mit 0

        # Last = Netzleistung + PV
        # Berechne die endgültige Last
        dataframes["Last"] = dataframes["Netzleistung"].merge(dataframes["PV"], on="timestamp", how="outer")
        dataframes["Last"]["Last_ohneWallbox"] = dataframes["Last"]["Netzleistung"] + dataframes["Last"]["PV"]
        dataframes["Last"]["Last"] = dataframes["Last"]["Netzleistung"] + dataframes["Last"]["PV"] - dataframes["Last"]["Wallbox_Leistung"]
        return dataframes["Last"].dropna()

    def get_last(self, start_date, end_date):
        dataframes = self.fetch_data(start_date, end_date)
        last_df = self.calculate_last(dataframes)
        return last_df
        
    def store_last_in_db(self, last_df):
        cursor = None
        try:
            cursor = self.conn.cursor()

            # Alte Daten für den Zeitraum löschen, um Überschneidungen zu vermeiden
            delete_query = "DELETE FROM last WHERE timestamp BETWEEN ? AND ?"
            start_timestamp = last_df['timestamp'].min()
            end_timestamp = last_df['timestamp'].max()
            cursor.execute(delete_query, (start_timestamp, end_timestamp))

            # Neue Daten einfügen
            insert_query = "INSERT INTO last (timestamp, topic, data) VALUES (?,?, ?)"
            for index, row in last_df.iterrows():
                cursor.execute(insert_query, (row['timestamp'],"last_haushalt", row['Last']))

            insert_query = "INSERT INTO last (timestamp, topic, data) VALUES (?,?, ?)"
            for index, row in last_df.iterrows():
                cursor.execute(insert_query, (row['timestamp'],"last", row['Last_ohneWallbox']))

            self.conn.commit()
        except mariadb.Error as e:
            print(f"Error inserting data: {e}")
        finally:
            if cursor:
                cursor.close()

            
        

class SolarDataProcessor:
    def __init__(self, config):
        self.config = config

    # Verbindung zur MariaDB herstellen
    def get_db_connection(self):
        if 'db' not in g:
            g.db = mariadb.connect(**self.config)
        return g.db


    # Daten aus der Datenbank abrufen
    def fetch_data(self):
        connection = self.get_db_connection()
        cursor = None
        try:
            query = """
                SELECT 
                    timestamp, 
                    data 
                FROM 
                    data 
                WHERE 
                    topic = 'solarallpower' AND timestamp >= NOW() - INTERVAL 1 WEEK 
                ORDER BY 
                    timestamp;
            """
            cursor = connection.cursor()
            cursor.execute(query)
            result = cursor.fetchall()

            # Konvertiere das Ergebnis in ein Pandas DataFrame
            df = pd.DataFrame(result, columns=['timestamp', 'data'])
            return df
        except mariadb.Error as e:
            print(f"Error fetching data: {e}")
        finally:
            if cursor:
                cursor.close()

    # Täglich erzeugte kWh berechnen mit Simpson-Regel
    def calculate_daily_kwh(self, data):
        data['timestamp'] = pd.to_datetime(data['timestamp'])
        data.set_index('timestamp', inplace=True)

        # Daten mitteln, um einen Wert pro Zeitstempel zu haben
        data = data.groupby(data.index).mean()

        # Konvertiere data in kW (angenommen, data ist in W)
        data['data'] = data['data']

        daily_kwh = {}

        for date, group in data.groupby(data.index.date):
            times = (group.index - group.index[0]).total_seconds().values
            values = group['data'].values

            # Simpson-Regel anwenden
            if len(values) > 2:
                integral = np.trapz(values, times) / 3600  # kWh
            else:
                integral = 0  # Nicht genug Datenpunkte

            daily_kwh[pd.Timestamp(date).strftime('%Y-%m-%d')] = integral

        return daily_kwh

    # kWh-Daten in die Datenbank einfügen
    def insert_kwh_data(self, daily_kwh):
        connection = self.get_db_connection()
        cursor = None
        try:
            cursor = connection.cursor()

            # Alte Daten löschen
            delete_query = "DELETE FROM data WHERE topic='solar1kwh' AND timestamp >= NOW() - INTERVAL 1 WEEK"
            cursor.execute(delete_query)

            # Neue Daten einfügen
            insert_query = "INSERT INTO data (timestamp, topic, data) VALUES (?, ?, ?)"
            for date, kwh in daily_kwh.items():
                cursor.execute(insert_query, (date, "solar1kwh", kwh))

            connection.commit()
        except mariadb.Error as e:
            print(f"Error inserting data: {e}")
        finally:
            if cursor:
                cursor.close()

# Datenbankverbindung schließen
@app.teardown_appcontext
def close_db_connection(exception):
    db = g.pop('db', None)
    if db is not None:
        db.close()

# Last berechnung und in DB speichern
# Last berechnung und in DB speichern
@app.route('/calculate_last', methods=['GET'])
def calculate_last():
    # Abrufen von Start- und Enddatum als Parameter
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')

    # Initialisiere den LastEstimator
    estimator = LastEstimator()

    # Berechne die Last und speichere sie in der Datenbank
    last_df = estimator.get_last(start_date, end_date)

    # Last in der Tabelle "last" in der Datenbank speichern
    if not last_df.empty:
        estimator.store_last_in_db(last_df)
        
        # Rückgabe der berechneten Last als JSON
        result = last_df[['timestamp', 'Last']].to_dict(orient='records')
        return jsonify(result)
    else:
        return jsonify({"error": "No data available"}), 404


@app.route('/calculate_last_7d', methods=['GET'])
def calculate_last_7d():
    # Standardmäßig die letzten 7 Tage, falls kein Parameter übergeben wird
    end_date = request.args.get('end_date', (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'))  # Einen Tag hinzufügen
    start_date = request.args.get('start_date', (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'))

    # Initialisiere den LastEstimator
    estimator = LastEstimator()

    # Berechne die Last
    last_df = estimator.get_last(start_date, end_date)

    # Last in der Tabelle "last" in der Datenbank speichern
    if not last_df.empty:
        estimator.store_last_in_db(last_df)
        
        # Rückgabe der berechneten Last als JSON
        result = last_df[['timestamp', 'Last']].to_dict(orient='records')
        return jsonify(result)
    else:
        return jsonify({"error": "No data available"}), 404



# Route definieren
@app.route('/daily_kwh', methods=['GET'])
def daily_kwh():
    processor = SolarDataProcessor(config=db_config)
    data = processor.fetch_data()
    if data is not None:
        daily_kwh = processor.calculate_daily_kwh(data)
        processor.insert_kwh_data(daily_kwh)
        return jsonify(daily_kwh)
    return jsonify({"error": "Error fetching data"}), 500


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5001)
