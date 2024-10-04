import numpy as np
import pandas as pd
import joblib, json
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense,Dropout
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l1, l2, l1_l2
from scipy.signal import savgol_filter
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, TimeDistributed, RepeatVector
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
import matplotlib.pyplot as plt


class BatterySocPredictor:
    def __init__(self):
        # Initialisierung von Scaler und Gaußschem Prozessmodell
        self.scaler = StandardScaler()
        kernel = WhiteKernel(1.0, (1e-7, 1e3)) + RBF(length_scale=(0.1,0.1), length_scale_bounds=((1e-7, 1e3),(1e-7, 1e3)))
        self.gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-2, normalize_y=True)

    def fit(self, X, y):
        # Transformiere die Zielvariable
        y_transformed = np.log(y / (101 - y))
        # Skaliere die Features
        X_scaled = self.scaler.fit_transform(X)
        # Trainiere das Modell
        self.gp.fit(X_scaled, y_transformed)

    def predict(self, X):
        # Skaliere die Features
        X_scaled = self.scaler.transform(X)
        # Vorhersagen und Unsicherheiten
        y_pred_transformed, sigma_transformed = self.gp.predict(X_scaled, return_std=True)
        # Rücktransformieren der Vorhersagen
        y_pred = 101 / (1 + np.exp(-y_pred_transformed))
        # Rücktransformieren der Unsicherheiten
        sigmoid_y_pred = 1 / (1 + np.exp(-y_pred_transformed))
        sigma = sigma_transformed * 101 * sigmoid_y_pred * (1 - sigmoid_y_pred)
        return float(y_pred), float(sigma)

    def save_model(self, file_path):
        # Speichere das gesamte Modell-Objekt
        joblib.dump(self, file_path)

    @staticmethod
    def load_model(file_path):
        # Lade das Modell-Objekt
        return joblib.load(file_path)
        
class BatterySoCPredictorLSTM:
    def __init__(self, model_path=None, scaler_path=None):
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        self.seq_length = 10  # Anzahl der Zeitschritte in der Eingabesequenz
        if model_path:
            self.model = load_model(model_path)
        else:
            self.model = self._build_model()
        
        if scaler_path:
            self.load_scalers(scaler_path)

    def _build_model(self):
        regu = 0.00  # Regularisierungsrate
        model = Sequential()
        model.add(LSTM(20, activation='relu', return_sequences=True, input_shape=(self.seq_length, 4), kernel_regularizer=l2(regu)))
        model.add(LSTM(20, activation='relu', return_sequences=False, kernel_regularizer=l2(regu)))
        model.add(Dense(1, kernel_regularizer=l2(regu)))

        optimizer = Adam(learning_rate=0.0005)
        model.compile(optimizer=optimizer, loss='mae')
        return model

    def fit(self, data_path, epochs=100, batch_size=50, validation_split=0.1, use_recursive=True
        data = pd.read_csv(data_path)
        data['Time'] = pd.to_datetime(data['Time'], unit='ms')
        data.set_index('Time', inplace=True)

        data.dropna(inplace=True)

        scaled_data = self.scaler.fit_transform(data[['battery_voltage', 'battery_current', 'data', 'battery_soc']].values)
        data['scaled_soc'] = self.target_scaler.fit_transform(data[['battery_soc']])

        X, y = self._create_sequences(scaled_data, self.seq_length)
        
        if use_recursive:
            tf.print("REUKUR")
            # Train with recursive prediction
            for epoch in range(epochs):
                print(f"Epoch {epoch + 1}/{epochs}")
                for i in range(0, len(X), batch_size):
                    batch_X = X[i:i + batch_size]
                    batch_y = y[i:i + batch_size]
                    batch_predictions = self._recursive_predict(batch_X)
                    self.model.train_on_batch(batch_X, batch_predictions)
        else:
            self.model.fit(X, y, epochs=epochs, batch_size=batch_size, validation_split=validation_split)

    def _create_sequences(self, data, seq_length):
        xs, ys = [], []
        for i in range(len(data) - seq_length):
            x = data[i:(i + seq_length)]
            y = data[i + seq_length, -1]  # The SoC is the last column in the data
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    def _recursive_predict(self, data):
        predictions = []
        for seq in data:
            current_seq = seq
            for _ in range(2):  # You can adjust the range if you want to predict multiple steps ahead
                current_seq = np.expand_dims(current_seq, axis=0)
                prediction = self.model.predict(current_seq)
                current_seq = np.append(current_seq[:, 1:, :], [[prediction]], axis=1)
            predictions.append(prediction[0, 0])
        
        return np.array(predictions)

    def predict(self, test_data_path, recursive=False):
        test_data = pd.read_csv(test_data_path)
        test_data['Time'] = pd.to_datetime(test_data['Time'], unit='ms')
        test_data.set_index('Time', inplace=True)
        test_data.replace('undefined', np.nan, inplace=True)
        test_data.dropna(inplace=True)

        test_data['battery_voltage'] = pd.to_numeric(test_data['battery_voltage'], errors='coerce')
        test_data['battery_current'] = pd.to_numeric(test_data['battery_current'], errors='coerce')
        test_data['battery_soc'] = pd.to_numeric(test_data['battery_soc'], errors='coerce')
        test_data['data.1'] = pd.to_numeric(test_data['data.1'], errors='coerce')
        test_data.dropna(inplace=True)

        scaled_test_data = self.scaler.transform(test_data[['battery_voltage', 'battery_current', 'data.1', 'battery_soc']])
        test_data['scaled_soc'] = self.target_scaler.transform(test_data[['battery_soc']])
        test_data.dropna(inplace=True)

        if not recursive:
            X_test, _ = self._create_sequences(scaled_test_data, self.seq_length)
            predictions = self.model.predict(X_test)
            predictions = self.target_scaler.inverse_transform(predictions.reshape(-1, 1))
            return predictions
        else:
            return self._recursive_predict(scaled_test_data)

    def predict_single(self, voltage_current_temp_soc_sequence):
        if len(voltage_current_temp_soc_sequence) != self.seq_length or len(voltage_current_temp_soc_sequence[0]) != 4:
            raise ValueError("Die Eingabesequenz muss die Form (seq_length, 4) haben.")
        
        scaled_sequence = self.scaler.transform(voltage_current_temp_soc_sequence)
        X = np.array([scaled_sequence])

        prediction = self.model.predict(X)
        prediction = self.target_scaler.inverse_transform(prediction)
        return prediction[0, 0]

    def save_model(self, model_path=None, scaler_path=None):
        self.model.save(model_path)
        
        scaler_params = {
            'scaler_min_': self.scaler.min_.tolist(),
            'scaler_scale_': self.scaler.scale_.tolist(),
            'target_scaler_min_': self.target_scaler.min_.tolist(),
            'target_scaler_scale_': self.target_scaler.scale_.tolist()
        }
        with open(scaler_path, 'w') as f:
            json.dump(scaler_params, f)

    def load_scalers(self, scaler_path):
        with open(scaler_path, 'r') as f:
            scaler_params = json.load(f)
        self.scaler.min_ = np.array(scaler_params['scaler_min_'])
        self.scaler.scale_ = np.array(scaler_params['scaler_scale_'])
        self.target_scaler.min_ = np.array(scaler_params['target_scaler_min_'])
        self.target_scaler.scale_ = np.array(scaler_params['target_scaler_scale_'])

    def plot_results(self, predictions, test_data_path):
        test_data = pd.read_csv(test_data_path)
        test_data['Time'] = pd.to_datetime(test_data['Time'], unit='ms')
        test_data.set_index('Time', inplace=True)
        test_data.replace('undefined', np.nan, inplace=True)
        test_data.dropna(inplace=True)

        voltage_data = test_data['battery_voltage'][self.seq_length + 1:]
        current_data = test_data['battery_current'][self.seq_length + 1:]

        fig, ax1 = plt.subplots(figsize=(12, 8))

        ax2 = ax1.twinx()
        ax3 = ax1.twinx()

        ax3.spines["right"].set_position(("axes", 1.2))
        ax3.set_frame_on(True)
        ax3.patch.set_visible(False)

        ax1.plot(test_data.index[self.seq_length + 1:], predictions.flatten(), label='Predicted SoC', color='tab:blue', alpha=0.7)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('State of Charge (%)', color='tab:blue')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.legend(loc='upper left')

        ax2.plot(test_data.index[self.seq_length + 1:], current_data, label='Battery Current (A)', color='tab:red')
        ax2.set_ylabel('Current (A)', color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:red')
        ax2.legend(loc='upper right')

        ax3.plot(test_data.index[self.seq_length + 1:], voltage_data, label='Battery Voltage (V)', color='tab:green')
        ax3.set_ylabel('Voltage (V)', color='tab:green')
        ax3.tick_params(axis='y', labelcolor='tab:green')
        ax3.legend(loc='center right')

        plt.title('Battery SoC, Voltage, and Current Over Time')
        plt.show()

if __name__ == '__main__':
    train_data_path = 'lstm_train/raw_data_clean.csv'
    test_data_path = 'Test_Data.csv'
    model_path = 'battery_soc_predictor_lstm_model.keras'
    scaler_path = 'battery_soc_predictor_scaler_model'

    predictor = BatterySoCPredictorLSTM()

    # Training mit rekursiver Vorhersage
    predictor.fit(train_data_path, epochs=30, batch_size=500, validation_split=0.1, use_recursive=True)

    # Speichern des Modells und der Scaler
    predictor.save_model(model_path=model_path, scaler_path=scaler_path)
    
    # Laden des Modells und der Scaler
    loaded_predictor = BatterySoCPredictorLSTM(model_path=model_path, scaler_path=scaler_path)

    # Vorhersagen ohne rekursive Funktion
    predictions = loaded_predictor.predict(test_data_path, recursive=False)
    print("Vorhersagen ohne rekursive Funktion:")
    print(predictions)
    
    # Vorhersagen mit rekursiver Funktion
    recursive_predictions = loaded_predictor.predict(test_data_path, recursive=True)
    print("Vorhersagen mit rekursiver Funktion:")
    print(recursive_predictions)

    voltage_current_soc_sequence = np.array([
        [12.5, 1.2, 25, 60],
        [12.6, 1.3, 26, 61],
        [12.7, 1.4, 27, 62],
        [12.8, 1.5, 28, 63],
        [12.9, 1.6, 29, 64],
        [13.0, 1.7, 30, 65],
        [13.1, 1.8, 31, 66],
        [13.2, 1.9, 32, 67],
        [13.3, 2.0, 33, 68],
        [13.4, 2.1, 34, 69]
    ])

    single_prediction = loaded_predictor.predict_single(voltage_current_soc_sequence)
    print(f"Einzelvorhersage für die gegebene Spannungs-, Strom- und SoC-Sequenz: {single_prediction}")




