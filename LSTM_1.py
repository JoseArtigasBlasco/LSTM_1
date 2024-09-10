# Este ejemplo está diseñado para predecir la serie temporal del precio de las acciones utilizando datos sintéticos.


import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras._tf_keras.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM


# Generar datos para la serie temporal
def generate_synthetic_data(size=1000):
    t = np.arange(0, size)
    data = np.sin(0.02 * t) + 0.5 * np.random.normal(size=size)
    return data

data = generate_synthetic_data()
plt.plot(data)
plt.title('Datos Sintéticos')
plt.show()

# Escalamiento los datos
scaler = MinMaxScaler(feature_range=(0, 1))
data_scaled = scaler.fit_transform(data.reshape(-1, 1))


def create_sequences(data, seq_length=50):
    x, y = [], []
    for i in range(len(data) - seq_length):
        x.append(data[i:i + seq_length])
        y.append(data[i + seq_length])

    return np.array(x), np.array(y)

seq_length = 50
x, y = create_sequences(data_scaled)

# Dividimos los datos en conjuntos de entrenamiento y prueba
train_size = int(len(x) * 0.8)
x_train, y_train = x[:train_size], y[:train_size]
x_test, y_test = x[train_size:], y[train_size:]

# Construimos el modelo LSTM
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(seq_length, 1)))
model.add(LSTM(50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# Entrenamos el modelo
history = model.fit(x_train, y_train, epochs=25, batch_size=32, validation_data=(x_test, y_test))


# Evaluar el modelo y hacer predicciones
y_pred = model.predict(x_test)


# Desescalar los datos
y_test = scaler.inverse_transform(y_test)
y_pred = scaler.inverse_transform(y_pred)


# Calculamos el error cuadrático medio
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'RMSE: {rmse}')


# Graficar resultados
plt.plot(y_test, label='Real')
plt.plot(y_pred, label='Predicción')
plt.title('Predicción de la Serie Temporal')
plt.legend()
plt.show()














