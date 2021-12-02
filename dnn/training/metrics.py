from tensorflow.keras.metrics import mean_squared_error, mean_absolute_percentage_error, mean_squared_logarithmic_error, mean_absolute_error

metrics = {
    'mse': mean_squared_error,
    'mape': mean_absolute_percentage_error,
    'msle': mean_squared_logarithmic_error,
    'mae': mean_absolute_error
}
