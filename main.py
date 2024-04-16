from flask import Flask, render_template, request, jsonify, url_for
import pandas as pd
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from decimal import Decimal, ROUND_HALF_UP
from sklearn.metrics import mean_squared_error
from flask import Flask, render_template, request, redirect
import database
from collections import Counter
from scipy.stats import mode
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


app = Flask(__name__)

# Load the dataset
data=pd.read_excel("blend_data_MOST_UPDATEDD_FINAL.xlsx")

# Define features and target
feature_columns = ['API', 'excipient', 'API_percent', 'API_coated', 'API_coat_percent', 'exc_coated', 'exc_coat_percent', 'API_silica_type', 'exc_silica_type']
target_columns = ['FF', 'FF_regime']

X1 = data[feature_columns]
y1 = data[target_columns]

# Split the data into training and test sets
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)

# Train the RandomForestRegressor with the optimized parameters
optimized_model = RandomForestRegressor(n_estimators=100, max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=42)
optimized_model.fit(X1_train, y1_train)

# Define the mappings for API and excipient
api_values = {
    1: {'API_d10': 3.1, 'API_d50': 20, 'API_d90': 82, 'API_d32': 7.3, 'API_surf_eng': 0.041, 'API_density': 1290,
        'API AR(-)': 0.55, 'API 1/AR': 1.818181818, 'API Sphericity(-)': 0.81, 'API Elongation(-)': 0.45},
    2: {'API_d10': 2.5, 'API_d50': 6.9, 'API_d90': 15, 'API_d32': 4.5, 'API_surf_eng': 0.039545, 'API_density': 1263,
        'API AR(-)': 0.66, 'API 1/AR': 1.515151515, 'API Sphericity(-)': 0.9, 'API Elongation(-)': 0.64},
    3: {'API_d10': 3, 'API_d50': 8, 'API_d90': 19.3, 'API_d32': 5.5, 'API_surf_eng': 0.04, 'API_density': 1430,
        'API AR(-)': 0.64, 'API 1/AR': 1.5625, 'API Sphericity(-)': 0.84, 'API Elongation(-)': 0.55},
    4: {'API_d10': 8.01, 'API_d50': 28.5, 'API_d90': 58.7, 'API_d32': 12.1, 'API_surf_eng': 0.039, 'API_density': 1120,
        'API AR(-)': 0.6, 'API 1/AR': 1.666666667, 'API Sphericity(-)': 0.84, 'API Elongation(-)': 0.49},
    5: {'API_d10': 10.6, 'API_d50': 35.2, 'API_d90': 65.1, 'API_d32': 14.1, 'API_surf_eng': 0.03892, 'API_density': 1120,
        'API AR(-)': 0.58, 'API 1/AR': 1.72, 'API Sphericity(-)': 0.82, 'API Elongation(-)': 0.46},
    6: {'API_d10': 9.5, 'API_d50': 41, 'API_d90': 98, 'API_d32': 19.3, 'API_surf_eng': 0.03892, 'API_density': 1120,
        'API AR(-)': 0.59, 'API 1/AR': 1.694915254, 'API Sphericity(-)': 0.83, 'API Elongation(-)': 0.47},
    7: {'API_d10': 9.1, 'API_d50': 52.2, 'API_d90': 135, 'API_d32': 15.5, 'API_surf_eng': 0.03892, 'API_density': 1120,
        'API AR(-)': 0.59, 'API 1/AR': 1.694915254, 'API Sphericity(-)': 0.83, 'API Elongation(-)': 0.47},
    8: {'API_d10': 2.5, 'API_d50': 10, 'API_d90': 27.8, 'API_d32': 5.25, 'API_surf_eng': 0.03643, 'API_density': 1365,
        'API AR(-)': 0, 'API 1/AR': 0, 'API Sphericity(-)': 0, 'API Elongation(-)': 0},
    9: {'API_d10': 2.42, 'API_d50': 8.4, 'API_d90': 27.6, 'API_d32': 5.2, 'API_surf_eng': 0.046, 'API_density': 1290,
        'API AR(-)': 0.61, 'API 1/AR': 1.639344262, 'API Sphericity(-)': 0.81, 'API Elongation(-)': 0},
    10: {'API_d10': 2.5, 'API_d50': 10, 'API_d90': 27.8, 'API_d32': 5.25, 'API_surf_eng': 0.03643, 'API_density': 1365,
        'API AR(-)': 0, 'API 1/AR': 0, 'API Sphericity(-)': 0, 'API Elongation(-)': 0},
}


excipient_values = {
    111: {'exc_d10': 0, 'exc_d50': 0, 'exc_d90': 0, 'exc_d32': 0, 'exc_surf_eng': 0, 'exc_density': 0,
        'exc AR(-)': 0, 'exc 1/AR': 0, 'exc Sphericity(-)': 0, 'exc Elongation(-)': 0},
    222: {'exc_d10': 21.2, 'exc_d50': 64.2, 'exc_d90': 143, 'exc_d32': 42.9, 'exc_surf_eng': 0.03234, 'exc_density': 1444,
        'exc AR(-)': 0, 'exc 1/AR': 0, 'exc Sphericity(-)': 0.78, 'exc Elongation(-)': 0.38},
    333: {'exc_d10': 31.3, 'exc_d50': 117, 'exc_d90': 165.4, 'exc_d32': 49.1, 'exc_surf_eng': 0.056, 'exc_density': 1563,
        'exc AR(-)': 0.59, 'exc 1/AR': 1.694915254, 'exc Sphericity(-)': 0.79, 'exc Elongation(-)': 0.45},
    444: {'exc_d10': 5.6, 'exc_d50': 19.3, 'exc_d90': 43.4, 'exc_d32': 10.8, 'exc_surf_eng': 0.048, 'exc_density': 1559,
        'exc AR(-)': 0.61, 'exc 1/AR': 1.639344262, 'exc Sphericity(-)': 0.86, 'exc Elongation(-)': 0.52},
    555: {'exc_d10': 53.4, 'exc_d50': 185.8, 'exc_d90': 322.3, 'exc_d32': 100.4, 'exc_surf_eng': 0.047, 'exc_density': 1562,
        'exc AR(-)': 0.73, 'exc 1/AR': 1.73, 'exc Sphericity(-)': 0.87, 'exc Elongation(-)': 0.59},
    666: {'exc_d10': 8, 'exc_d50': 14.4, 'exc_d90': 22.9, 'exc_d32': 15, 'exc_surf_eng': 0.03234, 'exc_density': 1444,
        'exc AR(-)': 0.73, 'exc 1/AR': 1.37, 'exc Sphericity(-)': 0.88, 'exc Elongation(-)': 0.6},
    777: {'exc_d10': 4.5, 'exc_d50': 28.2, 'exc_d90': 79.6, 'exc_d32': 10.1, 'exc_surf_eng': 0.042, 'exc_density': 1540,
        'exc AR(-)': 0.65, 'exc 1/AR': 1.538461538, 'exc Sphericity(-)': 0.88, 'exc Elongation(-)': 0.53},
    888: {'exc_d10': 3.65, 'exc_d50': 19.2, 'exc_d90': 48.6, 'exc_d32': 6.2, 'exc_surf_eng': 0.04469, 'exc_density': 1543,
        'exc AR(-)': 0.65, 'exc 1/AR': 1.538461538, 'exc Sphericity(-)': 0.83, 'exc Elongation(-)': 0.53},
    999: {'exc_d10': 48, 'exc_d50': 115, 'exc_d90': 202, 'exc_d32': 85.2, 'exc_surf_eng': 0.03948, 'exc_density': 1543,
        'exc AR(-)': 0, 'exc 1/AR': 0, 'exc Sphericity(-)': 0.88, 'exc Elongation(-)': 0.6},
    1111: {'exc_d10': 1.8, 'exc_d50': 8.6, 'exc_d90': 20.5, 'exc_d32': 4.29, 'exc_surf_eng': 0.043, 'exc_density': 1520,
        'exc AR(-)': 0.65, 'exc 1/AR': 0, 'exc Sphericity(-)': 0.86, 'exc Elongation(-)': 0.58},
    2222: {'exc_d10': 2.9, 'exc_d50': 12.5, 'exc_d90': 32.1, 'exc_d32': 6.4, 'exc_surf_eng': 0.043695, 'exc_density': 1780,
        'exc AR(-)': 0, 'exc 1/AR': 1.5625, 'exc Sphericity(-)': 0.87, 'exc Elongation(-)': 0.58},
    3333: {'exc_d10': 1.9, 'exc_d50': 8.5, 'exc_d90': 25.5, 'exc_d32': 4.5, 'exc_surf_eng': 0.04267, 'exc_density': 1710,
        'exc AR(-)': 0.64, 'exc 1/AR': 0, 'exc Sphericity(-)': 0.86, 'exc Elongation(-)': 0.58},
    4444: {'exc_d10': 25.3, 'exc_d50': 93.9, 'exc_d90': 163, 'exc_d32': 38, 'exc_surf_eng': 0.03746, 'exc_density': 1504,
        'exc AR(-)': 0, 'exc 1/AR': 1.62, 'exc Sphericity(-)': 0.85, 'exc Elongation(-)': 0.59},
    5555: {'exc_d10': 1.3, 'exc_d50': 4.2, 'exc_d90': 22, 'exc_d32': 3.23, 'exc_surf_eng': 0.042505, 'exc_density': 1734,
        'exc AR(-)': 0.62, 'exc 1/AR': 0, 'exc Sphericity(-)': 0.86, 'exc Elongation(-)': 0.53},
    6666: {'exc_d10': 65, 'exc_d50': 45, 'exc_d90': 103, 'exc_d32': 11.8, 'exc_surf_eng': 0.042505, 'exc_density': 1734,
        'exc AR(-)': 0.66, 'exc 1/AR': 1.612903226, 'exc Sphericity(-)': 0.86, 'exc Elongation(-)': 0.55},
    7777: {'exc_d10': 4.5, 'exc_d50': 27.9, 'exc_d90': 88.5, 'exc_d32': 10.3, 'exc_surf_eng': 0.03437, 'exc_density': 1528,
        'exc AR(-)': 0.67, 'exc 1/AR': 1.49, 'exc Sphericity(-)': 0.88, 'exc Elongation(-)': 0.59},
    8888: {'exc_d10': 3.5, 'exc_d50': 21.9, 'exc_d90': 56.2, 'exc_d32': 7.3, 'exc_surf_eng': 0.03437, 'exc_density': 1546,
        'exc AR(-)': 0.67, 'exc 1/AR': 1.49, 'exc Sphericity(-)': 0.9, 'exc Elongation(-)': 0.64},
    9999: {'exc_d10': 1.5, 'exc_d50': 9.6, 'exc_d90': 35.4, 'exc_d32': 4.1, 'exc_surf_eng': 0.04382, 'exc_density': 1642,
        'exc AR(-)': 0.66, 'exc 1/AR': 1.515151515, 'exc Sphericity(-)': 0.91, 'exc Elongation(-)': 0.71},
    11111: {'exc_d10': 0.88, 'exc_d50': 3.2, 'exc_d90': 7.9, 'exc_d32': 2.11, 'exc_surf_eng': 0.043475, 'exc_density': 1739,
        'exc AR(-)': 0.6, 'exc 1/AR': 1.666666667, 'exc Sphericity(-)': 0.91, 'exc Elongation(-)': 0.71},
}
#calling vae model
X = data[['API', 'excipient', 'API_percent', 'API_coat_percent', 'exc_coat_percent', 'API_silica_type', 'exc_silica_type']]
y = data[['FF']]

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np

# Assuming 'X' and 'y' are your features and labels respectively
scaler_X = StandardScaler()
X_scaled = scaler_X.fit_transform(X)

# Convert 'y' to a numpy array if it's a pandas DataFrame or Series
y = y.to_numpy() if hasattr(y, 'to_numpy') else y

# Splitting the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# Convert to TensorFlow tensors
X_train_tensor = tf.convert_to_tensor(X_train, dtype=tf.float32)
y_train_tensor = tf.convert_to_tensor(y_train, dtype=tf.float32)
X_test_tensor = tf.convert_to_tensor(X_test, dtype=tf.float32)
y_test_tensor = tf.convert_to_tensor(y_test, dtype=tf.float32)

# Create TensorFlow datasets
train_dataset = tf.data.Dataset.from_tensor_slices((X_train_tensor, y_train_tensor)).shuffle(buffer_size=1024).batch(64)
val_dataset = tf.data.Dataset.from_tensor_slices((X_test_tensor, y_test_tensor)).batch(64)

class Sampling(tf.keras.layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.keras.backend.shape(z_mean)[0]
        dim = tf.keras.backend.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.keras.backend.exp(0.5 * z_log_var) * epsilon
    
#initializing VAE MODEL
class VAE(tf.keras.Model):
    def __init__(self, input_size, latent_size, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.fc1 = tf.keras.layers.Dense(128, activation='relu')
        self.fc2 = tf.keras.layers.Dense(64, activation='relu')
        self.fc31 = tf.keras.layers.Dense(latent_size)
        self.fc32 = tf.keras.layers.Dense(latent_size)
        self.sampling = Sampling()
        self.fc4 = tf.keras.layers.Dense(64, activation='relu')
        self.fc5 = tf.keras.layers.Dense(128, activation='relu')
        self.fc6 = tf.keras.layers.Dense(input_size + 1)  # Adjusting for FF and FF_regime

    def encode(self, x):
        h1 = self.fc1(x)
        h2 = self.fc2(h1)
        return self.fc31(h2), self.fc32(h2)

    def reparameterize(self, mu, logvar):
        z = self.sampling((mu, logvar))
        return z

    def decode(self, z):
        h3 = self.fc4(z)
        h4 = self.fc5(h3)
        return self.fc6(h4)

    def call(self, inputs):
        mu, logvar = self.encode(inputs)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

# Example usage
input_size = 7  # Example input size, adjust as necessary
latent_size = 64
vae1 = VAE(input_size=input_size, latent_size=latent_size)

def custom_vae_loss(model, x, reconstructed, additional_outputs, y_true):
    # Reconstruction loss (e.g., MSE for the input data)
    reconstruction_loss = tf.keras.losses.mean_squared_error(x, reconstructed)

    # Additional target prediction loss (e.g., MSE for the additional outputs)
    additional_loss = tf.keras.losses.mean_squared_error(y_true, additional_outputs)

    # KL divergence loss (already added in the model as an added loss)
    kl_loss = sum(model.losses)

    # Total loss
    total_loss = tf.reduce_mean(reconstruction_loss + additional_loss) + kl_loss
    return total_loss

optimizer = tf.keras.optimizers.Adam(1e-4)

@tf.function
def train_step(model, x_batch, y_batch, optimizer):
    with tf.GradientTape() as tape:
        x_recon, mu, logvar = model(x_batch, training=True)
        recon_loss = tf.reduce_sum(tf.keras.losses.mse(tf.concat([x_batch, y_batch], axis=1), x_recon))
        kl_div = -0.5 * tf.reduce_sum(1 + logvar - tf.square(mu) - tf.exp(logvar))
        loss = recon_loss + kl_div
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

def train_vae(model, train_dataset, optimizer, epochs=100):
    for epoch in range(epochs):
        for x_batch, y_batch in train_dataset:
            loss = train_step(model, x_batch, y_batch, optimizer)
        print(f'Epoch {epoch}, Loss: {loss.numpy()}')

# Assuming train_dataset is a tf.data.Dataset object
train_vae(vae1, train_dataset, optimizer)

@app.route('/')
def index():
    # You can pass an error message to render on the page, which by default is empty.
    return render_template('index.html', error='')

@app.route('/RF')
def rf():
    return render_template('RF.html')

@app.route('/VAE')
def vae():
    return render_template('VAE.html')

@app.route('/GAN')
def gan():
    return render_template('GAN.html')

@app.route('/CNN')
def cnn():
    return render_template('CNN.html')

@app.route('/cimsepp', methods=['GET', 'POST'])
def cimsepp():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if database.check_credentials(username, password):
            return render_template('cimsepp.html')
        else:
            # Instead of flashing a message, pass an error message to the template.
            return render_template('index.html', error='You do not have access. Please contact Dr. Joshua Young at joshua.a.young@njit.edu to get the access!')
    return redirect(url_for('index'))


@app.route('/predictRF', methods=['POST'])
def predictRF():
    # Get data from request
    data = request.json

    # Safely get and convert silica type values, defaulting to 0 if conversion fails
    def safe_convert_silica_type(value):
        try:
            return int(value)
        except (ValueError, TypeError):
            return 0

    api_silica_type = safe_convert_silica_type(data.get('API_silica_type', 0))
    exc_silica_type = safe_convert_silica_type(data.get('exc_silica_type', 0))

    # Check if the silica type values are valid
    valid_silica_types = [0, 1, 2]
    if api_silica_type not in valid_silica_types:
        api_silica_type = 0
    if exc_silica_type not in valid_silica_types:
        exc_silica_type = 0

    # Prepare the data for prediction
    api = int(data['api'])
    excipient = int(data['excipient'])
    api_features = api_values[api]
    excipient_features = excipient_values[excipient]

    # Prepare the prediction features
    prediction_features = {
        'API': api,
        'excipient': excipient,
        'API_percent': int(data['api_percent']),
        'API_coated': int(data['api_coated']),
        'API_coat_percent': int(data['api_coat_percent']) if data['api_coated'] else 0,
        'exc_coated': int(data['exc_coated']),
        'exc_coat_percent': int(data['exc_coat_percent']) if data['exc_coated'] else 0,
        'API_silica_type': api_silica_type,
        'exc_silica_type': exc_silica_type
    }

    # Combine all features
    all_features = {**prediction_features, **api_features, **excipient_features}
    new_combination = pd.DataFrame([all_features])

    # Make the prediction
    predicted_ff_ff_regime = optimized_model.predict(new_combination[feature_columns])
    predicted_ff= round(predicted_ff_ff_regime[0][0], 4)
    ff_regime_value = predicted_ff_ff_regime[0][1]


    # Map the ff_regime_value to the corresponding range and label
    if ff_regime_value < 2:
        ff_regime_label = "Cohesive"
    elif 2 <= ff_regime_value < 4:
        ff_regime_label = "Cohesive"
    elif 4 <= ff_regime_value< 6:
        ff_regime_label = "Easy Flowing"
    elif 6 <= ff_regime_value < 10:
        ff_regime_label = "Easy Flowing"
    elif ff_regime_value >= 10:
        ff_regime_label = "Free Flowing"
    else:
        ff_regime_label = "Unknown"  # You can decide how to handle out-of-range values

    # Return the result along with all the additional features
    return jsonify({
        "api": api,
        "excipient": excipient,
        "api_percent": data['api_percent'],
        "api_coated": data['api_coated'],
        "api_coat_percent": data['api_coat_percent'],
        "exc_coated": data['exc_coated'],
        "exc_coat_percent": data['exc_coat_percent'],
        "api_silica_type": data['api_silica_type'],
        "exc_silica_type": data['exc_silica_type'],
        "API_surf_eng": api_features.get('API_surf_eng'),
        "API_density": api_features.get('API_density'),
        "API AR(-)": api_features.get('API AR(-)', None),
        "API 1/AR": api_features.get('API 1/AR', None),
        "API Sphericity(-)": api_features.get('API Sphericity(-)', None),
        "API Elongation(-)": api_features.get('API Elongation(-)', None),
        "exc_surf_eng": excipient_features.get('exc_surf_eng', None),
        "exc_density": excipient_features.get('exc_density', None),
        "exc AR(-)": excipient_features.get('exc AR(-)', None),
        "exc 1/AR": excipient_features.get('exc 1/AR', None),
        "exc Sphericity(-)": excipient_features.get('exc Sphericity(-)', None),
        "exc Elongation(-)": excipient_features.get('exc Elongation(-)', None),
        "predicted_ff": float(ff_regime_value),
        "predicted_ff_regime": ff_regime_label
    })

@app.route('/predictVAE', methods=['POST'])
def predictVAE():
    # Get data from request
    data = request.json

    # Safely get and convert silica type values, defaulting to 0 if conversion fails
    def safe_convert_silica_type(value):
        try:
            return int(value)
        except (ValueError, TypeError):
            return 0

    api_silica_type = safe_convert_silica_type(data.get('API_silica_type', 0))
    exc_silica_type = safe_convert_silica_type(data.get('exc_silica_type', 0))

    # Check if the silica type values are valid
    valid_silica_types = [0, 1, 2]
    if api_silica_type not in valid_silica_types:
        api_silica_type = 0
    if exc_silica_type not in valid_silica_types:
        exc_silica_type = 0

    # Prepare the data for prediction
    api = int(data['api'])
    excipient = int(data['excipient'])
    api_features = api_values[api]
    excipient_features = excipient_values[excipient]

    # Prepare the prediction features
    prediction_features = {
        'API': api,
        'excipient': excipient,
        'API_percent': int(data['api_percent']),
        #'API_coated': int(data['api_coated']),
        'API_coat_percent': int(data['api_coat_percent']) ,
        #'exc_coated': int(data['exc_coated']),
        'exc_coat_percent': int(data['exc_coat_percent']) ,
        'API_silica_type': api_silica_type,
        'exc_silica_type': exc_silica_type
    }

    # Combine all features
    all_features = {**prediction_features, **api_features, **excipient_features}
    new_combinations = pd.DataFrame([all_features])
    print("##### Debug #####")
    print(new_combinations)
    print(new_combinations.columns)
    print(new_combinations.shape)
    new_combinations = new_combinations[['API', 'excipient', 'API_percent', 'API_coat_percent', 'exc_coat_percent', 'API_silica_type', 'exc_silica_type']]
    print("##### Changes Completed #####")
    print(new_combinations)
    print(new_combinations.columns)
    print(new_combinations.shape)
    print("##### Debug Completed #####")


    def get_ff_regime_label(ff_regime_value):
        if ff_regime_value < 2:
            return "Cohesive"
        elif 2 <= ff_regime_value < 4:
            return "Cohesive"
        elif 4 <= ff_regime_value < 6:
            return "Easy-Flowing"
        elif 6 <= ff_regime_value < 10:
            return "Easy-Flowing"
        elif ff_regime_value >= 10:
            return "Free-Flowing"
        else:
            ff_regime_label = "Unknown" # You can decide how to handle out-of-range values
    
    # Initialize a list to collect ff_regime_labels
    ff_regime_labels = []

    #make predictions
    for _ in range(20):
        new_combination_scaled = scaler_X.fit_transform(new_combinations)
        new_combination_tensor = tf.convert_to_tensor(new_combination_scaled, dtype=tf.float32)
        predictions = vae1.predict(new_combination_tensor)
        reconstructed_features = predictions[0]
        ff_predictions = reconstructed_features[:, 7:]
        if hasattr(ff_predictions, 'numpy'):
            ff_regime_value = ff_predictions.numpy().flatten()[0]
        else:
            ff_regime_value = ff_predictions.flatten()[0]
        # Map the ff_regime_value to its corresponding label
        ff_regime_label = get_ff_regime_label(ff_regime_value)
        ff_regime_labels.append(ff_regime_label)
    
    print(ff_regime_labels)
    label_counter = Counter(ff_regime_labels)
    # Find the most common label
    most_common_label, occurrences = label_counter.most_common(1)[0]

    try:
        print(f"The most common label is '{most_common_label}' with {occurrences} occurrences.")
    except Exception as e:
        print("An error occurred:", e)


    # Return the result along with all the additional features
    return jsonify({
        "api": api,
        "excipient": excipient,
        "api_percent": data['api_percent'],
        #"api_coated": data['api_coated'],
        "api_coat_percent": data['api_coat_percent'],
        #"exc_coated": data['exc_coated'],
        "exc_coat_percent": data['exc_coat_percent'],
        "api_silica_type": data['api_silica_type'],
        "exc_silica_type": data['exc_silica_type'],
        "API_surf_eng": api_features.get('API_surf_eng'),
        "API_density": api_features.get('API_density'),
        "API AR(-)": api_features.get('API AR(-)', None),
        "API 1/AR": api_features.get('API 1/AR', None),
        "API Sphericity(-)": api_features.get('API Sphericity(-)', None),
        "API Elongation(-)": api_features.get('API Elongation(-)', None),
        "exc_surf_eng": excipient_features.get('exc_surf_eng', None),
        "exc_density": excipient_features.get('exc_density', None),
        "exc AR(-)": excipient_features.get('exc AR(-)', None),
        "exc 1/AR": excipient_features.get('exc 1/AR', None),
        "exc Sphericity(-)": excipient_features.get('exc Sphericity(-)', None),
        "exc Elongation(-)": excipient_features.get('exc Elongation(-)', None),
        "predicted_ff": float(ff_regime_value),
        "predicted_ff_regime": most_common_label
    })

@app.route('/logout', methods=['POST'])
def logout():
    # Here you would implement your logout logic, such as clearing session data
    return redirect(url_for('index'))


if __name__ == '__main__':
    database.setup()  # Set up the database (create table etc.)
    app.run(debug=True, use_reloader= False )
