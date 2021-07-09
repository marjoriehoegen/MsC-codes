import pandas as pd
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# Create sequences for network data input
def create_sequences(values, time_steps):
	output = []
	for i in range(len(values) - time_steps):
		output.append(values[i : (i + time_steps)])
	return np.stack(output)

# Prepare train and test data sets for convolutional model
def prepare_data_conv(variable,time_steps,t):

	h_training = variable[:t]
	training_mean = h_training.mean()
	training_std = h_training.std()
	h_training_norm = (h_training - training_mean) / training_std
	x_train = create_sequences(h_training_norm,time_steps)
	x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))

	h_testing = variable[t:]
	h_testing_norm = (h_testing - training_mean) / training_std
	x_test = create_sequences(h_testing_norm,time_steps)
	x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

	return x_train, x_test


# Convolutional autoencoder
def convolutional_autoencoder(x_train):

	# Model definition
	model = keras.Sequential(
	    [
	        layers.Input(shape=(x_train.shape[1], x_train.shape[2])),
	        layers.Conv1D(
	            filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
	        ),
	        layers.Dropout(rate=0.2),
	        layers.Conv1D(
	            filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
	        ),
	        layers.Conv1DTranspose(
	            filters=16, kernel_size=7, padding="same", strides=2, activation="relu"
	        ),
	        layers.Dropout(rate=0.2),
	        layers.Conv1DTranspose(
	            filters=32, kernel_size=7, padding="same", strides=2, activation="relu"
	        ),
	        layers.Conv1DTranspose(filters=1, kernel_size=7, padding="same"),
	    ]
	)

	model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")

	return model

# Model training
def train(x_train, model):

	history = model.fit(
	    x_train,
	    x_train,
	    epochs=50,
	    batch_size=50,
	    validation_split=0.1,
	    callbacks=[
	        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, mode="min")
	    ],
	)

	return history, model

# Calculate train and test MAE to detect anomalies
def detect_anomalies(model, x_train, x_test, limit):

	x_train_pred = model.predict(x_train)
	train_mae_loss = np.mean(np.abs(x_train_pred - x_train), axis=1)
	threshold = np.max(train_mae_loss)

	x_test_pred = model.predict(x_test)
	test_mae_loss = np.mean(np.abs(x_test_pred - x_test), axis=1)
	test_mae_loss = test_mae_loss.reshape((-1))

	anomalies = test_mae_loss > (limit*threshold)

	return anomalies, train_mae_loss, test_mae_loss

# Monitors variables from a df with a convolutional autoencoder
def test_all(df, t, limit,npts_anomaly):
	'''
		Monitors a df with variables as columns
		Returns column indexes of the altered variables

		Parameters:
		t: number of points/samples for training data set
		limit: detection limit, related to maximum training MAE
		npts_anomaly: number of minimum anomaly points to consider an alteration

	'''
	altered_variables = []

	for column in df:
		x = df[column]
		x = x.to_numpy()
		
		# Model training
		time_steps = 100
		x_train, x_test = prepare_data_conv(x,time_steps,t)
		model = convolutional_autoencoder(x_train)
		history, model = train(x_train, model)

		# Alteration detection
		anomalies, train_mae, test_mae = detect_anomalies(model, x_train, x_test, limit)

		if np.sum(anomalies) > npts_anomaly:
			altered_variables.append(column)
			
	return altered_variables