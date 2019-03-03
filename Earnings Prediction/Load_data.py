import tensorflow as tf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# The data has already been cleaned and split into training and test data.
# We load the training data(from the training data csv file) into a pandas data frame.
training_data_df = pd.read_csv('sales_data_training.csv', dtype=float)

# Retrieve the columns used to train the model in X_training,
# then the columns we want to predict in Y_training.

# To get values for X_training, we drop the total_earnings column. The result will be an array.
X_training = training_data_df.drop('total_earnings', axis=1).values

# To get values for Y_training, we retrieve only the earnings column into a one-dimensional array.
Y_training =training_data_df[['total_earnings']].values

# We load the TEST data(from the test data csv file) into a pandas data frame.
test_data_df = pd.read_csv('sales_data_test.csv', dtype=float)

# Retrieve the columns used to train the model in X_testing,
# then the columns we want to predict in Y_testing.

# To get values for X_testing, we drop the total_earnings column. The result will be an array.
X_testing = test_data_df.drop('total_earnings', axis=1).values

# To get values for Y_testing, we retrieve only the earnings column into a one-dimensional array.
Y_testing = test_data_df[['total_earnings']].values

# All data needs to be scaled to a small range like 0 to 1 for the neural network to work well.

# Define scaler objects for X and Y
X_scaler = MinMaxScaler(feature_range=(0, 1))
Y_scaler = MinMaxScaler(feature_range=(0, 1))

# Use the fit_transform function to scale both the training inputs and outputs
X_scaled_training = X_scaler.fit_transform(X_training)
Y_scaled_training = Y_scaler.fit_transform(Y_training)

# IMPORTANT: the training and test data should be scaled with the same scaler.
X_scaled_testing = X_scaler.transform(X_testing)
Y_scaled_testing = Y_scaler.transform(Y_testing)

# Print out the array shapes(dimensions) of the loaded test data
print(X_scaled_testing.shape)
print(Y_scaled_testing.shape)

# Print out information about how the data was scaled
print("Note: Y values were scaled by multiplying by {:.10f} and adding {:.4f}".format(Y_scaler.scale_[0], Y_scaler.min_[0]))
print("Note: X values were scaled by multiplying by {:.10f} and adding {:.4f}".format(X_scaler.scale_[0], X_scaler.min_[0]))
