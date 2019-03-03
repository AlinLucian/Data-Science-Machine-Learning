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
Y_training = training_data_df[['total_earnings']].values

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


# Start neural network model definition (Input layer, three computational layers, and one output layer)
# Define model parameters
learning_rate = 0.001
# an epoch is a full training pass over the full data set
training_epochs = 100
display_step = 5

# Define how many inputs and outputs are in our neural network
number_of_inputs = 9   # 9 input nodes because we have 9 columns in the frame we previously loaded
number_of_outputs = 1  # 1 output node because we are interested in predicting a number(profit)

# Define how many nodes(neurons) we want in each layer of our neural network
# Values can be adjusted in accordance with prediction accuracy
layer_1_nodes = 50
layer_2_nodes = 100
layer_3_nodes = 50


# Define the layers of the neural network:


# Define Input Layer:
with tf.variable_scope('input'):
    # node will accept batches of 9 inputs
    # of any size
    X = tf.placeholder(tf.float32, shape=(None, number_of_inputs))


# Define Layer 1:
with tf.variable_scope('layer_1'):
    # value for each connection between it's nodes and the nodes in the previous layer
    # shape is determined by the variable's value for each connection between layer
    # nodes and previous layer nodes
    # the weights array is initialize using the 'Xavier Initialization' algorithm
    weights = tf.get_variable(name='weights1', shape=[number_of_inputs, layer_1_nodes], initializer=tf.contrib.layers.xavier_initializer())

    # bias value for each node set as a variable of shape equal to layer's node size (set above)
    # bias variable initialized with zeroes filled in
    biases = tf.get_variable(name='biases1', shape=[layer_1_nodes], initializer=tf.zeros_initializer())

    # Multiply weights by inputs and call activation function that outputs the result of the layer
    # We will use matrix multiplication and the standard rectified linear unit ('relu') activation function
    # Multiply inputs by weights, add biases, then call the relu activation function for the result
    layer_1_output = tf.nn.relu(tf.matmul(X, weights) + biases)


# Define Layer 2 using the same method as for Layer 1
with tf.variable_scope('layer_2'):
    # variable shape changes, as the inputs are now the nodes of the first layer
    weights = tf.get_variable(name='weights2', shape=[layer_1_nodes, layer_2_nodes], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name='biases2', shape=[layer_2_nodes], initializer=tf.zeros_initializer())
    layer_2_output = tf.nn.relu(tf.matmul(layer_1_output, weights) + biases)


# Define Layer 3 using the same method as for Layer 1 & 2
with tf.variable_scope('layer_3'):
    weights = tf.get_variable(name='weights3', shape=[layer_2_nodes, layer_3_nodes], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name='biases3', shape=[layer_3_nodes], initializer=tf.zeros_initializer())
    layer_3_output = tf.nn.relu(tf.matmul(layer_2_output, weights) + biases)


# Define Output Layer
with tf.variable_scope('output'):
    # make sure to change the shapes according to the output layer specifics
    weights = tf.get_variable(name='weights4', shape=[layer_3_nodes, number_of_outputs], initializer=tf.contrib.layers.xavier_initializer())
    biases = tf.get_variable(name='biases4', shape=[number_of_outputs], initializer=tf.zeros_initializer())
    prediction = tf.nn.relu(tf.matmul(layer_3_output, weights) + biases)


# Define the cost function(or loss function) of the neural network
# The cost function is necessary in training the neural network
# The cost function tells us how wrong the neural network is when making
# a prediction based on a single serving of training data

with tf.variable_scope('cost'):
    Y = tf.placeholder(tf.float32, shape=(None, 1))

    # This will be the mean squared error between the prediction(calculated above)
    # and what we expected (Y)
    # Note: tf's reduce_mean and squared_difference functions
    # are used to obtains the mean squared difference
    cost = tf.reduce_mean(tf.squared_difference(prediction, Y))


# Define the optimizer function that will be run to optimize the neural network
# This represents the actual training, as the optimizer function will run until
# the cost function returns a value as low as possible, meaning the difference
# between the predicted values and expected values is minimal

with tf.variable_scope('train'):
    # We use a standard optimizer called 'Adam' which receives a learning rate parameter, defined above
    # Then we use the minimize function from the Adam optimizer tell it which variable we want it minimize
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

# Create a summary to log the progress of the network
with tf.variable_scope('logging'):
    tf.summary.scalar('current_cost', cost)
    summary = tf.summary.merge_all()  # when called, this node runs all summary nodes in the graph

# Object used to save the model after it is trained
saver = tf.train.Saver()

# To run an operation on a Tensorflow graph a session is needed
with tf.Session() as session:

    # Run the global variable initializer to initialize all variables and layers to their default values
    session.run(tf.global_variables_initializer())

    # Create log file writers to record the training progress
    # Training and testing data should be logged separately
    # Saving log files in the same parent folder allows TensorBoard to display them together
    # and gives you the ability to click between them
    training_writer = tf.summary.FileWriter('./logs/training', session.graph)
    testing_writer = tf.summary.FileWriter('./logs/testing', session.graph)

    # We run the optimizer to train the network for a fixed number of times or
    # until the cost function is within acceptable values
    # An epoch is a full run through the training data set.
    for epoch in range(training_epochs):

        # Feed the training data (X_scaled_training -- inputs, Y_scaled_training -- expected results)
        # to the optimizer function and do one step of neural network training
        session.run(optimizer, feed_dict={X: X_scaled_training, Y: Y_scaled_training})

    # Print the current training pass(epoch) to the screen
        print("Training pass: {}".format(epoch))

    # Every 5 epochs log progress to see if there is improved prediction accuracy
        if epoch % 5 == 0:
            training_cost, training_summary = session.run([cost, summary],  feed_dict={X: X_scaled_training, Y: Y_scaled_training})
            testing_cost, testing_summary = session.run([cost, summary],  feed_dict={X: X_scaled_testing, Y: Y_scaled_testing})

            print('Epoch: {}'.format(epoch), '-- Current training cost: {}'.format(training_cost), 'Current testing cost: {}'.format(testing_cost))

            # Write the current training status to the log files
            training_writer.add_summary(training_summary, epoch)
            testing_writer.add_summary(testing_summary, epoch)

    # Training is now complete!
    print("Training is complete!")

    # Print out final training and testing cost values
    final_training_cost = session.run(cost, feed_dict={X: X_scaled_training, Y: Y_scaled_training})
    final_testing_cost = session.run(cost, feed_dict={X: X_scaled_testing, Y: Y_scaled_testing})
    print("Final Training cost: {}".format(final_training_cost))
    print("Final Testing cost: {}".format(final_testing_cost))

    # Now that the neural network is trained, let's use it to make predictions for our test data.
    # Pass in the X testing data and run the prediciton operation
    Y_predicted_scaled = session.run(prediction, feed_dict={X: X_scaled_testing})

    # Rescale the data back to it's original units (dollars)
    Y_predicted = Y_scaler.inverse_transform(Y_predicted_scaled)

    # Select the first game and pass it to the model
    real_earnings = test_data_df['total_earnings'].values[0]
    predicted_earnings = Y_predicted[0][0]

    print("The actual earnings of Game #1 were ${}".format(real_earnings))
    print("Predicted earnings for Game #1 were ${}".format(predicted_earnings))

    # We save the trained model as a .ckpt -- checkpoint file
    # It can now be used in other sessions by calling the saver.restore() method
    save_path = saver.save(session, 'logs\/trained_model.ckpt')
    print('Model saved: {}'.format(save_path))
