import os
import tensorflow as tf

# Turn off TensorFlow warning messages in program output
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Define computational graph (nodes X and Y, which will be the inputs)
X = tf.placeholder(tf.float32, name='X')
Y = tf.placeholder(tf.float32, name='Y')

# Define the node that actually pulls data from nodes X and Y
# and does the addition--tf.add is used to do this
addition = tf.add(X, Y, name='addition')


# Create the session
with tf.Session() as session:

    # Pass in the operation we want to run
    # We also need to pass in the values for X and Y as arrays-- feed_dict function is used to do this
    result = session.run(addition, feed_dict={X: [1, 2, 10], Y: [4, 3, 43]})

    # Print the resulting tensor (array)
    print(result)
