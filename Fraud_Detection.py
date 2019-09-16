import pandas as pd
import numpy as np

# Import and store dataset
credit_card_data = pd.read_csv('creditcard.csv')
# print(credit_card_data)

# Splitting data into 4 sets
# 1. Shuffle/randomize data
# 2. One-hot encoding
# 3. Normalize
# 4. Splitting up X/Y values
# 5. Convert data_frames to numpy arrays (float32)
# 6. Splitting the final data into X/Y train/test

# 1. Shuffle/randomize data
shuffle_data = credit_card_data.sample(frac=1)
# 2. One-hot encoding
one_hot_data = pd.get_dummies(shuffle_data, columns=['Class'])
# 3. Normalize
normalized_data = (one_hot_data - one_hot_data.min()) / (one_hot_data.max() - one_hot_data.min())
# 4. Splitting up X/Y values
df_X = normalized_data.drop(['Class_0', 'Class_1'], axis=1)
df_y = normalized_data[['Class_0', 'Class_1']]
# 5. Convert data_frames to numpy arrays (float32)
ar_X, ar_y = np.asarray(df_X.values, dtype='float32'), np.asarray(df_y.values, dtype='float32')
# 6. Splitting the final data into X/Y train/test
train_size = int(0.8 * len(ar_X))
(raw_X_train, raw_y_train) = (ar_X[:train_size], ar_y[:train_size])
(raw_X_test, raw_y_test) = (ar_X[train_size:], ar_y[train_size:])

count_legit, count_fraud = np.unique(credit_card_data['Class'], return_counts=True)[1]
fraud_ratio = float(count_fraud / (count_legit + count_fraud))
print("Percent of fraudulent transactions: ", fraud_ratio)

weighting = 1 / fraud_ratio
raw_y_train[:, 1] = raw_y_train[:,  1] * weighting

import tensorflow as tf

input_dimensions = ar_X.shape[1]
output_dimensions = ar_y.shape[1]
num_layer_1_cells = 100
num_layer_2_cells = 150

X_train_node = tf.placeholder(tf.float32, [None, input_dimensions], name='X_train')
y_train_node = tf.placeholder(tf.float32, [None, output_dimensions], name='y_train')

X_test_node = tf.constant(raw_X_test, name='X_test')
y_test_node = tf.constant(raw_y_test, name='y_test')

weight_1_node = tf.Variable(tf.zeros([input_dimensions, num_layer_1_cells]), name='weight_1')
biases_1_node = tf.Variable(tf.zeros([num_layer_1_cells]), name='biases_1')

weight_2_node = tf.Variable(tf.zeros([num_layer_1_cells, num_layer_2_cells]), name='weight_2')
biases_2_node = tf.Variable(tf.zeros([num_layer_2_cells]), name='biases_2')

weight_3_node = tf.Variable(tf.zeros([num_layer_2_cells, output_dimensions]), name='weight_3')
biases_3_node = tf.Variable(tf.zeros([output_dimensions]), name='biases_3')






























