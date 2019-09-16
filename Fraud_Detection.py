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






































