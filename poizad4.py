import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix
from tensorflow import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical
import matplotlib.pyplot as plt

df = pd.read_csv("texture_features.csv")

X = df.drop("category", axis=1).values
y = df["category"].values

label_encoder = LabelEncoder()
y_int = label_encoder.fit_transform(y)

onehot_encoder = OneHotEncoder(sparse_output=False)
y_onehot = onehot_encoder.fit_transform(y_int.reshape(-1, 1))

x_train, x_test, y_train, y_test = train_test_split(X, y_onehot, test_size=0.3)

model = Sequential()
model.add(Dense(10, activation='sigmoid', input_dim=X.shape[1]))
model.add(Dense(y_onehot.shape[1], activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=100, batch_size=10, shuffle=True)

y_pred = model.predict(x_test)
y_pred_int = np.argmax(y_pred, axis=1)
y_test_int = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_test_int, y_pred_int)
print("Confusion Matrix:")
print(cm)