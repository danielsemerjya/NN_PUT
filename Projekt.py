import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import numpy as np


# DATA SET IMPORT
header_list = ['serial', 'date', 'age', 'distance', 'stores', 'latitude', 'longitude', 'price']
column_names = "nazwa kolumny"
df = pd.read_csv('data.csv', names=header_list)
df.head()

print("DATASET LOADED")

# DATA NORMALIZATION
df = df.iloc[:,1:]
df_norm = (df - df.mean()) / df.std()
df_norm.head()

# FUNCTION TO UN-NORMALIZE LABEL (price) TO REGULAR SIZE
y_mean = df['price'].mean()
y_std = df['price'].std()

def convert_label_value(pred):
    return int(pred * y_std + y_mean)

# CREATE TRAINING AND TEST SET
X = df_norm.iloc[:, :6]
X.head()   # zostawic to head czy nie? :)
Y = df_norm.iloc[:,-1]
Y.head()

X_arr = X.values  # REJECT COLUMN NAMES
Y_arr = Y.values

# TRAINING AND TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(X_arr, Y_arr, test_size = 0.05, shuffle = True, random_state=0)
print('X_train shape: ', X_train.shape)
print('y_train shape: ', y_train.shape)
print('X_test shape: ', X_test.shape)
print('y_test shape: ', y_test.shape)

# CREATING MODEL

def get_model():

    model = Sequential([
        Dense(10, input_shape = (6,), activation = 'relu'),
        Dense(20, activation='relu'),
        Dense(5, activation='relu'),
        Dense(1)
    ])
    model.compile(
        loss='mse',
        optimizer='adam'
    )
    return model

model = get_model()
model.summary()

print("Model created")


# MODEL TRAINING

early_stopping = EarlyStopping(monitor='val_loss', patience = 5)
model = get_model()
preds_on_untrained = model.predict(X_test)

history = model.fit(
    X_train, y_train,
    validation_data = (X_test, y_test),
    epochs = 1000,
    callbacks = [early_stopping]
)

# PLOT TRAINING AND VALIDATION LOSS

def plot_loss(history):
    h = history.history
    x_lim = len(h['loss'])
    plt.figure(figsize=(8, 8))
    plt.plot(range(x_lim), h['val_loss'], label = 'Validation Loss')
    plt.plot(range(x_lim), h['loss'], label = 'Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    print("Minimalna wartosc loss function", round(min(h['loss']), 4))
    return

plot_loss(history)

# PREDICTION

def compare_predictions(preds1, preds2, y_test):
    plt.figure(figsize=(8, 8))
    plt.plot(preds1, y_test, 'ro', label='Untrained Model')
    plt.plot(preds2, y_test, 'go', label='Trained Model')
    plt.xlabel('Preds')
    plt.ylabel('Labels')

    y_min = min(min(y_test), min(preds1), min(preds2))
    y_max = max(max(y_test), max(preds1), max(preds2))

    plt.xlim([y_min, y_max])
    plt.ylim([y_min, y_max])
    plt.plot([y_min, y_max], [y_min, y_max], 'b--')
    plt.legend()
    plt.show()
    return


preds_on_trained = model.predict(X_test)
compare_predictions(preds_on_untrained, preds_on_trained, y_test)

price_on_untrained = [convert_label_value(y) for y in preds_on_untrained]
price_on_trained = [convert_label_value(y) for y in preds_on_trained]
price_y_test = [convert_label_value(y) for y in y_test]

def error_indicator(pred, true):
    error = list(set(pred)-set(true))
    error_ind = sum(error)/len(error) / sum(true)/len(true)
    error = np.array(error)
    error_std = np.std(error, dtype=np.float64)
    return error_ind, error_std
error_ind = error_indicator(price_on_trained, price_y_test)

print("Wskaznik średniego błędu pomiędzy wartością predykowaną oraz średnią wartością ceny nieruchomości: ", round(error_ind[0]*100, 5), "%")
print("Odchylenie standardowe: ", round(error_ind[1], 5))
compare_predictions(price_on_untrained, price_on_trained, price_y_test)


