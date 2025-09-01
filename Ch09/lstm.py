from keras.datasets import imdb
from sklearn.model_selection import train_test_split
from keras.preprocessing.sequence import pad_sequences
import keras
import matplotlib.pyplot as plt


(train_input, train_target), (test_input, test_target) = imdb.load_data(num_words=500)
train_input, val_input, train_target, val_target = train_test_split(train_input, train_target, test_size=0.2, random_state=42)

train_seq = pad_sequences(train_input, maxlen=100)
val_seq = pad_sequences(val_input, maxlen=100)

def plot_loss(history):
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()
    plt.show()

""" LSTM 순환층 """
# model_lstm = keras.Sequential()
# model_lstm.add(keras.layers.Input(shape=(100,)))
# model_lstm.add(keras.layers.Embedding(500, 16))
# model_lstm.add(keras.layers.LSTM(8))
# model_lstm.add(keras.layers.Dense(1, activation='sigmoid'))
# # 모델 구조 확인
# model_lstm.summary()    # SimpleRnn 대비, 셀 4개만큼 4배 되어 총 800개 모델 파라미터
#
# ## 모델 컴파일 및 훈련
# model_lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# checkpoint_cb = keras.callbacks.ModelCheckpoint('best-lstm-model.keras', save_best_only=True)
# early_stopping_cb = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
# history = model_lstm.fit(train_seq, train_target, epochs=100, batch_size=64,
#                          validation_data=(val_seq, val_target),
#                          callbacks=[checkpoint_cb, early_stopping_cb])
#
# ## 손실 그래프 도식화
# plot_loss(history)


""" 순환층에 Dropout 적용 """
# model_dropout = keras.Sequential()
# model_dropout.add(keras.layers.Input(shape=(100,)))
# model_dropout.add(keras.layers.Embedding(500, 16))
# model_dropout.add(keras.layers.LSTM(8, dropout=0.2))
# model_dropout.add(keras.layers.Dense(1, activation='sigmoid'))
#
# # 모델 컴파일 및 훈련
# model_dropout.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# checkpoint_cb = keras.callbacks.ModelCheckpoint('beset-dropout-model.keras', save_best_only=True)
# early_stopping_cb = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)
# history = model_dropout.fit(train_seq, train_target, epochs=100, batch_size=64,
#                             validation_data=(val_seq, val_target),
#                             callbacks=[checkpoint_cb, early_stopping_cb])
#
# # 손실 그래프 도식화
# plot_loss(history)