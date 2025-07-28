import keras


# img_classification.py에서 훈련했던 CNN의 체크포인트 파일 불러오기
model = keras.models.load_model('best-cnn-model.keras')
print(model.layers)     # 층 확인