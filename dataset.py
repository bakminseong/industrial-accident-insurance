# from tensorflow.keras import layers, models
#
# # 간단한 Autoencoder 모델
# def create_autoencoder():
#     input_img = layers.Input(shape=(128, 128, 3))  # 예시: 128x128 크기의 RGB 이미지
#     x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(input_img)
#     x = layers.MaxPooling2D((2, 2), padding='same')(x)
#     x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
#     encoded = layers.MaxPooling2D((2, 2), padding='same')(x)
#
#     x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(encoded)
#     x = layers.UpSampling2D((2, 2))(x)
#     x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
#     x = layers.UpSampling2D((2, 2))(x)
#     decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
#
#     autoencoder = models.Model(input_img, decoded)
#     autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
#     return autoencoder
#
# # 모델 생성
# autoencoder = create_autoencoder()
#
# # 모델 훈련 (예시)
# # 이미지 데이터셋을 준비하고 훈련합니다.
# autoencoder.fit(x_train, x_train, epochs=50, batch_size=256, shuffle=True, validation_data=(x_test, x_test))
#
# # 모델 저장
# autoencoder.save('face_blending_model.h5')