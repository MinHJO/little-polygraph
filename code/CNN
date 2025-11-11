import tensorflow as tf
from tensorflow.keras import layers, models, Input

def build_hybrid_model(face_shape, audio_shape):
    # ───────────── 얼굴 입력 (시퀀스) ─────────────
    face_input = Input(shape=face_shape, name='face_input')  # (50, 6)
    x = layers.Conv1D(64, 3, padding='same', activation='relu')(face_input)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(128, 3, padding='same', activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    # ───────────── 오디오 입력 ─────────────
    audio_input = Input(shape=audio_shape, name='audio_input')  # (15,)
    y = layers.Dense(64, activation='relu')(audio_input)
    y = layers.Dropout(0.3)(y)
    y = layers.Dense(64, activation='relu')(y)

    # ───────────── 결합 후 출력 ─────────────
    combined = layers.Concatenate()([x, y])
    z = layers.Dense(64, activation='relu')(combined)
    z = layers.Dropout(0.3)(z)
    output = layers.Dense(1, activation='sigmoid')(z)

    # 모델 정의
    model = models.Model(inputs=[face_input, audio_input], outputs=output)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# 입력 shape 정의
face_input_shape = Xf_train.shape[1:]   # (50, 6)
audio_input_shape = Xa_train.shape[1:]  # (15,)

# 모델 생성
model = build_hybrid_model(face_input_shape, audio_input_shape)
model.summary()

# 훈련
history = model.fit(
    [Xf_train, Xa_train], y_train,
    validation_data=([Xf_val, Xa_val], y_val),
    epochs=50,
    batch_size=32,
    callbacks=[
        tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
    ]
)
