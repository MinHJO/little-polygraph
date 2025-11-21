import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dense, Dropout, Concatenate, Bidirectional, Lambda
from tensorflow.keras.callbacks import EarlyStopping

# 설정
T = 50          # 시퀀스 길이
D_face = 6      # 얼굴 feature 차원
audio_dim = 15  # 오디오 feature 차원

# 입력
face_input = Input(shape=(T, D_face), name="face_input")
audio_input = Input(shape=(audio_dim,), name="audio_input")

# Δxₜ 연산 함수
def temporal_difference(x):
    x_t = x[:, 1:, :]                   # x_t (시작: t=1)
    x_prev = x[:, :-1, :]              # x_{t-1} (시작: t=0)
    dx = x_t - x_prev                  # Δxₜ
    return tf.concat([x_t, dx], axis=-1)  # concat([x_t, Δxₜ])

# Δxₜ 포함된 시퀀스 생성
face_diff = Lambda(temporal_difference)(face_input)   # shape: (B, T-1, 2*D_face)

# 얼굴 branch
x1 = Bidirectional(GRU(64))(face_diff)
x1 = Dense(64, activation='relu')(x1)
x1 = Dropout(0.3)(x1)

# 오디오 branch
x3 = Dense(64, activation='relu')(audio_input)
x3 = Dropout(0.3)(x3)
x3 = Dense(32, activation='relu')(x3)

# 병합 및 출력
merged = Concatenate()([x1, x3])
merged = Dense(64, activation='relu')(merged)
merged = Dropout(0.4)(merged)
output = Dense(1, activation='sigmoid')(merged)

model_BiGRU = Model(inputs=[face_input, audio_input], outputs=output)
model_BiGRU.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_BiGRU.summary()

early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

history = model_BiGRU.fit(
    [Xf_train, Xa_train], y_train,
    validation_data=([Xf_val, Xa_val], y_val),
    epochs=100,
    batch_size=64,
    callbacks=[early_stop],
    verbose=2
)
