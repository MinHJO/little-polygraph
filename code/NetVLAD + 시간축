from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate, Bidirectional, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# --- Δxₜ 계산 레이어 ---
def temporal_difference(x):
    x_t = x[:, 1:, :]
    x_prev = x[:, :-1, :]
    dx = x_t - x_prev
    return tf.concat([x_t, dx], axis=-1)

def create_delta_features(X_seq):  # X_seq: (N, T, D)
    x_t = X_seq[:, 1:, :]          # (N, T-1, D)
    x_prev = X_seq[:, :-1, :]      # (N, T-1, D)
    dx = x_t - x_prev              # (N, T-1, D)
    return np.concatenate([x_t, dx], axis=-1)  # (N, T-1, 2D)

# Δxₜ 포함된 입력으로 변경
Xf_train = create_delta_features(Xf_train)  # shape: (N, 49, 12)
Xf_val = create_delta_features(Xf_val)      # shape: (N, 49, 12)


# --- 설정 ---
T = 50
D_face = 6
audio_dim = 15

# --- 입력 정의 ---
#AFTER: Δxₜ 포함 → shape=(T-1, 2D)
face_input = Input(shape=(T-1, 2*D_face), name="face_input")  # (49, 12)
audio_input = Input(shape=(audio_dim,), name="audio_input")

# Δxₜ 포함 시퀀스
#face_diff = Lambda(temporal_difference)(face_input)     # shape: (B, T-1, 2*D)

# NetVLAD 적용 (차원 = 2*D)
x1 = NetVLAD(num_clusters=8, dim=2*D_face)(face_input)
x1 = Dense(128, activation='relu')(x1)
x1 = Dropout(0.3)(x1)

x3 = Dense(64, activation='relu')(audio_input)
x3 = Dropout(0.3)(x3)
x3 = Dense(32, activation='relu')(x3)

# 병합 및 분류기
merged = Concatenate()([x1, x3])
merged = Dense(128, activation='relu')(merged)
merged = Dropout(0.4)(merged)
output = Dense(1, activation='sigmoid')(merged)

model_NVT = Model(inputs=[face_input, audio_input], outputs=output)
model_NVT.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
model_NVT.summary()

early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

# 모델 학습
history = model_NVT.fit(
    [Xf_train, Xa_train], y_train,
    validation_data=([Xf_val, Xa_val], y_val),
    epochs=400,
    batch_size=64,
    callbacks=[early_stop],
    verbose=2
)
