from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Concatenate, Bidirectional, GRU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

# 입력 정의
face_input = Input(shape=(max_frames, len(feature_cols)), name="face_input")
audio_input = Input(shape=(Xa_train.shape[1],), name="audio_input")

# 얼굴 branch (BiGRU → Dense)
x1 = Bidirectional(GRU(64, return_sequences=False))(face_input)  # output: (128,)
x1 = Dense(128, activation='relu')(x1)
x1 = Dropout(0.3)(x1)

# 오디오 branch (Dense)
x3 = Dense(64, activation='relu')(audio_input)
x3 = Dropout(0.3)(x3)
x3 = Dense(32, activation='relu')(x3)

# 병합 & 분류기
merged = Concatenate()([x1, x3])
merged = Dense(128, activation='relu')(merged)
merged = Dropout(0.4)(merged)
output = Dense(1, activation='sigmoid')(merged)

# 모델 정의
model_GRU = Model(inputs=[face_input, audio_input], outputs=output)
model_GRU.compile(optimizer=Adam(1e-4), loss='binary_crossentropy', metrics=['accuracy'])
model_GRU.summary()

early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

history = model_GRU.fit(
    [Xf_train, Xa_train], y_train,
    validation_data=([Xf_val, Xa_val], y_val),
    epochs=400,
    batch_size=64,
    callbacks=[early_stop],
    verbose=2
)
