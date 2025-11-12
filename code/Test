import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import StandardScaler

# ───────────────────────────────────────────────
# ① 테스트용 CSV 불러오기
# ───────────────────────────────────────────────
face_df = pd.read_csv("/data/face_features_test_dataset.csv")
audio_df = pd.read_csv("/data/audio_features_test_dataset.csv")

face_df['basename'] = face_df['File'].apply(lambda p: os.path.basename(p))
audio_df['basename'] = audio_df['File'].apply(lambda p: os.path.basename(p))

# 공통 basename 필터링
valid = set(audio_df['basename'])
face_df = face_df[face_df['basename'].isin(valid)].copy()
audio_df = audio_df[audio_df['basename'].isin(valid)].copy()

# ───────────────────────────────────────────────
# ② 얼굴 feature + Δxₜ 구성 (NetVLAD용 입력)
# ───────────────────────────────────────────────
feature_cols = ["EAR", "Eyebrows_Frown", "Eyebrows_Raise", "Lips_Up", "Lips_Protruded", "Head_Turn"]

seqs_diff = []
labels = []

for b, grp in face_df.groupby("basename"):
    grp = grp.sort_values("Frame")
    x = grp[feature_cols].values

    if len(x) < 2:
        continue

    # Δxₜ 계산 (t번째 - t-1번째)
    x_t = x[1:]
    x_prev = x[:-1]
    dx = x_t - x_prev

    # concat([x_t, Δxₜ]) → feature dimension 6 → 12
    x_combined = np.concatenate([x_t, dx], axis=-1)
    seqs_diff.append(x_combined)

    labels.append(1 if grp["Label"].iloc[0].lower() == "lie" else 0)

# padding: (T-1=49, 2*D=12)
Xf_test = pad_sequences(seqs_diff, maxlen=49, padding='post', truncating='post', dtype='float32')
y_test = np.array(labels)

# audio feature
audio_feats = ['Pitch', 'Energy'] + [f"MFCC_{i}" for i in range(1, 14)]

# audio_df는 이미 basename을 기준으로 필터링 되어 있음
audio_df = audio_df.set_index('basename').loc[face_df['basename'].unique()]
Xa_test_raw = audio_df[audio_feats].values
Xa_test = scaler.transform(Xa_test_raw)  # 기존 학습 때 사용한 scaler 그대로 사용

# 추론
y_pred = model_NVT.predict([Xf_test, Xa_test])
y_pred_label = (y_pred > 0.5).astype(int)

# 결과 출력
for i, bname in enumerate(face_df['basename'].unique()):
    if i >= len(y_pred):
        break
    p_lie = y_pred[i][0]           # 거짓말 확률
    p_truth = 1 - p_lie             # 진실 확률

    label = "거짓말" if p_lie > 0.5 else "진실"
    print(f"{bname}")
    print(f"  ▸ 예측 결과: {label}")
    print(f"  ▸ 거짓말일 확률: {p_lie:.4f}")
    print(f"  ▸ 진실일 확률: {p_truth:.4f}")
    print("-" * 40)
