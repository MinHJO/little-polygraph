# ─────────────────────────────────────────
# 데이터 로드 및 전처리
# ─────────────────────────────────────────
# 데이터 경로 설정
face_csv_path = "/data/face_features_dataset.csv"
audio_csv_path = "/data/audio_features_dataset.csv"

# 데이터 불러오기
face_df = pd.read_csv(face_csv_path)
audio_df = pd.read_csv(audio_csv_path)

# basename 열 추가
face_df['basename'] = face_df['File'].apply(lambda p: os.path.basename(p))
audio_df['basename'] = audio_df['File'].apply(lambda p: os.path.basename(p))

# 공통 비디오만 남기기
valid = set(audio_df['basename'])
face_df = face_df[face_df['basename'].isin(valid)].copy()
audio_df = audio_df[audio_df['basename'].isin(valid)].copy()

# 얼굴 feature
feature_cols = ["EAR", "Eyebrows_Frown", "Eyebrows_Raise", "Lips_Up", "Lips_Protruded", "Head_Turn"]

# 얼굴 시퀀스 & 레이블
seqs, labels = [], []
for b, grp in face_df.groupby("basename"):
    grp = grp.sort_values("Frame")
    seqs.append(grp[feature_cols].values)
    labels.append(1 if grp["Label"].iloc[0].lower() == "lie" else 0)

# 패딩
max_frames = 50
X_face = pad_sequences(seqs, maxlen=max_frames, padding='post', truncating='post', dtype='float32')
y = np.array(labels)

# 오디오 feature
audio_feats = ['Pitch', 'Energy'] + [f"MFCC_{i}" for i in range(1,14)]
audio_df = audio_df.set_index('basename').loc[face_df['basename'].unique()]
X_audio_raw = audio_df[audio_feats].values

# 스케일링
scaler = StandardScaler()
X_audio = scaler.fit_transform(X_audio_raw)

# 학습/검증 분할
Xf_train, Xf_val, Xa_train, Xa_val, y_train, y_val = train_test_split(
    X_face, X_audio, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)
