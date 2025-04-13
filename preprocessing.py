# Colab에서 Drive 마운트
from google.colab import drive
drive.mount('/content/drive')

# 필요한 라이브러리 임포트
import os
import cv2
import re
import numpy as np
import pandas as pd
from scipy.spatial import distance as dist
from tqdm import tqdm
import mediapipe as mp

# 음성 분석 관련 라이브러리
from moviepy.editor import VideoFileClip
import librosa
import parselmouth

# 모델 파일 및 기본 경로 (Drive 내 경로)
base_path = "/content/drive/MyDrive/Capstone"
detector_model_path = os.path.join(base_path, "mmod_human_face_detector.dat")
landmark_model_path = os.path.join(base_path, "shape_predictor_68_face_landmarks.dat")

# 자연정렬을 통해 파일 정렬
def natural_keys(s):
    # re.split로 숫자 부분(\d+)과 나머지 부분을 분리합니다.
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

# 파일명 또는 폴더명에서 truth/lie 라벨 자동 추출 함수
def get_truth_label(name):
    if 'truth' in name.lower():
        return "truth"
    elif 'lie' in name.lower():
        return "lie"
    else:
        return "unknown"

# ──────────────────────────────
# [얼굴 분석 부분] (mediapipe 사용)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# 눈 깜박임(EAR) 계산 함수 (dlib 코드와 동일 방식)
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# 한 프레임에서 얼굴 특징 추출 (mediapipe Face Mesh 이용)
def extract_features_from_frame(frame):
    try:
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None
        # 첫 번째 얼굴의 랜드마크(468 포인트) 사용
        landmarks = results.multi_face_landmarks[0].landmark
        # 정규화 좌표를 픽셀 좌표로 변환
        pts = np.array([[int(lmk.x * w), int(lmk.y * h)] for lmk in landmarks])

        # Mediapipe에서 주로 사용하는 인덱스 (예시):
        left_eye_indices = [33, 160, 158, 133, 153, 144]
        right_eye_indices = [362, 385, 387, 263, 373, 380]
        left_eye = pts[left_eye_indices]
        right_eye = pts[right_eye_indices]

        left_eye_center = np.mean(left_eye, axis=0)
        right_eye_center = np.mean(right_eye, axis=0)
        left_EAR = eye_aspect_ratio(left_eye)
        right_EAR = eye_aspect_ratio(right_eye)
        avg_EAR = (left_EAR + right_EAR) / 2.0

        # 눈썹 찡함: (예: 포인트 55와 285)
        eyebrows_frown = np.linalg.norm(pts[55] - pts[285])
        # 눈썹 올림: (예: 포인트 70, 300와 눈 중심의 세로 차이)
        left_eyebrow_raise = abs(pts[70][1] - left_eye_center[1])
        right_eyebrow_raise = abs(pts[300][1] - right_eye_center[1])
        avg_eyebrow_raise = (left_eyebrow_raise + right_eyebrow_raise) / 2.0

        # 입 관련 특징: (예: 포인트 13, 1, 14)
        lips_up = np.linalg.norm(pts[13] - pts[1])
        lips_protruded = np.linalg.norm(pts[13] - pts[14])

        # 머리 회전 정도: 코 포인트(예: 1)와 양쪽 눈 중심 간의 차이
        head_turn = np.linalg.norm(pts[1] - left_eye_center) - np.linalg.norm(pts[1] - right_eye_center)

        return [avg_EAR, eyebrows_frown, avg_eyebrow_raise, lips_up, lips_protruded, head_turn]
    except Exception as e:
        print(f"Face analysis error: {e}")
        return None

# 영상 파일에 대한 얼굴 분석 (프레임 단위)
def process_video_face(video_path, sample_rate=10):
    filename = os.path.basename(video_path)
    label = get_truth_label(filename)
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    data = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % sample_rate == 0:
            features = extract_features_from_frame(frame)
            if features:
                data.append([video_path, frame_count, label] + features)
        frame_count += 1
    cap.release()
    return data

# 개별 이미지 파일(프레임)에서 얼굴 분석하는 함수
def process_image_face(image_path):
    filename = os.path.basename(image_path)
    label = get_truth_label(filename)
    # 파일명에서 숫자를 추출하여 프레임 번호로 사용 (예: trial_truth_006_120.jpg → 120)
    try:
        frame_number = int(''.join(filter(str.isdigit, filename)))
    except:
        frame_number = 0
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Image load error: {image_path}")
        return []
    features = extract_features_from_frame(frame)
    if features:
        return [[image_path, frame_number, label] + features]
    return []

# 재귀적으로 폴더 내의 영상 파일 및 이미지 파일(프레임) 모두 탐색하여 얼굴 분석
def process_folder_face_recursive(root_folder, sample_rate=10):
    all_data = []
    # os.walk로 하위 폴더까지 모두 순회
    for root, dirs, files in os.walk(root_folder):
        # 자연 정렬 적용: 파일명 정렬 시, sorted(files, key=natural_keys)
        sorted_files = sorted(files, key=natural_keys)
        for file in sorted_files:
            filepath = os.path.join(root, file)
            if file.lower().endswith((".mp4", ".avi", ".mov")):
                all_data.extend(process_video_face(filepath, sample_rate))
            elif file.lower().endswith((".jpg", ".jpeg", ".png")):
                all_data.extend(process_image_face(filepath))
    return all_data

# 얼굴 분석 결과를 CSV로 저장
def process_folder_face(root_folder, output_csv, sample_rate=10):
    print(f"Processing face analysis recursively in: {root_folder}")
    all_data = process_folder_face_recursive(root_folder, sample_rate)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df = pd.DataFrame(all_data, columns=["File", "Frame", "Label", "EAR", "Eyebrows_Frown", "Eyebrows_Raise", "Lips_Up", "Lips_Protruded", "Head_Turn"])
    df.to_csv(output_csv, index=False)
    print(f"Face features saved to: {output_csv}")

# ──────────────────────────────
# [음성 분석 부분]
# 영상에서 오디오 추출 함수
def extract_audio_from_video(video_path, output_audio_path):
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(output_audio_path, verbose=False, logger=None)

# Librosa를 이용한 MFCC, Pitch, Energy 추출
def extract_librosa_features(wav_path):
    y, sr = librosa.load(wav_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = mfccs.mean(axis=1)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    # 강도가 중앙값보다 큰 피치들 선택 후 평균 계산
    pitch = pitches[magnitudes > np.median(magnitudes)]
    pitch_mean = np.mean(pitch) if len(pitch) > 0 else 0
    energy = librosa.feature.rms(y=y)
    energy_mean = np.mean(energy)
    return mfcc_mean, pitch_mean, energy_mean

# 영상 파일에서 음성 특징 추출
def process_video_audio(video_path):
    filename = os.path.basename(video_path)
    label = get_truth_label(filename)
    audio_path = "temp_audio.wav"
    # 영상에서 오디오 추출 (MoviePy 사용)
    extract_audio_from_video(video_path, audio_path)

    mfcc_mean, pitch_mean, energy_mean = extract_librosa_features(audio_path)

    # 데이터 구성: 파일명, 라벨, Pitch, Energy, 그리고 13개의 MFCC 평균값
    data = [video_path, label, pitch_mean, energy_mean]
    data.extend(mfcc_mean.tolist())

    if os.path.exists(audio_path):
        os.remove(audio_path)
    return data

# 재귀적으로 폴더 내 영상 파일을 탐색하여 음성 분석
def process_folder_audio_recursive(root_folder):
    all_data = []
    for root, dirs, files in os.walk(root_folder):
        video_files = [f for f in files if f.lower().endswith((".mp4", ".avi", ".mov"))]
        for file in video_files:
            video_path = os.path.join(root, file)
            all_data.append(process_video_audio(video_path))
    return all_data

# 음성 분석 결과를 CSV로 저장
def process_folder_audio(root_folder, output_csv):
    print(f"Processing audio analysis recursively in: {root_folder}")
    all_data = process_folder_audio_recursive(root_folder)
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    # 컬럼: File, Label, Pitch, Energy, MFCC_1 ~ MFCC_13
    columns = ["File", "Label", "Pitch", "Energy"]
    for i in range(13):
        columns.append(f"MFCC_{i+1}")
    df = pd.DataFrame(all_data, columns=columns)
    df.to_csv(output_csv, index=False)
    print(f"Audio features saved to: {output_csv}")

# ──────────────────────────────
# [Main 실행부]
if __name__ == "__main__":
    # 데이터가 있는 최상위 폴더 경로
    root_folder = os.path.join(base_path, "dataset")
    face_output_csv = os.path.join(base_path, "data/face_features_dataset.csv")
    audio_output_csv = os.path.join(base_path, "data/audio_features_dataset.csv")

    # 영상에 대해 얼굴 분석 (프레임 단위)
    process_folder_face(root_folder, face_output_csv, sample_rate=10)

    # 영상에 대해 음성 분석 수행
    process_folder_audio(root_folder, audio_output_csv)
