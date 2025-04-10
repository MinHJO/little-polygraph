# Colab에서 Drive 마운트
from google.colab import drive
drive.mount('/content/drive')

# 필요한 라이브러리 임포트
import os
import cv2
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

# 재귀적으로 폴더 내 영상 파일을 탐색하여 얼굴 분석
def process_folder_face_recursive(root_folder, sample_rate=10):
    all_data = []
    for root, dirs, files in os.walk(root_folder):
        video_files = [f for f in files if f.lower().endswith((".mp4", ".avi", ".mov"))]
        for file in video_files:
            video_path = os.path.join(root, file)
            all_data.extend(process_video_face(video_path, sample_rate))
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

def extract_voiced_segment(wav_path, top_db=30):
    """
    wav_path: 오디오 파일 경로 (WAV 파일)
    top_db: 무음 구간을 결정하는 dB 임계값 (기본 30)
    
    반환: (y_voiced, sr)
      y_voiced: 선택한 voiced segment의 오디오 데이터 (numpy array)
      sr: 샘플링 주파수
    """
    y, sr = librosa.load(wav_path, sr=None)
    # 비-무음 구간의 인덱스 배열을 구함
    intervals = librosa.effects.split(y, top_db=top_db)
    if len(intervals) == 0:
        return None, sr
    # 가장 긴 구간 선택 (안정적인 발화 구간)
    longest = max(intervals, key=lambda x: x[1] - x[0])
    y_voiced = y[longest[0]:longest[1]]
    return y_voiced, sr

# Parselmouth를 이용한 Jitter와 Shimmer 추출
def extract_jitter_shimmer_preprocessed(wav_path, top_db=30):
    # 안정적인 발화 구간 추출
    y_voiced, sr = extract_voiced_segment(wav_path, top_db=top_db)
    if y_voiced is None:
        print("No voiced segment found in", wav_path)
        return np.nan, np.nan

    # 추출된 구간으로부터 Sound 객체 생성 (배열과 샘플링 주파수를 직접 지정)
    try:
        snd = parselmouth.Sound(y_voiced, sr)
    except Exception as e:
        print("Error creating Sound object:", e)
        return np.nan, np.nan
        
    # PointProcess 객체 생성 (75 ~ 500 Hz 범위; 이는 대부분의 성인 음성에 적합)
    try:
        point_process = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 500)
    except Exception as e:
        print("Error extracting point process:", e)
        return np.nan, np.nan

    # jitter와 shimmer 계산: 인자 설명
    # 첫 번째 인자: 최소 주기 (여기서는 0초로 설정; Praat에서는 0을 사용)
    # 두 번째 인자: 최대 주기, 여기서는 0.02초 (즉, 20ms)
    # 세 번째 인자: 시간 간격 (0.0001초)
    # 네 번째 인자: 상대 임계값 (0.02)
    # 다섯 번째 인자: 최대 진폭 계수 (1.3)
    try:
        jitter = parselmouth.praat.call([snd, point_process], "Get jitter (local)", 0, 0.02, 0.0001, 0.02, 1.3)
    except Exception as e:
        print("Jitter extraction error:", e)
        jitter = np.nan
    try:
        shimmer = parselmouth.praat.call([snd, point_process], "Get shimmer (local, dB)", 0, 0.02, 0.0001, 0.02, 1.3)
    except Exception as e:
        print("Shimmer extraction error:", e)
        shimmer = np.nan
        
    return jitter, shimmer


# 영상 파일에서 음성 특징 추출
def process_video_audio(video_path):
    filename = os.path.basename(video_path)
    label = get_truth_label(filename)
    audio_path = "temp_audio.wav"
    # 영상에서 오디오 추출 (MoviePy 사용)
    extract_audio_from_video(video_path, audio_path)
    
    mfcc_mean, pitch_mean, energy_mean = extract_librosa_features(audio_path)
    # 전처리된 오디오를 사용하여 jitter와 shimmer 추출
    jitter, shimmer = extract_jitter_shimmer_preprocessed(audio_path, top_db=30)
    
    # 데이터 구성: 파일명, 라벨, Pitch, Energy, Jitter, Shimmer, 그리고 13개의 MFCC 평균값
    data = [video_path, label, pitch_mean, energy_mean, jitter, shimmer]
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
    # 컬럼: File, Label, Pitch, Energy, Jitter, Shimmer, MFCC_1 ~ MFCC_13
    columns = ["File", "Label", "Pitch", "Energy", "Jitter", "Shimmer"]
    for i in range(13):
        columns.append(f"MFCC_{i+1}")
    df = pd.DataFrame(all_data, columns=columns)
    df.to_csv(output_csv, index=False)
    print(f"Audio features saved to: {output_csv}")

# ──────────────────────────────
# [Main 실행부]
if __name__ == "__main__":
    # 데이터가 있는 최상위 폴더 경로 (영상 파일만 포함)
    root_folder = os.path.join(base_path, "dataset")  # 예: /content/drive/MyDrive/Capstone/dataset
    face_output_csv = os.path.join(base_path, "data/face_features_dataset.csv")
    audio_output_csv = os.path.join(base_path, "data/audio_features_dataset.csv")

    # 영상에 대해 얼굴 분석 (프레임 단위)
    process_folder_face(root_folder, face_output_csv, sample_rate=10)
    
    # 영상에 대해 음성 분석 수행
    process_folder_audio(root_folder, audio_output_csv)
