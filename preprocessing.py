import os
import cv2
import dlib
import numpy as np
import pandas as pd
from scipy.spatial import distance as dist
from tqdm import tqdm

# 파일명에서 truth/lie 라벨 자동 추출 함수
def get_truth_label(name):
    if 'truth' in name.lower():
        return "truth"
    elif 'lie' in name.lower():
        return "lie"
    else:
        return "unknown"

# ──────────────────────────────
# [얼굴 분석 부분]
# dlib 모델 로드 (모델 파일은 .py와 같은 폴더 또는 지정한 경로에 두세요)
cnn_detector = dlib.cnn_face_detection_model_v1("C:/Users/cmh06/vscode-workspace/Capstone/mmod_human_face_detector.dat")
predictor = dlib.shape_predictor("C:/Users/cmh06/vscode-workspace/Capstone/shape_predictor_68_face_landmarks.dat")

# 눈 깜박임(EAR) 계산 함수
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

# 한 프레임에서 얼굴 특징 추출 (dlib CNN + 랜드마크)
def extract_features_from_frame(frame):
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        dets = cnn_detector(rgb, 1)
        if len(dets) == 0:
            return None
        rect = dets[0].rect
        landmarks = predictor(rgb, rect)
        landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])
        
        # 부위별 좌표 추출
        left_eye = landmarks[36:42]
        right_eye = landmarks[42:48]
        left_eyebrow = landmarks[17:22]
        right_eyebrow = landmarks[22:27]
        upper_lip = landmarks[50:53]
        lower_lip = landmarks[56:59]
        nose_tip = landmarks[30]
        
        # 특징 계산
        left_EAR = eye_aspect_ratio(left_eye)
        right_EAR = eye_aspect_ratio(right_eye)
        avg_EAR = (left_EAR + right_EAR) / 2.0
        
        eyebrows_frown = np.linalg.norm(landmarks[21] - landmarks[22])
        left_eyebrow_height = np.linalg.norm(left_eyebrow[2] - left_eye[1])
        right_eyebrow_height = np.linalg.norm(right_eyebrow[2] - right_eye[1])
        avg_eyebrow_raise = (left_eyebrow_height + right_eyebrow_height) / 2.0
        
        lips_up = np.linalg.norm(upper_lip[1] - nose_tip)
        lips_protruded = np.linalg.norm(upper_lip[1] - lower_lip[1])
        
        left_eye_center = np.mean(left_eye, axis=0)
        right_eye_center = np.mean(right_eye, axis=0)
        head_turn = np.linalg.norm(nose_tip - left_eye_center) - np.linalg.norm(nose_tip - right_eye_center)
        
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
# [Main 실행부]
if __name__ == "__main__":
    # 처리할 영상 파일들이 있는 최상위 폴더 경로 (하위 폴더까지 모두 탐색)
    root_folder = "C:/Users/cmh06/vscode-workspace/Capstone/dataset"  # 자신의 경로에 맞게 수정
    face_output_csv = "data/face_features_dataset.csv"
    
    # 영상에 대해서만 얼굴 분석 수행
    process_folder_face(root_folder, face_output_csv, sample_rate=10)
