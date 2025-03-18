# from flask import Flask, request, jsonify
# import cv2
# import mediapipe as mp
# import numpy as np
# import base64
#
# app = Flask(__name__)
#
# # MediaPipe Face Detection 초기화
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(
#     static_image_mode=True,
#     max_num_faces=1,
#     refine_landmarks=True
# )
#
# def decode_image(base64_image):
#     """Base64로 인코딩된 이미지를 OpenCV 이미지로 변환"""
#     img_data = base64.b64decode(base64_image)
#     np_arr = np.frombuffer(img_data, np.uint8)
#     return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
#
# def analyze_mouth(image):
#     """입 모양 분석"""
#     h, w, _ = image.shape
#     results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#
#     if results.multi_face_landmarks:
#         landmarks = results.multi_face_landmarks[0]
#         # 입술 상하 좌표
#         mouth_top = landmarks.landmark[13]  # Upper lip
#         mouth_bottom = landmarks.landmark[14]  # Lower lip
#
#         # 입술 간 거리 계산
#         mouth_open_distance = abs(mouth_bottom.y - mouth_top.y) * h
#         return {"mouth_open_distance": mouth_open_distance}
#     else:
#         return {"error": "No face detected"}
#
# @app.route('/detect', methods=['POST'])
# def detect():
#     """이미지를 받아 입 모양 분석"""
#     try:
#         data = request.json
#         base64_image = data['image']
#         image = decode_image(base64_image)
#         result = analyze_mouth(image)
#         return jsonify(result)
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500
#
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)

from flask import Flask, request, jsonify
import cv2
import mediapipe as mp
import numpy as np
import base64

app = Flask(__name__)

# MediaPipe Face Detection 초기화
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True
)


def decode_image(base64_image):
    """Base64로 인코딩된 이미지를 OpenCV 이미지로 변환"""
    if not base64_image:
        print("Error: Received empty base64 image data")  # 로그 출력
        return None

    try:
        img_data = base64.b64decode(base64_image)
        np_arr = np.frombuffer(img_data, np.uint8)
        image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if image is None:
            print("Error: Failed to decode the image")  # 디코딩 실패 시 로그 출력
            return None
        print("Decoded image successfully")  # 디코딩 성공 로그
        return image
    except Exception as e:
        print(f"Error decoding image: {e}")  # 예외 발생 시 로그 출력
        return None


def analyze_mouth(image):
    """입 모양 분석"""
    h, w, _ = image.shape
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    if results.multi_face_landmarks:
        landmarks = results.multi_face_landmarks[0]
        # 입술 상하 좌표
        mouth_top = landmarks.landmark[13]  # Upper lip
        mouth_bottom = landmarks.landmark[14]  # Lower lip

        # 입술 간 거리 계산
        mouth_open_distance = abs(mouth_bottom.y - mouth_top.y) * h

        return {"mouth_open_distance": mouth_open_distance}
    else:
        return {"error": "No face detected"}


@app.route('/detect', methods=['POST'])
def detect():
    """이미지를 받아 입 모양 분석"""
    try:
        data = request.json
        base64_image = data['image']

        # base64_image가 비어있는지 확인
        if not base64_image:
            print("Error: Received empty base64 image data")  # 로그 출력
            return jsonify({"error": "No image data received"}), 400

        print(f"Received base64 image data: {base64_image[:100]}...")  # 첫 100자만 출력 (너무 긴 값을 확인)
        image = decode_image(base64_image)

        if image is None:
            return jsonify({"error": "Failed to decode the image"}), 400

        result = analyze_mouth(image)

        # 입술이 벌어졌는지 여부를 판별하여 클라이언트에 응답
        if "error" in result:
            return jsonify(result)

        # 입술이 벌어진 상태 또는 다문 상태에 대한 정보를 응답에 포함
        return jsonify({
            "mouth_open_distance": result["mouth_open_distance"]
        })
    except Exception as e:
        print(f"Error during request processing: {e}")  # 오류 메시지 출력
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

# from flask import Flask, render_template, Response
# from deepface import DeepFace
# import cv2
# import mediapipe as mp
#
# app = Flask(__name__)
#
# # Mediapipe 얼굴 인식 초기화
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
# mp_drawing = mp.solutions.drawing_utils
#
#
# # DeepFace를 사용하여 닮은 사람 찾기 (DB 없이)
# def analyze_face(image):
#     # DeepFace를 사용하여 얼굴 분석
#     analysis = DeepFace.analyze(image, actions=['age', 'gender', 'race', 'emotion'], enforce_detection=False)
#     return analysis
#
#
# # 카메라로 얼굴 인식 후 닮은 사람 찾기
# def generate_frames():
#     cap = cv2.VideoCapture(0)  # 카메라 연결
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         # 얼굴 인식 처리
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = face_mesh.process(frame_rgb)
#
#         if results.multi_face_landmarks:
#             for face_landmarks in results.multi_face_landmarks:
#                 # 얼굴 랜드마크 그리기
#                 mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS)
#
#             # 얼굴 영역만 잘라내기
#             height, width, _ = frame.shape
#             face_image = frame[0:height, 0:width]  # 전체 얼굴 영역 사용 (원하는 대로 수정 가능)
#
#             # 얼굴 분석 (나이, 성별, 감정 등)
#             analysis = analyze_face(face_image)
#             print("Face analysis result:", analysis)
#
#         # 비디오 프레임을 JPEG로 인코딩하여 클라이언트로 전달
#         _, buffer = cv2.imencode('.jpg', frame)
#         frame = buffer.tobytes()
#         yield (b'--frame\r\n'
#                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
#
#
# @app.route('/')
# def index():
#     return render_template('index.html')
#
#
# @app.route('/video_feed')
# def video_feed():
#     return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
#
#
# if __name__ == '__main__':
#     app.run(debug=True)
