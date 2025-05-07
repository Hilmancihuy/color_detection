import cv2
import numpy as np
from flask import Flask, render_template, Response

app = Flask(__name__)

# Rentang warna (HSV) untuk deteksi
color_ranges = {
    "Red": [(0, 120, 70), (10, 255, 255), (170, 120, 70), (180, 255, 255)],
    "Light Orange": [(10, 100, 100), (20, 255, 255)],
    "Orange": [(20, 150, 100), (25, 255, 255)],
    "Yellow": [(25, 150, 100), (35, 255, 255)],
    "Light Yellow": [(35, 150, 150), (45, 255, 255)],
    "Green": [(45, 100, 100), (85, 255, 255)],
    "Cyan": [(85, 100, 100), (95, 255, 255)],
    "Blue": [(95, 100, 100), (130, 255, 255)],
    "Purple": [(130, 50, 50), (145, 255, 255)],
    "Magenta": [(145, 50, 50), (160, 255, 255)],
    "Pink": [(160, 50, 50), (170, 255, 255)],
    "White": [(0, 0, 200), (180, 30, 255)],
    "Gray": [(0, 0, 50), (180, 50, 200)],
    "Brown": [(10, 100, 20), (25, 255, 200)],
    "Black": [(0, 0, 0), (180, 255, 50)]
}

# Fungsi untuk mendeteksi warna pada titik tengah
def detect_color(frame):
    height, width, _ = frame.shape
    center_x, center_y = width // 2, height // 2

    # Ambil warna pada titik tengah (BGR)
    b, g, r = frame[center_y, center_x]
    
    # Konversi ke HSV
    hsv_pixel = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]

    detected_color = "Unknown"

    # Bandingkan dengan rentang warna yang telah didefinisikan
    for color, ranges in color_ranges.items():
        for i in range(0, len(ranges), 2):  # Karena beberapa warna punya 2 rentang
            lower, upper = ranges[i], ranges[i + 1]
            if lower[0] <= hsv_pixel[0] <= upper[0] and lower[1] <= hsv_pixel[1] <= upper[1] and lower[2] <= hsv_pixel[2] <= upper[2]:
                detected_color = color
                break

    return detected_color

# Fungsi untuk menangkap video stream dari kamera
def gen_frames():
    cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # **Resizing agar optimal**
        frame = cv2.resize(frame, (1280, 720))

        # Deteksi warna dari titik tengah
        color_detected = detect_color(frame)

        # **Tampilkan warna yang terdeteksi di layar**
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = f"Detected Color: {color_detected}"
        cv2.putText(frame, text, (50, 50), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # **Gambarkan titik tengah sebagai referensi**
        height, width, _ = frame.shape
        center_x, center_y = width // 2, height // 2
        cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), -1)  # Titik hijau di tengah

        # Encode frame untuk ditampilkan di web
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
