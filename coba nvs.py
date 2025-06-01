import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("best (2).pt")  

# Inisialisasi kamera (0 untuk webcam default, 1/2 untuk USB camera)
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Kamera tidak ditemukan.")
    exit()

print("Deteksi real-time dengan night vision... Tekan 'q' untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # ===== NIGHT VISION MODE =====
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # ubah ke grayscale
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))  # kontras adaptif
    enhanced = clahe.apply(gray)
    enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)  # kembali ke 3 channel

    # Deteksi dengan YOLO
    results = model(enhanced_bgr)[0]

    # Gambar hasil deteksi
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls_id]

        # Hitung tinggi bounding box
        box_height = y2 - y1

        # Ukuran dinamis
        font_scale = max(0.3, min(0.6, box_height / 200))
        thickness = max(1, min(2, int(box_height / 100)))

        # Gambar bounding box dan label
        cv2.rectangle(enhanced_bgr, (x1, y1), (x2, y2), (0, 255, 0), thickness)
        cv2.putText(enhanced_bgr, f"{label} ({conf:.2f})", (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 0), thickness)

    # Tampilkan frame
    cv2.imshow("Night Vision Detection", enhanced_bgr)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Bersihkan
cap.release()
cv2.destroyAllWindows()
