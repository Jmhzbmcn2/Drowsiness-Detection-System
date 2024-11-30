import cv2
import time
import os
import winsound  # Dùng playsound nếu cần tương thích nhiều hệ điều hành

# Tạo thư mục để lưu ảnh nếu chưa tồn tại
output_folder = "captured_images_nhammat"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Tải bộ nhận diện khuôn mặt Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Khởi tạo webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Không thể truy cập webcam.")
    exit()

frame_count = 0
capture_interval = 0.05  # Thời gian giữa mỗi lần chụp ảnh (giây)
previous_face = None
alpha = 0.7  # Tỷ lệ làm mượt vị trí và kích thước khung viền
min_face_size = 100
max_face_size = 300
angles = [0, 15, -15, 30, -30, 45, -45, 60, -60]  # Các góc thử nghiệm

def rotate_image(image, angle):
    """Xoay ảnh với một góc cho trước."""
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h))

while True:
    ret, frame = cap.read()
    if not ret:
        print("Không thể lấy hình ảnh từ webcam.")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detected = False  # Cờ để kiểm tra nếu đã phát hiện khuôn mặt

    for angle in angles:
        # Xoay hình xám theo góc và thử nhận diện
        rotated_gray = rotate_image(gray, angle)
        faces = face_cascade.detectMultiScale(rotated_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            # Khi phát hiện khuôn mặt, chọn khuôn mặt đầu tiên
            x, y, w, h = faces[0]
            top_padding = 30

            if min_face_size <= w <= max_face_size and min_face_size <= h <= max_face_size:
                # Làm mượt vị trí và kích thước khung
                if previous_face is not None:
                    x = int(alpha * previous_face[0] + (1 - alpha) * x)
                    y = int(alpha * previous_face[1] + (1 - alpha) * y)
                    w = int(alpha * previous_face[2] + (1 - alpha) * w)
                    h = int(alpha * previous_face[3] + (1 - alpha) * h)

                previous_face = (x, y, w, h)

                # Vẽ khung khuôn mặt
                cv2.rectangle(frame, (x, y - top_padding), (x + w, y + h), (255, 0, 0), 2)

                # Cắt và lưu ảnh khuôn mặt
                face_only = frame[max(0, y - top_padding): y + h, x: x + w]
                height, width, _ = face_only.shape
                # text_width = f"W: {width}px"
                # text_height = f"H: {height}px"
                # cv2.putText(face_only, text_width, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                # cv2.putText(face_only, text_height, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                frame_path = os.path.join(output_folder, f"face_{frame_count}.jpg")
                cv2.imwrite(frame_path, face_only)
                print(f"Lưu ảnh khuôn mặt: {frame_path}")
                frame_count += 1
                detected = True
                break  # Dừng xoay khi đã tìm thấy khuôn mặt
            else:
                cv2.putText(frame, "Error!!!", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                print("Khuôn mặt nằm ngoài kích thước cho phép.")
                winsound.Beep(1000, 500)

        if detected:
            break  # Dừng thử các góc khi đã tìm thấy khuôn mặt

    if not detected:
        previous_face = None

    cv2.imshow('Face Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    time.sleep(capture_interval)

cap.release()
cv2.destroyAllWindows()
