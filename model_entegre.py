import cv2
from ultralytics import YOLO
import numpy as np
import winsound
from datetime import datetime, timedelta, time

# Model ve kamera bağlantısı
model = YOLO("runs/train/merged/weights/best.pt")
rtsp_url = "rtsp://****:****@***.***.**.**:***/cam/realmonitor?channel=***&subtype=0"
class_names = ['clean', 'fog']

cap = cv2.VideoCapture(rtsp_url)
if not cap.isOpened():
    print("Kameraya bağlanılamadı.")
    exit()

alert_played = False
fog_frame_count = 0  # Üst üste kaç kare fog görüldü

# Gündüz kontrolü (07:00 - 17:00 Türkiye saati)
def is_daytime():
    now_utc = datetime.utcnow()
    now_tr = now_utc + timedelta(hours=3)
    start = time(7, 0, 0)
    end = time(17, 0, 0)
    return start <= now_tr.time() <= end

while True:
    ret, frame = cap.read()
    if not ret:
        print("Görüntü alınamadı.")
        break

    frame = cv2.resize(frame, (960, 540))
    results = model(frame)[0]
    fog_detected = False

    if results.masks is not None:
        masks = results.masks.data.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy()
        confidences = results.boxes.conf.cpu().numpy()

        for i, cls_id in enumerate(classes):
            conf = confidences[i]

            # Sadece "fog" için işlem yap
            if class_names[int(cls_id)] == "fog" and conf > 0.5:
                fog_detected = True
                mask = masks[i].astype(np.uint8) * 255

                # Mask boyutunu frame boyutuna getir (kayma sorununu çözer)
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))

                # Kontur çiz
                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(frame, contours, -1, (0, 0, 255), 2)

    # Fog tespit sayaç sistemi (false positive azaltma)
    if fog_detected:
        fog_frame_count += 1
    else:
        fog_frame_count = 0

    if fog_frame_count >= 3 and not alert_played and is_daytime():
        winsound.PlaySound("alert_sound.wav", winsound.SND_FILENAME | winsound.SND_ASYNC)
        alert_played = True
    elif fog_frame_count == 0:
        alert_played = False

    cv2.imshow("Fog Segmentasyon Maskesi", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
