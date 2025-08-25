# 🚨 Factory Smoke Detection with YOLOv8  

Bu proje, **fabrikalarda oluşabilecek siyah dumanları gerçek zamanlı olarak tespit etmek** amacıyla geliştirilmiştir.  
Makine öğrenmesi tabanlı **YOLOv8 segmentasyon modeli** kullanılarak, kameradan alınan görüntüler üzerinde duman tespiti yapılır.  

---

## 📌 Özellikler
- 🔴 **YOLOv8** kullanılarak siyah duman segmentasyonu  
- 📹 **RTSP protokolü** ile canlı kamera bağlantısı  
- 🕒 **Zaman kontrolü (07:00 - 17:00)** → sadece gündüz saatlerinde alarm  
- ⚡ **False Positive azaltma** için ardışık kare sayacı (≥ 3 kare “fog”)  
- 🔊 Duman tespit edildiğinde **alarm sesi (alert_sound.wav)**  

---

## 🛠 Kullanılan Teknolojiler
- **Python**  
- **OpenCV** → Görüntü işleme  
- **Ultralytics YOLOv8** → Makine öğrenmesi modeli  
- **NumPy** → Sayısal işlemler  
- **Winsound** → Windows’ta alarm sesi  

---


## 🚀 Çalıştırma
1. Repoyu klonlayın:  
   ```bash
   git clone https://github.com/zeynepplbyk/Factory-Smoke-Detection-with-YOLOv8----Fabrika-Duman-tespiti-YOLOv8.git
   cd factory-smoke-detection
   ```

2. Gerekli kütüphaneleri yükleyin:  
   ```bash
   pip install ultralytics opencv-python numpy
   ```

3. `main.py` dosyasını çalıştırın:  
   ```bash
   python main.py
   ```

---

## ⚠️ Notlar
- Model ağırlıkları `runs/train/merged/weights/best.pt` yolunda olmalıdır.  
- `rtsp_url` kısmını kendi kamera bağlantınızla değiştirin.  
- Şu an `winsound` kullanıldığı için **Windows üzerinde çalışır**. Linux için `playsound` veya `pygame` kullanılabilir.  

---
