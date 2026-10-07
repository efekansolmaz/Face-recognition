import os
import urllib.request
import sys

MODELS = {
    "face_detection_yunet_2023mar.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "face_recognition_sface_2021dec.onnx": "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    "emotion-ferplus-8.onnx": "https://huggingface.co/onnxmodelzoo/emotion-ferplus-8/resolve/main/emotion-ferplus-8.onnx"
}

def download_models(target_dir="models"):
    os.makedirs(target_dir, exist_ok=True)
    for model_name, url in MODELS.items():
        file_path = os.path.join(target_dir, model_name)
        if os.path.exists(file_path) and os.path.getsize(file_path) > 10000:
            print(f"[OK] {model_name} zaten mevcut.")
            continue
        
        print(f"[INDIRILIYOR] {model_name}...")
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(file_path, 'wb') as out_file:
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                chunk_size = 1024 * 64
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out_file.write(chunk)
                    downloaded += len(chunk)
                    if total_size > 0:
                        percent = (downloaded / total_size) * 100
                        mb_down = downloaded / (1024 * 1024)
                        mb_total = total_size / (1024 * 1024)
                        sys.stdout.write(f"\r  -> {percent:.1f}% ({mb_down:.1f}/{mb_total:.1f} MB)")
                        sys.stdout.flush()
                print(f"\n[TAMAMLANDI] {model_name} basariyla indirildi.")
        except Exception as e:
            print(f"\n[HATA] {model_name} indirilemedi: {e}")
            if os.path.exists(file_path):
                os.remove(file_path)
            raise e

if __name__ == "__main__":
    download_models()
