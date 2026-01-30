import cv2
from predict_full_pipeline import predict_full_pipeline
import tensorflow as tf

DETECTOR_PATH = 'hand_detector_best.keras'
CLASSIFIER_PATH = 'gesture_classifier.keras'
CLASSES =  ['hello', 'iloveyou', 'no', 'yes']

def find_working_camera():
    """Szuka pierwszego działającego indeksu kamery."""
    for index in range(5):
        cap = cv2.VideoCapture(index)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            if ret:
                print(f"[SUKCES] Znaleziono kamerę pod indeksem: {index}")
                return index
    print("[BŁĄD] Nie znaleziono żadnej kamery!")
    return None

def run_live_inference():
  try:
      print("Ładowanie detektora...")
      detector = tf.keras.models.load_model(DETECTOR_PATH)
      
      print("Ładowanie klasyfikatora...")
      classifier = tf.keras.models.load_model(CLASSIFIER_PATH)
      
      print("Modele załadowane pomyślnie!")
  except Exception as e:
      print(f"BŁĄD PODCZAS ŁADOWANIA MODELI: {e}")
      print("Sprawdź ścieżki do plików .keras!")

  cap = cv2.VideoCapture(find_working_camera())
  
  # Ustawienie rozdzielczości kamery (opcjonalne, dla wydajności)
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

  if not cap.isOpened():
      print("Nie można otworzyć kamery!")
      return

  print("Kamera uruchomiona. Naciśnij 'q', aby wyjść.")
  
  while True:
    ret, frame = cap.read()
    if not ret:
      break

    # Kopia do wyświetlania (zeby rysowac na czystym)
    debug_frame = frame.copy()
    h_orig, w_orig, _ = frame.shape

    img_bgr = predict_full_pipeline(frame, detector, classifier, CLASSES)
    
    # Wyświetlenie klatki
    cv2.imshow('Hand Sign Recognition', img_bgr)

    # Wyjście klawiszem 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  cap.release()
  cv2.destroyAllWindows()

if __name__ == "__run_live_inference__":
  run_live_inference()