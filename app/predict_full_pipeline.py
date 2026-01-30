import cv2
import numpy as np
import tensorflow as tf

IMG_SIZE = 400
PADDING = 20

def crop_hand(image, model):
    # 1. Przygotuj obraz
    original_img = image.copy()
    img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
    input_tensor = np.expand_dims(img_resized, axis=0)

    # 2. Predykcja ramki
    box = model.predict(input_tensor)[0] # [x, y, w, h] (znormalizowane)
    
    # 3. Przelicz na piksele
    h_orig, w_orig, _ = original_img.shape
    x = int(box[0] * w_orig)
    y = int(box[1] * h_orig)
    w = int(box[2] * w_orig)
    h = int(box[3] * h_orig)

    # Zabezpieczenia (żeby nie wyjść poza obraz przy błędzie modelu)
    x = max(0, x)
    y = max(0, y)
    
    # 4. Wycięcie (Crop)
    # Dodajemy mały margines (padding), żeby nie uciąć palców
    padding = PADDING
    crop = original_img[y-padding : y+h+padding, x-padding : x+w+padding]
    
    return crop, (x, y, w, h)


def predict_full_pipeline(image, detector_model, classifier_model, classes_list):
    img_bgr = image.copy()
    hand_crop, coords = crop_hand(img_bgr.copy(), detector_model)
    x_pad = coords[0]
    y_pad = coords[1]
    w_pad = coords[2]
    h_pad = coords[3]

    # Zabezpieczenie przed pustym wycinkiem (gdyby detektor zawiódł)
    if hand_crop.size == 0:
        print("Nie udało się wyciąć dłoni (błąd detektora).")
        return

    # =========================================================
    # KROK 2: KLASYFIKACJA (Jaki to gest?)
    # =========================================================

    # Preprocessing dla klasyfikatora
    img_cls = cv2.resize(hand_crop, (IMG_SIZE, IMG_SIZE)) / 255.0
    img_cls_batch = np.expand_dims(img_cls, axis=0)

    # Predykcja gestu
    predictions = classifier_model.predict(img_cls_batch, verbose=0)[0]
    class_idx = np.argmax(predictions)       # Indeks najwyższego wyniku
    confidence = predictions[class_idx]      # Pewność (0.0 - 1.0)
    
    predicted_label = classes_list[class_idx]
    
    print(f"Predykcja: {predicted_label} z pewnością {confidence*100:.2f}%")

    # =========================================================
    # KROK 3: WIZUALIZACJA
    # =========================================================
    
    # Kolor ramki zależny od pewności (Zielony = pewny, Żółty = niepewny)
    color = (0, 255, 0) if confidence > 0.7 else (0, 255, 255)

    # Rysujemy ramkę na ORYGINALNYM obrazie BGR
    cv2.rectangle(img_bgr, (x_pad, y_pad), (x_pad + w_pad, y_pad + h_pad), color, 2)

    # Przygotowanie tekstu: "I love You (98%)"
    text = f"{predicted_label} ({confidence*100:.1f}%)"
    
    # Tło pod napisem (dla czytelności)
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(img_bgr, (x_pad, y_pad - 25), (x_pad + text_w, y_pad), color, -1)
    
    # Napis
    cv2.putText(img_bgr, text, (x_pad, y_pad - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # Pokazanie wyniku
    cropped_img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_bgr
    # plt.imshow(cropped_img_rgb)
    
if __name__ == "__predict_full_pipeline__":
  predict_full_pipeline()