import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageOps
import easyocr
import cv2

# CONFIGURATION
MODEL_PATH = "Model_V2.tflite"
IMG_HEIGHT = 32
IMG_WIDTH = 20

st.set_page_config(page_title="Lecture de compteurs", layout="wide")


@st.cache_resource
def load_resources():
    # allowlist='0123456789' force l'IA à ne chercher que des chiffres
    reader = easyocr.Reader(['en'], gpu=False)
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return reader, interpreter

def preprocess_for_tflite(image_crop):
    img = image_crop.resize((IMG_WIDTH, IMG_HEIGHT))
    img_array = np.array(img, dtype=np.float32)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_digit(interpreter, img_array):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    return np.argmax(output), np.max(output)

def is_dark_background(image_pil, bbox, threshold=130):
    """
    Vérifie si la zone découpée est majoritairement SOMBRE.
    Retourne: (True/False, luminosité_moyenne)
    """
    x1, y1, x2, y2 = bbox
    crop = image_pil.crop((x1, y1, x2, y2))
    
    # Convertir en niveaux de gris
    gray_crop = np.array(crop.convert('L'))
    
    # Calculer la moyenne de luminosité des pixels
    # 0 = Noir total, 255 = Blanc total
    avg_brightness = np.mean(gray_crop)
    
    # Si la moyenne est inférieure au seuil, c'est un fond sombre (Noir ou Rouge foncé)
    # Si c'est supérieur (ex: 200), c'est du blanc , on rejette
    return avg_brightness < threshold, avg_brightness

def merge_nearby_boxes(boxes):
    """ Fusionne les boîtes alignées (ex: Partie noire + Partie rouge) """
    if not boxes: return []
    boxes.sort(key=lambda b: b[0]) # Tri par X
    
    merged = []
    current_box = boxes[0]
    
    for next_box in boxes[1:]:
        x1, y1, x2, y2 = current_box
        nx1, ny1, nx2, ny2 = next_box
        
        # Alignement Y et Proximité X
        if abs(y1 - ny1) < 30 and (nx1 - x2) < 100:
            # Fusion
            current_box = [min(x1, nx1), min(y1, ny1), max(x2, nx2), max(y2, ny2)]
        else:
            merged.append(current_box)
            current_box = next_box
    merged.append(current_box)
    return merged

def auto_detect_meter(reader, image_pil):
    """
    Stratégie :
    1. Inverser l'image (negatif) pour que EasyOCR voit les chiffres blancs.
    2. Récupérer toutes les zones de chiffres.
    3. Rejeter tout ce qui a un fond blanc sur l'image originale.
    """
    # 1. Scan sur image inversée (Négatif)
    #  EasyOCR lit mal le blanc sur noir mais bien le noir sur blanc.
    img_inverted = ImageOps.invert(image_pil.convert('RGB'))
    results = reader.readtext(np.array(img_inverted), allowlist='0123456789')
    
    raw_boxes = []
    rejected_boxes = [] # Pour le debug
    
    for (bbox, text, prob) in results:
        (tl, tr, br, bl) = bbox
        x_min = int(min(tl[0], bl[0]))
        y_min = int(min(tl[1], tr[1]))
        x_max = int(max(tr[0], br[0]))
        y_max = int(max(bl[1], br[1]))
        
        box = [x_min, y_min, x_max, y_max]
        
        # Filtre de taille minimale
        if (x_max - x_min) < 20 or (y_max - y_min) < 10:
            continue

        #  LE FILTRE 
        # On regarde la couleur sur l'image originale (pas inversée)
        is_dark, brightness = is_dark_background(image_pil, box)
        
        if is_dark:
            # Si c'est sombre on garde
            raw_boxes.append(box)
        else:
            # Si c'est claire en reject
            rejected_boxes.append((box, brightness))

    # 2. Fusionner les morceaux (noir et rouge)
    merged_boxes = merge_nearby_boxes(raw_boxes)
    
    # 3. Choisir le meilleur candidat restant
    best_box = None
    max_score = 0
    
    for box in merged_boxes:
        w = box[2] - box[0]
        h = box[3] - box[1]
        
        # Le compteur est un rectangle large
        ratio = w / h if h > 0 else 0
        if ratio > 2.0:
            # On privilégie la zone la plus grande
            if (w * h) > max_score:
                max_score = w * h
                best_box = tuple(box)
                
    return best_box, rejected_boxes


#  ################ INTERFACE

st.title("Lecture Automatique de Compteurs ")
st.caption("Détecte et predit les chiffres de compteurs automatiquement.")

num_digits = st.sidebar.number_input("Nombre de chiffres", 4, 10, 8)
margin_fix = st.sidebar.slider("Rogner bords (px)", 0, 10, 2)

try:
    reader, interpreter = load_resources()
except Exception as e:
    st.error(f"Erreur : {e}")
    st.stop()

uploaded_file = st.file_uploader("Image du compteur", type=["jpg", "png", "jpeg"])

if uploaded_file:
    col1, col2 = st.columns(2)
    image_pil = Image.open(uploaded_file).convert('RGB')
    
    with col1:
        st.image(image_pil, caption="Image Originale", use_column_width=True)

    if st.button("Lancer l'analyse"):
        with st.spinner("Analyse et filtrage ..."):
            
            #  COEUR DE LA DETECTION 
            final_box, rejected = auto_detect_meter(reader, image_pil)
            
            # Dessin de debug
            debug_img = image_pil.copy()
            draw = ImageDraw.Draw(debug_img)
            
            # Dessiner en rouge ce qu'on a rejeté 
            for (r_box, lum) in rejected:
                draw.rectangle(r_box, outline="red", width=2)
            
            if final_box:
                # Dessiner en vert ce qu'on a gardé 
                draw.rectangle(final_box, outline="#00FF00", width=4)
                
                with col2:
                    st.image(debug_img, caption="Vert = Retenu | Rouge = Rejeté", use_column_width=True)
                
                #  DÉCOUPAGE ET LECTURE 
                x1, y1, x2, y2 = final_box
                meter_zone = image_pil.crop((x1, y1, x2, y2))
                zone_w, zone_h = meter_zone.size
                
                step_x = zone_w / num_digits
                digits_read = []
                crops = []
                
                for i in range(num_digits):
                    left = i * step_x
                    right = (i + 1) * step_x
                    digit_crop = meter_zone.crop((left + margin_fix, 0, right - margin_fix, zone_h))
                    
                    input_arr = preprocess_for_tflite(digit_crop)
                    digit, conf = predict_digit(interpreter, input_arr)
                    digits_read.append(str(digit))
                    crops.append(digit_crop)
                
                raw_value = ''.join(digits_read)
                if len(raw_value) > 3:
                    formatted_value = f"{raw_value[:-3]},{raw_value[-3:]}"
                else:
                    formatted_value = f"0,{raw_value.zfill(3)}"

                st.success(f"Valeur du compteur : {formatted_value}")

                
                # Affichage des chiffres
                cols = st.columns(num_digits)
                for i, c in enumerate(cols):
                    with c:
                        st.image(crops[i], use_column_width=True)
                        st.markdown(f"<div style='text-align: center;'><b>{digits_read[i]}</b></div>", unsafe_allow_html=True)
            
            else:
                st.image(debug_img, caption="Tout a été rejeté", use_column_width=True)
                st.error("Aucune zone à fond sombre détectée.")