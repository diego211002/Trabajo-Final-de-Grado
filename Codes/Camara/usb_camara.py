import cv2
import time
import io
import zipfile
from collections import deque
from ultralytics import YOLO

# 1. Cargar modelo y cámara USB externa
model = YOLO("yolo26n.pt")
cap = cv2.VideoCapture(2) # Cambia a 0 o 2 si tu cámara USB usa otro índice

# Parámetros de control y memoria
UMBRAL_AREA = 0.25          # 25% de la pantalla = peligro
INTERVALO_CAPTURA = 1.0     # 1.0 segundo = frecuencia de 1 Hz
MAX_IMAGENES_ZIP = 5        # Máximo de fotos en memoria (evita llenar la RAM/disco)

# Cola circular que descarta automáticamente las fotos viejas al llegar a 5
buffer_imagenes = deque(maxlen=MAX_IMAGENES_ZIP)
ultimo_registro = 0

def generar_y_enviar_zip():
    """Crea un ZIP en RAM con las últimas capturas y lo guarda/envía."""
    zip_en_memoria = io.BytesIO()
    
    with zipfile.ZipFile(zip_en_memoria, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for nombre_archivo, bytes_jpg in buffer_imagenes:
            zf.writestr(nombre_archivo, bytes_jpg)
            
    # Guardar el ZIP resultante en disco (siempre sobrescribe el mismo archivo para no ocupar espacio)
    with open("evidencia_obstaculo.zip", "wb") as f:
        f.write(zip_en_memoria.getvalue())
        
    print(f"[ZIP Actualizado] Contiene {len(buffer_imagenes)} imagen(es). Peso: {len(zip_en_memoria.getvalue()) / 1024:.1f} KB")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error al leer la cámara USB.")
        break

    alto_img, ancho_img, _ = frame.shape
    area_total = alto_img * ancho_img
    peligro_detectado = False

    results = model(frame, conf=0.5, verbose=False)

    for r in results:
        for box in r.boxes:
            # CORRECCIÓN AQUÍ: se añade [0] para extraer las 4 coordenadas del tensor
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            proporcion = ((x2 - x1) * (y2 - y1)) / area_total

            if proporcion >= UMBRAL_AREA:
                peligro_detectado = True
                color = (0, 0, 255)
                etiqueta = f"OBSTACULO ({proporcion:.0%})"
            else:
                color = (0, 255, 0)
                etiqueta = f"OK ({proporcion:.0%})"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, etiqueta, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 2. Acción ante obstáculo cercano
    if peligro_detectado:
        cv2.putText(frame, "SISTEMA DETENIDO", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        # serial.write(b'STOP\n')  <-- Tu orden de parada aquí

        tiempo_actual = time.time()
        if (tiempo_actual - ultimo_registro) >= INTERVALO_CAPTURA:
            exito, buffer_jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])
            
            if exito:
                nombre = f"captura_{int(tiempo_actual)}.jpg"
                buffer_imagenes.append((nombre, buffer_jpg.tobytes()))
                generar_y_enviar_zip()
                
            ultimo_registro = tiempo_actual

    cv2.imshow("Detector con ZIP ligero", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
