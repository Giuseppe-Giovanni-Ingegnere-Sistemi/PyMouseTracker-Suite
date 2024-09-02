import cv2
import mediapipe as mp
import pyautogui

# Inicializar captura de cámara
cap = cv2.VideoCapture(0)

# Verificar si la cámara se abrió correctamente
if not cap.isOpened():
    print("No se pudo abrir la cámara")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Inicializar detector de manos de mediapipe
hand_detector = mp.solutions.hands.Hands()
drawing_utils = mp.solutions.drawing_utils

# Obtener tamaño de pantalla
screen_width, screen_height = pyautogui.size()

# Variables para almacenar posiciones de dedos
index_x, index_y = 0, 0
middle_x, middle_y = 0, 0
thumb_x, thumb_y = 0, 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Reflejar el fotograma horizontalmente
    frame = cv2.flip(frame, 1)
    
    # Convertir el fotograma a RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Procesar el fotograma con el detector de manos
    output = hand_detector.process(rgb_frame)
    
    # Verificar si se detectaron manos
    if output.multi_hand_landmarks:
        for hand_landmarks in output.multi_hand_landmarks:
            # Dibujar landmarks de la mano en el fotograma
            drawing_utils.draw_landmarks(frame, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
            
            # Obtener coordenadas de los dedos índice, medio y pulgar
            index_x = int(hand_landmarks.landmark[8].x * frame_width)
            index_y = int(hand_landmarks.landmark[8].y * frame_height)
            middle_x = int(hand_landmarks.landmark[12].x * frame_width)
            middle_y = int(hand_landmarks.landmark[12].y * frame_height)
            thumb_x = int(hand_landmarks.landmark[4].x * frame_width)
            thumb_y = int(hand_landmarks.landmark[4].y * frame_height)

            # Verificar si el dedo medio está más bajo que el índice
            if middle_y > index_y:
                # Si está más bajo, hacer clic
                pyautogui.click()
                pyautogui.sleep(1)
            else:
                # Si no, mover el cursor a la posición del índice
                pyautogui.moveTo(index_x, index_y)
    
    # Mostrar el fotograma con la detección
    cv2.imshow('Virtual Mouse', frame)
    
    # Salir del bucle al presionar 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar la captura de cámara y cerrar ventanas
cap.release()
cv2.destroyAllWindows()
