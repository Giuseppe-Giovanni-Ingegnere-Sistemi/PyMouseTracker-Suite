import cv2
import mediapipe as mp
import numpy as np
from math import acos, degrees
import random

def palm_centroid(coordinates_list):
     coordinates = np.array(coordinates_list)
     centroid = np.mean(coordinates, axis=0)
     centroid = int(centroid[0]), int(centroid[1])
     return centroid

def fingers_up_down(hand_results, thumb_points, palm_points, fingertips_points, finger_base_points):
     fingers = None
     coordinates_thumb = []
     coordinates_palm = []
     coordinates_ft = []
     coordinates_fb = []
     for hand_landmarks in hand_results.multi_hand_landmarks:
          for index in thumb_points:
               x = int(hand_landmarks.landmark[index].x * width)
               y = int(hand_landmarks.landmark[index].y * height)
               coordinates_thumb.append([x, y])
          
          for index in palm_points:
               x = int(hand_landmarks.landmark[index].x * width)
               y = int(hand_landmarks.landmark[index].y * height)
               coordinates_palm.append([x, y])
          
          for index in fingertips_points:
               x = int(hand_landmarks.landmark[index].x * width)
               y = int(hand_landmarks.landmark[index].y * height)
               coordinates_ft.append([x, y])
          
          for index in finger_base_points:
               x = int(hand_landmarks.landmark[index].x * width)
               y = int(hand_landmarks.landmark[index].y * height)
               coordinates_fb.append([x, y])
          ##########################
          # Pulgar
          p1 = np.array(coordinates_thumb[0])
          p2 = np.array(coordinates_thumb[1])
          p3 = np.array(coordinates_thumb[2])

          l1 = np.linalg.norm(p2 - p3)
          l2 = np.linalg.norm(p1 - p3)
          l3 = np.linalg.norm(p1 - p2)

          # Calcular el ángulo
          to_angle = (l1**2 + l3**2 - l2**2) / (2 * l1 * l3)
          if int(to_angle) == -1:
               angle = 180
          else:
               angle = degrees(acos(to_angle))
          thumb_finger = np.array(False)
          if angle > 150:
               thumb_finger = np.array(True)
          
          ################################
          # Índice, medio, anular y meñique
          nx, ny = palm_centroid(coordinates_palm)
          cv2.circle(frame, (nx, ny), 3, (0, 255, 0), 2)
          coordinates_centroid = np.array([nx, ny])
          coordinates_ft = np.array(coordinates_ft)
          coordinates_fb = np.array(coordinates_fb)

          # Distancias
          d_centrid_ft = np.linalg.norm(coordinates_centroid - coordinates_ft, axis=1)
          d_centrid_fb = np.linalg.norm(coordinates_centroid - coordinates_fb, axis=1)
          dif = d_centrid_ft - d_centrid_fb
          fingers = dif > 0
          fingers = np.append(thumb_finger, fingers)

          mp_drawing.draw_landmarks(
               frame,
               hand_landmarks,
               mp_hands.HAND_CONNECTIONS,
               mp_drawing_styles.get_default_hand_landmarks_style(),
               mp_drawing_styles.get_default_hand_connections_style())
     return fingers

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Pulgar
thumb_points = [1, 2, 4]

# Índice, medio, anular y meñique
palm_points = [0, 1, 2, 5, 9, 13, 17]
fingertips_points = [8, 12, 16, 20]
finger_base_points =[6, 10, 14, 18]

# FINGERS COMBINATIONS
TO_ACTIVATE = np.array([True, False, False, False, False])
# Piedra, papel, tijeras
PIEDRA = np.array([False, False, False, False, False])
PAPEL = np.array([True, True, True, True, True])
TIJERAS = np.array([False, True, True, False, False])

# REGLAS PIEDRA, PAPEL, TIJERAS (0, 1, 2)
WIN_GAME = ["02", "10", "21"]

pc_option = False # Si la pc ha escogido o no
detect_hand = True

THRESHOLD = 10
THRESHOLD_RESTART = 50

count_like = 0
count_piedra = 0
count_papel = 0
count_tijeras = 0
count_restart = 0

# Images to show
image1 = cv2.imread("1.jpg")
image2 = cv2.imread("2.jpg")
image_winner = cv2.imread("3.jpg")
image_tie = cv2.imread("4.jpg")
image_loser = cv2.imread("5.jpg")
# Image to concat
imAux = image1

player = None