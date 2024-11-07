# -- coding: utf-8 --
"""AutoGluon + CLIP inference script"""

import sys
import logging
import cv2
from autoGluonCLI import AutoGluonClip

# Set up logging
logging.basicConfig(level=logging.INFO)

# Add repository path
REPO = '/home/sebas/Documents/PuntoFisicoV2'
sys.path.append(REPO)

# Define constants
SAVEPATH = '/home/sebas/Documents/PuntoFisicoV2/Models' 
MODEL_NAME = 'AutoGluon-1'
FEATURE_EXTRACTOR_PATH = '/home/sebas/Documents/PuntoFisicoV2/ViT-B-32.pt'
MODEL_PATH = f'/home/sebas/Documents/PuntoFisicoV2/{MODEL_NAME}'
CONFIDENCE = 0.5

# Class labels
LABELS = {
    -1: 'Incierto',
    0: 'Botella',
    1: 'Empaques impresos',
    2: 'Envase',
    3: 'Lata',
    4: 'Orgánico',
    5: 'Otros',
    6: 'Papel no reciclable',
    7: 'Papeles',
}

BIN_ID = {
    0: 'Orgánico',
    1: 'No aprovechable',
    2: 'Reciclable',
    3: 'Papeles'
}

# Model parameters
BIN_CENTERS = [
    [200, 260],
    [500, 260],
    [811, 260],
    [1080, 270]
]
BIN_AREA = [
    [80, 1180],
    [100, 510]
]
MODE0_AREA = [
    (256, 976),
    (0, 720)
]

# Load images
im_root = '/home/sebas/Documents/PuntoFisicoV2/Test_images'
im0 = cv2.imread(f'{im_root}/t0.jpg')
im1 = cv2.imread(f'{im_root}/mode0.jpg')

# Display centers and areas
display = im1.copy()
for c in BIN_CENTERS:
    cv2.circle(display, c, 5, (0, 255, 0), -1)
display = cv2.rectangle(display, (MODE0_AREA[0][0], MODE0_AREA[1][0]),
                        (MODE0_AREA[0][1], MODE0_AREA[1][1]), (255, 0, 0), 2)
#cv2.imshow('Display', display)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Load model
model = AutoGluonClip(MODEL_PATH, FEATURE_EXTRACTOR_PATH, BIN_CENTERS, BIN_AREA, MODE0_AREA)

# Predict
res, dis = model(im0, im1, mode=0, confidence=CONFIDENCE)
print(f'Residuo:      {LABELS[res]} \nContenedor:   {dis}')

# Load another image and predict
im1 = cv2.imread(f'{im_root}/mode0-1.jpg')
res, dis = model(im0, im1, mode=0, confidence=CONFIDENCE)
print(f'Residuo:      {LABELS[res]} \nContenedor:   {dis}')