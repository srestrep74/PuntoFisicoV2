"""
Class for inference of AutoGluon models
with CLIP image encoder as feature extractor.

LIBRARIES REQUIRED
    >> pip install uv
    >> uv pip install autogluon
    >> uv pip install git+https://github.com/openai/CLIP.git
    
RUN (autocluon_clip conda env):
>> python -m Tests.Exploratory.autogluonCLIP_test

JCA
"""
import logging
logger = logging.getLogger(__name__)

import cv2
import torch
import clip
import numpy as np
from autogluon.tabular import TabularPredictor
from PIL import Image
import pandas as  pd



class AutoGluonClip():
    """ Inference of AutoGluon models with CLIP image encoder as feature extractor. """
    def __init__(self, model_path, feature_extractor_path,
                 bin_centers, bin_area, mode0_area,
                 min_change=30, border=0.2) -> None:
        """
        Params:
        : model_path (str) : Dirección a la carpeta del modelo
        : feature_extractor_path (str): Dirección al archivo del modelo CLIP
        : bin_centers (List): Lista de los centros de las canecas en la imagen
            [[x,y], [x,y], ...]
        : bin_area : (list) Área de la imagen correspondiente a las canecas
            [[x_init, x_end], [y_init, y_end]]
        : mode0_area (list) Area de predicción del modelo en modo 1 
             [[x_init, x_end], [y_init, y_end]]
        """
        # Feature extractor
        logger.info('Loading CLIP model...')
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.feature_extractor, self.preprocess = clip.load(feature_extractor_path, device=self.device)

        # Autogluon model from training folder
        logger.info('Loading AutoGluon model...')
        self.model = TabularPredictor.load(model_path, require_py_version_match=False)

        # Fisical device distances viewed from camera
        self.bc = np.array(bin_centers)
        self.bin_area = bin_area
        # Parameters of foreground identification
        self.min_change = min_change
        self.border = border

        # Area del modo 1
        # En modo 1 recorta siempre esta área predefinida
        #  [[x_init, x_end], [y_init, y_end]]
        self.mode0_area = mode0_area 


    def get_hull(self, contours, convex_hull=True):
        """Return the largest contour"""
        hull = None
        hull_area = 0
        hull_center = None
        main_contour = None
        # Only consider the larges convex hull of the image
        for c in contours:
            # calculate moments for each contour
            M = cv2.moments(c)
            if M['m00'] > hull_area:
                im_hull = cv2.convexHull(c) if convex_hull else c

                hull = im_hull
                hull_area = M['m00']

                # calculate x,y coordinate of center
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                hull_center = (cX, cY)

                main_contour = c

        return hull, hull_area, hull_center, main_contour

    def distance_points(self, waste_center, bin_centers):
        """
        Devuelte el índice del contenedor calculando la distancia mínima L2
        entre los centros de las canecas y el centro del desecho.

        Parametros
        : waste_center : (np.array) coordenadas del centro del desecho (x, y)
        : bin_centers : (np.array) matriz con las coordenadas de los centros de
                        los contenedores [[x,y], [x,y], ...]
        """

        dif = waste_center - bin_centers
        dist = np.linalg.norm(dif, axis=1)
        idx = np.argmin(dist)
        return idx

    def get_coords(self, main_contour, width, height):
        """Return bounding box coordinates based the contour.
            return (x,y,w,h): x and y are the initial coordinates
        """
        x,y,w,h  = cv2.boundingRect(main_contour)
        x_i, y_i, w_i, h_i = x,y,w,h
        if self.border:
            dx = self.border * w/2
            dy = self.border * h/2
            x = int(max(0, x-dx))
            y = int(max(0, y-dy))

            w = w+2*dx if x+w+2*dx < width else width-x
            h = h+2*dy if y+h+2*dy < height else height-y

            w = int(w)
            h = int(h)
        return x,y,w,h

    def get_fgd(self, im0, im1):
        """Función que quita el fondo, y recorta area solo con desecho
        Params:
        : im0 (np.array) : Fondo (BGR) cargada con OpenCV
        : im1 (np.array) : Desecho (BGR) cargada con
        : min_change : (int) Minimum pixel chamge to be considered

        """

        # # in LAB there are 2 color channels and 1 brightness channel:
        # #   L-channel: Brightness value in the image
        # #   A-channel: Red and green color in the image
        # #   B-channel: Blue and yellow color in the image
        lab0 = cv2.cvtColor(im0, cv2.COLOR_BGR2LAB)
        lab1 = cv2.cvtColor(im1, cv2.COLOR_BGR2LAB)
        diff_a = cv2.absdiff(lab0[:,:,1], lab1[:,:,1])
        diff_b = cv2.absdiff(lab0[:,:,2], lab1[:,:,2])
        diff = diff_a * diff_b
        diff = cv2.dilate(diff,(5,5),iterations=1)

        thresh = cv2.threshold(
                diff, self.min_change, 255,
                cv2.THRESH_BINARY)[1]/255

        # Dilate to get internal parts
        dilation = cv2.dilate(thresh, (5,5), iterations=1)

        # Find contours to only get the largest one
        contours, hierarchy = cv2.findContours(dilation.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        hull, hull_area, hull_center, main_contour = self.get_hull(contours, convex_hull=False)

        # Crop rectangle around largest contour
        x,y,w,h = self.get_coords(main_contour, im1.shape[1], im1.shape[0])


        # Recortar contour principa
        crop = im1[y:y+h, x:x+w]


        return crop, hull_center

    def crop_mode0(self, im1):
        """Crop area of image for mode0
            Parameters:
            : im1 (np.array) : Desecho (BGR) cargada con OpenCV
        """
        return im1[self.mode0_area[1][0]:self.mode0_area[1][1],  self.mode0_area[0][0]:self.mode0_area[0][1]]


    def __call__(self, im0, im1, mode=0, confidence=0.5):
        """Predice usando dos imágenes cargadas de OpenCV para quitar el fondo.
        Una de las impagenes contiene solo el fondo y la otra el fondo + el desecho
        Params:
        : im0 (np.array) : Fondo (BGR) cargada con OpenCV
        : im1 (np.array) : Desecho (BGR) cargada con OpenCV
        : show (Bool) : Muestra imagen procesada antes del modelo para debug
        : mode (int): : Indica el modo en el que debe trabajar el modelo Modo 1: 0 modo 2: 1

        Return
        : rec (int) : ID correspondiente al tipo de reciclaje
        : disp (int) : ID correspondiente al contenedor donde se desechó
        """
        # Las imágenes son modificadas y como son mutables evitar
        # Distorsionarlas enla predicción
        im0_src = im0.copy()
        im1_src = im1.copy()

        

        if mode==1:
            # Recortar área de la imágen del dispositivo
            im0 = im0[self.bin_area[1][0]:self.bin_area[1][1],  self.bin_area[0][0]:self.bin_area[0][1]]
            im1 = im1[self.bin_area[1][0]:self.bin_area[1][1],  self.bin_area[0][0]:self.bin_area[0][1]]
            # Remover fondo y obtener centro de contorno más grande
            fgd, center = self.get_fgd(im0, im1)
        elif mode==0:
            center = None
            fgd = self.crop_mode0(im1_src)

        # if show:
        #     showim(fgd)

        # Convertir OpenCV a PIL (RGB) 
        rgb = cv2.cvtColor(fgd, cv2.COLOR_BGR2RGB)
        im_pil = Image.fromarray(rgb) 

        # Agregar una dimensión más porque el modelo recibe un bache de imágenes
        im_prep = self.preprocess(im_pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            image_features = self.feature_extractor.encode_image(im_prep)

        # Pasar por modelo de predicción de tipo de reciclaje
        features_pd = pd.DataFrame(image_features)
        #res = self.model.predict(features_pd).iloc[0]
        prob = self.model.predict_proba(features_pd).to_numpy()

        res = np.argmax(prob) if np.max(prob) >= confidence else -1
        
        # Identificar contenedor donde se desechó
        if mode==1:
            disp = self.distance_points(center, self.bc)
        else:
            disp=None

        return int(res), disp