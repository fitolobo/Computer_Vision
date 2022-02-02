# Construccion de la base de datos
# Rodolfo Lobo C. 26/07/2021, Globant Chile.
import os
import glob
import utils
import matplotlib.pyplot as plt
from utils import make_kmeans_palette, changing_palette, image_palette,augment_image, centers_to_lab_vectors, viz_color_palette
from typing import Dict, Tuple, Sequence
import collections
import pathlib
import random
import pickle
import cv2
#!pip install instagram-scraper
cwd = os.getcwd()

def run():
    '''
        Este codigo permite procesar y crear la base de datos para el modelo, a partir de un grupo de imagenes previas.
        Tras haber utilizado el scrapper y guardado las imagenes en el directorio "grafftales" (usando pip install instagram-scraper)
        https://github.com/arc298/instagram-scraper
        Para descargar todas las imagenes necesitas usar tu usuario personal y debes ser un contacto del instagram de interes.
        Para efectos de esta POC, descargué todas las imágenes de una página de grafitis que traen la paleta de colores a un costado.
    '''

  # Guardando nombre de los archivos
    images = glob.glob(cwd+'/grafftales/'+"*.jpg")
    filenames = list()
    for image in images:
        filenames.append(image)

    image_to_palette = collections.defaultdict(set)
    pathlist = pathlib.Path(cwd + "/grafftales/").glob("*.png")


    for i in range(len(filenames)):
        path = filenames[i]
        # Reading Image
        img = cv2.imread(str(path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Extraccion de la paleta
        #palette = centers_to_lab_vectors(make_kmeans_palette(img))
        palette = viz_color_palette(make_kmeans_palette(img))
        # Variando la paleta de colores original
        hue_shift = random.random()
        augmented_image = augment_image(img, "Image", hue_shift)
        # Guardando la nueva paleta
        #augmented_palette = centers_to_lab_vectors(make_kmeans_palette(augmented_image))
        augmented_palette  =  viz_color_palette(make_kmeans_palette(augmented_image))
        name = 'img_'+str(i)+'.png'
        # Guardando la paleta, la imagen aumentada y la paleta aumentada
        cv2.imwrite(f'data/train/input/'+name, img)
        pickle.dump(palette, open(f'data/train/old_palette/'+'img_'+str(i)+'.pkl', 'wb'))
        cv2.imwrite(f'data/train/output/'+name, augmented_image)
        pickle.dump(augmented_palette, open(f'data/train/new_palette/'+'img_'+str(i)+'.pkl', 'wb'))
        # Progress bar
        print('\r'+f' Processed Iamge {i}/{len(filenames)}',end='')
if __name__ == "__main__":
    run()
