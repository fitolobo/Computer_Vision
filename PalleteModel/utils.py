from sklearn.cluster import KMeans
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from skimage import io, color
from skimage.color import rgb2hsv,hsv2rgb,rgb2lab,lab2rgb
import PIL
import random
from colormath.color_objects import sRGBColor, LabColor
from colormath.color_conversions import convert_color

'''
    Este grupo de funciones permite extraer las paletas de colores de las imágenes.
    Además permite realizar rotaciones en el espacio de colores para obtener
    imágenes nuevas.
'''
################################################################################
def rgb_to_cielab(a):
    """
    a is a pixel with RGB coloring
    """
    a1,a2,a3 = a/255
    color1_rgb = sRGBColor(a1, a2, a3);
    color1_lab = convert_color(color1_rgb, LabColor);
    final = [color1_lab.lab_l,color1_lab.lab_a,color1_lab.lab_b]
    return final

################################################################################
def centers_to_lab_vectors(centers):
    row,col = centers.shape
    target_palette = np.zeros((row,col))
    for i in range(row):
        target_palette[i,:] = rgb_to_cielab(centers[i,:])

    return np.reshape(target_palette,(row*col,1))

################################################################################
def read_image(path):
    with open(path,"rb") as f:
        return np.array(Image.open(f))

################################################################################
def preprocess_image(img):
    return img.reshape((-1,3)).astype("float32") / 255

################################################################################
def get_kmeans_centers(img,nclusters):
    return KMeans(n_clusters=nclusters).fit(img).cluster_centers_

################################################################################
def make_kmeans_palette(img,nclusters=6):
    pixels = preprocess_image(img)
    centers = get_kmeans_centers(pixels,nclusters)
    return centers

################################################################################
def augment_image(img, title, hue_shift):
    img_HSV = matplotlib.colors.rgb_to_hsv(img)
    a_2d_index = np.array([[1,0,0] for _ in range(img_HSV.shape[1])]).astype('bool')
    img_HSV[:, a_2d_index] = (img_HSV[:, a_2d_index] + hue_shift) % 1
    new_img = matplotlib.colors.hsv_to_rgb(img_HSV).astype(int)
    img = img.astype(np.float) / 255.0
    new_img = new_img.astype(np.float) / 255.0
    ori_img_LAB = rgb2lab(img)
    new_img_LAB = rgb2lab(new_img)
    new_img_LAB[:, :, 0] = ori_img_LAB[:, :, 0]
    new_img_augmented = (lab2rgb(new_img_LAB)*255.0).astype(int)
    return new_img_augmented

################################################################################
def shift_channel(c, amount):
    if amount > 0:
        lim = 255 - amount
        c[c >= lim] = 255
        c[c < lim] += amount
    elif amount < 0:
        amount = -amount
        lim = amount
        c[c <= lim] = 0
        c[c > lim] -= amount
    return c

################################################################################
# Agrega la variacion de color y mantiene la luminosidad
def changing_palette(img,var):

    imagen_original = img.copy()
    lab = color.rgb2lab(imagen_original)
    L = lab[:,:,0]
    hsv_img = rgb2hsv(imagen_original)
    h = hsv_img[:,:,0]
    rand_h = var
    shift_h = random.randint(-rand_h, rand_h)
    hsv_img[:,:,0]=shift_channel(h, shift_h)
    new_rgb_image = hsv2rgb(hsv_img)
    new_lab_image = rgb2lab(new_rgb_image)
    new_lab_image[:,:,0] = L
    return lab2rgb(new_lab_image)

################################################################################
# Visualizar la paleta de colores
def image_palette(centers,name):

    plt.figure(figsize=(14,8))
    # crea repeticiones del mismo color para poder formar cubos de color uniforme en el plot
    x = centers[np.concatenate([[i] * 100 for i in range(len(centers))]).reshape((-1,10)).T]
    plt.imshow(x)
    plt.xticks([])
    plt.yticks([])
    plt.savefig(name)

    return x

################################################################################
def centers_to_hexa(centers):
    hexa_colors = list()
    for i in range(centers.shape[0]):
        hexa_colors.append(matplotlib.colors.to_hex(centers[i,:]))
    return hexa_colors

################################################################################
def viz_color_palette(centers):
    """
    visualize color palette
    """
    hexcodes = centers_to_hexa(centers)
    while len(hexcodes) < 6:
        hexcodes = hexcodes + hexcodes
    hexcodes = hexcodes[:6]
    palette = []
    for hexcode in hexcodes:
        rgb = np.array(list(int(hexcode.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)))
        palette.append(rgb)
    palette = np.array(palette)[np.newaxis, :, :]
    return palette

################################################################################
def resize_image(img):
# Tamaño propuesto en el paper
    SIZE = (288,432)
    img_pil = PIL.Image.fromarray(img).resize(SIZE, resample=PIL.Image.BILINEAR)
    return img_pil
