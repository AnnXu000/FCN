import random
import math
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf


def load_image(img_path):
    return np.asarray(Image.open(img_path))

#resize the image keeping the same aspect ratio    
def resize_image(img, img_size, interpolation=cv2.INTER_LINEAR):
    h, w = img.shape[:2]
    size = img_size[0]
    c = None if len(img.shape) < 3 else img.shape[2]
    if h == w: return cv2.resize(img, (size, size), interpolation)
    if h > w: dif = h
    else:     dif = w
    x_pos = int((dif - w)/2.)
    y_pos = int((dif - h)/2.)
    if c is None:
        mask = np.zeros((dif, dif), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w] = img[:h, :w]
    else:
        mask = np.zeros((dif, dif, c), dtype=img.dtype)
        mask[y_pos:y_pos+h, x_pos:x_pos+w, :] = img[:h, :w, :]
    return cv2.resize(mask, (size, size), interpolation)


class VOCDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, img_files, label_files, img_size=(224, 2224), batch_size=32, aug=False):
        self.img_files = img_files
        self.label_files = label_files
        self.batch_size = batch_size
        self.img_size = img_size
        self.aug = False
    
    def __len__(self):
        return math.ceil(len(self.img_files)/self.batch_size)
    
    def __getitem__(self, idx):
        batch_x = self.img_files[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.label_files[idx * self.batch_size:(idx + 1) * self.batch_size]
        images=[]
        labels=[]
        for img_path, label_path in zip(batch_x, batch_y):
            img, label = load_image(img_path).astype('float32'), load_image(label_path)
            #normalize as per imagenet settings
            img[:,:,0] -= 123.68
            img[:,:,1] -= 116.779
            img[:,:,2] -= 103.939
            #resize the image to given size
            img, label = resize_image(img, self.img_size), resize_image(label, self.img_size)
            images.append(img)
            labels.append(label)
        return np.stack(images), np.stack(labels)
    
    def on_epoch_end(self):
        z = list(zip(self.img_files, self.label_files))
        random.shuffle(z)
        self.img_files, self.label_files = zip(*z)
