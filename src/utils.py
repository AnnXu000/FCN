import numpy as np
import cv2

def get_topk_predictions(predictions, k=3):
    """function returns top k predictions"""
    top_k_indexes = predictions.argsort()[::-1][:k]
    top_k_predictions = [(labels_dict[index], predictions[index])for index in top_k_indexes]
    return top_k_predictions

def preprocess_input(img):
    """
       Function to normalize image as per imagenet papaer
       Accept image in rgb format
    """
    #convert to bgr format
    img = cv2.resize(img[:, :, ::-1], (224, 224)).astype(np.float32)
    img[:,:,0] -= 103.939
    img[:,:,1] -= 116.779
    img[:,:,2] -= 123.68
    img = np.expand_dims(img, axis=0)
    return img