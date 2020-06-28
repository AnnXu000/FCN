from .vgg16 import VGG16
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Add
from tensorflow.keras.models import Model


def fcn(image_size=(224, 224), upsampling_factor=8, num_classes=21, backbone_weights_path=None):
    """Function to create a fully convolutional network for semantic segmentation
       Arguments:
       image_size: Input image size
       upsampling_factor: Either 8, 16 or 32, upsampling stride back to original image pixels in one step
       num_classes: Number of classes
       backbone_weights_path: file path to pretrained imagenet weights for backbone network"""
    #upsampling factor should be either 8, 16 or 32
    assert (upsampling_factor == 8 or upsampling_factor == 16 or upsampling_factor == 32), "upsampling_factor should be either 8, 16 or 32"

    #create backbone network, subsampling rate is 32
    #load pretrained imagenet weights if weight file is given
    backbone = VGG16(img_height=image_size[0], img_width=image_size[1], weights_path=backbone_weights_path)
    
    #extract pool3, pool4, pool5 output tensors from backback network
    pool3 = backbone.get_layer(name='block3_pool').output
    pool4 = backbone.get_layer(name='block4_pool').output
    pool5 = backbone.get_layer(name='block5_pool').output
    
    #predictions over num_classes
    predictions = Conv2D(num_classes, (1, 1), activation='relu', name='pool5_predictions')(pool5)
    
    #FCN-32s
    if upsampling_factor == 32:
        #32x uplsampled predictions
        predictions_32up = Conv2DTranspose(num_classes, (3, 3), strides=(32, 32), padding='same', activation='softmax', name='32x_upsampled_predictions')(predictions)
        return Model(inputs=backbone.input, outputs=predictions_32up, name='FCN-32s')
    
    #pool4 prediction
    predictions_pool4 = Conv2D(num_classes, (1, 1), activation='relu', name='pool4_predictions')(pool4)
    #2x upsampled predictions
    predictions_2up = Conv2DTranspose(num_classes, (3, 3), strides=(2, 2), padding='same', activation='relu', name='2x_upsampled_predictions')(predictions)
    #combine coarse and fine predictions
    predictions_2up_pool4 = Add(name='add_2x_pool4_predictions')([predictions_2up, predictions_pool4])
    
    #FCN-16s
    if upsampling_factor == 16:
        #16x uplsampled predictions
        predictions_16up = Conv2DTranspose(num_classes, (3, 3), strides=(16, 16), padding='same', activation='softmax', name='16x_upsampled_predictions')(predictions_2up_pool4)
        return Model(inputs=backbone.input, outputs=predictions_16up, name='FCN-16s')
    
    #pool3 predictions, change number of filters to num_classes 
    predictions_pool3 = Conv2D(num_classes, (1, 1), activation='relu', name='pool3_predictions')(pool3)
    #4x upsampled predictions
    predictions_4up = Conv2DTranspose(num_classes, (3, 3), strides=(2, 2), padding='same', activation='relu', name='4x_upsampled_predictions')(predictions_2up_pool4)
    #combine coarse and fine predictions
    predictions_4up_pool3 = Add(name='add_4x_pool3_predictions')([predictions_4up, predictions_pool3])
    
    #FCN-8s
    #8x upsampled predictions
    predictions_8up = Conv2DTranspose(num_classes, (3, 3), strides=(8, 8), padding='same', activation='softmax', name='8x_upsampled_predictions')(predictions_4up_pool3)
    return Model(inputs=backbone.input, outputs=predictions_8up, name='FCN-8s')