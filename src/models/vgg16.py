from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Input, Flatten, Dropout
from tensorflow.keras.models import Model

def VGG16(img_height=224, img_width =224, weights_path=None, include_top=False):

    #input tensor
    input_img = Input(shape=(img_height, img_width, 3), name='input')

    #block1
    x = Conv2D(64, (3, 3), padding='same', activation='relu', name='block1_conv1')(input_img)
    x = Conv2D(64, (3, 3), padding='same', activation='relu', name='block1_conv2')(x)
    x = MaxPool2D(pool_size=(2, 2), name='block1_pool')(x)

    #block2
    x = Conv2D(128, (3, 3), padding='same', activation='relu', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), padding='same', activation='relu', name='block2_conv2')(x)
    x = MaxPool2D(pool_size=(2,2), name='block2_pool')(x)

    #block3
    x = Conv2D(256, (3, 3), padding='same', activation='relu', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), padding='same', activation='relu', name='block3_conv3')(x)
    x = MaxPool2D(pool_size=(2, 2), name='block3_pool')(x)

    #block4
    x = Conv2D(512, (3, 3), padding='same', activation='relu', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu', name='block4_conv3')(x)
    x = MaxPool2D(pool_size=(2, 2), name='block4_pool')(x)

    #block5
    x = Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), padding='same', activation='relu', name='block5_conv3')(x)
    x = MaxPool2D(pool_size=(2, 2), name='block5_pool')(x)

    

    #add layers for classification
    if include_top:
        #flatten the input
        x = Flatten(name='flatten')(x)
        #add fc layers
        x = Dense(4096, activation='relu', name='fc1')(x)
        x = Dropout(0.5)(x)
        x = Dense(4096, activation='relu', name='fc2')(x)
        x = Dropout(0.5)(x)
        x = Dense(1000, activation='softmax', name='predictions')(x)

    model = Model(inputs=input_img, outputs=x)
    
    if weights_path:
        model.load_weights(weights_path, by_name=True)

    return model