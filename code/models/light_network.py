from keras.layers import Conv2D, Conv2DTranspose, Input, MaxPooling2D, GlobalMaxPool2D, Dense, Dropout, Flatten
from keras.layers import concatenate, BatchNormalization, Activation, AveragePooling2D, Add, ZeroPadding2D, UpSampling2D
from keras.models import Model


def get_light_global_net(input_channels = 3, pretrained_weights_file = None, model_output_names = ['L'],
              final_activation = 'linear'):

    img_input = Input(shape=(144, 256, 6), name='RGB_and_normals_input')

    x = Conv2D(32, (3,3), activation='relu')(img_input)
    x = MaxPooling2D(pool_size = (2,2))(x)
    
    x = Conv2D(32, (3, 3), activation = 'relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(64, (3, 3), activation = 'relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = Conv2D(128, (3, 3), activation = 'relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    x = GlobalMaxPool2D()(x)

    x = Dense(units = 64, activation = 'relu')(x)
    x = Dropout(0.5)(x)
    
    ## output for theta (sin and cos)
    #out1 = Dense(units = 2, activation = final_activation, name = model_output_names[0])(x)
    ## output for phi (sin and cos)
    #out2 = Dense(units = 2, activation = final_activation, name = model_output_names[1])(x)

    out_vector = Dense(units = 3, activation = final_activation)(x)

    model = Model(inputs = [img_input], outputs = [out_vector], name = model_output_names[0])

    if pretrained_weights_file:
        model.load_weights(pretrained_weights_file)
        print("loaded weights from {}".format(pretrained_weights_file))
        
    return model

