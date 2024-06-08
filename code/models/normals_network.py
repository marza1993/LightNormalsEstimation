import tensorflow as tf
import keras
from keras.layers import Conv2D, Conv2DTranspose, Input, MaxPooling2D, GlobalMaxPool2D, Dense, Dropout, Flatten
from keras.layers import concatenate, BatchNormalization, Activation, AveragePooling2D, Add, ZeroPadding2D, UpSampling2D
from keras.models import Model


# parametric inception module
#    - with BatchNorm between conv and relu activation
#    - max pool layer can be disabled
@tf.keras.utils.register_keras_serializable(package='Custom', name=None)
class MyInception(keras.layers.Layer):

    def __init__(self, N_channels1, conv_dims, use_max_pool = False, N_channels_pool_conv = 16, **kwargs):
        super(MyInception, self).__init__(**kwargs)
        
        # ***********
        # needed for get_config method 
        self.N_channels1 = N_channels1
        self.conv_dims = conv_dims
        self.use_max_pool = use_max_pool
        self.N_channels_pool_conv = N_channels_pool_conv
        # ***********

        # 1x1 conv
        self.conv1 = keras.Sequential([
                                        Conv2D(N_channels1, (1,1), strides = 1, padding = 'valid'),
                                        BatchNormalization(),
                                        Activation('relu')
                                      ])
        # parallel conv layers
        N_parallel_convs = len(conv_dims)
        self.hidden_layers = []
        for i in range(N_parallel_convs):

            filter_size = conv_dims[i][0]
            N_channels_out_a = conv_dims[i][1]
            N_channels_out_b = conv_dims[i][2]
            pad_size = int((filter_size[0] - 1) / 2)

            # 1x1 conv + kxk conv
            curr_layer = keras.Sequential([
                                            Conv2D(N_channels_out_a, (1,1), strides = 1, padding = 'valid'),
                                            BatchNormalization(),
                                            Activation('relu'),
                                            ZeroPadding2D(padding = (pad_size, pad_size)),
                                            Conv2D(N_channels_out_b, filter_size, strides = 1, padding = 'valid'),
                                            BatchNormalization(),
                                            Activation('relu')
                                          ])
            self.hidden_layers.append(curr_layer)


    def call(self, input_layer):
        output1 = self.conv1(input_layer)
        outputs = [output1]
        for i in range(len(self.hidden_layers)):
            outputs.append(self.hidden_layers[i](input_layer))
        return concatenate(outputs, axis = -1)


    def get_config(self):
        config = super().get_config()
        config.update({
            "N_channels1": self.N_channels1,
            "conv_dims": self.conv_dims,
            "use_max_pool": self.use_max_pool,
            "N_channels_pool_conv": self.N_channels_pool_conv
        })
        return config


    @classmethod
    def from_config(cls, config):
        return cls(**config)

                
@tf.keras.utils.register_keras_serializable(package='Custom', name=None)
class Module4(keras.layers.Layer):

    def __init__(self, **kwargs):
        super(Module4, self).__init__(**kwargs)

        self.layer1 = keras.Sequential([
                                        MyInception(64, [((3,3),32,64), ((5,5),32,64), ((7,7),32,64)]),
                                        MyInception(64, [((3,3),32,64), ((5,5),32,64), ((7,7),32,64)])
                                        ])
        self.layer2 = keras.Sequential([
                                        AveragePooling2D(pool_size = (2,2), strides = 2),
                                        MyInception(64, [((3,3),32,64), ((5,5),32,64), ((7,7),32,64)]),
                                        MyInception(64, [((3,3),32,64), ((5,5),32,64), ((7,7),32,64)]),
                                        MyInception(64, [((3,3),32,64), ((5,5),32,64), ((7,7),32,64)]),
                                        UpSampling2D(size = (2,2), interpolation = "nearest")
                                        ])
    def call(self, input_layer):

        output1 = self.layer1(input_layer)
        output2 = self.layer2(input_layer)
        return Add()([output1, output2])


    def get_config(self):
        config = super().get_config()
        return config


    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable(package='Custom', name=None)
class Module3(keras.layers.Layer):

    def __init__(self, module4, **kwargs):
        super(Module3, self).__init__(**kwargs)

        self.module4 = module4
        self.layer1 = keras.Sequential([
                                        MyInception(64, [((3,3),32,64), ((5,5),32,64), ((7,7),32,64)]),
                                        MyInception(64, [((3,3),64,64), ((7,7),64,64), ((11,11),64,64)])
                                        ])
        self.layer2 = keras.Sequential([
                                        AveragePooling2D(pool_size = (2,2), strides = 2),
                                        MyInception(64, [((3,3),32,64), ((5,5),32,64), ((7,7),32,64)]),
                                        MyInception(64, [((3,3),32,64), ((5,5),32,64), ((7,7),32,64)]),
                                        module4,
                                        MyInception(64, [((3,3),32,64), ((5,5),32,64), ((7,7),32,64)]),
                                        MyInception(64, [((3,3),64,64), ((7,7),64,64), ((11,11),64,64)]),
                                        UpSampling2D(size = (2,2), interpolation = "nearest")
                                        ])

    def call(self, input_layer):

        output1 = self.layer1(input_layer)
        output2 = self.layer2(input_layer)
        return Add()([output1, output2])


    def get_config(self):
        config = super().get_config()
        config.update({
            "module4": self.module4
        })
        return config


    @classmethod
    def from_config(cls, config):
        return cls(**config)


@tf.keras.utils.register_keras_serializable(package='Custom', name=None)
class Module2(keras.layers.Layer):

    def __init__(self, module3, **kwargs):
        super(Module2, self).__init__(**kwargs)

        self.module3 = module3
        self.layer1 = keras.Sequential([
                                        MaxPooling2D((2,2), strides=(2,2)),
                                        MyInception(32, [((3,3),32,32), ((5,5),32,32), ((7,7),32,32)]),
                                        MyInception(64, [((3,3),32,64), ((5,5),32,64), ((7,7),32,64)]),
                                        module3,
                                        MyInception(64, [((3,3),32,64), ((5,5),32,64), ((7,7),32,64)]),
                                        MyInception(32, [((3,3),32,32), ((5,5),32,32), ((7,7),32,32)]),
                                        UpSampling2D(size = (2,2), interpolation = "nearest")
                                        ])
        self.layer2 = keras.Sequential([
                                        MyInception(32, [((3,3),32,32), ((5,5),32,32), ((7,7),32,32)]),
                                        MyInception(32, [((3,3),64,32), ((7,7),64,32), ((11,11),64,32)])
                                        ])

    def call(self, input_layer):

        output1 = self.layer1(input_layer)
        output2 = self.layer2(input_layer)
        return Add()([output1, output2])


    def get_config(self):
        config = super().get_config()
        config.update({
            "module3": self.module3
        })
        return config


    @classmethod
    def from_config(cls, config):
        return cls(**config)



@tf.keras.utils.register_keras_serializable(package='Custom', name=None)
class Module1(keras.layers.Layer):

    def __init__(self, module2, **kwargs):
        super(Module1, self).__init__(**kwargs)

        self.module2 = module2
        self.layer1 = keras.Sequential([
                                        MaxPooling2D((2,2), strides=(2,2)),
                                        MyInception(32, [((3,3),32,32), ((5,5),32,32), ((7,7),32,32)]),
                                        MyInception(32, [((3,3),32,32), ((5,5),32,32), ((7,7),32,32)]),
                                        module2,
                                        MyInception(32, [((3,3),64,32), ((5,5),64,32), ((7,7),64,32)]),
                                        MyInception(16, [((3,3),32,16), ((7,7),32,16), ((11,11),32,16)]),
                                        UpSampling2D(size = (2,2), interpolation = "nearest")
                                        ])
        self.layer2 = keras.Sequential([
                                        MyInception(16, [((3,3),64,16), ((7,7),64,16), ((11,11),64,16)])
                                        ])

    def call(self, input_layer):

        output1 = self.layer1(input_layer)
        output2 = self.layer2(input_layer)
        return Add()([output1, output2])


    def get_config(self):
        config = super().get_config()
        config.update({
            "module2": self.module2
        })
        return config


    @classmethod
    def from_config(cls, config):
        return cls(**config)


#def get_norm_net(input_layer, pretrained_weights_file = None):
    
#    ## first block
#    #img_input = Input(shape=(256, 256, 3), name='RGB_input')

#    module4 = Module4()
#    module3 = Module3(module4)
#    module2 = Module2(module3)
#    module1 = Module1(module2)

    
#    x = ZeroPadding2D(padding = (3,3))(input_layer)
#    x = Conv2D(128, (7,7), strides = 1, padding = 'valid')(x) #NB: no activation
#    x = BatchNormalization()(x)
#    x = Activation('relu')(x)
#    x = module1(x)
#    x = ZeroPadding2D(padding = (1,1))(x)
#    out = Conv2D(3, (3,3), strides = 1, padding = 'valid')(x)
    
#    model = Model(inputs = [input_layer], outputs = [out], name='normal_net')
#    #model.summary()
    
#    if pretrained_weights_file:
#        model.load_weights(pretrained_weights_file)
#        print("loaded weights from {}".format(pretrained_weights_file))
    
#    return model


def get_norm_net(pretrained_weights_file = None):
    
    ## first block
    img_input = Input(shape=(144, 256, 3), name='RGB_input')

    module4 = Module4()
    module3 = Module3(module4)
    module2 = Module2(module3)
    module1 = Module1(module2)

    
    x = ZeroPadding2D(padding = (3,3))(img_input)
    x = Conv2D(128, (7,7), strides = 1, padding = 'valid')(x) #NB: no activation
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = module1(x)
    x = ZeroPadding2D(padding = (1,1))(x)
    out = Conv2D(3, (3,3), strides = 1, padding = 'valid')(x)
    
    model = Model(inputs = [img_input], outputs = [out], name='normal_net')
    #model.summary()
    
    if pretrained_weights_file:
        model.load_weights(pretrained_weights_file)
        print("loaded weights from {}".format(pretrained_weights_file))
    
    return model
