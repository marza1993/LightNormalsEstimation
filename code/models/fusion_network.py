from models.normals_network import get_norm_net
from models.light_network import get_light_global_net
from keras.layers import Concatenate, Input, Conv2D, Flatten, Dense
from keras.models import Model
import keras
import tensorflow as tf



@tf.keras.utils.register_keras_serializable(package='Custom', name=None)
class DepthDotProduct(keras.layers.Layer):
    def __init__(self, **kwargs):
        #self.name_other = name
        super(DepthDotProduct, self).__init__(**kwargs)


    def build(self, input_shape):
        self.w = self.add_weight(
            name = 'w_DepthDot',
            shape=(1,),
            trainable=False
        )
        self.b = self.add_weight(
            name = 'b_DepthDot',
            shape=(2,), initializer="random_normal", trainable=True
        )


    def call(self, input_normals, input_L):

        depthProd = keras.layers.Dot(axes = [-1,-1])([input_normals, input_L])
        return self.b[0] * depthProd + self.b[1]


    def get_config(self):
        config = super().get_config()
        #config.update({
        #    "name_other": self.name_other
        #})
        return config


    @classmethod
    def from_config(cls, config):
        return cls(**config)





class FusionNetWrap:

    def __init__(self, pretrained_weights_file= None, output_names = ['L', 'L_img']):

        self.pretrained_weights_file = pretrained_weights_file
        self.output_names = output_names

        self.normal_net = get_norm_net()
        self.light_net = get_light_global_net(model_output_names = [output_names[0]])

        # create light prediction model -> it takes as input both RGB and normals image
        RGB_normals_input = Concatenate()([self.normal_net.input, self.normal_net.output])

        light_net_output = self.light_net(RGB_normals_input)

        # phong model approximation
        out_luminance_img = DepthDotProduct(name = self.output_names[1])(self.normal_net.output, light_net_output)

        self.fusion_model = Model(inputs = [self.normal_net.input], outputs = [light_net_output, out_luminance_img], name='fusion_net')
        self.fusion_model.summary()

        if self.pretrained_weights_file:
            self.fusion_model.load_weights(self.pretrained_weights_file)
            print("loaded weights from {}".format(self.pretrained_weights_file))


    def get_fusion_net(self):

        return self.fusion_model


    def get_light_net(self):
        return self.light_net
    

    def get_normal_net(self):
        return self.normal_net











def get_fusion_net(pretrained_weights_file = None, output_names = ['L', 'L_img']):

    
    normal_net = get_norm_net()
    light_net = get_light_global_net(model_output_names = [output_names[0]])

    # create light prediction model -> it takes as input both RGB and normals image
    RGB_normals_input = Concatenate()([normal_net.input, normal_net.output])

    light_net_output = light_net(RGB_normals_input)

    # phong model approximation
    out_luminance_img = DepthDotProduct(name = output_names[1])(normal_net.output, light_net_output)
    #out_luminance_img = DepthDotProduct()(normal_net.output, light_net_output)


    fusion_model = Model(inputs = [normal_net.input], outputs = [light_net_output, out_luminance_img], name='fusion_net')
    fusion_model.summary()

    if pretrained_weights_file:
        fusion_model.load_weights(pretrained_weights_file)
        print("loaded weights from {}".format(pretrained_weights_file))

    return fusion_model






# def get_fusion_net_test2():

#    # RGB input
#    img_input = Input(shape=(256, 256, 3), name ='RGB_input')

#    # normals net
#    out_normals = Conv2D(3, (7,7), strides = 1, padding = 'same')(img_input) #NB: no activation

#    # concatenation RGB - normals
#    RGB_normals_input = Concatenate()([img_input, out_normals])

#    # light net
#    x = Conv2D(32, (3,3), activation='relu', padding='same')(RGB_normals_input)
#    x = Flatten()(x)
#    out_L = Dense(units = 3, activation = 'linear', name = "out_L")(x)

#    # phong level
#    out_luminance_img = DepthDotProduct()(out_normals, out_L)

#    fusion_model = Model(inputs = [img_input], outputs = [out_luminance_img, out_L])

#    return fusion_model


# def get_fusion_net_test():

#    # RGB input
#    img_input = Input(shape=(256, 256, 3), name ='RGB_input')

#    # normals net
#    out_normals = Conv2D(3, (7,7), strides = 1, padding = 'same')(img_input) #NB: no activation
#    normal_net = Model(inputs = [img_input], outputs = [out_normals])

#    # concatenation RGB - normals
#    # light net

#    RGB_normals_input = Concatenate()([normal_net.input, normal_net.output])


#    light_net_input = Input(shape = (256, 256, 6), name = 'RGB_normals')

#    x = Conv2D(32, (3,3), activation='relu', padding='same')(light_net_input)
#    x = Flatten()(x)
#    out_L = Dense(units = 3, activation = 'linear', name = "out_L")(x)
#    light_net = Model(inputs = [light_net_input], outputs = [out_L])

#    # phong level
#    out_luminance_img = DepthDotProduct()(normal_net.output, light_net(RGB_normals_input))

#    fusion_model = Model(inputs = [img_input], outputs = [light_net(RGB_normals_input), out_luminance_img])

#    return fusion_model





