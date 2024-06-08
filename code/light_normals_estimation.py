from models.fusion_network import get_fusion_net
from tensorflow.keras.utils import plot_model
from keras.callbacks import TensorBoard

from keras.layers import Input, Conv2D
from keras.models import Model
import os
from DataLoader import Data_loader


dataset_path = "C:/Users/mar-z/progetti/data/outdoor_cpp"
gt_path = os.path.join(dataset_path, "light.csv")
train_dir = os.path.join(dataset_path, "train")
val_dir = os.path.join(dataset_path, "val")

batch_size = 4

model_output_names = ['L', 'L_img']
img_size = (256,256)
train_loader = Data_loader(batch_size, train_dir, gt_path, img_size = img_size, model_output_names=model_output_names)
val_loader = Data_loader(batch_size, val_dir, gt_path, img_size = img_size, model_output_names=model_output_names)

train_loader.visualize_batch()
val_loader.visualize_batch()


#model = get_fusion_net()
#model.summary()
#plot_model(model, "fusion_model.png", expand_nested = True, dpi=100)


## descrizione esperimento (comparirà nel path)
#nome_esperimento = f"_light_normals_fusion_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
#tensorboard_log_dir = ".\\log_esperimenti\\" + nome_esperimento

#tensorboard_callback = TensorBoard(log_dir=tensorboard_log_dir, histogram_freq=1, update_freq = 'epoch')




#img_input = Input(shape=(256, 256, 6), name='RGB_and_normals_input')
#x = Conv2D(32, (3,3), activation='relu')(img_input)
#model = Model(inputs = img_input, outputs = x)

#a = model.output

