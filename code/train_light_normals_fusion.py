
from models.fusion_network import get_fusion_net
from tensorflow.keras.utils import plot_model
from keras.callbacks import TensorBoard, ModelCheckpoint
import os
from DataLoader import Data_loader
import re
import random
import argparse


parser = argparse.ArgumentParser(description = "Script for training and testing 3d outdoor light direction estimation based on physical illumination model and deep learning")
                   
parser.add_argument("-d", "--dataset_folder", type=str, help="base dataset folder (must contain train, test, val folders).", required=True)
parser.add_argument("-m", "--model_dir", type=str, help="path where models will be saved.", required=True)
parser.add_argument("-b", "--batch_size", type=int, help="batch size", required=False, default=8)
parser.add_argument("-e", "--epochs", type=int, help="number of training epochs", required=False, default=10)
parser.add_argument("-p", "--pretrained_weights", type=str, help="path to pretrained weights file", required=False, default=None)
parser.add_argument("-f", "--dataset_fraction", type=float, help="portion of dataset to use (e.g., 0.1)", required=False, default=1)
args = parser.parse_args()


# parse arguments
dataset_path = args.dataset_folder
gt_path = os.path.join(dataset_path, "light.csv")
model_path = args.model_dir
pretrained_weights = args.pretrained_weights
batch_size = args.batch_size
N_epochs = args.epochs
portion = args.dataset_fraction


seed = 101 # seed for reproducibility of random split


train_dir = os.path.join(dataset_path, "train")
val_dir = os.path.join(dataset_path, "val")
test_dir = os.path.join(dataset_path, "test")

if not os.path.exists(train_dir):
    os.makedirs(train_dir)

if not os.path.exists(val_dir):
    os.makedirs(val_dir)

if not os.path.exists(test_dir):
    os.makedirs(test_dir)


# get list of images
regex_img = r".*\.(png|jpg)$"
list_imgs = [f for f in os.listdir(dataset_path) if re.search(regex_img,f)]
print("found {} images".format(len(list_imgs)))

random.seed(seed)
random.shuffle(list_imgs)

splits = [int(0.7 * len(list_imgs)), int(0.8 * len(list_imgs))]

train_imgs = list_imgs[:splits[0]]
val_imgs = list_imgs[splits[0]:splits[1]]
test_imgs = list_imgs[splits[1]:]


print("train set: {} images".format(len(train_imgs)))
print("val set: {} images".format(len(val_imgs)))
print("test set: {} images".format(len(test_imgs)))


for i in range(len(train_imgs)):
    os.rename(os.path.join(dataset_path, train_imgs[i]), os.path.join(train_dir, train_imgs[i]))

for i in range(len(val_imgs)):
    os.rename(os.path.join(dataset_path, val_imgs[i]), os.path.join(val_dir, val_imgs[i]))

for i in range(len(test_imgs)):
    os.rename(os.path.join(dataset_path, test_imgs[i]), os.path.join(test_dir, test_imgs[i]))

print("done.")


# run this code if "," is used for floating point values string (depending on geographic location)

with open(gt_path) as f:
    lines = f.readlines()

lines = [line.replace(",", ".") for line in lines]

with open(gt_path, "w") as f:
    f.writelines(lines)

print("done.")

# bugfix gt file 
with open(gt_path) as f:
    lines = f.readlines()

out_lines = []
for line in lines:
    match = re.search('scene.*', line)
    if(match):
        out_lines.append(match.group(0) + '\n')

gt_path = os.path.join(dataset_path, "light_out.csv")
with open(gt_path, "w") as f:
    f.writelines(out_lines)

print("done.")


train_dir = os.path.join(dataset_path, "train")
val_dir = os.path.join(dataset_path, "val")


model_output_names = ['L', 'L_img']
img_size = (144,256)
train_loader = Data_loader(batch_size, train_dir, gt_path, img_size = img_size, model_output_names=model_output_names)
val_loader = Data_loader(batch_size, val_dir, gt_path, img_size = img_size, model_output_names=model_output_names)

print("created data loaders.")

if(not os.path.exists(model_path)):
    os.makedirs(model_path)

model = get_fusion_net(pretrained_weights_file = pretrained_weights)

losses = {model_output_names[0]: 'mean_absolute_error', model_output_names[1]: 'mean_squared_error'}

# define weight for each loss
lossWeights = {model_output_names[0]: 1.0, model_output_names[1]: 1.0}

# compile model
model.compile(loss=losses, loss_weights=lossWeights, optimizer='adam', metrics=["accuracy"])

model_file_name = os.path.join(model_path, 'model_{epoch:03d}-{val_loss:.3f}.hdf5')
checkpoint = ModelCheckpoint(model_file_name, monitor='val_loss', verbose=1, save_best_only=True, mode='min')

print("start training ..")
history = model.fit(
        train_loader,
        steps_per_epoch= int(portion * (train_loader.N_samples() // batch_size)),
        epochs=N_epochs,
        validation_data=val_loader,
        validation_steps=int(portion * (val_loader.N_samples() // batch_size)),
        callbacks = [checkpoint]
        # workers=50, # uncomment if hardware supports it
        # use_multiprocessing=True
        )


print("finished training.")

