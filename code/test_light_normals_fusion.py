

import math
import re
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import cv2
import argparse
import os
from models.fusion_network import get_fusion_net
import matplotlib.pyplot as plt


def read_gt_light(light_gt_path):

    gt_light = {}
    with open(light_gt_path) as f:
        lines = f.readlines()

    for line in lines[2:]:
        tokens = line.split(";")
        name = tokens[0]
        theta = float(tokens[1])
        phi = float(tokens[2])
        Ly = math.sin(phi)
        Lx = math.cos(phi) * math.cos(theta)
        Lz = math.cos(phi) * math.sin(theta)
        
        gt_light[name] = [Lx,Ly,Lz]

    return gt_light



# preprocess input image for network
# img: PIL image
def preprocess_input(img, img_size=None):

    pil_img = Image.fromarray(img)

    if img_size != None:
        pil_img = pil_img.resize(img_size[::-1]) # resize method need size as (Width, Height)

    x = np.array(pil_img, dtype = np.float32)
    
    # ignore alpha channel
    if(x.shape[-1] == 4):
        x = x[...,:-1]
    generator_preprocess = ImageDataGenerator(rescale=1./255)
    x = generator_preprocess.standardize(x)
    return x


def evaluate(model, data_dir, img_size, gt_light_path, save_path = None):

    # read groundtruth light angles
    gt_light = read_gt_light(gt_light_path)

    # dictionary img_name -> [gt_angle_rad, model_angle_rad]
    predictions = {}

    cont = 0

    pixel_factor = 50

    if save_path and not os.path.exists(save_path):
        os.makedirs(save_path)

    regex_img = r".*\.(png|jpg)$"
    list_imgs = [f for f in os.listdir(data_dir) if re.search(regex_img,f)]

    print("found {} images".format(len(list_imgs)))

    for i, img_name in enumerate(list_imgs):

        print("processing img {} of {}".format(i+1, len(list_imgs)))

        #read image
        img = np.array(Image.open(os.path.join(data_dir, img_name)))
        
        # preprocess
        x = preprocess_input(img, img_size = img_size)

        # predict
        out = model.predict(np.expand_dims(x, axis = 0))
        L_hat = out[0][0]
        
        # groundtruth
        L = gt_light[img_name]

        predictions[img_name] = [L, L_hat] # per passsare da array di un elemento ad elemneto semplice

        if save_path:
            start = (int(img.shape[1] / 2), int(img.shape[0] / 2))
            end = (start[0] + int(pixel_factor * L[0]), start[1] - int(pixel_factor * L[1]))
            end_pred = (start[0] + int(pixel_factor * L_hat[0]), start[1] - int(pixel_factor * L_hat[1]))

            cv2.arrowedLine(img, start, end, (0, 255, 0), thickness = 2)
            cv2.arrowedLine(img, start, end_pred, (255, 255, 0), thickness = 2)

            Image.fromarray(img).save(os.path.join(save_path, img_name))
        
        cont+=1

    return predictions


parser = argparse.ArgumentParser(description = "Script for training and testing 3d outdoor light direction estimation based on physical illumination model and deep learning")
                   
parser.add_argument("-d", "--dataset_folder", type=str, help="base dataset folder (must contain train, test, val folders).", required=True)
parser.add_argument("-p", "--pretrained_weights", type=str, help="path to pretrained weights file", required=False, default=None)
parser.add_argument("-s", "--save_path", type=str, help="path where the image predictions will be saved", required=False, default=None)
args = parser.parse_args()


# parse arguments
dataset_path = args.dataset_folder
gt_path = os.path.join(dataset_path, "light.csv")
pretrained_weights = args.pretrained_weights
save_path = args.save_path


test_dir = os.path.join(dataset_path, "test")


if save_path is not None and not os.path.exists(save_path):
    os.makedirs(save_path)

model = get_fusion_net(pretrained_weights_file = pretrained_weights)

model_output_names = ['L', 'L_img']
losses = {model_output_names[0]: 'mean_absolute_error', model_output_names[1]: 'mean_squared_error'}

# define weight for each loss
lossWeights = {model_output_names[0]: 1.0, model_output_names[1]: 1.0}

# compile model
model.compile(loss=losses, loss_weights=lossWeights, optimizer='adam', metrics=["accuracy"])

img_size = (144,256)
predictions = evaluate(model, test_dir, img_size, gt_path, save_path=save_path)


gt_light_vectors = np.array([predictions[img_name][0] for img_name in predictions.keys()])
pred_light_vectors = np.array([predictions[img_name][1] for img_name in predictions.keys()])

print(pred_light_vectors.shape)
print(gt_light_vectors.shape)

angular_errors = []
for i in range(len(pred_light_vectors)):
    norm_prod = np.linalg.norm(pred_light_vectors[i]) * np.linalg.norm(gt_light_vectors[i])
    alpha = math.acos(np.dot(pred_light_vectors[i], gt_light_vectors[i]) / norm_prod)
    angular_errors.append(alpha)
angular_errors = np.array(angular_errors) * 180.0 / math.pi
print(angular_errors.shape)


plot_dir = os.path.join(save_path, "plots")
if not os.path.exists(plot_dir):
    os.makedirs(plot_dir)

plt.title("angular error distribution")
values, bins = np.histogram(angular_errors, bins=180, range = [0, 180])
plt.plot(bins[:-1],values)
plt.savefig(os.path.join(plot_dir,"angular_error_distribution.jpg"), bbox_inches="tight", dpi=600)
# plt.show()
plt.close()

# cumulative distribution
values, bins = np.histogram(angular_errors, bins=180, range = [0, 180])
cumul = np.cumsum(values)
tot_sum = np.sum(cumul) / (len(angular_errors) * 180.0)
print(tot_sum)

plt.title("angular error distribution cumulative, norm AUC: {:6.3f}".format(tot_sum))

plt.plot(bins[:-1], cumul, label="cumulative of prediction error distribution")
plt.plot(bins[:-1], np.linspace(0, len(angular_errors), num=len(bins[:-1])), '--', label="cumulative of uniform distribution")
plt.legend()

plt.savefig(os.path.join(plot_dir,"angular_error_distribution_cumulative.jpg"), bbox_inches="tight", dpi=600)
# plt.show()
plt.close()


