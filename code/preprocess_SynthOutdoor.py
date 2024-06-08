import os
import re
import random


dataset_path = "C:/Users/mar-z/progetti/data/SynthOutdoor/images"
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
gt_path = os.path.join(dataset_path, "light.csv")
with open(gt_path) as f:
    lines = f.readlines()

lines = [line.replace(",", ".") for line in lines]

with open(gt_path, "w") as f:
    f.writelines(lines)

print("done.")