import os
import re
import math
import random

dataset_path = "C:\\Users\\mar-z\\progetti\\data\\VIDIT\\"


def create_gt():
    global dataset_path
    regex_img = r".*\.(png|jpg)$"
    list_imgs = [f for f in os.listdir(dataset_path) if re.search(regex_img,f)]

    cardinal_to_theta_conversion = {"E": 0, 
                                   "NE": math.pi/4, 
                                   "N": math.pi/2, 
                                   "NW": 3.0/4.0 * math.pi, 
                                   "W": math.pi, 
                                   "SW": 5.0/4.0 * math.pi, 
                                   "S": 3.0/2.0 * math.pi, 
                                   "SE": 7.0/4.0 * math.pi
                                   }


    gt_file = dataset_path + "light.csv"
    with open(gt_file, 'w') as f:

        f.write("\"sep=;\"\n")
        f.write("img_name;theta;phi\n")

        # generate groundtruth file
        for i,img_name in enumerate(list_imgs):
            #print(img_name)
            print("image {} of {}".format(i+1,len(list_imgs)))
            tokens = img_name.split("_")
            temp_color = tokens[1]
            direction = tokens[2].split(".")[0]

            theta = cardinal_to_theta_conversion[direction]
            phi = math.pi/4

            f.write(img_name + ";" + str(theta) + ";" + str(phi) + "\n")
    print("done.")
    

def split():

    global dataset_path

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

    random.seed(100)
    random.shuffle(list_imgs)

    splits = [int(0.7 * len(list_imgs)), int(0.8*len(list_imgs))]

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


#create_gt()
split()
