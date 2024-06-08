import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import load_img
import os
import re
import keras
import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image
import cv2
import math


class Data_loader(tf.keras.utils.Sequence):
    
    # img_size può essere None oppure (W,H),
    def __init__(self, batch_size, imgs_path, gt_path, img_size = None, apply_augmentation = False, 
                 model_output_names = ['L', 'L_img']):

        self.batch_size = batch_size
        self.imgs_path = imgs_path
        self.gt_path = gt_path
        self.img_size = img_size
        self.model_output_names = model_output_names


        # se questo flag é false l'immagine viene solo riscalat
        self.apply_augmentation = apply_augmentation

        # carico la lista dei nomi delle immagini nel percorso passato
        regex_img = r".*\.(png|jpg)$"
        self.list_imgs = [f for f in os.listdir(imgs_path) if re.search(regex_img,f)]

        random.seed(100)
        random.shuffle(self.list_imgs)

        # leggo il file del groundtruth e creo una mappa: nome_immagine -> direzione luce
        with open(self.gt_path) as f:
            lines = f.readlines()
        
        self.gt = {}
        for line in lines[2:]:
            tokens = line.split(";")
            name = tokens[0]
            theta = float(tokens[1])
            phi = float(tokens[2])
            Ly = math.sin(phi)
            Lx = math.cos(phi) * math.cos(theta)
            Lz = math.cos(phi) * math.sin(theta)

            self.gt[name] = [Lx,Ly,Lz]
            

        print("n. imgs: {}".format(len(self.list_imgs)))


        # per la data augmentation
        self.img_augmenter = ImageDataGenerator(
                samplewise_center=True,
                samplewise_std_normalization=True,
                rescale=1./255,
                #shear_range=0.5,
                #zoom_range=0.1,
                #rotation_range=60,
                rotation_range=0,
                width_shift_range=0,
                height_shift_range=0,
                horizontal_flip=True,
                #vertical_flip=True,
                vertical_flip=False,
                fill_mode = 'wrap'
                )


        self.rescaler = ImageDataGenerator(
                rescale=1./255,
                #samplewise_center=True,
                
                #samplewise_std_normalization=True
                )

    
    def N_samples(self):
        return len(self.list_imgs)


    def shuffle(self):
        random.shuffle(self.list_imgs)

    def get_gt(self):
        return self.gt


    def __len__(self):
        return len(self.list_imgs) // self.batch_size

    def my_horiz_flip(self, img):
        return img[:,::-1,:]
    

    # ********************************************************************************************************
    # - se img_size � None e do_train_val � False il batch avr� dimensione: (batch_size, None), perch� ogni
    #      immagine avr� la sua dimensione
    # - se img_size � None e do_train_val � True allora viene effettuato un padding delle immagini in modo che il
    #      batch sia un tensore con una dimensione fissa: (batch_size, max_H, max_W, depth), dove max_H � il numero max
    #      di righe tra tutte le immagini del batch, analogo per max_W.
    # ********************************************************************************************************
    def get_batch(self, idx, img_size = None, augment = False, do_train_val = True):
        
        start = idx * self.batch_size
        end = np.min((len(self.list_imgs), start + self.batch_size))

        if start >= end:
            return None, None

        batch_img_list = self.list_imgs[start : end]
        
        x = []
        
        # two outputs: L -> light direction [1x3], L_img [WxHx1] -> Luminance component of image
        y1 = []
        y2 = []
        # dictionary that will contain the two outputs
        y = {}

        
        data_manipulator = self.img_augmenter if augment else self.rescaler
        #data_manipulator = self.rescaler
        
        for i in range(self.batch_size):
            # carico l'immagine
            orig_im = load_img(os.path.join(self.imgs_path, batch_img_list[i]), target_size=img_size, color_mode = 'rgb')

            hsv_im = np.array(orig_im.convert('HSV'), dtype = np.float32)

            # retrieve luminance image as one of the outputs -> V channel
            L_im = hsv_im[...,-1]

            x_i = np.array(orig_im, dtype = np.float32)# / 255.0
            
            # apply transormation to input
            random_seed = int(np.random.randint(0, 1000, 1))
            params_img_transf = data_manipulator.get_random_transform(x_i.shape, seed = random_seed)
            #print(params_img_transf['flip_horizontal'])
            #print("img trans params: {}".format(params_img_transf))
            x_i = data_manipulator.apply_transform(data_manipulator.standardize(x_i), params_img_transf)
            x.append(x_i)

            # apply transormation to output
            L_im = data_manipulator.apply_transform(data_manipulator.standardize(L_im), params_img_transf)
            y2.append(L_im)

            # load gt light direction
            Lx,Ly,Lz = self.gt[batch_img_list[i]]
            if params_img_transf['flip_horizontal']:
                Lx = -Lx
                print("{} is flipped".format(i))
                
            y1.append(np.array((Lx, Ly, Lz)))
            
        y[self.model_output_names[0]] = np.array(y1)
        y[self.model_output_names[1]] = np.array(y2)
        
        
        ## TODO
        #if do_train_val and self.img_size == None:
            
        #    # ottengo la risoluzione massima nel batch
        #    max_H = np.max([im.shape[0] for im in x[:]])
        #    max_W = np.max([im.shape[1] for im in x[:]])

        #    x_padded = np.zeros((self.batch_size,) + (max_H, max_W, 3), dtype="float32")
            
        #    # effettuo il padding delle immagini e del gt:
        #    for i in range(self.batch_size):
        #        x_padded[i, :x[i].shape[0], :x[i].shape[1], :] = x[i]
                
        #    return x_padded, y

        return np.array(x), y
         
        


    def __getitem__(self, idx):

        """Returns tuple (input_img, mask) correspond to batch #idx."""
        return self.get_batch(idx, img_size = self.img_size, augment = self.apply_augmentation)



    def visualize_batch(self, batch_index = -1, img_size = None, view_gt = True, do_augment = False):

        N_batches = self.__len__()

        if batch_index >= N_batches:
            print("indice batch oltre il limite di {}".format(N_batches))
            return

        if batch_index == -1:
            batch_index = int(np.random.randint(0, N_batches, 1))

        batch_img, batch_gt = self.get_batch(batch_index, img_size = img_size, augment = do_augment, do_train_val = False)
            
        print("batch_img.shape: {}".format(batch_img.shape))
        print("batch_gt[{}].shape: {}".format(self.model_output_names[0], batch_gt[self.model_output_names[0]].shape))
        print("batch_gt[{}].shape: {}".format(self.model_output_names[1], batch_gt[self.model_output_names[1]].shape))

        N_subplots = 2 if view_gt else 1
        pixel_factor = 100
        for i in range(0,len(batch_img)):

            vis_img = batch_img[i]
            Lx,Ly,Lz = batch_gt[self.model_output_names[0]][i]
            L_im = batch_gt[self.model_output_names[1]][i]
            
            print("Lx: {}, Ly: {}, Lz: {}".format(Lx,Ly,Lz))

            if view_gt:
                vis_img = np.copy(batch_img[i])

                # disegno il vettore della luce
                M, N = batch_img[i].shape[:-1]
                start = (int(N/2), int(M/2))
                end = (start[0] + int(pixel_factor * Lx), start[1] - int(pixel_factor * Ly))
                cv2.arrowedLine(vis_img, start, end, (0,255,0), thickness = 5, line_type = cv2.LINE_AA)

                f,arraxis = plt.subplots(1, N_subplots, figsize=(10,10))
                arraxis[0].imshow(vis_img)
                arraxis[0].title.set_text('RGB input')
                
                arraxis[1].imshow(L_im)
                arraxis[1].title.set_text('Brightness image')
            else:
                
                plt.imshow(vis_img)
            plt.show()
            
            
            

