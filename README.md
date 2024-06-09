# LightNormalsEstimation
An end-to-end deep learning approach for Outdoor 3d light direction estimation based the integration of a physical illumination model in the pipeline.


## How to run this code
1. Open a terminal (powershell on Windows or bash in Linux) and clone this repository:
    ```
    git clone https://github.com/marza1993/LightNormalsEstimation.git
    ```
2. Enter the repository's folder and Install virtualenv:
    ```
    cd LightNormalsEstimation
    pip install virtualenv
    ```
3. Create virtual environment
    ```
    virtualenv <venv-name>
    ```
4. Activate virtual environment
    ```
    .\<venv-name>\Scripts\activate
    ```
    on Windows or
    ```
    source <venv-name>/bin/activate
    ```
    on Linux

5. Install all required packages
    ```
    pip install -r requirements.txt
    ```
    **Note:** Check GPU drivers vs cuda capabilities to run this code (in particular the training scripts) with GPU. This project uses the 2.7.0 version of tensorflow, which is compatible with Cuda 11.2 and newer versions.
    
    As an alternative, a suitable docker image with tensorflow and cuda support can be installed and the virtual environment with the required packages can be run inside the generated container (see [https://www.tensorflow.org/install/docker?hl=it](https://www.tensorflow.org/install/docker?hl=it) and [https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorflow](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tensorflow))
6. To **train** the deep learning model, run the training script as follows:
    ```
    python .\code\train_light_normals_fusion.py -d <dataset_path> -m <models_dir_save_path> -b <batch_size> -e <n_epochs> -f <portion_of_dataset_to_use>
    ```
    the dataset path must point to a folder with 3 subfolders: *train*, *val* and *test* containing the images and a ground truth file with name *light.csv* with the 3d light direction information for each image.

    An example dataset that can be used to train our end-to-end deep learning approach is **SynthOutdoor**, a synthetic dataset that we created for training outdoor light estimation models (such as the approach described in this project), which is available [here](https://www.scidb.cn/en/detail?dataSetId=304a5d88dba04226957b6215c171c0c2). Also, the interested reader can re-generate it (or variations of it) by running our [code](https://github.com/marza1993/SynthOutdoor).

7. To run our trained model to perform inference on test images:
     ```
    python .\code\test_light_normals_fusion.py -d <dataset_path> -p <pretrained_weights_path> -s <output_images_save_path>
    ```
Different jupyter notebooks (`light_normals_fusion.ipynb`, `light_normals_fusion_fine_tuning.ipynb`, etc.) with both training and testing can be interactively run in a jupyter environment or in visual studio code.



## Acknowledgments
This code can be used for reprudcing the results contained in our work accepted (not yet published) on the [CBMI 2024 conference](https://cbmi2024.org/). 
If the reader wants to use this code in his/her research or projects an acknowledgment to our paper must be included as follows:

```
@article{LightNormalsEstimation,
  title={<insert>},
  author={<insert>},
  conference={},
  volume={},
  number={},
  pages={},
  year={2014},
  publisher={Taylor \& Francis}
}
```

