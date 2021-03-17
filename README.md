The mlpack **models** repository provides **ready-to-use** implementations of popular
and cutting-edge machine learning models---mostly deep learning models.  The
implementations in this repository are intended to be compiled into command-line
programs and bindings to Python and other languages.

_(This README contains various TODO comments like this one , so if you are
helping with the transition from the `examples/` repository, be sure to look for
comments like this one.  Once the transition is done, we can remove this
comment (and the others).)_

We provide ability to download datasets as well as pretrained weights using our
utility functions, by default we assume the server to be mlpack.org.
To download any file from mlpack.org simple use the following command.

NOTE: Our dataloader and models automatically download weights if necessary during
runtime.

```cpp
Utils::DownloadFile(url, downloadPath);
```

_(If this repository gets set up as a submodule to the main mlpack repository
and that is how everything in it should be compiled, then we should point that
out here!)_

### 0. Contents

  1. [Introduction](#1-introduction)
  2. [Dependencies](#2-dependencies)
  3. [Building-From-Source](#3-building-from-source)
  4. [Using Dataloaders](#4-using-dataloaders)
  5. [Using Augmentation](#5-using-augmentation)
  6. [Supported Models](#6-supported-models)
  7. [Datasets](#7-datasets)

###  1. Introduction

This repository contains a number of different models implemented in C++ using
mlpack. To understand more about mlpack, refer to the [mlpack
repository](https://github.com/mlpack/mlpack/) or the [mlpack
website](https://www.mlpack.org/).

In order to compile and build the programs in this repository, you'll need to
make sure that you have the same dependencies available that mlpack requires, in
addition to mlpack itself.

_(If this should only be built as a submodule, we should probably remove this
part about dependencies and instruct users to build this as a submodule of the
main mlpack repository.)_

      mlpack
      Armadillo      >= 8.400.0
      Boost (program_options, math_c99, unit_test_framework, serialization,
             spirit) >= 1.58
      CMake          >= 3.3.2
      ensmallen      >= 2.10.0

To install mlpack refer to the [installation
guide](https://www.mlpack.org/docs.html) that's available in the mlpack
documentation.

All of those dependencies should be available in your distribution's package
manager. If not, you will have to compile each of them by hand. See the
documentation for each of those packages for more information.

### 3. Building from source

To install this project run the following command.

  `mkdir build && cd build && cmake ../`

Use the optional command `-D DEBUG=ON ` to enable debugging.

Once CMake is configured, compile:

  `make`

You can also build with multiple cores using the `-j` option.  For example,
building with 4 cores can be done with the following command:

  `make -j4`

### 4. Using Dataloaders

This repository provides dataloaders and data preprocessing modules for mlpack library.
It also provides utility function required required for downloading, extracting and processing
image, text and sequential data. For more information about dataloaders and utility functions,
Refer to our wiki page.

#### 1. Dataloaders for popular datasets.

Creating and processing data can be done in just a single line. Don't have the dataset downloaded,
No worries, we will download and preprocess it for you. Kindly refer to sample code given below.

```cpp
const string datasetName = "mnist";
bool shuffleData = true;
double ratioForTrainTestSplit = 0.75;

// Create the DataLoader object.
DataLoader<> dataloader(datasetName, shuffleData, ratioForTrainTestSplit);
```

To train or test your model with our dataloaders is very simple.
```cpp
// Use the dataloader for training.
model.Train(dataloader.TrainFeatures(), dataloader.TrainLabels());
 
// Use the dataloader for prediction.
model.Predict(dataloader.TestFeatures(), dataloader.TestLabels());
```

Currently supported datasets are mentioned below :
|  **Dataset** | **Usage** | **Details** |
| --- | --- | --- |
|  MNIST | DataLoader<>&nbsp;("mnist"); | MNIST dataset is the de facto “hello world” dataset of computer vision.<br/> Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. The first column, called "label", is the digit that was drawn by the user. The rest of the columns contain the pixel-values of the associated image.|
|  Pascal VOC Detection | DataLoader<mat, field<vec>>&nbsp;("voc-detection") | The Pascal VOC challenge is a very popular dataset for building and evaluating algorithms for image classification, object detection and segmentation.<br/> VOC detection dataset provides support for loading object detection dataset in PASCAL VOC. Note : By default we refer to VOC - 2012 dataset as VOC dataset.|
| CIFAR 10 | DataLoader<>&nbsp;("cifar10"); | The CIFAR-10 dataset consists of 60000 32x32 color images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.|

#### 2. Loading Other Datasets.

We are continuously adding new datasets to this repository, However you can also
use our dataloaders to load other datasets. Refer to our dataloaders wiki for more
information.

##### a. Loading CSV Datasets.
Use our `LoadCSV` function to load and process CSV datasets.

```cpp
DataLoader<> irisDataloader;

const string datasetPath = "mnist";
bool shuffleData = true;
double ratioForTrainTestSplit = 0.75;
bool isTrainingData = true;
bool useFeatureScaling = true;
bool dropHeader = false;

// Starting column index for Training Features.
size_t startInputFeatures = 0;
// Ending column index for training Features.
// We also support wrapped index i.e. -1 implies last column and so on.
size_t endInputFeatures = -2;

irisDataloader(datasetPath, isTrainingData, shuffleData, ratioForTrainTestSplit,
    useFeatureScaling, dropHeader, startInputFeatures, endInputFeatures);
```

##### b. Loading Image Dataset.

Use our `LoadImageDatasetFromDirectory` to load image dataset in given directory. Directory should contain folders with folder name as class label and each folder should contain images corresponding to the class name.

```cpp
DataLoader<> dataloader;
dataloader.LoadImageDatasetFromDirectory("path/to/directory", imageWidth, imageHeight, imageDepth);
```

For advanced usage, refer to our wiki page.

##### c. Loading Object Detection Dataset.

We provide support to load annotations represented in XML files and their corresponding images. If your dataset contains fixed number of objects in each annotation use matrix type to load your dataset else use field type for labels / annotations. If images are not of same size pass a vector containing resize parameter. By default, each image is resized to 64 x 64.

 ```cpp
 DataLoader<> dataloader;
 vector<string> classes = {"class-name-0", "class-name-1", "class-name-2"}
 dataloader.LoadObjectDetectionDataset("path/to/annotations/", "path/to/images/", classes);
 ```

#### 3. Preprocessing.

For all datasets that we support we provide, We preprocess them internally. We also
provide access to preprocessor functions for standard datasets in case one needs to
apply them to their datasets.

They can simply be called as follows by calling static functions of ProProcess class i.e.
PreProcess::SupportedDatasetName

```cpp
PreProcess<>::MNIST(dataloader.TrainFeatures(), dataloader.TrainLabels(),
    dataloader.ValidFeatures(), dataloader.ValidLabels(), dataloader.TestFeatures());
```

This is especially useful when preprocessing of your dataset resembles any other standard
dataset that we support.
### 5. Using Augmentation

To prevent overfitting on training data, we provide support for native augmentation. The constructor takes in a list / vector of strings which contain supported augmentation. Augmentation can be applied to the dataset by calling the `Transform` function. For more information about augmentation, take a look at our wiki page.

```cpp
Augmentation augmentation({"horizontal-flip", "resize : (64, 64)"}, 0.2);
augmentation.Transform(dataset, imageWidth, imageHeight, imageDepth);
```

### 6. Supported Models

Currently model-zoo project has the following models implemented:
|  **Model** | **Usage** | **Available Weights** | **Paper** |
| --- | --- | --- | --- |
|  Darknet&nbsp;19 | DarkNet<NegativeLogLikelihood<>, HeInitialization, 19>&nbsp;darknet19({imageDepth, imageWidth, imageHeight}, numClasses)| ImageNet |[YOLO9000](https://pjreddie.com/media/files/papers/YOLO9000.pdf)|
|  Darknet&nbsp;53 | DarkNet<NegativeLogLikelihood<>, HeInitialization, 53>&nbsp;darknet19({imageDepth, imageWidth, imageHeight}, numClasses)| ImageNet |[YOLOv3](https://pjreddie.com/media/files/papers/YOLOv3.pdf)|

All models can be included as shown below :
```cpp
#include <models/Model-ClassName/Model_ClassName.hpp>
```

For more information about usage, take a look at our wiki page.
