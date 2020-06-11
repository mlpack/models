The mlpack **models** repository provides **ready-to-use** implementations of popular
and cutting-edge machine learning models---mostly deep learning models.  The
implementations in this repository are intended to be compiled into command-line
programs and bindings to Python and other languages.

_(This README contains various TODO comments like this one , so if you are
helping with the transition from the `examples/` repository, be sure to look for
comments like this one.  Once the transition is done, we can remove this
comment (and the others).)_

We provide ability to download datasets as well as pretrained weights using our
utility functions, by default we asume the server to be mlpack.org.
To dowload any file from mlpack.org simple use the following command.

NOTE: Our dataloader and models automatically download weights if neccesary during
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
  6. [Running Models](#6-running-models)
  7. [Current Models](#7-current-models)
  8. [Datasets](#8-datasets)

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
| CIFAR 10 | DataLoader<>&nbsp;("cifar10"); | The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.|

#### 2. Loading Other Datasets.

We are continously adding new datasets to this repository, However you can also
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
provide access to preprocessor functions for standard datasets incase one needs to
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

### 6. Running Models

_(This section needs significant overhaul once we clean up our build system.)_

### 7. Current Models

_(This section also needs some cleanup once we know what we're keeping and what
we're not keeping.)_

Currently model-zoo project has the following models implemented:

  - Simple Convolutional Neural Network on MNIST dataset.
  - Multivariate Time Series prediction using LSTM on Google Stock Prices.
  - Univariate Time Series prediction using LSTM on Electricity Consumption Dataset.
  - Variational Auto-Encoder on MNIST dataset.
  - Variational Convolutional Auto-Encoder on MNIST.

### 8. Datasets

_(This section will also need to be overhauled, but we should wait until we
overhaul the sections above too.)_

Model-Zoo project has the following datasets available:

#### 1. MNIST

[MNIST](http://yann.lecun.com/exdb/mnist/)("Modified National Institute of Standards and Technology") is the de facto “hello world” dataset of computer vision. 
Since its release in 1999, this classic dataset of handwritten images has served as the basis for benchmarking classification 
algorithms. As new machine learning techniques emerge, MNIST remains a reliable resource for researchers and learners alike.

Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. Each pixel has a single pixel-
value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-
value is an integer between 0 and 255, inclusive.
The training data set, (train.csv), has 785 columns. The first column, called "label", is the digit that was drawn by the 
user. The rest of the columns contain the pixel-values of the associated image. For more information refer to this [MNIST Database](http://yann.lecun.com/exdb/mnist/).

#### 2. Google Stock-Prices Dataset

Google Stock-Prices Dataset consists of stock prices for each day from 27th June, 2016 to 27th June, 2019. Each tuple is 
seperated from its adjacent tuple by 1 day. It consists of following rows that indicate opening, closing, volume and high and 
low of stocks associated with Google on that day.

#### 3. Electricity Consumption Dataset

Contains electricity consumption of a city for 2011 to 2012, where each tuple is seperated from its adjacent tuple by 1 day.  
Each tuple has consumption in kWH and binary values for each Off-peak, Mid-peak, On-peak rows.
