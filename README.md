The mlpack **models** repository provides **ready-to-use** implementations of popular
and cutting-edge machine learning models---mostly deep learning models.  The
implementations in this repository are intended to be compiled into command-line
programs and bindings to Python and other languages.

_(This README contains various TODO comments like this one , so if you are
helping with the transition from the `examples/` repository, be sure to look for
comments like this one.  Once the transition is done, we can remove this
comment (and the others).)_

_(If we have functionality to download datasets and also to download
pretrained model weights, we should put a comment about that here in the main
description of the repository.)_

_(If this repository gets set up as a submodule to the main mlpack repository
and that is how everything in it should be compiled, then we should point that
out here!)_

### 0. Contents

  1. [Introduction](#1-introduction)
  2. [Dependencies](#2-dependencies)
  3. [Building-From-Source](#3-building-from-source)
  5. [Running Models](#5-running-models)
  5. [Current Models](#5-current-models)
  6. [Datasets](#6-datasets)

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

### 4. Using Dataloaders.

This repository provides dataloaders and data preprocessing modules for mlpack library.
It also provides utility function required required for downloading, extracting and processing
image, text and sequential data. For more information about dataloaders and utility functions,
Refer to our wiki page.

#### 1. Dataloaders for popular datasets.

Creating and processing data can be done in just a single line. Don't have the dataset downloaded,
No worries, we will download and preprocess it for you. Kindly refer to sample code given below.

```
const string datasetName = "mnist";
bool shuffleData = true;
double ratioForTrainTestSplit = 0.75;

// Create the DataLoader object.
DataLoader<> dataloader(datasetName, shuffleData, ratioForTrainTestSplit);
```

To train or test your model with our dataloaders is very simple.
```
// Use the dataloader for training.
 model.Train(dataloader.TrainFeatures(), dataloader.TrainLabels());
 
 // Use the dataloader for prediction.
 model.Predict(dataloader.TestFeatures(), dataloader.TestLabels());
```

Currently supported datasets are mentioned below :
##### 1. MNIST Dataset

#### 2. Loading Other Datasets.

We are continously adding new datasets to this repository, However you can also
use our dataloaders to load other datasets. Refer to our dataloaders wiki for more
information.

```
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

### 5. Running Models

_(This section needs significant overhaul once we clean up our build system.)_

### 6. Current Models

_(This section also needs some cleanup once we know what we're keeping and what
we're not keeping.)_

Currently model-zoo project has the following models implemented:

  - Simple Convolutional Neural Network on MNIST dataset.
  - Multivariate Time Series prediction using LSTM on Google Stock Prices.
  - Univariate Time Series prediction using LSTM on Electricity Consumption Dataset.
  - Variational Auto-Encoder on MNIST dataset.
  - Variational Convolutional Auto-Encoder on MNIST.

### 7. Datasets

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
