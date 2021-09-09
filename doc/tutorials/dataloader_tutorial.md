Models provide an easy to use data loader to load popular datasets in just a single line of code. Maybe you want to use it some other dataset and you can do that too.

**Template Parameters:**

Dataloader requires defaults to using arma::mat for training features and prediction however those of who want to play around with other armadillo types can simply pass template parameters to change them.

```
DatasetX : Datatype for loading input features.
DatasetY : Datatype for prediction features.
ScalerType : mlpack's Scaler Object for scaling features.
```

### Loading Popular datasets

Use our constructor to pass the required information about the dataset that you want to load and we will download it, extract it and process it so that it's ready to use. For supported datasets take a look at this [list](#4-supported-datasets).

**Constructor Parameters**

For simply trying out our dataloader without any hassle, just pass the dataset name and whether or not to shuffle data and you are done.
For example, Loading mnist dataset is very simple.

**Simple Usage**

```
Dataloader<> dataloader("mnist", true);
```

This will fill TrainFeatures, TrainLabels, ValidationFeatures, ValidationLabels and TestFeatures for the dataloader. We will discuss them in detail below.

Advanced parameters: 
Currently we are working on providing support for augmentation support and we will update the tutorial with the same.


```
datasetPath : Path or name of dataset.
shuffle : Whether or not to shuffle the data.
ratio : Ratio for train-test split. Defaults to 0.75.
useScaler : Use feature scaler for pre-processing the dataset. Defaults to false.
augmentation : Adds augmentation to training data only. Defaults to an empty vector.
augmentationProbability : Probability of applying augmentation on dataset. Defaults to 0.2.
```

**Advanced Usage**

With the help of the above parameters you can use features such as scaling and augmentation to make your model robust. A sample usage is shown below.

Note: Augmentation class is under development.

```cpp
Dataloader<arma::mat, arma::mat, mlpack::data::MinMaxScaler> dataloader("Pascal-VOC-detection",
    true, 0.7, true, {"horizontal-flip", "vertical-flip"}, 0.2);
```

Refer to [accessor methods](#3-Accessor-Methods-Using-DataLoader-object-for-training-and-inference) in data loader to understand how to use data loader for training and testing. 

### Loading Other datasets

You can use our data loaders to load any type of dataset you want. We are currently developing Image data loaders to get images path from either CSVs or directories. Till then, we only support CSV datasets as part of our data loader.

Note : We support wrapped indices in our data loader i.e. using index such as -1 implies last column / row and so on.

**Downloading datasets**

In case your dataset is hosted on a server somewhere, you can use our utility functions to download it.

```cpp
Utils::DownloadFile("path-in-mlpack-server", "path-where-to-save-the-dataset")
```

For more details on how to use it to download files from other servers refer to our Utils tutorial wiki page.

**Usage**

Use the default constructor to create the data loader object. Then use one of our data loader methods to load the data.

**Load CSV Method**

This method can be used to load CSVs and preprocess the loaded data.

**Load CSV Usage**

You can simply load a CSV, scale it, perform train-test split and split the data into input features and output labels.

```
datasetPath : Path to the dataset.
loadTrainData : Boolean to determine whether data will be stored for
                training or testing. If true, data will be loaded for training.
                Note: This option augmentation to NULL, set ratio to 1 and
                scaler will be used to only transform the test data.
shuffle : Boolean to determine whether or not to shuffle the data.
ratio : Ratio for train-test split.
useScaler : Fits the scaler on training data and transforms dataset.
dropHeader : Drops the first row from CSV.
startInputFeatures : First Index which will be fed into the model as input.
endInputFeature : Last Index which will be fed into the model as input.
startPredictionFeatures : First Index which be predicted by the model as output.
endPredictionFeatures : Last Index which be predicted by the model as output.
augmentation : Vector strings of augmentations supported by mlpack.
augmentationProbability : Probability of applying augmentation to a particular cell.
```

An example of code is given below : 

```cpp
DataLoader<> irisDataloader;

std::string datasetPath = "./iris.csv";
// Starting column index for Training Features.
size_t startInputFeatures = 0;
// Ending column index for training Features.
size_t endInputFeatures = -2;
// Prediction columns. 
size_t startInputLabels = -1;

irisDataloader(datasetPath, isTrainingData, shuffleData, ratioForTrainTestSplit,
    useFeatureScaling, dropHeader, startInputFeatures, endInputFeatures, startInputLabels);
```

**Load Image Dataset**

Use our `LoadImageDatasetFromDirectory` to load image dataset in given directory. Directory should contain folders with folder name as class label and each folder should contain images corresponding to the class name. A sample directory structure is given below.

```
-- Directory
   -- class-name-1
       -- image1.jpg
       -- image2.jpg
   -- class-name-2
       -- image1.jpg
       -- image2.jpg
```

**Simple Usage of Load Image Dataset**

```
pathToDataset Path to all folders containing all images.
imageWidth Width of images in dataset.
imageHeight Height of images in dataset.
imageDepth Depth of images in dataset.
trainData Determines whether data is training set or test set.
shuffle Boolean to determine whether or not to shuffle the data.
validRatio Ratio of dataset to be used for validation set.
augmentation Vector strings of augmentations supported by mlpack.
augmentationProbability Probability of applying augmentation to a particular image.
```

A sample code snippet is given below.

```cpp
DataLoader<> dataloader;
std::string pathToDataset = "./path/to/dataset";
size_t imageWidth = 32, imageHeight = 32, imageDepth = 3;
dataloader.LoadImageDatasetFromDirectory(pathToDataset, imageWidth, imageHeight, imageDepth);
```

**Advanced Usage**

Use parameters such as augmentation and validRatio to increase robustness of model and create validation dataset.

```cpp
DataLoader<> dataloader;
std::string pathToDataset = "./path/to/dataset";
bool trainData = true;
double validRatio = 0.2;
std::vector<std::string> augmentation = {"resize 64", "horizontal-flip"};   
size_t imageWidth = 32, imageHeight = 32, imageDepth = 3;
dataloader.LoadImageDatasetFromDirectory(pathToDataset, imageWidth, imageHeight, imageDepth,
    trainData, validRatio, augmentation);
```

**Load Object Detection Dataset**

We provide support to load annotations represented in XML files and their corresponding images. If your dataset contains fixed number of objects in each annotation use matrix type to load your dataset else use field type for labels / annotations. If images are not of same size pass a vector containing resize parameter. By default, each image is resized to 64 x 64. Each XML file should correspond to a single image in images folder. XML file should containg the following :

1. Each XML file should be wrapped in XML-annotation tag.
2. Filename of image in images folder will be depicted by XML-filename tag.
3. XML-Object tag depicting characteristics of bounding box.
4. Each object tag should contain name tag i.e. class of the object.
5. Each object tag should contain bndbox tag containing xmin, ymin, xmax, ymax.

NOTE : Labels are assigned using classes vector. Set verbose to 1 to print labels and their corresponding class. The labels type should be field type here.

```
pathToAnnotations Path to the folder containing XML type annotation files.
pathToImages Path to folder containing images corresponding to annotations.
classes Vector of strings containing list of classes. Labels are assigned according to this vector.
validRatio Ratio of dataset that will be used for validation.
shuffle Boolean to determine whether the dataset is shuffled.
augmentation Vector strings of augmentations supported by mlpack.
augmentationProbability Probability of applying augmentation to a particular cell.
absolutePath Boolean to determine if absolute path is used. Defaults to false.
baseXMLTag XML tag name which wraps around the annotation file.
imageNameXMLTag XML tag name which holds the value of image filename.
objectXMLTag XML tag name which holds details of bounding box i.e. class and coordinates of bounding box.
bndboxXMLTag XML tag name which holds coordinates of bounding box.
classNameXMLTag XML tag name inside objectXMLTag which holds the name of the class of bounding box.
x1XMLTag XML tag name inside bndboxXMLTag which hold value of lower most x coordinate of bounding box.
y1XMLTag XML tag name inside bndboxXMLTag which hold value of lower most y coordinate of bounding box.
x2XMLTag XML tag name inside bndboxXMLTag which hold value of upper most x coordinate of bounding box.
y2XMLTag XML tag name inside bndboxXMLTag which hold value of upper most y coordinate of bounding box.
```

**Simple Usage**

```cpp
DataLoader<> dataloader;
vector<string> classes = {"class-name-0", "class-name-1", "class-name-2"};
dataloader.LoadObjectDetectionDataset("path/to/annotations/", "path/to/images/", classes);
```

**Advanced Usage of Object Detection Dataloader**

Use XML-Tag parameters to specify tags that the dataloader should look for. Also use parameters like augmentation to make model robust.

```cpp
DataLoader<> dataloader;
// Class names in augmentations.
vector<string> classes = {"class-name-0", "class-name-1", "class-name-2"};
// Percentage of data to be used for validation dataset.
double validRatio = 0.2;
// Transforms that will be applied to the dataset.
std::vector<std::string> augmentation = {"resize 64", "horizontal-flip"};
double augmentationProbability = 0.2;

// Lets assume annotation files are wrapped around XML_Dataset.
std::string baseXMLTag = "XML_Dataset";

dataloader.LoadObjectDetectionDataset("path/to/annotations/", "path/to/images/", classes,
    validRatio, augmentation, augmentationProbability, false, baseXMLTag);
```


Refer to [accessor methods](#3-Accessor-Methods-Using-DataLoader-object-for-training-and-inference) in data loader to understand how to use data loader for training and testing. 


### Accessor Methods : Using DataLoader object for training and inference

We provide access to loaded data using accessor and modifiers functions. This will allow you to perform extra pre-processing on dataset if you want. Details about the data loader members are given below.

```
TrainFeatures() : Returns input features to be used by model during training.
TrainLabels() :  Returns ground truth for training input features.

TestFeatures() : Returns input features to be used by model during testing.
TestLabels() : Return predictions made by model for test input features. Initially empty.

ValidFeatures() : Returns input features to be used by model during validation.
ValidLabels() : Returns ground truth for validation input features.

TrainSet() : Returns a tuple containing both TrainFeatures and TrainLabels.

ValidSet() : Returns a tuple containing both ValidFeatures and ValidLabels.

TestSet() : Returns a tuple containing both TestFeatures and TestLabels.
```

### Supported Datasets

Currently supported datasets are mentioned below :

|  **Dataset** | **Usage** | **Details** |
| --- | --- | --- |
|  MNIST | DataLoader<>&nbsp;("mnist"); | MNIST dataset is the de facto “hello world” dataset of computer vision.<br/> Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. The first column, called "label", is the digit that was drawn by the user. The rest of the columns contain the pixel-values of the associated image.|
|  Pascal VOC Detection | DataLoader<mat, field<vec>>&nbsp;("voc-detection") | The Pascal VOC challenge is a very popular dataset for building and evaluating algorithms for image classification, object detection and segmentation.<br/> VOC detection dataset provides support for loading object detection dataset in PASCAL VOC. Note : By default we refer to VOC - 2012 dataset as VOC dataset.|
| CIFAR 10 | DataLoader<>&nbsp;("cifar10"); | The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.|

We are an open source organization and we really appreciate it if you take the time to add any popular dataset in the dataloader or you can open an issue and someone will get to it.
