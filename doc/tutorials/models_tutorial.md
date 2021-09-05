models repository provide easy to use state of the art models with pre-trained weights. The pre-trained weights can be used for transfer learning or inference. The model can also be trained from scratch very easily.

### Contents

 1. [Properties of Models](#1-properties-of-models)
 2. [Object Classification Models](#2-object-classification-models)

### 1. Properties of Models

**i. Including Models**

Each model can be included as follows : 

```cpp
#include <models/ModelClassName/ModelClassName.hpp>
```

**ii. Template Parameters**

Each model accepts at least the following parameters.

```
OutputLayerType The output layer type used to evaluate the network.
InitializationRuleType Rule used to initialize the weight matrix.
```

Refer to default value of each model. There might be additional template parameters for some models.

**iii. Functions of Model**
Each model in the models repository has the following functions:

**1. GetModel()**

This is member function that returns the model. The object is returned by reference and hence can be used for training / inference.

**Usage:** 

```
FFN<> model = modelObject.GetModel();
```

**2. LoadModel(std::string filePath)**

**Parameters:**

```
filePath : Path to determine the model the file where the model will be loaded from.
```

Model will be loaded from the specified file.

**3. SaveModel(std::string filePath)**

**Parameters:**

```
filePath : Path to determine the model the file where the model will be saved.
```

Model will be saved to the specified file.

### 2. Object Classification Models

List of supported Object classification models is given below.

|  **Model** | **Usage** | **Available Weights** | **Paper** |
| --- | --- | --- | --- |
|  DarkNet&nbsp;19 | DarkNet<CrossEntropyError<>, RandomInitialization, 19>&nbsp;darknet19({imageChannel, imageWidth, imageHeight}, numClasses)| ImageNet |[YOLO9000](https://pjreddie.com/media/files/papers/YOLO9000.pdf)|
|  DarkNet&nbsp;53 | DarkNet<CrossEntropyError<>, RandomInitialization, 53>&nbsp;darknet53({imageChannel, imageWidth, imageHeight}, numClasses)| ImageNet |[YOLOv3](https://pjreddie.com/media/files/papers/YOLOv3.pdf)|
|  ResNet18 | ResNet<CrossEntropyError<>, RandomInitialization, 18> resnet18(imageChannel, imageWidth, imageHeight, includeTop, preTrained, numClasses) | ImageNet | [Deep Residual Learning](https://arxiv.org/pdf/1512.03385)|
|  ResNet34 | ResNet<CrossEntropyError<>, RandomInitialization, 34> resnet34(imageChannel, imageWidth, imageHeight, includeTop, preTrained, numClasses) | ImageNet | [Deep Residual Learning](https://arxiv.org/pdf/1512.03385)|
|  ResNet50 | ResNet<CrossEntropyError<>, RandomInitialization, 50> resnet50(imageChannel, imageWidth, imageHeight, includeTop, preTrained, numClasses) | ImageNet | [Deep Residual Learning](https://arxiv.org/pdf/1512.03385)|
|  ResNet101 | ResNet<CrossEntropyError<>, RandomInitialization, 101> resnet101(imageChannel, imageWidth, imageHeight, includeTop, preTrained, numClasses) | ImageNet | [Deep Residual Learning](https://arxiv.org/pdf/1512.03385)|
|  ResNet152 | ResNet<CrossEntropyError<>, RandomInitialization, 152> resnet152(imageChannel, imageWidth, imageHeight, includeTop, preTrained, numClasses) | ImageNet | [Deep Residual Learning](https://arxiv.org/pdf/1512.03385)|
|  MobileNetV1 | MobilenetV1 mobilenetv1(imageChannel, imageWidth, imageHeight, alpha, depthMultiplier, includeTop, preTrained, numClasses) | ImageNet | [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/pdf/1704.04861)|

### i. DarkNet Family

**Including DarkNet Models**

The models can be included using : 

```cpp
#include <models/darknet/darknet.hpp>
```

**Template Parameters**

```
OutputLayerType The output layer type used to evaluate the network. Defaults to CrossEntropyError.
InitializationRuleType Rule used to initialize the weight matrix. Defaults to RandomInitialization.
DaknetVersion Version of DarkNet. Defaults to version 19. Possible values are 19 and 53.
```
   
**Constructor Parameters**

Darknet supports two constructors that accept image dimensions as separate constructor parameters or as a tuple.

Parameters of the first constructor are given below:

```
inputShape : A three-valued tuple indicating input shape. First value is number of channels (channels-First). Second value is input height.
             Third value corresponds to the input width.
numClasses : Optional number of classes to classify images into, only to be specified if includeTop is  true.
weights : One of 'none', 'imagenet'(pre-training on ImageNet) or path to weights.
```

Parameters of the second constructor are given below:

```
inputChannels : Number of input channels of the input image.
inputWidth : Width of the input image.
inputHeight : Height of the input image.
numClasses : Optional number of classes to classify images into, only to be specified if includeTop is  true.
weights : One of 'none', 'imagenet'(pre-training on ImageNet) or path to weights.
includeTop : Must be set to true if weights are set.
```
