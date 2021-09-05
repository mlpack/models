models provide easy to use augmentation techniques to prevent overfitting. Take a look at our tutorial below.

### Constructor Parameters

Augmentation class takes in two parameters that are mentioned below :

```
augmentation : List of strings containing one of the supported augmentations.
augmentationProbability : Probability of applying augmentation on the dataset.
```

Note : Augmentation probability is set to 1 for operations that change the shape or size of the object i.e.    
       Operations such as resize and reshape are applied to all images.

Take a look at our list of [supported augmentations](#supported-augmentations).

### Usage

Use the `Transform` function to apply augmentation to the dataset.

```
dataset : Dataset on which augmentation will be applied.
datapointWidth : Width of a single data point i.e. Since each column represents a separate data point.
datapointHeight : Height of a single data point.
datapointDepth : Depth of a single data point. For 2-dimensional data point, set it to 1. Defaults to 1.
```

An example code snippet is given below : 

```
// Resize image to 8 x 8 and apply horizontal flip to 20 % of images / data points.
std::vector<std::string> augmentationVector = {"horizontal-flip",
    "resize : 8"};
Augmentation augmentation(augmentationVector, 0.2);

// Transform function called.
augmentation.Transform(input, inputWidth, inputHeight, depth);
```

### Supported Augmentations

Currently we only support `resize` augmentation. There are many more augmentations that will be added over the next few months. We are an open source organization and we really appreciate it if you take the time to add any augmentation.

#### Usage of Resize Transform.

We use regex to parse the string and obtain desired width and desired height. If only a single number is found then desired width and desired height are set to the same number.
An example for square output,

```cpp
Augmentation augmentation({"resize : 8"});
```

The above object will transform each data point in the dataset to 8 x 8.

Another usage includes resizing the data point to a rectangular shape. Here we need to specify both the desired width and desired height in the same order.

```cpp
Augmentation augmentation({"resize : (8, 10)"});
```

The above object will transform each data point in the dataset to 8 x 10.
