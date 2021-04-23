/**
 * @file yolo.hpp
 * @author Kartik Dutt
 * 
 * Definition of Yolo models.
 * 
 * For more information, kindly refer to the following paper.
 * 
 * Paper for YOLOv1.
 *
 * @code
 * @article{Redmon2016,
 *  author = {Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi},
 *  title = {You Only Look Once : Unified, Real-Time Object Detection},
 *  year = {2016},
 *  url = {https://arxiv.org/pdf/1506.02640.pdf}
 * }
 * @endcode
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MODELS_MODELS_YOLO_YOLO_HPP
#define MODELS_MODELS_YOLO_YOLO_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/layer/layer_types.hpp>
#include <mlpack/methods/ann/init_rules/random_init.hpp>


namespace mlpack {
namespace ann /** Artificial Neural Network. */{

/**
 * Definition of a YOLO object detection models.
 * 
 * @tparam OutputLayerType The output layer type used to evaluate the network.
 * @tparam InitializationRuleType Rule used to initialize the weight matrix.
 * @tparam YOLOVersion Version of YOLO model.
 */
template<
  typename OutputLayerType = NegativeLogLikelihood<>,
  typename InitializationRuleType = RandomInitialization
>
class YOLO
{
 public:
  //! Create the YOLO model.
  YOLO();

  /**
   * YOLO constructor intializes input shape and number of classes.
   *
   * @param inputChannels Number of input channels of the input image.
   * @param inputWidth Width of the input image.
   * @param inputHeight Height of the input image.
   * @param yoloVersion Version of YOLO model.
   * @param numClasses Optional number of classes to classify images into,
   *     only to be specified if includeTop is  true.
   * @param numBoxes Number of bounding boxes per image.
   * @param featureSizeWidth Width of output feature map.
   * @param featureSizeHeight Height of output feature map.
   * @param weights One of 'none', 'voc'(pre-training on VOC-2012) or path to weights.
   * @param includeTop Must be set to true if weights are set.
   */
  YOLO(const size_t inputChannel,
       const size_t inputWidth,
       const size_t inputHeight,
       const std::string yoloVersion = "v1-tiny",
       const size_t numClasses = 20,
       const size_t numBoxes = 2,
       const size_t featureSizeWidth = 7,
       const size_t featureSizeHeight = 7,
       const std::string& weights = "none",
       const bool includeTop = true);

  /**
   * YOLO constructor intializes input shape and number of classes.
   *
   * @param inputShape A three-valued tuple indicating input shape.
   *     First value is number of Channels (Channels-First).
   *     Second value is input height. Third value is input width.
   * @param yoloVersion Version of YOLO model.
   * @param numClasses Optional number of classes to classify images into,
   *     only to be specified if includeTop is  true.
   * @param numBoxes Number of bounding boxes per image.
   * @param featureShape A twp-valued tuple indicating width and height of output feature
   *     map.
   * @param weights One of 'none', 'voc'(pre-training on VOC) or path to weights.
   */
  YOLO(const std::tuple<size_t, size_t, size_t> inputShape,
       const std::string yoloVersion = "v1-tiny",
       const size_t numClasses = 1000,
       const size_t numBoxes = 2,
       const std::tuple<size_t, size_t> featureShape = {7, 7},
       const std::string& weights = "none",
       const bool includeTop = true);

  //! Get Layers of the model.
  FFN<OutputLayerType, InitializationRuleType>& GetModel() { return yolo; }

  //! Load weights into the model.
  void LoadModel(const std::string& filePath);

  //! Save weights for the model.
  void SaveModel(const std::string& filePath);

 private:
  /**
   * Adds Convolution Block.
   *
   * @tparam SequentialType Layer type in which convolution block will
   *     be added.
   *
   * @param inSize Number of input maps.
   * @param outSize Number of output maps.
   * @param kernelWidth Width of the filter/kernel.
   * @param kernelHeight Height of the filter/kernel.
   * @param strideWidth Stride of filter application in the x direction.
   * @param strideHeight Stride of filter application in the y direction.
   * @param padW Padding width of the input.
   * @param padH Padding height of the input.
   * @param batchNorm Boolean to determine whether a batch normalization
   *     layer is added.
   * @param baseLayer Layer in which Convolution block will be added, if
   *     NULL added to YOLO FFN.
   */
  template<typename SequentialType = Sequential<>>
  void ConvolutionBlock(const size_t inSize,
                        const size_t outSize,
                        const size_t kernelWidth,
                        const size_t kernelHeight,
                        const size_t strideWidth = 1,
                        const size_t strideHeight = 1,
                        const size_t padW = 0,
                        const size_t padH = 0,
                        const bool batchNorm = false,
                        SequentialType* baseLayer = NULL)
  {
    Sequential<>* bottleNeck = new Sequential<>();
    bottleNeck->Add(new Convolution<>(inSize, outSize, kernelWidth,
        kernelHeight, strideWidth, strideHeight, padW, padH, inputWidth,
        inputHeight));

    mlpack::Log::Info << "Conv Layer.  ";
    mlpack::Log::Info << "(" << inputWidth << ", " << inputHeight <<
        ", " << inSize << ") ----> ";

    inputWidth = ConvOutSize(inputWidth, kernelWidth, strideWidth, padW);
    inputHeight = ConvOutSize(inputHeight, kernelHeight, strideHeight, padH);
    mlpack::Log::Info << "(" << inputWidth << ", " << inputHeight <<
        ", " << outSize << ")" << std::endl;

    if (batchNorm)
      bottleNeck->Add(new BatchNorm<>(outSize, 1e-8, false));

    bottleNeck->Add(new LeakyReLU<>(0.01));

    if (baseLayer != NULL)
      baseLayer->Add(bottleNeck);
    else
      yolo.Add(bottleNeck);
  }

  /**
   * Adds Pooling Block.
   *
   * @param factor The factor by which input dimensions will be divided.
   * @param type One of "max" or "mean". Determines whether add mean pooling
   *     layer or max pooling layer.
   */
  void PoolingBlock(const size_t factor = 2,
                    const std::string type = "max")
  {
    if (type == "max")
    {
      yolo.Add(new AdaptiveMaxPooling<>(std::ceil(inputWidth * 1.0 / factor),
          std::ceil(inputHeight * 1.0 / factor)));
    }
    else
    {
      yolo.Add(new AdaptiveMeanPooling<>(std::ceil(inputWidth * 1.0 /
          factor), std::ceil(inputHeight * 1.0 / factor)));
    }

    mlpack::Log::Info << "Pooling Layer.  ";
    mlpack::Log::Info << "(" << inputWidth << ", " << inputHeight <<
        ") ----> ";
    // Update inputWidth and inputHeight.
    inputWidth = std::ceil(inputWidth * 1.0 / factor);
    inputHeight = std::ceil(inputHeight * 1.0 / factor);

    mlpack::Log::Info << "(" << inputWidth << ", " << inputHeight << ")" << std::endl;
  }

  /**
   * Return the convolution output size.
   *
   * @param size The size of the input (row or column).
   * @param k The size of the filter (width or height).
   * @param s The stride size (x or y direction).
   * @param padding The size of the padding (width or height) on one side.
   * @return The convolution output size.
   */
  size_t ConvOutSize(const size_t size,
                     const size_t k,
                     const size_t s,
                     const size_t padding)
  {
    return std::floor(size + 2 * padding - k) / s + 1;
  }

  //! Locally stored YOLO Model.
  FFN<OutputLayerType, InitializationRuleType> yolo;

  //! Locally stored number of channels in the image.
  size_t inputChannel;

  //! Locally stored width of the image.
  size_t inputWidth;

  //! Locally stored height of the image.
  size_t inputHeight;

  //! Locally stored number of output classes.
  size_t numClasses;

  //! Locally stored number of output bounding boxes.
  size_t numBoxes;

  //! Locally stored width of output feature map.
  size_t featureWidth;

  //! Locally stored height of output feature map.
  size_t featureHeight;

  //! Locally stored type of pre-trained weights.
  std::string weights;

  //! Locally stored version of yolo model.
  std::string yoloVersion;
}; // YOLO class.

} // namespace ann
} // namespace mlpack

# include "yolo_impl.hpp"

#endif
