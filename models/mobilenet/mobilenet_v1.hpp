/**
 * @file mobilenet_v1.hpp
 * @author Aakash Kaushik
 * 
 * Definition of MobileNet V1 model.
 * 
 * For more information, kindly refer to the following paper.
 *
 * @code
 * @article{Andrew G2017,
 *  author = {Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko,
 *      Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam},
 *  title = {MobileNets: Efficient Convolutional Neural Networks for Mobile
 *      Vision Applications},
 *  year = {2017},
 *  url = {https://arxiv.org/pdf/1704.04861}
 * }
 * @endcode
 * 
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MODELS_MODELS_MOBILENET_MOBILENET_V1_HPP
#define MODELS_MODELS_MOBILENET_MOBILENET_V1_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/layer/layer_types.hpp>
#include <mlpack/methods/ann/init_rules/random_init.hpp>
#include <mlpack/methods/ann/loss_functions/binary_cross_entropy_loss.hpp>
#include <mlpack/methods/ann/init_rules/he_init.hpp>

#include "./../../utils/utils.hpp"

namespace mlpack {
namespace models {

/**
 * Definition of a MobileNet V1 CNN.
 * 
 * @tparam OutputLayerType The output layer type used to evaluate the network.
 * @tparam InitializationRuleType Rule used to initialize the weight matrix.
 */
template<
  typename OutputLayerType = ann::CrossEntropyError<>,
  typename InitializationRuleType = ann::RandomInitialization
>
class MobileNetV1{
 public:
  //! Create the MobileNet model.
  MobileNetV1();

  /**
   * MobileNetV1 constructor intializes input shape and number of classes.
   *
   * @param inputChannels Number of input channels of the input image.
   * @param inputWidth Width of the input image.
   * @param inputHeight Height of the input image.
   * @param includeTop Must be set to true if preTrained is set to true.
   * @param preTrained True for pre-trained weights of ImageNet,
   *    default is false.
   * @param numClasses Optional number of classes to classify images into,
   *     only to be specified if includeTop is true, default is 1000.
   */
  MobileNetV1(const size_t inputChannel,
         const size_t inputWidth,
         const size_t inputHeight,
         const bool includeTop = true,
         const bool preTrained = false,
         const size_t numClasses = 1000);

  /**
   * MobileNetV1 constructor intializes input shape and number of classes.
   *
   * @param inputShape A three-valued tuple indicating input shape.
   *     First value is number of channels (channels-first).
   *     Second value is input height. Third value is input width.
   * @param preTrained True for pre-trained weights of ImageNet,
   *    default is false.
   * @param numClasses Optional number of classes to classify images into,
   *     only to be specified if includeTop is  true.
   */
  MobileNetV1(std::tuple<size_t, size_t, size_t> inputShape,
         const bool includeTop = true,
         const bool preTrained = false,
         const size_t numClasses = 1000);

  //! Get Layers of the model.
  ann::FFN<OutputLayerType, InitializationRuleType>&
      GetModel() { return resNet; }

  //! Load weights into the model and assumes the internal matrix to be
  //  named "MobileNetV1".
  void LoadModel(const std::string& filePath);

  //! Save weights for the model and assumes the internal matrix to be
  //  named "MobileNetV1".
  void SaveModel(const std::string& filepath);

 private:
  /**
   * Adds a ReLU6 Layer.
   *
   * @param baseLayer Sequential layer type in which ReLU6 layer will be added if it's not NULL ot
   *     added to mobileNet.
   */
  void ReLU6Layer(ann::Sequential<>* baseLayer = NULL)
  {
    if (baseLayer != NULL)
    {
      baseLayer->Add(new ann::ReLU6<>);
      mlpack::Log::Info << "RelU6" << std::endl;
      return;
    }

    mobileNet.Add(new ann::ReLU6<>);
    mlpack::Log::Info << "RelU6" << std::endl;
  }

  /**
   * Adds DepthWiseConvBlock block.
   *
   * It's represented as:
   * 
   * @code
   * @endcode
   * 
   * @param inSize Number of input maps.
   * @param outSize Number of output maps.
   * @param strideWidth Stride of filter application in the x direction.
   * @param strideHeight Stride of filter application in the y direction.
   * @param downSample If there will be a downSample block or not, default
   *     false.
   */
  void DepthWiseConvBlock(const size_t inSize,
                          const size_t alpha,
                          const size_t depthMultiplier = 1,
                          const size_t stride = 1)
{
    size_t pointwiseOutSize = size_t(outSize * alpha);
    size_t depthMultipliedOutSize = size_t(inSize * depthMultiplier);
    ann::Sequential<>* sequentialBlock = new ann::Sequential<>();
    if (stride != 1)
    {
      sequentialBlock->Add(new ann::Padding<>(0, 1, 0, 1, inputWidth,
          inputHeight));
      mlpack::Log::Info << "Padding: " << "(" << inSize << ", " << inputWidth
          << ", " << inputWidth << " ---> (";
      inputWidth += 1;
      inputHeight += 1;
      mlpack::Log::Info << inSize << ", " << inputWidth << ", " << inputHeight
          << ")" << std::endl;
      paddingType = "valid"
    }

    sequentialBlock->Add(new SeparableConvolution<>(inSize,
        depthMultipliedOutSize, 3, 3, stride, stride, 0, 0, inputWidth,
        inputHeight, paddingType));
    mlpack::Log::Info << "Convolution: " << "(" << inSize << ", " <<
        inputWidth << ", " << inputHeight << ")" << " ---> (";

    if (paddingType == "valid")
    {
      inputWidth = ConvOutSize(inputWidth, 3, stride, 0);
      inputHeight = ConvOutSize(inputHeight, 3, stride, 0);
    }

    mlpack::Log::Info << depthMultipliedOutSize << ", " << inputWidth << ", "
        << inputHeight << ")" << std::endl;

    sequentialBlock->Add(new ann::BatchNorm<>(depthMultipliedOutSize, 1e-3,
        true, 0.99));
    mlpack::Log::Info << "BatchNorm: " << "(" << depthMultipliedOutSize << ")"
        << " ---> (" << depthMultipliedOutSize << ")" << std::endl;
    ReLU6Layer(sequentialBlock);
    sequentialBlock->Add(new ann::Convolution<>(depthMultipliedOutSize,
        pointwiseOutSize, 1, 1, 1, 1, 0, 0, inputWidth, inputHeight, "same"));
    mlpack::Log::Info << "Convolution: " << "(" << inSize << ", " << inputWidth
        << ", " << inputHeight << ")" << " ---> (" << depthMultipliedOutSize
        << ", " << inputWidth << ", " << inputHeight << ")" << std::endl;
    sequentialBlock->Add(new ann::BatchNorm<>(pointwiseOutSize, 1e-3, true,
        0.99));
    mlpack::Log::Info << "BatchNorm: " << "(" << pointwiseOutSize << ")"
        << " ---> (" << pointwiseOutSize << ")" << std::endl;
    ReLU6Layer(sequentialBlock);

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
    return std::floor((size - k + 2 * padding) / s) + 1;
  }

  //! Locally stored DarkNet Model.
  ann::FFN<OutputLayerType, InitializationRuleType> mobileNet;

  //! Locally stored number of channels in the image.
  size_t inputChannel;

  //! Locally stored width of the image.
  size_t inputWidth;

  //! Locally stored height of the image.
  size_t inputHeight;

  //! Locally stored number of output classes.
  size_t numClasses;

  //! InSize for ResNet block creation.
  size_t downSampleInSize = 64;

  //! Locally stored block string from which to build the model.
  std::string paddingType = "same";

  //! Locally stored path string for pre-trained model.
  std::string preTrainedPath;
}; // MobileNetV1 class

} // namespace models
} // namespace mlpack

#include "mobilenet_v1_impl.hpp"

#endif
