/**
 * @file resnet.hpp
 * @author Aakash Kaushik
 * 
 * Definition of ResNet models.
 * 
 * For more information, kindly refer to the following paper.
 * 
 * Paper for ResNet.
 *
 * @code
 * @article{Kaiming He2015,
 *  author = {Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun},
 *  title = {Deep Residual Learning for Image Recognition},
 *  year = {2015},
 *  url = {https://arxiv.org/pdf/1512.03385.pdf}
 * }
 * @endcode
 * 
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MODELS_MODELS_RESNET_RESNET_HPP
#define MODELS_MODELS_RESNET_RESNET_HPP

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
 * Definition of a ResNet CNN.
 * 
 * @tparam OutputLayerType The output layer type used to evaluate the network.
 * @tparam InitializationRuleType Rule used to initialize the weight matrix.
 * @tparam ResNetVersion Version of ResNet.
 */
template<
  typename OutputLayerType = ann::CrossEntropyError<>,
  typename InitializationRuleType = ann::RandomInitialization,
  size_t ResNetVersion = 18
>
class ResNet{
 public:
  //! Create the ResNet model.
  ResNet();

  /**
   * ResNet constructor intializes input shape and number of classes.
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
  ResNet(const size_t inputChannel,
         const size_t inputWidth,
         const size_t inputHeight,
         const bool includeTop = true,
         const bool preTrained = false,
         const size_t numClasses = 1000);

  /**
   * ResNet constructor intializes input shape and number of classes.
   *
   * @param inputShape A three-valued tuple indicating input shape.
   *     First value is number of channels (channels-first).
   *     Second value is input height. Third value is input width.
   * @param preTrained True for pre-trained weights of ImageNet,
   *    default is false.
   * @param numClasses Optional number of classes to classify images into,
   *     only to be specified if includeTop is  true.
   */
  ResNet(std::tuple<size_t, size_t, size_t> inputShape,
         const bool includeTop = true,
         const bool preTrained = false,
         const size_t numClasses = 1000);

  //! Get Layers of the model.
  ann::FFN<OutputLayerType, InitializationRuleType>& GetModel()
  {
    return resNet;
  }

  //! Load weights into the model and assumes the internal matrix to be
  //  named "ResNet"
  void LoadModel(const std::string& filePath);

  //! Save weights for the model and assumes the internal matrix to be
  //  named "ResNet"
  void SaveModel(const std::string& filepath);

 private:
  /**
   * Adds a 3x3 Convolution Block.
   *
   * @tparam SequentialType Layer type in which 3x3 convolution block will
   *     be added.
   *
   * @param inSize Number of input maps.
   * @param outSize Number of output maps.
   * @param strideWidth Stride of filter application in the x direction.
   * @param strideHeight Stride of filter application in the y direction.
   * @param kernelWidth Width of the filter/kernel.
   * @param kernelHeight Height of the filter/kernel.
   * @param padW Padding width of the input.
   * @param padH Padding height of the input.
   */
  template<typename SequentialType = ann::Sequential<>>
  void ConvolutionBlock3x3(SequentialType* baseLayer,
                           const size_t inSize,
                           const size_t outSize,
                           const size_t strideWidth = 1,
                           const size_t strideHeight = 1,
                           const size_t kernelWidth = 3,
                           const size_t kernelHeight = 3,
                           const size_t padW = 1,
                           const size_t padH = 1)
  {
    baseLayer->Add(new ann::Convolution<>(inSize, outSize, kernelWidth,
        kernelHeight, strideWidth, strideHeight, padW, padH, inputWidth,
        inputHeight));

    mlpack::Log::Info << "Convolution: " << "(" << inSize << ", " << inputWidth
        << ", " << inputHeight << ")" << " ---> (";

    // Updating input dimesntions.
    inputWidth = ConvOutSize(inputWidth, kernelWidth, strideWidth, padW);
    inputHeight = ConvOutSize(inputHeight, kernelHeight, strideHeight, padH);

    mlpack::Log::Info << outSize << ", " << inputWidth << ", " << inputHeight
        << ")" << std::endl;
  }

  /**
   * Adds a 1x1 Convolution Block.
   *
   * @tparam SequentialType Layer type in which 1x1 convolution block will
   *     be added.
   *
   * @param inSize Number of input maps.
   * @param outSize Number of output maps.
   * @param downSampleInputWidth Input widht for downSample block.
   * @param downSampleInputHeight Input height for downSample block.
   * @param strideWidth Stride of filter application in the x direction.
   * @param strideHeight Stride of filter application in the y direction.
   * @param kernelWidth Width of the filter/kernel.
   * @param kernelHeight Height of the filter/kernel.
   * @param padW Padding width of the input.
   * @param padH Padding height of the input.
   * @param downSample Bool if it's a downsample block or not. default is false.
   */
  template<typename SequentialType = ann::Sequential<>>
  void ConvolutionBlock1x1(SequentialType* baseLayer,
                           const size_t inSize,
                           const size_t outSize,
                           const size_t downSampleInputWidth = 0,
                           const size_t downSampleInputHeight = 0,
                           const size_t strideWidth = 1,
                           const size_t strideHeight = 1,
                           const size_t kernelWidth = 1,
                           const size_t kernelHeight = 1,
                           const size_t padW = 0,
                           const size_t padH = 0,
                           const bool downSample = false)
  {
    if (downSample)
    {
      baseLayer->Add(new ann::Convolution<>(inSize, outSize, kernelWidth,
          kernelHeight, strideWidth, strideHeight, padW, padH,
          downSampleInputWidth, downSampleInputHeight));

      mlpack::Log::Info << "  Convolution: " << "(" << inSize << ", " <<
          downSampleInputWidth << ", " << downSampleInputHeight << ")" <<
          " ---> (" << outSize << ", " << downSampleInputWidth << ", " <<
          downSampleInputHeight << ")" << std::endl;
    }
    else
    {
      baseLayer->Add(new ann::Convolution<>(inSize, outSize, kernelWidth,
          kernelHeight, strideWidth, strideHeight, padW, padH,
          inputWidth, inputHeight));

      mlpack::Log::Info << "Convolution: " << "(" << inSize << ", " <<
          inputWidth << ", " << inputHeight << ")" << " ---> (";

      // Updating input dimesntions.
      inputWidth = ConvOutSize(inputWidth, kernelWidth, strideWidth, padW);
      inputHeight = ConvOutSize(inputHeight, kernelHeight, strideHeight, padH);

      mlpack::Log::Info << outSize << ", " << inputWidth << ", " << inputHeight
          << ")" << std::endl;
    }
  }

  /**
   * Adds 1x1 Convolution Block and a batch norm layer to constrcut downSample
   *     block.
   *
   * @tparam AddMergeType Layer type in which downSample block will
   *     be added.
   *
   * @param inSize Number of input maps.
   * @param outSize Number of output maps.
   * @param downSampleInputWidth Input widht for down-sample block.
   * @param downSampleInputHeight Input height for down-sample block.
   * @param kernelWidth Width of the filter/kernel.
   * @param kernelHeight Height of the filter/kernel.
   * @param strideWidth Stride of filter application in the x direction.
   * @param strideHeight Stride of filter application in the y direction.
   * @param padW Padding width of the input.
   * @param padH Padding height of the input.
   */
  template <typename AddMergeType = ann::AddMerge<>>
  void DownSample(AddMergeType* resBlock,
                  const size_t inSize,
                  const size_t outSize,
                  const size_t downSampleInputWidth,
                  const size_t downSampleInputHeight,
                  const size_t kernelWidth = 1,
                  const size_t kernelHeight = 1,
                  const size_t strideWidth = 2,
                  const size_t strideHeight = 2,
                  const size_t padW = 0,
                  const size_t padH = 0)
  {
    ann::Sequential<>* downSampleBlock = new ann::Sequential<>();
    ConvolutionBlock1x1(downSampleBlock, inSize, outSize, downSampleInputWidth,
        downSampleInputHeight, strideWidth, strideHeight, kernelWidth,
        kernelHeight, padW, padH, true);

    downSampleBlock->Add(new ann::BatchNorm<>(outSize, 1e-5));
    mlpack::Log::Info << "  BatchNorm: " << "(" << outSize << ")" << " ---> ("
        << outSize << ")" << std::endl;
    resBlock->Add(downSampleBlock);
  }

  /**
   * Adds basicBlock block for ResNet 18 and 34.
   *
   * It's represented as:
   * 
   * resBlock - AddMerge layer
   * {
   *   sequentialBlock - sequentialLayer
   *   {
   *     ConvolutionBlock3x3(inSize, outSize, strideWidth, strideHeight)
   *     BatchNorm(outSize, 1e-5)
   *     ReLU
   *     ConvolutionBlock3x3(inSize, outSize)
   *     BatchNorm(outSize, 1e-5)
   *   }
   *
   *   sequentialLayer
   *   {
   *     if downsample == true
   *       ConvolutionBlock1x1(inSize, outSize, downSampleInputWidth,
   *           downSampleInputHeight)
   *       BatchNorm(outSize, 1e-5)
   *
   *     else
   *       IdentityLayer
   *   }
   * 
   *   ReLU
   * }
   *
   * @param inSize Number of input maps.
   * @param outSize Number of output maps.
   * @param strideWidth Stride of filter application in the x direction.
   * @param strideHeight Stride of filter application in the y direction.
   * @param downSample If there will be a downSample block or not, default
   *     false.
   */
  void BasicBlock(const size_t inSize,
                  const size_t outSize,
                  const size_t strideWidth = 1,
                  const size_t strideHeight = 1,
                  const bool downSample = false)
  {
    downSampleInputWidth = inputWidth;
    downSampleInputHeight = inputHeight;

    ann::Sequential<>* basicBlock = new ann::Sequential<>();
    ann::AddMerge<>* resBlock = new ann::AddMerge<>(true, true);
    ann::Sequential<>* sequentialBlock = new ann::Sequential<>();
    ConvolutionBlock3x3(sequentialBlock, inSize, outSize, strideWidth,
        strideHeight);
    sequentialBlock->Add(new ann::BatchNorm<>(outSize, 1e-5));
    mlpack::Log::Info << "BatchNorm: " << "(" << outSize << ")" << " ---> ("
        << outSize << ")" << std::endl;
    sequentialBlock->Add(new ann::ReLULayer<>);
    mlpack::Log::Info << "Relu" << std::endl;
    ConvolutionBlock3x3(sequentialBlock, outSize, outSize);
    sequentialBlock->Add(new ann::BatchNorm<>(outSize, 1e-5));
    mlpack::Log::Info << "BatchNorm: " << "(" << outSize << ")" << " ---> ("
        << outSize << ")" << std::endl;

    resBlock->Add(sequentialBlock);

    if (downSample == true)
    {
      mlpack::Log::Info << "DownSample (" << std::endl;
      DownSample(resBlock, inSize, outSize, downSampleInputWidth,
          downSampleInputHeight);
      mlpack::Log::Info << ")" <<std::endl;
    }
    else
    {
      mlpack::Log::Info << "IdentityLayer" << std::endl;
      resBlock->Add(new ann::IdentityLayer<>);
    }

    basicBlock->Add(resBlock);
    basicBlock->Add(new ann::ReLULayer<>);
    mlpack::Log::Info << "Relu" << std::endl;
    resNet.Add(basicBlock);
  }

  /**
   * Adds bottleNeck block for ResNet 50, 101 and 152.
   *
   * It's represented as:
   * 
   * resBlock - AddMerge layer
   * {
   *   sequentialBlock
   *   {
   *     ConvolutionBlock1x1(inSize, width)
   *     BatchNorm(width, 1e-5)
   *     ReLU
   *     ConvolutionBlock3x3(width, width, strideWidth, strideHeight)
   *     BatchNorm(width, 1e-5)
   *     ReLU
   *     ConvolutionBlock1x1(width, outSize * bottleNeckExpansion)
   *     BatchNorm(outSize * bottleNeckExpansion, 1e-5)
   *   }
   *
   *   sequentialLayer
   *   {
   *     if downsample == true
   *       ConvolutionBlock1x1(inSize, outSize * bottleNeckExpansion,
   *           downSampleInputWidth, downSampleInputHeight, 1, 1, strideWidth,
   *           strideHeight)
   *       BatchNorm(outSize, 1e-5)
   *
   *     else
   *       IdentityLayer
   *   }
   * 
   *   ReLU
   * }
   *
   * @param inSize Number of input maps.
   * @param outSize Number of output maps.
   * @param strideWidth Stride of filter application in the x direction.
   * @param strideHeight Stride of filter application in the y direction.
   * @param downSample If there will be a downSample block or not, default
   *     false.
   * @param baseWidth Parameter for calculating width.
   * @param groups Parameter for calculating width.
   */
  void BottleNeck(const size_t inSize,
                  const size_t outSize,
                  const size_t strideWidth = 1,
                  const size_t strideHeight = 1,
                  const bool downSample = false,
                  const size_t baseWidth = 64,
                  const size_t groups = 1)
  {
    downSampleInputWidth = inputWidth;
    downSampleInputHeight = inputHeight;

    size_t width = int((baseWidth / 64.0) * outSize) * groups;
    ann::Sequential<>* basicBlock = new ann::Sequential<>();
    ann::AddMerge<>* resBlock = new ann::AddMerge<>(true, true);
    ann::Sequential<>* sequentialBlock = new ann::Sequential<>();
    ConvolutionBlock1x1(sequentialBlock, inSize, width);
    sequentialBlock->Add(new ann::BatchNorm<>(width, 1e-5));
    mlpack::Log::Info << "BatchNorm: " << "(" << width << ")" << " ---> ("
        << width << ")" << std::endl;
    sequentialBlock->Add(new ann::ReLULayer<>);
    mlpack::Log::Info << "Relu" << std::endl;
    ConvolutionBlock3x3(sequentialBlock, width, width, strideWidth,
        strideHeight);
    sequentialBlock->Add(new ann::BatchNorm<>(width, 1e-5));
    mlpack::Log::Info << "BatchNorm: " << "(" << width << ")" << " ---> ("
        << width << ")" << std::endl;
    sequentialBlock->Add(new ann::ReLULayer<>);
    mlpack::Log::Info << "Relu" << std::endl;
    ConvolutionBlock1x1(sequentialBlock, width, outSize * bottleNeckExpansion);
    sequentialBlock->Add(new ann::BatchNorm<>(outSize * bottleNeckExpansion,
        1e-5));
    mlpack::Log::Info << "BatchNorm: " << "(" << outSize * bottleNeckExpansion
    << ")" << " ---> (" << outSize * bottleNeckExpansion << ")" << std::endl;

    resBlock->Add(sequentialBlock);

    if (downSample == true)
    {
      mlpack::Log::Info << "DownSample (" << std::endl;
      DownSample(resBlock, inSize, outSize * bottleNeckExpansion,
          downSampleInputWidth, downSampleInputHeight, 1, 1, strideWidth,
          strideHeight);
      mlpack::Log::Info << ")" << std::endl;
    }
    else
    {
      mlpack::Log::Info << "IdentityLayer" << std::endl;
      resBlock->Add(new ann::IdentityLayer<>);
    }

    basicBlock->Add(resBlock);
    basicBlock->Add(new ann::ReLULayer<>);
    mlpack::Log::Info << "Relu" << std::endl;
    resNet.Add(basicBlock);
  }

  /**
   * Creates model layers based on the type of layer and parameters supplied.
   *
   * @param block Type of block to use for layer creation.
   * @param outSize Number of output maps.
   * @param numBlocks Number of layers to create.
   * @param stride Single parameter for StrideHeight and strideWidth.
   */
  void MakeLayer(const std::string& block,
                 const size_t outSize,
                 const size_t numBlocks,
                 const size_t stride = 1)
  {
    bool downSample = false;

    if (block == "basicblock")
    {
      if (stride != 1 || downSampleInSize != outSize * basicBlockExpansion)
        downSample = true;
      BasicBlock(downSampleInSize, outSize * basicBlockExpansion, stride,
          stride, downSample);
      downSampleInSize = outSize * basicBlockExpansion;
      for (size_t i = 1; i != numBlocks; ++i)
        BasicBlock(downSampleInSize, outSize);
    }

    else if (block == "bottleneck")
    {
      if (stride != 1 || downSampleInSize != outSize * bottleNeckExpansion)
        downSample = true;
      BottleNeck(downSampleInSize, outSize, stride, stride, downSample);
      downSampleInSize = outSize * bottleNeckExpansion;
      for (size_t i = 1; i != numBlocks; ++i)
        BottleNeck(downSampleInSize, outSize);
    }
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
  ann::FFN<OutputLayerType, InitializationRuleType> resNet;

  //! Locally stored number of channels in the image.
  size_t inputChannel;

  //! Locally stored width of the image.
  size_t inputWidth;

  //! Locally stored height of the image.
  size_t inputHeight;

  //! Locally stored number of output classes.
  size_t numClasses;

  //! Locally stored width of image for downSample block.
  size_t downSampleInputWidth;

  //! Locally stored height of image for downSample block.
  size_t downSampleInputHeight;

  //! Locally stored expansion for BasicBlock.
  size_t basicBlockExpansion = 1;

  //! Locally stored expansion for BottleNeck.
  size_t bottleNeckExpansion = 4;

  //! InSize for ResNet block creation.
  size_t downSampleInSize = 64;

  //! Locally stored map to constructor different ResNet versions.
  std::map<size_t, std::map<std::string, std::array<size_t, 4>>> ResNetConfig =
      {
        {18, {{"basicblock", {2, 2, 2, 2}}}},
        {34, {{"basicblock", {3, 4, 6, 3}}}},
        {50, {{"bottleneck", {3, 4, 6, 3}}}},
        {101, {{"bottleneck", {3, 4, 23, 3}}}},
        {152, {{"bottleneck", {3, 8, 36, 3}}}}
      };

  //! Locally stored array to constructor different ResNet versions.
  std::array<size_t , 4> numBlockArray;

  //! Locally stored block string from which to build the model.
  std::string builderBlock;

  //! Locally stored path string for pretrained model.
  std::string preTrainedPath;  
}; // ResNet class

// convenience typedefs for different ResNet models.
typedef ResNet<ann::CrossEntropyError<>, ann::RandomInitialization, 18>
    ResNet18;
typedef ResNet<ann::CrossEntropyError<>, ann::RandomInitialization, 34>
    ResNet34;
typedef ResNet<ann::CrossEntropyError<>, ann::RandomInitialization, 50>
    ResNet50;
typedef ResNet<ann::CrossEntropyError<>, ann::RandomInitialization, 101>
    ResNet101;
typedef ResNet<ann::CrossEntropyError<>, ann::RandomInitialization, 152>
    ResNet152;

} // namespace models
} // namespace mlpack

#include "resnet_impl.hpp"

#endif
