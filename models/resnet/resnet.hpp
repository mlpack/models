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

namespace mlpack {
namespace models {

template<
  typename OutputLayerType = ann::CrossEntropyError<>,
  typename InitializationRuleType = ann::RandomInitialization,
  size_t ResNetVersion = 18
>
class ResNet{
 public:

  ResNet();

  ResNet(const size_t inputChannel,
         const size_t inputWidth,
         const size_t inputHeight,
         const bool includeTop = true,
         const bool preTrained = false,
         const size_t numClasses = 1000);

  ResNet(std::tuple<size_t, size_t, size_t> inputShape,
         const bool includeTop = true,
         const bool preTrained = false,
         const size_t numClasses = 1000);

  //! Get Layers of the model.
  ann::FFN<OutputLayerType, InitializationRuleType>& GetModel()
      { return resNet; }

  //! Load weights into the model.
  void LoadModel(const std::string& filePath);

  //! Save weights for the model.
  void SaveModel(const std::string& filepath);

 private:

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
    
    // Updating input dimesntions.
    inputWidth = ConvOutSize(inputWidth, kernelWidth, strideWidth, padW);
    inputHeight = ConvOutSize(inputHeight, kernelHeight, strideHeight, padH);

    mlpack::Log::Info << "Convolution: " << "(" << inSize << " " << outSize
        << " " << kernelWidth << " " << kernelHeight << " " << strideWidth <<
        " " << strideHeight << " " << padW << " " << padH << " " << inputWidth
        << " " << inputHeight << ")" << std::endl;
  }

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

      mlpack::Log::Info << "Convolution: " << "(" << inSize << " " << outSize
          << " " << kernelWidth << " " << kernelHeight << " " << strideWidth
          << " " << strideHeight << " " << padW << " " << padH << " " <<
          downSampleInputWidth << " " << downSampleInputHeight << ")"
          << std::endl;
    }
    else
    {
      baseLayer->Add(new ann::Convolution<>(inSize, outSize, kernelWidth,
          kernelHeight, strideWidth, strideHeight, padW, padH,
          inputWidth, inputHeight));

      // Updating input dimesntions.
      inputWidth = ConvOutSize(inputWidth, kernelWidth, strideWidth, padW);
      inputHeight = ConvOutSize(inputHeight, kernelHeight, strideHeight, padH);
      
      mlpack::Log::Info << "Convolution: " << "(" << inSize << " " << outSize
          << " " << kernelWidth << " " << kernelHeight << " " << strideWidth
          << " " << strideHeight << " " << padW << " " << padH << " " <<
          inputWidth << " " << inputHeight << ")" << std::endl;
    }
  }

  template <typename AddmergeType = ann::AddMerge<>>
  void DownSample(AddmergeType* resBlock,
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

    downSampleBlock->Add(new ann::BatchNorm<>(outSize));
    mlpack::Log::Info << "BatchNorm: " << "(" <<outSize << ")" << std::endl;
    resBlock->Add(downSampleBlock);
  }

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
    sequentialBlock->Add(new ann::BatchNorm<>(outSize));
    mlpack::Log::Info << "BatchNorm: " << "(" << outSize << ")" << std::endl;
    sequentialBlock->Add(new ann::ReLULayer<>);
    mlpack::Log::Info << "Relu" << std::endl;
    ConvolutionBlock3x3(sequentialBlock, outSize, outSize);
    sequentialBlock->Add(new ann::BatchNorm<>(outSize));
    mlpack::Log::Info << "BatchNorm: " << "(" << outSize << ")" << std::endl;

    resBlock->Add(sequentialBlock);

    if (downSample == true)
    {  
      mlpack::Log::Info << "DownSample below" << std::endl;
      DownSample(resBlock, inSize, outSize, downSampleInputWidth,
          downSampleInputHeight);
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
    sequentialBlock->Add(new ann::BatchNorm<>(width));
    mlpack::Log::Info << "BatchNorm: " << "(" << width << ")" << std::endl;
    sequentialBlock->Add(new ann::ReLULayer<>);
    mlpack::Log::Info << "Relu" << std::endl;
    ConvolutionBlock3x3(sequentialBlock, width, width, strideWidth,
        strideHeight);
    sequentialBlock->Add(new ann::BatchNorm<>(width));
    mlpack::Log::Info << "BatchNorm: " << "(" << outSize << ")" << std::endl;
    sequentialBlock->Add(new ann::ReLULayer<>);
    mlpack::Log::Info << "Relu" << std::endl;
    ConvolutionBlock1x1(sequentialBlock, width, outSize * bottleNeckExpansion);
    sequentialBlock->Add(new ann::BatchNorm<>(outSize * bottleNeckExpansion));
    mlpack::Log::Info << "BatchNorm: " << "(" << outSize * bottleNeckExpansion
        << ")" << std::endl;

    resBlock->Add(sequentialBlock);

    if (downSample == true)
    {  
      mlpack::Log::Info << "DownSample below" << std::endl;
      DownSample(resBlock, inSize, outSize * bottleNeckExpansion,
          downSampleInputWidth, downSampleInputHeight, 1, 1, strideWidth,
          strideHeight);
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

  //! Locally stored vector to constructor different ResNet versions.
  std::vector<size_t> numBlockArray;

  //! Locally stored block string from which to build the model.
  std::string builderBlock;
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
