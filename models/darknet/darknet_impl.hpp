/**
 * @file darknet_impl.hpp
 * @author Kartik Dutt
 *
 * Implementation of LeNet using mlpack.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MODELS_DARKNET_IMPL_HPP
#define MODELS_DARKNET_IMPL_HPP

#include "darknet.hpp"

namespace mlpack {
namespace ann {

template<
     typename OutputLayerType,
     typename InitializationRuleType,
     size_t DarkNetVer
>
DarkNet<OutputLayerType, InitializationRuleType, DarkNetVer>::DarkNet() :
    inputChannel(0),
    inputWidth(0),
    inputHeight(0),
    numClasses(0),
    weights("none")
{
  // Nothing to do here.
}

template<
     typename OutputLayerType,
     typename InitializationRuleType,
     size_t DarkNetVer
>
DarkNet<OutputLayerType, InitializationRuleType, DarkNetVer>::DarkNet(
  const size_t inputChannel,
  const size_t inputWidth,
  const size_t inputHeight,
  const size_t numClasses,
  const std::string& weights,
  const bool includeTop) :
  DarkNet<OutputLayerType, InitializationRuleType, DarkNetVer>(
    std::tuple<size_t, size_t, size_t>(
      inputChannel,
      inputWidth,
      inputHeight),
      numClasses,
      weights,
      includeTop)
{
  // Nothing to do here.
}

template<
     typename OutputLayerType,
     typename InitializationRuleType,
     size_t DarkNetVer
>
DarkNet<OutputLayerType, InitializationRuleType, DarkNetVer>::DarkNet(
    const std::tuple<size_t, size_t, size_t> inputShape,
    const size_t numClasses,
    const std::string& weights,
    const bool includeTop) :
    inputChannel(std::get<0>(inputShape)),
    inputWidth(std::get<1>(inputShape)),
    inputHeight(std::get<2>(inputShape)),
    numClasses(numClasses),
    weights(weights)
{
  mlpack::Log::Assert(DarkNetVer == 19 || DarkNetVer == 53,
      "Incorrect DarkNet version. Possible values are 19 and 53. Trying \
      to find version : " + std::to_string(DarkNetVer) + ".");

  if (weights == "cifar10")
  {
    // Download weights here.
    LoadModel("./../weights/darknet/darknet" + std::to_string(DarkNetVer) +
        "_cifar10.bin");
    return;
  }
  else if (weights != "none")
  {
    LoadModel(weights);
    return;
  }

  if (DarkNetVer == 19)
  {
    darkNet.Add<IdentityLayer<>>();

    // Convolution and activation function in a block.
    ConvolutionBlock(inputChannel, 32, 3, 3, 1, 1, 1, 1);
    PoolingBlock(2, 2, 2, 2);
    ConvolutionBlock(32, 64, 3, 3, 1, 1, 1, 1);
    PoolingBlock(2, 2, 2, 2);
    DarkNet19SequentialBlock(64, 3, 3, 1, 1);
    PoolingBlock(2, 2, 2, 2);
    DarkNet19SequentialBlock(128, 3, 3, 1, 1);
    PoolingBlock(2, 2, 2, 2);
    DarkNet19SequentialBlock(256, 3, 3, 1, 1);
    ConvolutionBlock(512, 256, 1, 1, 1, 1);
    ConvolutionBlock(256, 512, 3, 3, 1, 1, 1, 1);
    PoolingBlock(2, 2, 2, 2);
    DarkNet19SequentialBlock(512, 3, 3, 1, 1);
    ConvolutionBlock(1024, 512, 1, 1, 1, 1);
    ConvolutionBlock(512, 1024, 3, 3, 1, 1, 1, 1);
    ConvolutionBlock(1024, 1000, 1, 1, 1, 1);
    PoolingBlock(inputWidth, inputHeight, inputWidth,
        inputHeight, "mean");

    if (includeTop)
    {
      darkNet.Add<Linear<>>(1000, numClasses);
      darkNet.Add<LogSoftMax<>>();
    }

    darkNet.ResetParameters();
  }
  else if (DarkNetVer == 53)
  {
    darkNet.Add<IdentityLayer<>>();
    ConvolutionBlock(inputChannel, 32, 3, 3, 1, 1, 1, 1, true);
    ConvolutionBlock(inputChannel, 64, 3, 3, 2, 2, 1, 1, true);
    // Let's automate this a bit.
    size_t curChannels = 64;

    // Residual block configuration for DarkNet 53.
    std::vector<size_t> residualBlockConfig = {1, 2, 8, 8, 4};
    for (size_t blockCount : residualBlockConfig)
    {
      for (size_t i = 0; i < blockCount; i++)
      {
        DarkNet53ResidualBlock(curChannels);
      }

      ConvolutionBlock(curChannels, curChannels * 2, 3, 3,
          2, 2, 1, 1, true);
      curChannels = curChannels * 2;
    }

    darkNet.Add<Linear<>>(inputWidth * inputHeight * curChannels, numClasses);
    if (includeTop)
    {
      //darkNet.Add<Linear<>>(1000, numClasses);
      darkNet.Add<LogSoftMax<>>();
    }

    darkNet.ResetParameters();
  }
}

template<
     typename OutputLayerType,
     typename InitializationRuleType,
     size_t DarkNetVer
>
void DarkNet<
    OutputLayerType, InitializationRuleType, DarkNetVer
>::LoadModel(const std::string& filePath)
{
  data::Load(filePath, "DarkNet" + std::to_string(DarkNetVer), darkNet);
  Log::Info << "Loaded model" << std::endl;
}

template<
     typename OutputLayerType,
     typename InitializationRuleType,
     size_t DarkNetVer
>
void DarkNet<
    OutputLayerType, InitializationRuleType, DarkNetVer
>::SaveModel(const std::string& filePath)
{
  Log::Info<< "Saving model." << std::endl;
  data::Save(filePath, "DarkNet" + std::to_string(DarkNetVer), darkNet);
  Log::Info << "Model saved in " << filePath << "." << std::endl;
}

} // namespace ann
} // namespace mlpack

#endif
