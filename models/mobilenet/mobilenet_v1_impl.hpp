/**
 * @file resnet_impl.hpp
 * @author Aakash Kaushik
 *
 * Implementation of ResNet using mlpack.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MODELS_MODELS_MOBILENET_MOBILENET_V1_IMPL_HPP
#define MODELS_MODELS_MOBILENET_MOBILENET_V1_IMPL_HPP

#include "mobilenet_v1.hpp"

namespace mlpack {
namespace models {

template<typename OutputLayerType, typename InitializationRuleType>
void MobileNetV1<OutputLayerType, InitializationRuleType>::MobileNetV1() :
    inputChannel(0),
    inputWidth(0),
    inputHeight(0),
    numClasses(0)
{
  // Nothing to do here.
}

template<typename OutputLayerType, typename InitializationRuleType>
void MobileNetV1<OutputLayerType, InitializationRuleType>::MobileNetV1(
    const size_t inputChannel,
    const size_t inputWidth,
    const size_t inputHeight,
    const bool includeTop,
    const bool preTrained,
    const size_t numClasses) :
    ResNet<OutputLayerType, InitializationRuleType, ResNetVersion>(
        std::tuple<size_t, size_t, size_t>(
        inputChannel,
        inputWidth,
        inputHeight),
        includeTop,
        preTrained,
        numClasses)
{
  // Nothing to do here.
}

template<typename OutputLayerType, typename InitializationRuleType>
void MobileNetV1<OutputLayerType, InitializationRuleType>::MobileNetV1(
    std::tuple<size_t, size_t, size_t> inputShape,
    const bool includeTop,
    const bool preTrained,
    const size_t numClasses) :
    inputChannel(std::get<0>(inputShape)),
    inputWidth(std::get<1>(inputShape)),
    inputHeight(std::get<2>(inputShape)),
    numClasses(numClasses)
{
  if (inputWidth < 32 || inputHeight < 32)
  {
    mlpack::Log::Fatal << "input width and input height cannot be smaller than" 
        " 32.\nGiven input width and height: (" << inputWidth << ", "
        << inputHeight << ")" << std::endl;
  }

  mobileNet.Add(new ann::Convolution<>(3, int(32 * alpha), 3, 3, 2, 2, 0, 0,
      inputWidth, inputHeight, "same"));
  mlpack::Log::Info << "Convolution: " << "(" << "3, " << inputWidth << ", "
      << inputHeight << ")" << " ---> (" << int(32 * alpha) << ", "
      << inputWidth << ", " << inputHeight << ")" << std::endl;
  mobileNet.Add(new ann::BatchNorm<>(int(32 * alpha), 1e-3, true, 0.99))
  mlpack::Log::Info << "BatchNorm: " << "(" << int(32 * alpha) << ")"
        << " ---> (" << int(32 * alpha) << ")" << std::endl;
  ReLU6Layer();

  
  // Reset parameters for a new network.
  mobileNet.ResetParameters();
}

template<typename OutputLayerType, typename InitializationRuleType>
void MobileNetV1<OutputLayerType, InitializationRuleType>::LoadModel(
    const std::string& filePath)
{
  data::Load(filePath, "ResNet", resNet);
  Log::Info << "Loaded model" << std::endl;
}

template<typename OutputLayerType, typename InitializationRuleType>
void MobileNetV1<OutputLayerType, InitializationRuleType>::SaveModel(
    const std::string& filePath)
{
  Log::Info<< "Saving model." << std::endl;
  data::Save(filePath, "ResNet", resNet);
  Log::Info << "Model saved in " << filePath << "." << std::endl;
}

} // namespace models
} // namespace mlpack

#endif
