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
#ifndef MODELS_MODELS_RESNET_RESNET_IMPL_HPP
#define MODELS_MODELS_RESNET_RESNET_IMPL_HPP

#include "resnet.hpp"

namespace mlpack {
namespace models {

template<typename OutputLayerType, typename InitializationRuleType,
    size_t ResNetVersion>
ResNet<OutputLayerType, InitializationRuleType, ResNetVersion>::ResNet() :
    inputChannel(0),
    inputWidth(0),
    inputHeight(0),
    numClasses(0),
{
  // Nothing to do here.
}

template<typename OutputLayerType, typename InitializationRuleType,
    size_t ResNetVersion>
ResNet<OutputLayerType, InitializationRuleType, ResNetVersion>::ResNet(
    const bool includeTop,
    const bool preTrained,
    const size_t inputChannel,
    const size_t inputWidth,
    const size_t inputHeight,
    const size_t numClasses) :
    ReNet<OutputLayerType, InitializationRule,  ResNetVersion>(
        includeTop,
        preTrained,
        std::tuple<size_t, size_t, size_t>(
        inputChannel,
        inputWidth,
        inputHeight),
        numClasses)
{
  // Nothing to do here.
}

template<typename OutputLayerType, typename InitializationRuleType,
    size_t ResNetVersion>
ResNet<OutputLayerType, InitializationRuleType, ResNetVersion>::ResNet(
    const bool includeTop,
    const bool preTrained,
    std::tuple<size_t, size_t, size_t> inputShape,
    const size_t numClasses,
    ) :
    inputChannel(std::get<0>(inputShape)),
    inputWidth(std::get<1>(inputShape)),
    inputHeight(std::get<2>(inputShape)),
    numClasses(numClasses)
{

  resNet.Add(ann::Convolution<>(3, 64, 7, 7, 2, 2, 3, 3, inputWidth, inputHeight));
  resNet.Add(ann::batchNorm<>(64));
  resNet.Add(ann::ReLULayer<>);
  // how to add a padding of 1 for the maxpool operation ? 
  resNet.Add(ann::MaxPooling<>(3, 3, 2, 2));

  if (ResNetVersion == 18)
  {

  }
}


} // namespace models
} // namespace mlpack


#endif
