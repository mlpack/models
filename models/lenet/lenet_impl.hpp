/**
 * @file lenet_impl.hpp
 * @author Eugene Freyman
 * @author Daivik Nema
 * @author Kartik Dutt
 * 
 * Implementation of LeNet using mlpack.
 * 
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MODELS_LENET_IMPL_HPP
#define MODELS_LENET_IMPL_HPP

#include "lenet.hpp"
namespace mlpack {
namespace ann {

template<
    typename OutputLayerType,
    typename InitializationRuleType,
    size_t leNetVer
>
LeNet<OutputLayerType, InitializationRuleType, leNetVer>::LeNet(
    const size_t inputChannel,
    const size_t inputWidth,
    const size_t inputHeight,
    const size_t numClasses,
    const std::string& weights) :
    LeNet<OutputLayerType, InitializationRuleType, leNetVer>(
        std::tuple<size_t, size_t, size_t>(
            inputChannel,
            inputWidth,
            inputHeight),
          numClasses,
          weights)
{
  // Nothing to do here.
}

template<
    typename OutputLayerType,
    typename InitializationRuleType,
    size_t leNetVer
>
LeNet<OutputLayerType, InitializationRuleType, leNetVer>::LeNet(
    const std::tuple<size_t, size_t, size_t> inputShape,
    const size_t numClasses,
    const std::string &weights) :
    inputChannel(std::get<0>(inputShape)),
    inputWidth(std::get<1>(inputShape)),
    inputHeight(std::get<2>(inputShape)),
    numClasses(numClasses),
    weights(weights)
{
  mlpack::Log::Assert(leNetVer == 1 || leNetVer == 4 || leNetVer == 5,
      "Incorrent LeNet version. Possible values are 1, 4 and 5.");

  if (weights == "mnist")
  {
    LoadModel("./../weights/lenet/lenet" + std::to_string(leNetVer) +
        "_mnist.bin");
  }
  else if (weights != "none")
  {
    LoadModel(weights);
  }
  else
  {
    leNet.Add(new IdentityLayer<>());
    ConvolutionBlock(inputChannel, 6, 5, 5, 1, 1, 2, 2);
    PoolingBlock(2, 2, 2, 2);
    ConvolutionBlock(6, 16, 5, 5, 1, 1, 2, 2);
    PoolingBlock(2, 2, 2, 2);
    // Add linear layer for LeNet.
    if (leNetVer == 1)
    {
      leNet.Add(new Linear<>(16 * inputWidth * inputHeight, numClasses));
    }
    else if (leNetVer == 4)
    {
      leNet.Add(new Linear<>(16 * inputWidth * inputHeight, 120));
      leNet.Add(new LeakyReLU<>());
      leNet.Add(new Linear<>(120, numClasses));
    }
    else if (leNetVer == 5)
    {
      leNet.Add(new Linear<>(16 * inputWidth * inputHeight, 120));
      leNet.Add(new LeakyReLU<>());
      leNet.Add(new Linear<>(120, 84));
      leNet.Add(new LeakyReLU<>());
      leNet.Add(new Linear<>(84, 10));
    }

    leNet.Add(new LogSoftMax<>());
  }
}

template<
    typename OutputLayerType,
    typename InitializationRuleType,
    size_t leNetVer
>
Sequential<> LeNet<
    OutputLayerType, InitializationRuleType, leNetVer
>::AsSequential()
{
  return Sequential<>().Add(leNet);
}

template<
    typename OutputLayerType,
    typename InitializationRuleType,
    size_t leNetVer
>
void LeNet<
    OutputLayerType, InitializationRuleType, leNetVer
>::LoadModel(const std::string& filePath)
{
  data::Load(filePath, "LeNet" + std::to_string(leNetVer), leNet);
  std::cout << "Loaded model" << std::endl;
}

template<
    typename OutputLayerType,
    typename InitializationRuleType,
    size_t leNetVer
>
void LeNet<
    OutputLayerType, InitializationRuleType, leNetVer
>::SaveModel(const std::string& filePath)
{
  std::cout << "Saving model" << std::endl;
  data::Save(filePath, "LeNet" + std::to_string(leNetVer), leNet);
  std::cout << "Model saved in " << filePath << std::endl;
}

} // namespace ann
} // namespace mlpack

#endif
