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
    numClasses(0)
{
  // Nothing to do here.
}

template<typename OutputLayerType, typename InitializationRuleType,
    size_t ResNetVersion>
ResNet<OutputLayerType, InitializationRuleType, ResNetVersion>::ResNet(
    const size_t inputChannel,
    const size_t inputWidth,
    const size_t inputHeight,
    const bool includeTop,
    const bool preTrained,
    const size_t numClasses) :
    ResNet<OutputLayerType, InitializationRuleType,  ResNetVersion>(
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

template<typename OutputLayerType, typename InitializationRuleType,
    size_t ResNetVersion>
ResNet<OutputLayerType, InitializationRuleType, ResNetVersion>::ResNet(
    std::tuple<size_t, size_t, size_t> inputShape,
    const bool includeTop,
    const bool preTrained,
    const size_t numClasses) :
    inputChannel(std::get<0>(inputShape)),
    inputWidth(std::get<1>(inputShape)),
    inputHeight(std::get<2>(inputShape)),
    numClasses(numClasses)
{

  resNet.Add(new ann::Convolution<>(3, 64, 7, 7, 2, 2, 3, 3, inputWidth, inputHeight));
  std::cout<<"Convolution: "<<3<<" "<<64<<" "<<7<<" "<<7<<" "
        <<2<<" "<<2<<" "<<3<<" "<<3<<" "
        <<inputWidth<<" "<<inputHeight<<std::endl;
  
  // Updating input dimesntions.
  inputWidth = ConvOutSize(inputWidth, 7, 2, 3);
  inputHeight = ConvOutSize(inputHeight, 7, 2, 3);
  
  resNet.Add(new ann::BatchNorm<>(64));
  std::cout<<"BatchNorm: "<<64<<std::endl;

  resNet.Add(new ann::ReLULayer<>);
  std::cout<<"Relu"<<std::endl;

  
  resNet.Add(new ann::Padding<>(1, 1, 1, 1));
  std::cout<<"Padding: "<<"1,1,1,1"<<" ";

  inputWidth += 2;
  inputHeight += 2; 

  std::cout<<inputWidth<<" "<<inputHeight<<std::endl;

  resNet.Add(new ann::MaxPooling<>(3, 3, 2, 2));
  std::cout<<"MaxPool: "<<"3,3,2,2"<<" ";

  inputWidth = ConvOutSize(inputWidth, 3, 2, 0);
  inputHeight = ConvOutSize(inputHeight, 3, 2, 0);

  std::cout<<inputWidth<<" "<<inputHeight<<std::endl;

  if (ResNetVersion == 18)
  {
    numBlockArray = {2 ,2, 2, 2};
  }

  MakeLayer("basicblock", 64, numBlockArray[0]);
  MakeLayer("basicblock", 128, numBlockArray[1], 2);
  MakeLayer("basicblock", 256, numBlockArray[2], 2);
  MakeLayer("basicblock", 512, numBlockArray[3], 2);

  if (includeTop)
  {
    // What would be the Pytroch equivalent of nn.AdaptiveAvgPool2d((1, 1))
    // reference: https://pytorch.org/docs/stable/generated/torch.nn.AdaptiveAvgPool2d.html 
    resNet.Add(new ann::AdaptiveMeanPooling<>(1, 1));
    std::cout<<"AdaptiveMeanPooling: "<<"1,1"<<std::endl;


    if (ResNetVersion == 18 || ResNetVersion == 34)
    {
      resNet.Add(new ann::Linear<>(512 * basicBlockExpansion, numClasses));
      std::cout<<"Linear: "<<512 * basicBlockExpansion<<" "<<numClasses<<std::endl;
    }
    else
    { 
      resNet.Add(new ann::Linear<>(512 * bottleNeckExpansion, numClasses));
      std::cout<<"Linear: "<<512 * bottleNeckExpansion<<" "<<numClasses<<std::endl;
    } 
  }

}


} // namespace models
} // namespace mlpack


#endif
