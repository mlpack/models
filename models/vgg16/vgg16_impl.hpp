/**
 * @file vgg16_impl.hpp
 * @author Vishwas Chepuri
 * 
 * Implementation of VGG16 using mlpack.
 * 
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MODELS_VGG16_IMPL_HPP
#define MODELS_VGG16_IMPL_HPP

#include "vgg16.hpp"

namespace mlpack {
namespace ann {

template<
      typename OutputLayerType, 
      typename InitializationRuleType
>
VGG16<OutputLayerType, InitializationRuleType>::VGG16() :
    VGG16<OutputLayerType, InitializationRuleType>(
      std::tuple<size_t, size_t, size_t>(3, 224, 224),
      true,
      "imagenet",
      1000)
{
  // Nothing to do here.
}

template<
      typename OutputLayerType, 
      typename InitializationRuleType
>
VGG16<OutputLayerType, InitializationRuleType>::VGG16(
    const std::tuple<size_t, size_t, size_t> inputShape,
    const bool includeTop,
    const std::string& weights,
    const size_t numClasses) : 
    inputChannel(std::get<0>(inputShape)),
    inputWidth(std::get<1>(inputShape)),
    inputHeight(std::get<2>(inputShape)),
    numClasses(numClasses),
    weights(weights)
{

  mlpack::Log::Assert(!(weights == "imagenet" && includeTop && numClasses != 1000),
          "If using `weights` as `imagenet` with `includeTop` as true, `numClasses` should be `1000`");

  // TODO : Add loading weights
  

  // Block 1
  vgg16Network.Add(new Convolution<>(inputChannel, 64, 3, 3, 1, 1, 1, 1, inputWidth, inputHeight, "same"));
  vgg16Network.Add(new ReLULayer<>());

  inputWidth = ConvOutSize(inputWidth, 3, 1, "same");
  inputHeight = ConvOutSize(inputHeight, 3, 1, "same");

  vgg16Network.Add(new Convolution<>(64, 64, 3, 3, 1, 1, 1, 1, inputWidth, inputHeight, "same"));
  vgg16Network.Add(new ReLULayer<>());

  inputWidth = ConvOutSize(inputWidth, 3, 1, "same");
  inputHeight = ConvOutSize(inputHeight, 3, 1, "same");

  vgg16Network.Add(new MaxPooling<>(2, 2, 2, 2));

  inputWidth = ConvOutSize(inputWidth, 2, 2, "valid");
  inputHeight = ConvOutSize(inputHeight, 2, 2, "valid");


  // Block 2
  vgg16Network.Add(new Convolution<>(64, 128, 3, 3, 1, 1, 1, 1, inputWidth, inputHeight, "same"));
  vgg16Network.Add(new ReLULayer<>());

  inputWidth = ConvOutSize(inputWidth, 3, 1, "same");
  inputHeight = ConvOutSize(inputHeight, 3, 1, "same");

  vgg16Network.Add(new Convolution<>(128, 128, 3, 3, 1, 1, 1, 1, inputWidth, inputHeight, "same"));
  vgg16Network.Add(new ReLULayer<>());

  inputWidth = ConvOutSize(inputWidth, 3, 1, "same");
  inputHeight = ConvOutSize(inputHeight, 3, 1, "same");

  vgg16Network.Add(new MaxPooling<>(2, 2, 2, 2));

  inputWidth = ConvOutSize(inputWidth, 2, 2, "valid");
  inputHeight = ConvOutSize(inputHeight, 2, 2, "valid");


  // Block 3
  vgg16Network.Add(new Convolution<>(128, 256, 3, 3, 1, 1, 1, 1, inputWidth, inputHeight, "same"));
  vgg16Network.Add(new ReLULayer<>());

  inputWidth = ConvOutSize(inputWidth, 3, 1, "same");
  inputHeight = ConvOutSize(inputHeight, 3, 1, "same");

  vgg16Network.Add(new Convolution<>(256, 256, 3, 3, 1, 1, 1, 1, inputWidth, inputHeight, "same"));
  vgg16Network.Add(new ReLULayer<>());

  inputWidth = ConvOutSize(inputWidth, 3, 1, "same");
  inputHeight = ConvOutSize(inputHeight, 3, 1, "same");

  vgg16Network.Add(new Convolution<>(256, 256, 3, 3, 1, 1, 1, 1, inputWidth, inputHeight, "same"));
  vgg16Network.Add(new ReLULayer<>());

  inputWidth = ConvOutSize(inputWidth, 3, 1, "same");
  inputHeight = ConvOutSize(inputHeight, 3, 1, "same");

  vgg16Network.Add(new MaxPooling<>(2, 2, 2, 2));

  inputWidth = ConvOutSize(inputWidth, 2, 2, "valid");
  inputHeight = ConvOutSize(inputHeight, 2, 2, "valid");


  // Block 4
  vgg16Network.Add(new Convolution<>(256, 512, 3, 3, 1, 1, 1, 1, inputWidth, inputHeight, "same"));
  vgg16Network.Add(new ReLULayer<>());

  inputWidth = ConvOutSize(inputWidth, 3, 1, "same");
  inputHeight = ConvOutSize(inputHeight, 3, 1, "same");

  vgg16Network.Add(new Convolution<>(512, 512, 3, 3, 1, 1, 1, 1, inputWidth, inputHeight, "same"));
  vgg16Network.Add(new ReLULayer<>());

  inputWidth = ConvOutSize(inputWidth, 3, 1, "same");
  inputHeight = ConvOutSize(inputHeight, 3, 1, "same");

  vgg16Network.Add(new Convolution<>(512, 512, 3, 3, 1, 1, 1, 1, inputWidth, inputHeight, "same"));
  vgg16Network.Add(new ReLULayer<>());

  inputWidth = ConvOutSize(inputWidth, 3, 1, "same");
  inputHeight = ConvOutSize(inputHeight, 3, 1, "same");

  vgg16Network.Add(new MaxPooling<>(2, 2, 2, 2));

  inputWidth = ConvOutSize(inputWidth, 2, 2, "valid");
  inputHeight = ConvOutSize(inputHeight, 2, 2, "valid");


  // Block 5
  vgg16Network.Add(new Convolution<>(512, 512, 3, 3, 1, 1, 1, 1, inputWidth, inputHeight, "same"));
  vgg16Network.Add(new ReLULayer<>());

  inputWidth = ConvOutSize(inputWidth, 3, 1, "same");
  inputHeight = ConvOutSize(inputHeight, 3, 1, "same");

  vgg16Network.Add(new Convolution<>(512, 512, 3, 3, 1, 1, 1, 1, inputWidth, inputHeight, "same"));
  vgg16Network.Add(new ReLULayer<>());

  inputWidth = ConvOutSize(inputWidth, 3, 1, "same");
  inputHeight = ConvOutSize(inputHeight, 3, 1, "same");

  vgg16Network.Add(new Convolution<>(512, 512, 3, 3, 1, 1, 1, 1, inputWidth, inputHeight, "same"));
  vgg16Network.Add(new ReLULayer<>());

  inputWidth = ConvOutSize(inputWidth, 3, 1, "same");
  inputHeight = ConvOutSize(inputHeight, 3, 1, "same");

  vgg16Network.Add(new MaxPooling<>(2, 2, 2, 2));

  inputWidth = ConvOutSize(inputWidth, 2, 2, "valid");
  inputHeight = ConvOutSize(inputHeight, 2, 2, "valid");


  if (includeTop)
  {
    vgg16Network.Add(new Linear<>(inputWidth * inputHeight * 512, 4096));
    vgg16Network.Add(new ReLULayer<>());

    vgg16Network.Add(new Linear<>(4096, 4096));
    vgg16Network.Add(new ReLULayer<>());

    vgg16Network.Add(new Linear<>(4096, numClasses));
    vgg16Network.Add(new Softmax<>());
  }
  
}


} // namespace ann
} // namespace MLPACK

#endif
