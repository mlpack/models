/**
 * @file mobilenet_impl.hpp
 * @author Aakash Kaushik
 *
 * Implementation of MobileNet using mlpack.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MODELS_MODELS_MOBILENET_MOBILENET_IMPL_HPP
#define MODELS_MODELS_MOBILENET_MOBILENET_IMPL_HPP

#include "mobilenet.hpp"

namespace mlpack {
namespace models {

template<typename MatType>
MobileNetType<MatType>::MobileNetType(
    const size_t numClasses,
    const bool includeTop,
    const float alpha,
    const size_t depthMultiplier) :
    MultiLayer<MatType>(),
    numClasses(numClasses),
    includeTop(includeTop),
    alpha(alpha),
    depthMultiplier(depthMultiplier)
{
  MakeModel();
}

template<typename MatType>
MobileNetType<MatType>::MobileNetType(
    const MobileNetType& other) :
    MultiLayer<MatType>(other),
    numClasses(other.numClasses),
    includeTop(other.includeTop),
    alpha(other.alpha),
    depthMultiplier(other.depthMultiplier)
{
  // Nothing to do here.
}

template<typename MatType>
MobileNetType<MatType>::MobileNetType(
    MobileNetType&& other) :
    MultiLayer<MatType>(std::move(other)),
    numClasses(std::move(other.numClasses)),
    includeTop(std::move(other.includeTop)),
    alpha(std::move(other.alpha)),
    depthMultiplier(std::move(other.depthMultiplier))
{
  // Nothing to do here.
}

template<typename MatType>
MobileNetType<MatType>&
MobileNetType<MatType>::operator=(
    const MobileNetType& other)
{
  if (this != &other)
  {
    MultiLayer<MatType>::operator=(other);
    numClasses = other.numClasses;
    includeTop = other.includeTop;
    alpha = other.alpha;
    depthMultiplier = other.depthMultiplier;
  }

  return *this;
}

template<typename MatType>
MobileNetType<MatType>&
MobileNetType<MatType>::operator=(
    MobileNetType&& other)
{
  if (this != &other)
  {
    MultiLayer<MatType>::operator=(std::move(other));
    numClasses = std::move(other.numClasses);
    includeTop = std::move(other.includeTop);
    alpha = std::move(other.alpha);
    depthMultiplier = std::move(other.depthMultiplier);
  }

  return *this;
}

template<typename MatType>
template<typename Archive>
void MobileNetType<MatType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(cereal::base_class<MultiLayer<MatType>>(this));

  ar(CEREAL_NVP(numClasses));
  ar(CEREAL_NVP(includeTop));
  ar(CEREAL_NVP(alpha));
  ar(CEREAL_NVP(depthMultiplier));
}

template<typename MatType>
void MobileNetType<MatType>::MakeModel()
{
  this->template Add<Convolution>(32, 3, 3, 2, 2, 0, 0, "none", false);
  this->template Add<BatchNorm>();

  size_t outSize = size_t(32 * alpha);

  this->template Add<SeparableConvolution>(outSize, outSize*depthMultiplier,
                    alpha, depthMultiplier);
  outSize = size_t(64 * alpha);
  for (const auto& blockConfig : mobileNetConfig){
    this->template Add<SeparableConvolution>(outSize, outSize*depthMultiplier,
                      alpha, depthMultiplier, 2);
    this->template Add<BatchNorm>();
    this->template Add<ReLU6>();

    outSize = size_t(blockConfig.first * alpha);

    for (size_t numBlock = 1; numBlock < blockConfig.second; ++numBlock)
    {
      this->template Add<SeparableConvolution>(outSize, outSize*depthMultiplier,
                         alpha, depthMultiplier);
      this->template Add<BatchNorm>();
      this->template Add<ReLU6>();

      outSize = size_t(blockConfig.first * alpha);
    }

    this->template Add<AdaptiveMeanPooling>(1,1);
  }

  if (includeTop)
  {
    this->template Add<Dropout>(1e-3);
    this->template Add<Convolution>(size_t(1024 * alpha), numClasses,
                      1, 1, 1, 0, 0, 1, 1, "same");
    this->template Add<Softmax>();
  }
}

} // namespace models
} // namespace mlpack

#endif // MODELS_MODELS_MOBILENET_MOBILENET_IMPL_HPP
