/**
 * @file squeezenet_impl.hpp
 * @author Shubham Agrawal
 *
 * Implementation of SqueezeNet using mlpack.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MODELS_MODELS_SQUEEZENET_SQUEEZENET_IMPL_HPP
#define MODELS_MODELS_SQUEEZENET_SQUEEZENET_IMPL_HPP

#include "squeezenet.hpp"

namespace mlpack {
namespace models {

template<typename MatType, size_t SqueezeNetVersion>
SqueezeNetType<MatType, SqueezeNetVersion>::SqueezeNetType(
    const size_t numClasses,
    const bool includeTop) :
    MultiLayer<MatType>(),
    numClasses(numClasses),
    includeTop(includeTop)
{
  MakeModel();
}

template<typename MatType, size_t SqueezeNetVersion>
SqueezeNetType<MatType, SqueezeNetVersion>::SqueezeNetType(
    const SqueezeNetType& other) :
    MultiLayer<MatType>(other),
    numClasses(other.numClasses),
    includeTop(other.includeTop)
{
  // Nothing to do here.
}

template<typename MatType, size_t SqueezeNetVersion>
SqueezeNetType<MatType, SqueezeNetVersion>::SqueezeNetType(
    SqueezeNetType&& other) :
    MultiLayer<MatType>(std::move(other)),
    numClasses(std::move(other.numClasses)),
    includeTop(std::move(other.includeTop))
{
  // Nothing to do here.
}

template<typename MatType, size_t SqueezeNetVersion>
SqueezeNetType<MatType, SqueezeNetVersion>&
SqueezeNetType<MatType, SqueezeNetVersion>::operator=(
    const SqueezeNetType& other)
{
  if (this != &other)
  {
    MultiLayer<MatType>::operator=(other);
    numClasses = other.numClasses;
    includeTop = other.includeTop;
  }

  return *this;
}

template<typename MatType, size_t SqueezeNetVersion>
SqueezeNetType<MatType, SqueezeNetVersion>&
SqueezeNetType<MatType, SqueezeNetVersion>::operator=(SqueezeNetType&& other)
{
  if (this != &other)
  {
    MultiLayer<MatType>::operator=(std::move(other));
    numClasses = std::move(other.numClasses);
    includeTop = std::move(other.includeTop);
  }

  return *this;
}

template<typename MatType, size_t SqueezeNetVersion>
template<typename Archive>
void SqueezeNetType<MatType, SqueezeNetVersion>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(cereal::base_class<MultiLayer<MatType>>(this));

  ar(CEREAL_NVP(numClasses));
  ar(CEREAL_NVP(includeTop));
}

template<typename MatType, size_t SqueezeNetVersion>
void SqueezeNetType<MatType, SqueezeNetVersion>::Fire(
    const size_t squeezePlanes,
    const size_t expand1x1Planes,
    const size_t expand3x3Planes)
{
  this->template Add<Convolution>(squeezePlanes, 1, 1);
  this->template Add<ReLU>();

  MultiLayer<MatType>* expand1x1 = new MultiLayer<MatType>();
  expand1x1->template Add<Convolution>(expand1x1Planes, 1, 1);
  expand1x1->template Add<ReLU>();

  MultiLayer<MatType>* expand3x3 = new MultiLayer<MatType>();
  expand3x3->template Add<Convolution>(expand3x3Planes, 3, 3, 1, 1, 1, 1);
  expand3x3->template Add<ReLU>();

  Concat* catLayer = new Concat(2);
  catLayer->template Add(expand1x1);
  catLayer->template Add(expand3x3);

  this->template Add(catLayer);
}

template<typename MatType, size_t SqueezeNetVersion>
void SqueezeNetType<MatType, SqueezeNetVersion>::MakeModel()
{
  if (SqueezeNetVersion == 0)
  {
    this->template Add<Convolution>(96, 7, 7, 2, 2);
    this->template Add<ReLU>();
    this->template Add<MaxPooling>(3, 3, 2, 2, false);
    Fire(16, 64, 64);
    Fire(16, 64, 64);
    Fire(32, 128, 128);
    this->template Add<MaxPooling>(3, 3, 2, 2, false);
    Fire(32, 128, 128);
    Fire(48, 192, 192);
    Fire(48, 192, 192);
    Fire(64, 256, 256);
    this->template Add<MaxPooling>(3, 3, 2, 2, false);
    Fire(64, 256, 256);
  }
  else if (SqueezeNetVersion == 1)
  {
    this->template Add<Convolution>(64, 3, 3, 2, 2);
    this->template Add<ReLU>();
    this->template Add<MaxPooling>(3, 3, 2, 2, false);
    Fire(16, 64, 64);
    Fire(16, 64, 64);
    this->template Add<MaxPooling>(3, 3, 2, 2, false);
    Fire(32, 128, 128);
    Fire(32, 128, 128);
    this->template Add<MaxPooling>(3, 3, 2, 2, false);
    Fire(48, 192, 192);
    Fire(48, 192, 192);
    Fire(64, 256, 256);
    Fire(64, 256, 256);
  }
  else
  {
    mlpack::Log::Fatal << "Unsupported SqueezeNet version." << std::endl;
  }
  if (includeTop)
  {
    this->template Add<Dropout>();
    this->template Add<Convolution>(numClasses, 1, 1);
    this->template Add<ReLU>();
    this->template Add<AdaptiveMeanPooling>(1, 1);
  }
}

} // namespace models
} // namespace mlpack

#endif
