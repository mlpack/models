/**
 * @file alexnet_impl.hpp
 * @author Shubham Agrawal
 *
 * Implementation of AlexNet using mlpack.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MODELS_MODELS_ALEXNET_ALEXNET_IMPL_HPP
#define MODELS_MODELS_ALEXNET_ALEXNET_IMPL_HPP

#include "alexnet.hpp"

namespace mlpack {
namespace models {

template<typename MatType>
AlexNetType<MatType>::AlexNetType(
    const size_t numClasses,
    const bool includeTop) :
    MultiLayer<MatType>(),
    numClasses(numClasses),
    includeTop(includeTop)
{
  MakeModel();
}

template<typename MatType>
AlexNetType<MatType>::AlexNetType(const AlexNetType& other) :
    MultiLayer<MatType>(other),
    numClasses(other.numClasses),
    includeTop(other.includeTop)
{
  // Nothing to do here.
}

template<typename MatType>
AlexNetType<MatType>::AlexNetType(AlexNetType&& other) :
    MultiLayer<MatType>(std::move(other)),
    numClasses(std::move(other.numClasses)),
    includeTop(std::move(other.includeTop))
{
  // Nothing to do here.
}

template<typename MatType>
AlexNetType<MatType>& AlexNetType<MatType>::operator=(const AlexNetType& other)
{
  if (this != &other)
  {
    MultiLayer<MatType>::operator=(other);
    numClasses = other.numClasses;
    includeTop = other.includeTop;
  }

  return *this;
}

template<typename MatType>
AlexNetType<MatType>& AlexNetType<MatType>::operator=(AlexNetType&& other)
{
  if (this != &other)
  {
    MultiLayer<MatType>::operator=(std::move(other));
    numClasses = std::move(other.numClasses);
    includeTop = std::move(other.includeTop);
  }

  return *this;
}

template<typename MatType>
template<typename Archive>
void AlexNetType<MatType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(cereal::base_class<MultiLayer<MatType>>(this));

  ar(CEREAL_NVP(numClasses));
  ar(CEREAL_NVP(includeTop));
}

template<typename MatType>
void AlexNetType<MatType>::MakeModel()
{
  this->template Add<Convolution>(64, 11, 11, 4, 4, 2, 2);
  this->template Add<ReLU>();
  this->template Add<MaxPooling>(3, 3, 2, 2);
  this->template Add<Convolution>(192, 5, 5, 1, 1, 2, 2);
  this->template Add<ReLU>();
  this->template Add<MaxPooling>(3, 3, 2, 2);
  this->template Add<Convolution>(384, 3, 3, 1, 1, 1, 1);
  this->template Add<ReLU>();
  this->template Add<Convolution>(256, 3, 3, 1, 1, 1, 1);
  this->template Add<ReLU>();
  this->template Add<Convolution>(256, 3, 3, 1, 1, 1, 1);
  this->template Add<ReLU>();
  this->template Add<MaxPooling>(3, 3, 2, 2);
  if (includeTop)
  {
    this->template Add<Dropout>();
    this->template Add<Linear>(4096);
    this->template Add<ReLU>();
    this->template Add<Dropout>();
    this->template Add<Linear>(4096);
    this->template Add<ReLU>();
    this->template Add<Linear>(numClasses);
  }
}

} // namespace models
} // namespace mlpack

#endif
