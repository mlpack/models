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
SqueezeNetType<MatType, SqueezeNetVersion>::SqueezeNetType() :
    ann::MultiLayer<MatType>(),
    numClasses(1000),
    includeTop(true)
{
  makeModel();
}

template<typename MatType, size_t SqueezeNetVersion>
SqueezeNetType<MatType, SqueezeNetVersion>::SqueezeNetType(
    const size_t numClasses,
    const bool includeTop) :
    ann::MultiLayer<MatType>(),
    numClasses(numClasses),
    includeTop(includeTop)
{
  makeModel();
}

template<typename MatType, size_t SqueezeNetVersion>
SqueezeNetType<MatType, SqueezeNetVersion>::SqueezeNetType(
    const SqueezeNetType& other) :
    ann::MultiLayer<MatType>(other),
    numClasses(other.numClasses),
    includeTop(other.includeTop)
{
  // Nothing to do here.
}

template<typename MatType, size_t SqueezeNetVersion>
SqueezeNetType<MatType, SqueezeNetVersion>::SqueezeNetType(
    SqueezeNetType&& other) :
    ann::MultiLayer<MatType>(std::move(other)),
    numClasses(std::move(other.numClasses)),
    includeTop(std::move(other.includeTop))
{
  // Nothing to do here.
}

template<typename MatType, size_t SqueezeNetVersion>
SqueezeNetType<MatType, SqueezeNetVersion>& 
SqueezeNetType<MatType, SqueezeNetVersion>::operator=(const SqueezeNetType& other)
{
  if (this != &other)
  {
    ann::MultiLayer<MatType>::operator=(other);
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
    ann::MultiLayer<MatType>::operator=(std::move(other));
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
  ar(cereal::base_class<ann::MultiLayer<MatType>>(this));

  ar(CEREAL_NVP(numClasses));
  ar(CEREAL_NVP(includeTop));
}

} // namespace models
} // namespace mlpack

#endif
