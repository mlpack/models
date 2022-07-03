/**
 * @file vgg_impl.hpp
 * @author Shubham Agrawal
 *
 * Implementation of VGG using mlpack.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MODELS_MODELS_ALEXNET_ALEXNET_IMPL_HPP
#define MODELS_MODELS_ALEXNET_ALEXNET_IMPL_HPP

#include "vgg.hpp"

namespace mlpack {
namespace models {

template<typename MatType, size_t VGGVersion, bool IsBatchNorm>
VGGType<MatType, VGGVersion, IsBatchNorm>::VGGType() :
    ann::MultiLayer<MatType>(),
    numClasses(1000),
    includeTop(true)
{
  makeModel();
}

template<typename MatType, size_t VGGVersion, bool IsBatchNorm>
VGGType<MatType, VGGVersion, IsBatchNorm>::VGGType(
    const size_t numClasses,
    const bool includeTop) :
    ann::MultiLayer<MatType>(),
    numClasses(numClasses),
    includeTop(includeTop)
{
  makeModel();
}

template<typename MatType, size_t VGGVersion, bool IsBatchNorm>
VGGType<MatType, VGGVersion, IsBatchNorm>::VGGType(
    const VGGType& other) :
    ann::MultiLayer<MatType>(other),
    numClasses(other.numClasses),
    includeTop(other.includeTop)
{
  // Nothing to do here.
}

template<typename MatType, size_t VGGVersion, bool IsBatchNorm>
VGGType<MatType, VGGVersion, IsBatchNorm>::VGGType(
    VGGType&& other) :
    ann::MultiLayer<MatType>(std::move(other)),
    numClasses(std::move(other.numClasses)),
    includeTop(std::move(other.includeTop))
{
  // Nothing to do here.
}

template<typename MatType, size_t VGGVersion, bool IsBatchNorm>
VGGType<MatType, VGGVersion, IsBatchNorm>& 
VGGType<MatType, VGGVersion, IsBatchNorm>::operator=(const VGGType& other)
{
  if (this != &other)
  {
    ann::MultiLayer<MatType>::operator=(other);
    numClasses = other.numClasses;
    includeTop = other.includeTop;
  }

  return *this;
}

template<typename MatType, size_t VGGVersion, bool IsBatchNorm>
VGGType<MatType, VGGVersion, IsBatchNorm>& 
VGGType<MatType, VGGVersion, IsBatchNorm>::operator=(VGGType&& other)
{
  if (this != &other)
  {
    ann::MultiLayer<MatType>::operator=(std::move(other));
    numClasses = std::move(other.numClasses);
    includeTop = std::move(other.includeTop);
  }

  return *this;
}

template<typename MatType, size_t VGGVersion, bool IsBatchNorm>
template<typename Archive>
void VGGType<MatType, VGGVersion, IsBatchNorm>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(cereal::base_class<ann::MultiLayer<MatType>>(this));

  ar(CEREAL_NVP(numClasses));
  ar(CEREAL_NVP(includeTop));
}

} // namespace models
} // namespace mlpack

#endif
