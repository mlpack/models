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
#ifndef MODELS_MODELS_VGG_VGG_IMPL_HPP
#define MODELS_MODELS_VGG_VGG_IMPL_HPP

#include "vgg.hpp"

namespace mlpack {
namespace models {

template<typename MatType, size_t VGGVersion, bool UsesBatchNorm>
VGGType<MatType, VGGVersion, UsesBatchNorm>::VGGType(
    const size_t numClasses,
    const bool includeTop) :
    ann::MultiLayer<MatType>(),
    numClasses(numClasses),
    includeTop(includeTop)
{
  MakeModel();
}

template<typename MatType, size_t VGGVersion, bool UsesBatchNorm>
VGGType<MatType, VGGVersion, UsesBatchNorm>::VGGType(
    const VGGType& other) :
    ann::MultiLayer<MatType>(other),
    numClasses(other.numClasses),
    includeTop(other.includeTop)
{
  // Nothing to do here.
}

template<typename MatType, size_t VGGVersion, bool UsesBatchNorm>
VGGType<MatType, VGGVersion, UsesBatchNorm>::VGGType(
    VGGType&& other) :
    ann::MultiLayer<MatType>(std::move(other)),
    numClasses(std::move(other.numClasses)),
    includeTop(std::move(other.includeTop))
{
  // Nothing to do here.
}

template<typename MatType, size_t VGGVersion, bool UsesBatchNorm>
VGGType<MatType, VGGVersion, UsesBatchNorm>&
VGGType<MatType, VGGVersion, UsesBatchNorm>::operator=(const VGGType& other)
{
  if (this != &other)
  {
    ann::MultiLayer<MatType>::operator=(other);
    numClasses = other.numClasses;
    includeTop = other.includeTop;
  }

  return *this;
}

template<typename MatType, size_t VGGVersion, bool UsesBatchNorm>
VGGType<MatType, VGGVersion, UsesBatchNorm>&
VGGType<MatType, VGGVersion, UsesBatchNorm>::operator=(VGGType&& other)
{
  if (this != &other)
  {
    ann::MultiLayer<MatType>::operator=(std::move(other));
    numClasses = std::move(other.numClasses);
    includeTop = std::move(other.includeTop);
  }

  return *this;
}

template<typename MatType, size_t VGGVersion, bool UsesBatchNorm>
template<typename Archive>
void VGGType<MatType, VGGVersion, UsesBatchNorm>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(cereal::base_class<ann::MultiLayer<MatType>>(this));

  ar(CEREAL_NVP(numClasses));
  ar(CEREAL_NVP(includeTop));
}

template<typename MatType, size_t VGGVersion, bool UsesBatchNorm>
void VGGType<MatType, VGGVersion, UsesBatchNorm>::MakeModel()
{
  /**
   * @brief `contruct` variable contains the number of output maps for each
   *        layer. Also, 0 means MaxPooling layer here.
   */
  std::map<size_t, std::vector<size_t>> const construct {
    { 11, {64, 0, 128, 0, 256, 256, 0, 512, 512, 0, 512, 512, 0} },
    { 13, {64, 64, 0, 128, 128, 0, 256, 256, 0, 512, 512, 0, 512, 512, 0} },
    { 16, {64, 64, 0, 128, 128, 0, 256, 256, 256, 0, 512, 512, 512, 0, 512,
        512, 512, 0} },
    { 19, {64, 64, 0, 128, 128, 0, 256, 256, 256, 256, 0, 512, 512, 512, 512,
        0, 512, 512, 512, 512, 0} }
  };
  std::vector<size_t> layers = construct.at(VGGVersion);
  for (size_t i = 0; i < layers.size(); i++)
  {
    if (layers[i] == 0)
    {
      this->template Add<ann::MaxPooling>(2, 2, 2, 2);
    }
    else
    {
      this->template Add<ann::Convolution>(layers[i], 3, 3, 1, 1, 1, 1);
      if (UsesBatchNorm)
        this->template Add<ann::BatchNorm>(2, 2, 1e-5, false, 0.1);
      this->template Add<ann::ReLU>();
    }
  }
  if (includeTop)
  {
    this->template Add<ann::Linear>(4096);
    this->template Add<ann::ReLU>();
    this->template Add<ann::Dropout>();
    this->template Add<ann::Linear>(4096);
    this->template Add<ann::ReLU>();
    this->template Add<ann::Dropout>();
    this->template Add<ann::Linear>(numClasses);
  }
}

} // namespace models
} // namespace mlpack

#endif
