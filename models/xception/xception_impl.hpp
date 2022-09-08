/**
 * @file xception_impl.hpp
 * @author Shubham Agrawal
 *
 * Implementation of Xception using mlpack.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MODELS_MODELS_XCEPTION_XCEPTION_IMPL_HPP
#define MODELS_MODELS_XCEPTION_XCEPTION_IMPL_HPP

#include "xception.hpp"

namespace mlpack {
namespace models {

template<typename MatType>
XceptionType<MatType>::XceptionType(
    const size_t numClasses,
    const bool includeTop) :
    ann::MultiLayer<MatType>(),
    numClasses(numClasses),
    includeTop(includeTop)
{
  MakeModel();
}

template<typename MatType>
XceptionType<MatType>::XceptionType(
    const XceptionType& other) :
    ann::MultiLayer<MatType>(other),
    numClasses(other.numClasses),
    includeTop(other.includeTop)
{
  // Nothing to do here.
}

template<typename MatType>
XceptionType<MatType>::XceptionType(
    XceptionType&& other) :
    ann::MultiLayer<MatType>(std::move(other)),
    numClasses(std::move(other.numClasses)),
    includeTop(std::move(other.includeTop))
{
  // Nothing to do here.
}

template<typename MatType>
XceptionType<MatType>&
XceptionType<MatType>::operator=(const XceptionType& other)
{
  if (this != &other)
  {
    ann::MultiLayer<MatType>::operator=(other);
    numClasses = other.numClasses;
    includeTop = other.includeTop;
  }

  return *this;
}

template<typename MatType>
XceptionType<MatType>&
XceptionType<MatType>::operator=(XceptionType&& other)
{
  if (this != &other)
  {
    ann::MultiLayer<MatType>::operator=(std::move(other));
    numClasses = std::move(other.numClasses);
    includeTop = std::move(other.includeTop);
  }

  return *this;
}

template<typename MatType>
template<typename Archive>
void XceptionType<MatType>::serialize(
    Archive& ar, const uint32_t /* version */)
{
  ar(cereal::base_class<ann::MultiLayer<MatType>>(this));

  ar(CEREAL_NVP(numClasses));
  ar(CEREAL_NVP(includeTop));
}

template<typename MatType>
void XceptionType<MatType>::SeparableConv(
    ann::MultiLayer<MatType>* block,
    const size_t inMaps,
    const size_t outMaps,
    const size_t kernelSize,
    const size_t stride = 1,
    const size_t padding = 0,
    const bool useBias = false)
{
  block->template Add<ann::GroupedConvolution>(inMaps, kernelSize, kernelSize,
      inMaps, stride, stride, padding, padding, "none", useBias);
  block->template Add<ann::Convolution>(outMaps, 1, 1, 1, 1, 0, 0, "none",
      useBias);
}

template<typename MatType>
void XceptionType<MatType>::Block(
    const size_t inMaps,
    const size_t outMaps,
    const size_t reps,
    const size_t strides = 1,
    const bool startWithRelu = true,
    const bool growFirst = true)
{
  ann::MultiLayer<MatType>* block = new ann::MultiLayer<MatType>();
  size_t filter = inMaps;
  if (reps < 2)
  {
    if (startWithRelu)
      block->template Add<ann::ReLU>();
    SeparableConv(block, inMaps, outMaps, 3, 1, 1, false);
    block->template Add<ann::BatchNorm>();
  }
  else
  {
    if (growFirst)
    {
      if (startWithRelu)
        block->template Add<ann::ReLU>();
      SeparableConv(block, inMaps, outMaps, 3, 1, 1, false);
      block->template Add<ann::BatchNorm>();
      filter = outMaps;
    }
    if (startWithRelu || growFirst)
      block->template Add<ann::ReLU>();
    SeparableConv(block, filter, filter, 3, 1, 1, false);
    block->template Add<ann::BatchNorm>();
    if (reps > 2)
    {
      for (size_t i = 0; i < reps - 2; i++)
      {
        block->template Add<ann::ReLU>();
        SeparableConv(block, filter, filter, 3, 1, 1, false);
        block->template Add<ann::BatchNorm>();
      }
    }
    if (!growFirst)
    {
      block->template Add<ann::ReLU>();
      SeparableConv(block, inMaps, outMaps, 3, 1, 1, false);
      block->template Add<ann::BatchNorm>();
    }
  }
  if (strides != 1)
  {
    block->template Add<ann::Padding>(1, 1, 1, 1);
    block->template Add<ann::MaxPooling>(3, 3, strides, strides);
  }
  if (inMaps != outMaps || strides != 1)
  {
    ann::MultiLayer<MatType>* block2 = new ann::MultiLayer<MatType>();
    block2->template Add<ann::Convolution>(outMaps, 1, 1, strides, strides,
        0, 0, "none", false);
    block2->template Add<ann::BatchNorm>();

    ann::AddMerge* merge = new ann::AddMerge();
    merge->template Add(block);
    merge->template Add(block2);

    this->template Add(merge);
  }
  else
  {
    ann::AddMerge* merge = new ann::AddMerge();
    merge->template Add(block);
    merge->template Add<ann::Identity>();

    this->template Add(merge);
  }
}

template<typename MatType>
void XceptionType<MatType>::MakeModel()
{
  this->template Add<ann::Convolution>(32, 3, 3, 2, 2, 0, 0, "none", false);
  this->template Add<ann::BatchNorm>();
  this->template Add<ann::ReLU>();

  this->template Add<ann::Convolution>(64, 3, 3, 1, 1, 0, 0, "none", false);
  this->template Add<ann::BatchNorm>();
  this->template Add<ann::ReLU>();

  Block(64, 128, 2, 2, false, true);
  Block(128, 256, 2, 2);
  Block(256, 728, 2, 2);

  Block(728, 728, 3, 1);
  Block(728, 728, 3, 1);
  Block(728, 728, 3, 1);
  Block(728, 728, 3, 1);

  Block(728, 728, 3, 1);
  Block(728, 728, 3, 1);
  Block(728, 728, 3, 1);
  Block(728, 728, 3, 1);

  Block(728, 1024, 2, 2, true, false);

  SeparableConv(this, 1024, 1536, 3, 1, 1);
  this->template Add<ann::BatchNorm>();
  this->template Add<ann::ReLU>();

  SeparableConv(this, 1536, 2048, 3, 1, 1);
  this->template Add<ann::BatchNorm>();

  if (includeTop)
  {
    this->template Add<ann::ReLU>();
    this->template Add<ann::AdaptiveMeanPooling>(1, 1);
    this->template Add<ann::Linear>(numClasses);
  }
}

} // namespace models
} // namespace mlpack

#endif
