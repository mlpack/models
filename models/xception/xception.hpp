/**
 * @file xception.hpp
 * @author Shubham Agrawal
 * 
 * Definition of Xception model.
 * 
 * For more information, kindly refer to the following paper.
 * 
 * Paper for Xception.
 *
 * @code
 * @article{Chollet2016,
 *  author = {François Chollet},
 *  title = {Xception: Deep Learning with Depthwise Separable Convolutions},
 *  year = {2016},
 *  url = {https://arxiv.org/pdf/1610.02357.pdf}
 * }
 * @endcode
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MODELS_MODELS_XCEPTION_XCEPTION_HPP
#define MODELS_MODELS_XCEPTION_XCEPTION_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/layer/layer_types.hpp>
#include <mlpack/methods/ann/init_rules/random_init.hpp>
#include <mlpack/methods/ann/loss_functions/binary_cross_entropy_loss.hpp>

namespace mlpack {
namespace models {

/**
 * Definition of a VGG CNN.
 * 
 * NOTE: Note that output size will be 1x1xN. Here, N is number of classes.
 * 
 * @tparam MatType Matrix representation to accept as input and use for
 *    computation.
 * @tparam VGGVersion Version of VGG.
 * @tparam IsBatchNorm Whether to apply Batch Norm layer.
 */
template<
  typename MatType = arma::mat
>
class XceptionType : public ann::MultiLayer<MatType>
{
 public:
  //! Create the XceptionType layer.
  XceptionType();

  /**
   * XceptionType constructor intializes number of classes and weights.
   *
   * @param numClasses Optional number of classes to classify images into,
   *     only to be specified if includeTop is  true.
   * @param includeTop Must be set to true if weights are set.
   */
  XceptionType(
    const size_t numClasses,
    const bool includeTop = true);
  
  //! Copy the given XceptionType.
  XceptionType(const XceptionType& other);
  //! Take ownership of the layers of the given XceptionType.
  XceptionType(XceptionType&& other);
  //! Copy the given XceptionType.
  XceptionType& operator=(const XceptionType& other);
  //! Take ownership of the given XceptionType.
  XceptionType& operator=(XceptionType&& other);

  //! Virtual destructor: delete all held layers.
  virtual ~XceptionType()
  {
    // Nothing to do here. 
  }

  //! Create a copy of the XceptionType (this is safe for polymorphic use).
  XceptionType* Clone() const { return new XceptionType(*this); }

  /**
   * Get Layers of the model.
   * 
   * @tparam OutputLayerType The output layer type used to evaluate the network.
   * @tparam InitializationRuleType Rule used to initialize the weight matrix.
   */
  template<
    typename OutputLayerType = ann::CrossEntropyError,
    typename InitializationRuleType = ann::RandomInitialization
  >
  ann::FFN<OutputLayerType, InitializationRuleType, MatType>* GetModel()
  {
    ann::FFN<OutputLayerType, InitializationRuleType, MatType>* xception = 
        new ann::FFN<OutputLayerType, InitializationRuleType, MatType>();
    xception->Add(this);
    return xception;
  }

  //! Serialize the XceptionType.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  void SeparableConv(
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
    block->template Add<ann::Convolution>(outMaps, 1, 1, 1, 1, 0, 0, "none", useBias);
  }

  void Block(
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
      block2->template Add<ann::Convolution>(outMaps, 1, 1, strides, strides, 0, 0, "none", false);
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
      merge->template Add<ann::Idenity>();

      this->template Add(merge);
    }
  }

  void makeModel()
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

    SeparableConv(this, 1024, 1536, 3, 1, 1);
    this->template Add<ann::BatchNorm>();

    if (includeTop)
    {
      this->template Add<ann::ReLU>();
      this->template Add<ann::AdaptiveMeanPooling>(1, 1);
      this->template Add<ann::Linear>(numClasses);
    }
  }

  //! Locally stored number of output classes.
  size_t numClasses;

  //! Locally stored if classifier layers are included or not.
  bool includeTop;
}; // XceptionType class.

// Convenience typedefs for different VGG layer.
typedef XceptionType<arma::mat> Xception;


} // namespace models
} // namespace mlpack

CEREAL_REGISTER_TYPE(mlpack::models::XceptionType<arma::mat>);

#include "xception_impl.hpp"

#endif
