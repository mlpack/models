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
 * Definition of a Xception CNN.
 * 
 * @tparam MatType Matrix representation to accept as input and use for
 *    computation.
 */
template<
  typename MatType = arma::mat
>
class XceptionType : public ann::MultiLayer<MatType>
{
 public:
  /**
   * XceptionType constructor intializes number of classes and weights.
   *
   * @param numClasses Number of classes to classify images into,
   *     only to be specified if includeTop is true.
   * @param includeTop Must be set to true if classifier layers are set.
   */
  XceptionType(const size_t numClasses = 1000,
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
  { /* Nothing to do here. */ }

  //! Create a copy of the XceptionType (this is safe for polymorphic use).
  XceptionType* Clone() const { return new XceptionType(*this); }

  /**
   * Get the FFN object representing the network.
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
  /**
   * Adds Seperable Convolution to the given block.
   *
   * @param block Block to add the separable convolution to.
   * @param inMaps Number of input maps.
   * @param outMaps Number of output maps.
   * @param kernelSize Kernel size of the convolution.
   * @param stride Stride of the convolution.
   * @param padding Padding of the convolution.
   * @param useBias Whether to use bias in the convolution.
   */
  void SeparableConv(ann::MultiLayer<MatType>* block,
                     const size_t inMaps,
                     const size_t outMaps,
                     const size_t kernelSize,
                     const size_t stride = 1,
                     const size_t padding = 0,
                     const bool useBias = false);

  /**
   * Adds block.
   *
   * @param inMaps Number of input maps.
   * @param outMaps Number of output maps.
   * @param reps Number of repetitions of the convolution.
   * @param strides Stride of the convolution.
   * @param startWithRelu Whether to use ReLU at the start of the block.
   * @param growFirst Whether to map to outMaps at beginning only. If false,
   *                  it maps to inMaps at last convolution layer.
   */
  void Block(const size_t inMaps,
             const size_t outMaps,
             const size_t reps,
             const size_t strides = 1,
             const bool startWithRelu = true,
             const bool growFirst = true);

  //! Generate the layers of the Xception.
  void MakeModel();

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
