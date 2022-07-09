/**
 * @file vgg.hpp
 * @author Shubham Agrawal
 * 
 * Definition of VGG models.
 * 
 * For more information, kindly refer to the following paper.
 * 
 * Paper for VGG.
 *
 * @code
 * @article{Simonyan2014,
 *  author = {Karen Simonyan, Andrew Zisserman},
 *  title = {Very Deep Convolutional Networks for Large-Scale Image Recognition},
 *  year = {2014},
 *  url = {https://arxiv.org/pdf/1409.1556.pdf}
 * }
 * @endcode
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MODELS_MODELS_VGG_VGG_HPP
#define MODELS_MODELS_VGG_VGG_HPP

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
  typename MatType = arma::mat,
  size_t VGGVersion = 11,
  bool IsBatchNorm = false
>
class VGGType : public ann::MultiLayer<MatType>
{
 public:
  //! Create the VGGType layer.
  VGGType();

  /**
   * VGGType constructor intializes number of classes and weights.
   *
   * @param numClasses Optional number of classes to classify images into,
   *     only to be specified if includeTop is  true.
   * @param includeTop Must be set to true if weights are set.
   */
  VGGType(
    const size_t numClasses,
    const bool includeTop = true);
  
  //! Copy the given VGGType.
  VGGType(const VGGType& other);
  //! Take ownership of the layers of the given VGGType.
  VGGType(VGGType&& other);
  //! Copy the given VGGType.
  VGGType& operator=(const VGGType& other);
  //! Take ownership of the given VGGType.
  VGGType& operator=(VGGType&& other);

  //! Virtual destructor: delete all held layers.
  virtual ~VGGType()
  {
    // Nothing to do here. 
  }

  //! Create a copy of the VGGType (this is safe for polymorphic use).
  VGGType* Clone() const { return new VGGType(*this); }

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
    ann::FFN<OutputLayerType, InitializationRuleType, MatType>* vgg = 
        new ann::FFN<OutputLayerType, InitializationRuleType, MatType>();
    vgg->Add(this);
    return vgg;
  }

  //! Serialize the VGGType.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  void makeModel()
  {
    std::map<size_t, std::vector<size_t>> const construct {
      { 11, {64, 0, 128, 0, 256, 256, 0, 512, 512, 0, 512, 512, 0} },
      { 13, {64, 64, 0, 128, 128, 0, 256, 256, 0, 512, 512, 0, 512, 512, 0} },
      { 16, {64, 64, 0, 128, 128, 0, 256, 256, 256, 0, 512, 512, 512, 0, 512,
          512, 512, 0} },
      { 19, {64, 64, 0, 128, 128, 0, 256, 256, 256, 256, 0, 512, 512, 512, 512,
          0, 512, 512, 512, 512, 0} }
    };
    std::vector<size_t> layers = construct[VGGVersion];
    for (size_t i = 0; i < layers.size(); i++) 
    {
      if (layers[i] == 0)
      {
        this->template Add<ann::MaxPooling>(2, 2, 2, 2);
      }
      else
      {
        this->template Add<ann::Convolution>(layers[i], 3, 3, 1, 1, 1, 1);
        if (IsBatchNorm)
          this->template Add<ann::BatchNorm>();
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

  //! Locally stored number of output classes.
  size_t numClasses;

  //! Locally stored if classifier layers are included or not.
  bool includeTop;
}; // VGGType class.

// Convenience typedefs for different VGG layer.
typedef VGGType<arma::mat, 11, false> VGG11;
typedef VGGType<arma::mat, 13, false> VGG13;
typedef VGGType<arma::mat, 16, false> VGG16;
typedef VGGType<arma::mat, 19, false> VGG19;

typedef VGGType<arma::mat, 11, true> VGG11BN;
typedef VGGType<arma::mat, 13, true> VGG13BN;
typedef VGGType<arma::mat, 16, true> VGG16BN;
typedef VGGType<arma::mat, 19, true> VGG19BN;


} // namespace models
} // namespace mlpack

CEREAL_REGISTER_TYPE(mlpack::models::VGGType<arma::mat, 11, false>);
CEREAL_REGISTER_TYPE(mlpack::models::VGGType<arma::mat, 13, false>);
CEREAL_REGISTER_TYPE(mlpack::models::VGGType<arma::mat, 16, false>);
CEREAL_REGISTER_TYPE(mlpack::models::VGGType<arma::mat, 19, false>);

CEREAL_REGISTER_TYPE(mlpack::models::VGGType<arma::mat, 11, true>);
CEREAL_REGISTER_TYPE(mlpack::models::VGGType<arma::mat, 13, true>);
CEREAL_REGISTER_TYPE(mlpack::models::VGGType<arma::mat, 16, true>);
CEREAL_REGISTER_TYPE(mlpack::models::VGGType<arma::mat, 19, true>);


#include "vgg_impl.hpp"

#endif
