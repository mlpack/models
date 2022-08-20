/**
 * @file squeezenet.hpp
 * @author Shubham Agrawal
 * 
 * Definition of SqueezeNet models.
 * 
 * For more information, kindly refer to the following paper.
 * 
 * Paper for SqueezeNet.
 *
 * @code
 * @article{Iandola2016,
 *  author = {Forrest N. Iandola, Song Han, Matthew W. Moskewicz, Khalid Ashraf,
 *            William J. Dally, Kurt Keutzer},
 *  title = {SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and
 *          <0.5MB model size},
 *  year = {2016},
 *  url = {https://arxiv.org/pdf/1602.07360.pdf}
 * }
 * @endcode
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MODELS_MODELS_SQUEEZENET_SQUEEZENET_HPP
#define MODELS_MODELS_SQUEEZENET_SQUEEZENET_HPP

#include <mlpack/core.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/layer/layer_types.hpp>
#include <mlpack/methods/ann/init_rules/random_init.hpp>
#include <mlpack/methods/ann/loss_functions/binary_cross_entropy_loss.hpp>

namespace mlpack {
namespace models {

/**
 * Definition of a SqueezeNet CNN.
 * 
 * NOTE: Note that output size will be 1x1xN. Here, N is number of classes.
 *       Especially, if we are using this network as layer in a network, then
 *       the output size of the network will matter.
 * 
 * @tparam MatType Matrix representation to accept as input and use for
 *    computation.
 * @tparam SqueezeNetVersion Version of SqueeeNet (Possible Config - 0, 1).
 */
template<
  typename MatType = arma::mat,
  size_t SqueezeNetVersion = 0
>
class SqueezeNetType : public ann::MultiLayer<MatType>
{
 public:
  //! Create the SqueezeNetType layer.
  SqueezeNetType();

  /**
   * SqueezeNetType constructor intializes number of classes and weights.
   *
   * @param numClasses Optional number of classes to classify images into,
   *     only to be specified if includeTop is true.
   * @param includeTop Must be set to true if classifier layers are set.
   */
  SqueezeNetType(const size_t numClasses,
                 const bool includeTop = true);

  //! Copy the given SqueezeNetType.
  SqueezeNetType(const SqueezeNetType& other);
  //! Take ownership of the layers of the given SqueezeNetType.
  SqueezeNetType(SqueezeNetType&& other);
  //! Copy the given SqueezeNetType.
  SqueezeNetType& operator=(const SqueezeNetType& other);
  //! Take ownership of the given SqueezeNetType.
  SqueezeNetType& operator=(SqueezeNetType&& other);

  //! Virtual destructor: delete all held layers.
  virtual ~SqueezeNetType()
  { /* Nothing to do here. */ }

  //! Create a copy of the SqueezeNetType (this is safe for polymorphic use).
  SqueezeNetType* Clone() const { return new SqueezeNetType(*this); }

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
    ann::FFN<OutputLayerType, InitializationRuleType, MatType>* squeezeNet =
        new ann::FFN<OutputLayerType, InitializationRuleType, MatType>();
    squeezeNet->Add(this);
    return squeezeNet;
  }

  //! Serialize the SqueezeNetType.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  /**
   * Adds Fire Block.
   *
   * @param squeezePlanes Number of squeeze maps.
   * @param expand1x1Planes Number of expansion maps with 1x1 kernel.
   * @param expand3x3Planes Number of expansion maps with 3x3 kernel.
   */
  void Fire(const size_t squeezePlanes,
            const size_t expand1x1Planes,
            const size_t expand3x3Planes);

  //! Generate the layers of the SqueezeNet.
  void MakeModel();

  //! Locally stored number of output classes.
  size_t numClasses;

  //! Locally stored if classifier layers are included or not.
  bool includeTop;
}; // SqueezeNetType class.

// Convenience typedefs for different SqueezeNet layer.
typedef SqueezeNetType<arma::mat, 0> SqueezeNet0;

typedef SqueezeNetType<arma::mat, 1> SqueezeNet1;

} // namespace models
} // namespace mlpack

CEREAL_REGISTER_TYPE(mlpack::models::SqueezeNetType<arma::mat, 0>);
CEREAL_REGISTER_TYPE(mlpack::models::SqueezeNetType<arma::mat, 1>);

#include "squeezenet_impl.hpp"

#endif
