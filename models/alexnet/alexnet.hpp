/**
 * @file alexnet.hpp
 * @author Shubham Agrawal
 * 
 * Definition of AlexNet model.
 * 
 * For more information, kindly refer to the following paper.
 * 
 * Paper for AlexNet.
 *
 * @code
 * @article{Krizhevsky2012,
 *  author = {Alex Krizhevsky, Ilya Sutskever, Geoffrey E. Hinton},
 *  title = {ImageNet Classification with Deep Convolutional Neural Networks},
 *  year = {2012},
 *  url = {https://proceedings.neurips.cc/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf}
 * }
 * @endcode
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MODELS_MODELS_ALEXNET_ALEXNET_HPP
#define MODELS_MODELS_ALEXNET_ALEXNET_HPP

#define MLPACK_ENABLE_ANN_SERIALIZATION
#include <mlpack.hpp>

namespace mlpack {
namespace models {

/**
 * Definition of a AlexNet CNN.
 * 
 * @tparam MatType Matrix representation to accept as input and use for
 *    computation.
 */
template<typename MatType = arma::mat>
class AlexNetType : public MultiLayer<MatType>
{
 public:
  /**
   * AlexNetType constructor intializes number of classes and weights.
   *
   * @param numClasses Number of classes to classify images into,
   *     only to be specified if includeTop is true.
   * @param includeTop Must be set to true if classifier layers are set.
   */
  AlexNetType(const size_t numClasses = 1000,
              const bool includeTop = true);

  //! Copy the given AlexNetType.
  AlexNetType(const AlexNetType& other);
  //! Take ownership of the layers of the given AlexNetType.
  AlexNetType(AlexNetType&& other);
  //! Copy the given AlexNetType.
  AlexNetType& operator=(const AlexNetType& other);
  //! Take ownership of the given AlexNetType.
  AlexNetType& operator=(AlexNetType&& other);

  //! Virtual destructor: delete all held layers.
  virtual ~AlexNetType()
  { /* Nothing to do here. */ }

  //! Create a copy of the AlexNetType (this is safe for polymorphic use).
  AlexNetType* Clone() const { return new AlexNetType(*this); }

  /**
   * Get the FFN object representing the network.
   * 
   * @tparam OutputLayerType The output layer type used to evaluate the network.
   * @tparam InitializationRuleType Rule used to initialize the weight matrix.
   */
  template<
    typename OutputLayerType = CrossEntropyError,
    typename InitializationRuleType = RandomInitialization
  >
  FFN<OutputLayerType, InitializationRuleType, MatType>* GetModel()
  {
    FFN<OutputLayerType, InitializationRuleType, MatType>* alexNet =
        new FFN<OutputLayerType, InitializationRuleType, MatType>();
    alexNet->Add(this);
    return alexNet;
  }

  //! Serialize the AlexNetType.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Generate the layers of the AlexNet.
  void MakeModel();

  //! Locally stored number of output classes.
  size_t numClasses;

  //! Locally stored if classidier layers are included or not.
  bool includeTop;
}; // AlexNetType class.

// Convenience typedefs for different AlexNet layer.
typedef AlexNetType<arma::mat> AlexNet;

} // namespace models
} // namespace mlpack

CEREAL_REGISTER_TYPE(mlpack::models::AlexNetType<arma::mat>);

#include "alexnet_impl.hpp"

#endif
