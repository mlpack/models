/**
 * @file mobilenet.hpp
 * @author Aakash Kaushik
 * 
 * Definition of MobileNet model.
 * 
 * For more information, kindly refer to the following paper.
 *
 * @code
 * @article{Andrew G2017,
 *  author = {Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko,
 *      Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam},
 *  title = {MobileNets: Efficient Convolutional Neural Networks for Mobile
 *      Vision Applications},
 *  year = {2017},
 *  url = {https://arxiv.org/pdf/1704.04861}
 * }
 * @endcode
 * 
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MODELS_MODELS_MOBILENET_MOBILENET_HPP
#define MODELS_MODELS_MOBILENET_MOBILENET_HPP

#define MLPACK_ENABLE_ANN_SERIALIZATION
#include <mlpack.hpp>

#include "separable_convolution.hpp"

#include "./../../utils/utils.hpp"

namespace mlpack {
namespace models {

/**
 * Definition of a MobileNet CNN.
 * 
 * @tparam OutputLayerType The output layer type used to evaluate the network.
 * @tparam InitializationRuleType Rule used to initialize the weight matrix.
 */
template<
  typename MatType = arma::mat
>
class MobileNetType : public MultiLayer<MatType>
{
 public:
  //! Create the MobileNet model.
  //MobileNetType();

  /**
   * MobileNetType constructor initializes input shape and number of classes.
   *
   * @param numClasses Number of classes to classify images into,
   *     only to be specified if includeTop is true.
   * @param includeTop Must be set to true if classifier layers are set.
   * * @param alpha Controls the number of output channels in pointwise
   *     convolution: outSize * depthMultiplier.
   * @param depthMultiplier Controls the number of output channels in depthwise
   *     convolution: inSize * depthMultiplier.
   */
  MobileNetType(const size_t numClasses = 1000,
                  const bool includeTop = true,
                  const float alpha = 1.0,
                  const size_t depthMultiplier = 1.0);

  //! Copy the given MobileNetType.
  MobileNetType(const MobileNetType& other);
  //! Take ownership of the layers of the given MobileNetType.
  MobileNetType(MobileNetType&& other);
  //! Copy the given MobileNetType.
  MobileNetType& operator=(const MobileNetType& other);
  //! Take ownership of the given MobileNetType.
  MobileNetType& operator=(MobileNetType&& other);

  //! Virtual destructor: delete all held layers.
  virtual ~MobileNetType()
  { /* Nothing to do here. */ }

  //! Create a copy of the MobileNetType (this is safe for polymorphic use).
  MobileNetType* Clone() const { return new MobileNetType(*this); }

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
    FFN<OutputLayerType, InitializationRuleType, MatType>* mobileNet =
        new FFN<OutputLayerType, InitializationRuleType, MatType>();
    mobileNet->Add(this);
    return mobileNet;
  }

  //! Serialize the MobileNetType.
  template<typename Archive>
  void serialize(Archive& ar, const uint32_t /* version */);

 private:
  //! Generate the layers of the MobileNet.
  void MakeModel();

  //! Locally stored number of output classes.
  size_t numClasses;

  //! Locally stored if classifier layers are included or not.
  bool includeTop;

  //! Locally stored alpha for mobileNet block creation.
  float alpha;

  //! Locally stored Depth multiplier for mobileNet block creation.
  float depthMultiplier;

  //! Locally stored map to construct mobileNetV1 blocks.
  std::map<size_t, size_t> mobileNetConfig = {
                                                {128, 2},
                                                {256, 2},
                                                {512, 6},
                                                {1024, 2},
                                              };
}; // MobileNetType class

// convenience typedef.
typedef MobileNetType<arma::mat> Mobilenet;

} // namespace models
} // namespace mlpack

CEREAL_REGISTER_TYPE(mlpack::models::MobileNetType<arma::mat>);

#include "mobilenet_impl.hpp"

#endif
