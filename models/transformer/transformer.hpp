/**
 * @file models/transformer/transformer.hpp
 * @author Mikhail Lozhnikov
 * @author Mrityunjay Tripathi
 *
 * Definition of the Transformer model.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MODELS_TRANSFORMER_TRANSFORMER_HPP
#define MODELS_TRANSFORMER_TRANSFORMER_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/layer/layer_types.hpp>
#include <mlpack/methods/ann/layer/base_layer.hpp>
#include <mlpack/methods/ann/regularizer/no_regularizer.hpp>

#include "encoder.hpp"
#include "decoder.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * @tparam ActivationType The type of activation function to be used in the
 *         position-wise feed forward neural network.
 * @tparam RegularizerType The regularizer type to be applied on layer
 *         parameters.
 */
template <
  typename ActivationFunction = ReLULayer<>,
  typename RegularizerType = NoRegularizer
>
class Transformer
{
 public:
  /**
   * Create the Transformer object using the specified parameters.
   *
   * @param numLayers The number of encoder and decoder layers.
   * @param tgtSeqLen Target Sequence Length.
   * @param srcSeqLen Source Sequence Length.
   * @param tgtVocabSize Target vocabulary size.
   * @param srcVocabSize Source vocabulary size.
   * @param dModel The number of features in the input. Also, same as the
   *               `embedDim` in `MultiheadAttention` layer.
   * @param numHeads The number of attention heads.
   * @param dimFFN The dimentionality of feedforward network.
   * @param dropout The dropout rate.
   * @param attentionMask The attention mask to be applied to the sequences.
   * @param keyPaddingMask The key padding mask applied to the sequences.
   * @param ownMemory Whether to delete pointer-type transformer object.
   */
  Transformer(const size_t numLayers,
              const size_t tgtSeqLen,
              const size_t srcSeqLen,
              const size_t tgtVocabSize,
              const size_t srcVocabSize,
              const size_t dModel = 512,
              const size_t numHeads = 12,
              const size_t dimFFN = 1024,
              const double dropout = 0.1,
              const arma::mat& attentionMask = arma::mat(),
              const arma::mat& keyPaddingMask = arma::mat(),
              const bool ownMemory = false);

  /**
   * Destructor.
   */
  ~Transformer()
  {
    if (ownMemory)
      delete transformer;
  }

  /**
   * Copy constructor.
   */
  Transformer(const Transformer& /* transformer */) = delete;

  /**
   * Move constructor.
   */
  Transformer(const Transformer&& /* transformer */) = delete;

  /**
   * Copy assignment operator.
   */
  Transformer& operator = (const Transformer& /* transformer */) = delete;

  /**
   * Get the Transformer Encoder Model.
   */
  Sequential<>* Model()
  {
    return transformer;
  }

  //! Get the attention mask.
  arma::mat const& AttentionMask() const { return attentionMask; }

  //! Modify the attention mask.
  arma::mat& AttentionMask() { return attentionMask; }

  //! Get the key padding mask.
  arma::mat const& KeyPaddingMask() const { return keyPaddingMask; }

  //! Modify the key padding mask.
  arma::mat& KeyPaddingMask() { return keyPaddingMask; }

 private:
  //! Locally-stored number of encoder and decoder layers.
  size_t numLayers;

  //! Locally-stored target sequence length.
  size_t tgtSeqLen;

  //! Locally-stored source sequence length.
  size_t srcSeqLen;

  //! Locally-stored vocabulary size of the target.
  size_t tgtVocabSize;

  //! Locally-stored vocabulary size of the source.
  size_t srcVocabSize;

  //! Locally-stored number of features in the input.
  size_t dModel;

  //! Locally-stored number attention heads.
  size_t numHeads;

  //! Locally-stored dimensionality of the position-wise feed forward network.
  size_t dimFFN;

  //! Locally-stored dropout rate.
  double dropout;

  //! Locally-stored attention mask.
  arma::mat attentionMask;

  //! Locally-stored key padding mask.
  arma::mat keyPaddingMask;

  //! Whether to delete the pointer-type transformer object.
  bool ownMemory;

  //! Locally-stored transformer model.
  Sequential<>* transformer;
}; // class Transformer

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "transformer_impl.hpp"

#endif
