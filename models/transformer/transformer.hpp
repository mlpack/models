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
 * @tparam InputDataType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputDataType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template <
  typename ActivationFunction = ReLULayer<>,
  typename RegularizerType = NoRegularizer,
  typename InputDataType = arma::mat,
  typename OutputDataType = arma::mat
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
              const InputDataType& attentionMask = InputDataType(),
              const InputDataType& keyPaddingMask = InputDataType());

  /**
   * Get the Transformer Encoder Model.
   */
  Sequential<InputDataType, OutputDataType, false>* Model()
  {
    return transformer;
  }

  /**
   * Load the Transformer model from a local directory.
   *
   * @param filepath The location of the stored model.
   */
  void LoadModel(const std::string& filepath);

  /**
   * Save the Transformer model locally.
   *
   * @param filepath The location where the model is to be saved.
   */
  void SaveModel(const std::string& filepath);

  //! Get the attention mask.
  InputDataType const& AttentionMask() const { return attentionMask; }

  //! Modify the attention mask.
  InputDataType& AttentionMask() { return attentionMask; }

  //! Get the key padding mask.
  InputDataType const& KeyPaddingMask() const { return keyPaddingMask; }

  //! Modify the key padding mask.
  InputDataType& KeyPaddingMask() { return keyPaddingMask; }

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

  //! Locally-stored number of input units.
  size_t dModel;

  //! Locally-stored number of output units.
  size_t numHeads;

  //! Locally-stored weight object.
  size_t dimFFN;

  //! Locally-stored weight parameters.
  double dropout;

  //! Locally-stored attention mask.
  InputDataType attentionMask;

  //! Locally-stored key padding mask.
  InputDataType keyPaddingMask;

  //! Locally-stored transformer model.
  Sequential<InputDataType, OutputDataType>* transformer;

}; // class Transformer

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "transformer_impl.hpp"

#endif
