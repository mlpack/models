/**
 * @file models/transformer/encoder.hpp
 * @author Mikhail Lozhnikov
 * @author Mrityunjay Tripathi
 *
 * Definition of the Transformer Encoder layer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MODELS_TRANSFORMER_ENCODER_HPP
#define MODELS_TRANSFORMER_ENCODER_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/methods/ann/layer/layer_types.hpp>
#include <mlpack/methods/ann/layer/base_layer.hpp>
#include <mlpack/methods/ann/regularizer/no_regularizer.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * The Transformer Encoder layer has two sub-layers. The first is a multi-head
 * self-attention mechanism, and the second is a simple, position-wise
 * fully connected feed-forward network. We employ a residual connection around
 * each of the two sub-layers, followed by layer normalization. Hence the output
 * of each sub-layer is 'LayerNorm(x + Sublayer(x))', where 'Sublayer(x)' is the
 * function implemented by the sub-layer itself. To facilitate these residual
 * connections, all sub-layers in the model, as well as the embedding layers,
 * produce outputs of dimension 'dModel'.
 *
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
class TransformerEncoder
{
 public:
  /**
   * Create the TransformerEncoder object using the specified parameters.
   *
   * @param numLayers The number of encoder blocks.
   * @param srcSeqLen Source Sequence Length.
   * @param dModel The number of features in the input. It is same as the
   *               `embedDim` in `MultiheadAttention` layer.
   * @param numHeads The number of attention heads.
   * @param dimFFN The dimentionality of feedforward network.
   * @param dropout The dropout rate.
   * @param attentionMask The attention mask to be applied to the sequences.
   * @param keyPaddingMask The key padding mask applied to the sequences.
   */
  TransformerEncoder(const size_t numLayers,
                     const size_t srcSeqLen,
                     const size_t dModel = 512,
                     const size_t numHeads = 2,
                     const size_t dimFFN = 1024,
                     const double dropout = 0.1,
                     const InputDataType& attentionMask = InputDataType(),
                     const InputDataType& keyPaddingMask = InputDataType());

  /**
   * Get the Transformer Encoder Model.
   */
  Sequential<InputDataType, OutputDataType, false>* Model()
  {
    return encoder;
  }

  /**
   * Load the encoder block from a local directory.
   *
   * @param filepath The location of the stored model.
   */
  void LoadModel(const std::string& filepath);

  /**
   * Save the encoder block locally.
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
  /**
   * The method adds attention block to the encoder block.
   */
  void AttentionBlock()
  {
    Concat<>* input = new Concat<>();
    input->Add<IdentityLayer<>>();
    input->Add<IdentityLayer<>>();
    input->Add<IdentityLayer<>>();

    /* Self attention layer. */
    Sequential<>* selfAttention = new Sequential<>();
    selfAttention->Add(input);
    selfAttention->Add<MultiheadAttention<
        InputDataType, OutputDataType, RegularizerType>
        >(srcSeqLen, srcSeqLen, dModel, numHeads);

    /* This layer adds a residual connection. */
    AddMerge<>* residualAdd = new AddMerge<>();
    residualAdd->Add(selfAttention);
    residualAdd->Add<IdentityLayer<>>();

    encoder->Add(residualAdd);
    encoder->Add<LayerNorm<>>(dModel * srcSeqLen);
  }

  /**
   * This method adds position-wise feed forward block to the encoder.
   */
  void PositionWiseFFNBlock()
  {
    Sequential<>* positionWiseFFN = new Sequential<>();
    positionWiseFFN->Add<Linear3D<>>(dModel, dimFFN);
    positionWiseFFN->Add<ActivationFunction>();
    positionWiseFFN->Add<Linear3D<>>(dimFFN, dModel);
    positionWiseFFN->Add<Dropout<>>(dropout);

    /* This layer adds a residual connection. */
    AddMerge<>* residualAdd = new AddMerge<>();
    residualAdd->Add(positionWiseFFN);
    residualAdd->Add<IdentityLayer<>>();

    encoder->Add(residualAdd);
    encoder->Add<LayerNorm<>>(dModel * srcSeqLen);
  }

  //! Locally-stored number of encoder blocks.
  size_t numLayers;

  //! Locally-stored source sequence length.
  size_t srcSeqLen;

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

  //! Locally-stored encoder block.
  Sequential<InputDataType, OutputDataType, false>* encoder;

}; // class TransformerEncoder

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "encoder_impl.hpp"

#endif
