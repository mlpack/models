/**
 * @file models/transformer/decoder.hpp
 * @author Mikhail Lozhnikov
 * @author Mrityunjay Tripathi
 *
 * Definition of the Transformer Decoder layer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MODELS_TRANSFORMER_DECODER_HPP
#define MODELS_TRANSFORMER_DECODER_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/methods/ann/layer/layer_types.hpp>
#include <mlpack/methods/ann/layer/base_layer.hpp>
#include <mlpack/methods/ann/regularizer/no_regularizer.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * In addition to the two sub-layers in each encoder layer, the decoder inserts
 * a third sub-layer, which performs multi-head attention over the output of the
 * encoder stack. Similar to the encoder, we employ residual connections around
 * each of the sub-layers, followed by layer normalization. We also modify the
 * self-attention sub-layer in the decoder stack to prevent positions from
 * attending to subsequent positions. This masking, combined with fact that the
 * output embeddings are offset by one position, ensures that the predictions
 * for position i can depend only on the known outputs at positions less than i.
 *
 * @tparam ActivationFunction The type of the activation function to be used in
 *         the position-wise feed forward neural network.
 * @tparam RegularizerType The type of regularizer to be applied to layer
 *         parameters.
 */
template <
  typename ActivationFunction = ReLULayer<>,
  typename RegularizerType = NoRegularizer
>
class TransformerDecoder
{
 public:
  TransformerDecoder();

  /**
   * Create the TransformerDecoder object using the specified parameters.
   *
   * @param numLayers The number of decoder blocks.
   * @param tgtSeqLen Target Sequence Length.
   * @param srcSeqLen Source Sequence Length.
   * @param memoryModule The last Encoder module.
   * @param dModel The number of features in the input. Also, same as the
   *        `embedDim` in `MultiheadAttention` layer.
   * @param numHeads The number of attention heads.
   * @param dimFFN The dimentionality of feedforward network.
   * @param dropout The dropout rate.
   * @param attentionMask The attention mask used to black-out future sequences.
   * @param keyPaddingMask The padding mask used to black-out particular token.
   * @param ownMemory Whether to delete the pointer-type decoder object.
   */
  TransformerDecoder(const size_t numLayers,
                     const size_t tgtSeqLen,
                     const size_t srcSeqLen,
                     const size_t dModel = 512,
                     const size_t numHeads = 8,
                     const size_t dimFFN = 1024,
                     const double dropout = 0.1,
                     const arma::mat& attentionMask = arma::mat(),
                     const arma::mat& keyPaddingMask = arma::mat(),
                     const bool ownMemory = false);

  /**
   * Destructor.
   */
  ~TransformerDecoder()
  {
    if (ownMemory)
      delete decoder;
  }

  /**
   * Copy constructor.
   */
  TransformerDecoder(const TransformerDecoder& ) = delete;

  /**
   * Move constructor.
   */
  TransformerDecoder(const TransformerDecoder&& ) = delete;

  /**
   * Copy assignment operator.
   */
  TransformerDecoder& operator = (const TransformerDecoder& ) = delete;

  /**
   * Get the Transformer Decoder model.
   */
  Sequential<>* Model() { return decoder; }

  /**
   * Load the network from a local directory.
   *
   * @param filepath The location of the stored model.
   */
  void LoadModel(const std::string& filepath);

  /**
   * Save the network locally.
   *
   * @param filepath The location where the model is to be saved.
   */
  void SaveModel(const std::string& filepath);

  //! Get the attention mask.
  arma::mat const& AttentionMask() const { return attentionMask; }

  //! Modify the attention mask.
  arma::mat& AttentionMask() { return attentionMask; }

  //! Get the key padding mask.
  arma::mat const& KeyPaddingMask() const { return keyPaddingMask; }

  //! Modify the key padding mask.
  arma::mat& KeyPaddingMask() { return keyPaddingMask; }

 private:
  /**
   * This method adds the attention block to the decoder.
   */
  Sequential<>* AttentionBlock()
  {
    Sequential<>* decoderBlockBottom = new Sequential<>(false);
    decoderBlockBottom->Add<Subview<>>(1, 0, dModel * tgtSeqLen - 1, 0, -1);

    // Broadcast the incoming input to decoder
    // i.e. query into (query, key, value).
    Concat<>* decoderInput = new Concat<>();
    decoderInput->Add<IdentityLayer<>>();
    decoderInput->Add<IdentityLayer<>>();
    decoderInput->Add<IdentityLayer<>>();

    // Masked Self attention layer.
    Sequential<>* maskedSelfAttention = new Sequential<>(false);
    maskedSelfAttention->Add(decoderInput);
    maskedSelfAttention->Add<MultiheadAttention<
        arma::mat, arma::mat, RegularizerType>>(
          tgtSeqLen,
          tgtSeqLen,
          dModel,
          numHeads,
          attentionMask);

    // Residual connection.
    AddMerge<>* residualAdd1 = new AddMerge<>();
    residualAdd1->Add(maskedSelfAttention);
    residualAdd1->Add<IdentityLayer<>>();

    decoderBlockBottom->Add(residualAdd1);

    // Add the LayerNorm layer with required parameters.
    decoderBlockBottom->Add<LayerNorm<>>(dModel * tgtSeqLen);

    // This layer broadcasts the output of encoder i.e. key into (key, value).
    Concat<>* broadcastEncoderOutput = new Concat<>();
    broadcastEncoderOutput->Add<Subview<>>(1, dModel * tgtSeqLen, -1, 0, -1);
    broadcastEncoderOutput->Add<Subview<>>(1, dModel * tgtSeqLen, -1, 0, -1);

    // This layer concatenates the output of the bottom decoder block (query)
    // and the output of the encoder (key, value).
    Concat<>* encoderDecoderAttentionInput = new Concat<>();
    encoderDecoderAttentionInput->Add(decoderBlockBottom);
    encoderDecoderAttentionInput->Add(broadcastEncoderOutput);

    // Encoder-decoder attention.
    Sequential<>* encoderDecoderAttention = new Sequential<>(false);
    encoderDecoderAttention->Add(encoderDecoderAttentionInput);
    encoderDecoderAttention->Add<MultiheadAttention<
        arma::mat, arma::mat, RegularizerType>>(
          tgtSeqLen,
          srcSeqLen,
          dModel,
          numHeads,
          arma::mat(), // No attention mask to encoder-decoder attention.
          keyPaddingMask);

    // Residual connection.
    AddMerge<>* residualAdd2 = new AddMerge<>();
    residualAdd2->Add(encoderDecoderAttention);
    residualAdd2->Add<IdentityLayer<>>();

    Sequential<>* decoderBlock = new Sequential<>(false);
    decoderBlock->Add(residualAdd2);
    decoderBlock->Add<LayerNorm<>>(dModel * tgtSeqLen);
    return decoderBlock;
  }

  /**
   * This method adds the position-wise feed forward network to the decoder.
   */
  Sequential<>* PositionWiseFFNBlock()
  {
    Sequential<>* positionWiseFFN = new Sequential<>(false);
    positionWiseFFN->Add<Linear3D<>>(dModel, dimFFN);
    positionWiseFFN->Add<ActivationFunction>();
    positionWiseFFN->Add<Linear3D<>>(dimFFN, dModel);
    positionWiseFFN->Add<Dropout<>>(dropout);

    /* Residual connection. */
    AddMerge<>* residualAdd = new AddMerge<>();
    residualAdd->Add(positionWiseFFN);
    residualAdd->Add<IdentityLayer<>>();

    Sequential<>* decoderBlock = new Sequential<>(false);
    decoderBlock->Add(residualAdd);
    return decoderBlock;
  }

  //! Locally-stored number of decoder layers.
  size_t numLayers;

  //! Locally-stored target sequence length.
  size_t tgtSeqLen;

  //! Locally-stored source sequence length.
  size_t srcSeqLen;

  //! Locally-stored number of features in the input.
  size_t dModel;

  //! Locally-stored number of attention heads.
  size_t numHeads;

  //! Locally-stored dimensionality of position-wise feed forward network.
  size_t dimFFN;

  //! Locally-stored dropout rate.
  double dropout;

  //! Locally-stored attention mask.
  arma::mat attentionMask;

  //! Locally-stored key padding mask.
  arma::mat keyPaddingMask;

  //! Whether to delete pointer-type decoder object.
  bool ownMemory;

  //! Locally-stored complete decoder network.
  Sequential<>* decoder;
}; // class TransformerDecoder

} // namespace ann
} // namespace mlpack

// Include implementation.
#include "decoder_impl.hpp"

#endif
