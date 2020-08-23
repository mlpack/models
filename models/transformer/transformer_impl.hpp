/**
 * @file models/transformer/transformer_impl.hpp
 * @author Mikhail Lozhnikov
 * @author Mrityunjay Tripathi
 *
 * Implementation of the Transformer model.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MODELS_TRANSFORMER_TRANSFORMER_IMPL_HPP
#define MODELS_TRANSFORMER_TRANSFORMER_IMPL_HPP

#include "transformer.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template <typename ActivationFunction, typename RegularizerType>
Transformer<ActivationFunction, RegularizerType>::Transformer(
    const size_t numLayers,
    const size_t tgtSeqLen,
    const size_t srcSeqLen,
    const size_t tgtVocabSize,
    const size_t srcVocabSize,
    const size_t dModel,
    const size_t numHeads,
    const size_t dimFFN,
    const double dropout,
    const arma::mat& attentionMask,
    const arma::mat& keyPaddingMask,
    const bool ownMemory) :
    numLayers(numLayers),
    tgtSeqLen(tgtSeqLen),
    srcSeqLen(srcSeqLen),
    tgtVocabSize(tgtVocabSize),
    srcVocabSize(srcVocabSize),
    dModel(dModel),
    numHeads(numHeads),
    dimFFN(dimFFN),
    dropout(dropout),
    attentionMask(attentionMask),
    keyPaddingMask(keyPaddingMask),
    ownMemory(ownMemory)
{
  transformer = new Sequential<>(false);

  Sequential<>* encoder = new Sequential<>(false);

  // Pull out the sequences of source language which is stacked above in the
  // input matrix. Here 'lastCol = -1' denotes upto last batch of input matrix.
  encoder->Add<Subview<>>(1, 0, srcSeqLen - 1, 0, -1);
  encoder->Add<Lookup<>>(srcVocabSize, dModel);
  encoder->Add<PositionalEncoding<>>(dModel, srcSeqLen);

  Sequential<>* encoderStack = mlpack::ann::TransformerEncoder<
    ActivationFunction, RegularizerType>(
      numLayers,
      srcSeqLen,
      dModel,
      numHeads,
      dimFFN,
      dropout,
      attentionMask,
      keyPaddingMask).Model();

  encoder->Add(encoderStack);

  Sequential<>* decoderPE = new Sequential<>(false);

  // Pull out the sequences of target language which is stacked below in the
  // input matrix. Here 'lastRow = -1' and 'lastCol = -1' denotes upto last
  // row and last batch of the input matrix respectively.
  decoderPE->Add<Subview<>>(1, srcSeqLen, -1, 0, -1);
  decoderPE->Add<Lookup<>>(tgtVocabSize, dModel);
  decoderPE->Add<PositionalEncoding<>>(dModel, tgtSeqLen);

  Concat<>* encoderDecoderConcat = new Concat<>();
  encoderDecoderConcat->Add(encoder);
  encoderDecoderConcat->Add(decoderPE);

  transformer->Add(encoderDecoderConcat);

  Sequential<>* decoderStack = mlpack::ann::TransformerDecoder<
    ActivationFunction, RegularizerType>(
      numLayers,
      tgtSeqLen,
      srcSeqLen,
      dModel,
      numHeads,
      dimFFN,
      dropout,
      attentionMask,
      keyPaddingMask).Model();

  transformer->Add(decoderStack);
}

} // namespace ann
} // namespace mlpack

#endif
