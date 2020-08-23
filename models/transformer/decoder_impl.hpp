/**
 * @file models/transformer/decoder_impl.hpp
 * @author Mikhail Lozhnikov
 * @author Mrityunjay Tripathi
 *
 * Implementation of the Transformer Decoder class.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MODELS_TRANSFORMER_DECODER_IMPL_HPP
#define MODELS_TRANSFORMER_DECODER_IMPL_HPP

#include "decoder.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename ActivationFunction, typename RegularizerType>
TransformerDecoder<ActivationFunction, RegularizerType>::TransformerDecoder() :
    tgtSeqLen(0),
    srcSeqLen(0),
    dModel(0),
    numHeads(0),
    dimFFN(0),
    dropout(0),
    ownMemory(true)
{
  // Nothing to do here.
}

template<typename ActivationFunction, typename RegularizerType>
TransformerDecoder<ActivationFunction, RegularizerType>::TransformerDecoder(
    const size_t numLayers,
    const size_t tgtSeqLen,
    const size_t srcSeqLen,
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
    dModel(dModel),
    numHeads(numHeads),
    dimFFN(dimFFN),
    dropout(dropout),
    attentionMask(attentionMask),
    keyPaddingMask(keyPaddingMask),
    ownMemory(ownMemory)
{
  decoder = new Sequential<>(false);

  for (size_t n = 0; n < numLayers; ++n)
  {
    if (n + 1 == numLayers)
    {
      decoder->Add(AttentionBlock());
      decoder->Add(PositionWiseFFNBlock());
      break;
    }

    Sequential<>* decoderBlock = new Sequential<>(false);
    decoderBlock->Add(AttentionBlock());
    decoderBlock->Add(PositionWiseFFNBlock());

    Concat<>* concatQueryKey = new Concat<>();
    concatQueryKey->Add(decoderBlock);
    concatQueryKey->Add<Subview<>>(1, dModel * tgtSeqLen, -1, 0, -1);

    decoder->Add(concatQueryKey);
  }
}

template<typename ActivationFunction, typename RegularizerType>
void TransformerDecoder<ActivationFunction, RegularizerType>::
LoadModel(const std::string& filepath)
{
  data::Load(filepath, "TransformerDecoder", decoder);
  std::cout << "Loaded model" << std::endl;
}

template<typename ActivationFunction, typename RegularizerType>
void TransformerDecoder<ActivationFunction, RegularizerType>::
SaveModel(const std::string& filepath)
{
  std::cout << "Saving model" << std::endl;
  data::Save(filepath, "TransformerDecoder", decoder);
  std::cout << "Model saved in " << filepath << std::endl;
}

} // namespace ann
} // namespace mlpack

#endif
