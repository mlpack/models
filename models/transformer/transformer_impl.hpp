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

template <typename ActivationFunction, typename RegularizerType,
    typename InputDataType, typename OutputDataType>
Transformer<ActivationFunction, RegularizerType, InputDataType,
OutputDataType>::Transformer(
    const size_t numLayers,
    const size_t tgtSeqLen,
    const size_t srcSeqLen,
    const size_t tgtVocabSize,
    const size_t srcVocabSize,
    const size_t dModel,
    const size_t numHeads,
    const size_t dimFFN,
    const double dropout,
    const InputDataType& attentionMask,
    const InputDataType& keyPaddingMask) :
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
    keyPaddingMask(keyPaddingMask)
{
  transformer = new Sequential<>();

  Sequential<>* encoder = new Sequential<>();

  // Pull out the sequences of source language which is stacked above in the
  // input matrix. Here 'lastCol = -1' denotes upto last batch of input matrix.
  encoder->Add<Subview<>>(1, 0, srcSeqLen - 1, 0, -1);
  encoder->Add<Lookup<>>(srcVocabSize, dModel);
  encoder->Add<PositionalEncoding<>>(dModel, srcSeqLen);

  Sequential<>* encoderStack = mlpack::ann::TransformerEncoder<
    ActivationFunction, RegularizerType, InputDataType, OutputDataType>(
      numLayers,
      srcSeqLen,
      dModel,
      numHeads,
      dimFFN,
      dropout,
      attentionMask,
      keyPaddingMask,
  ).Model();

  encoder->Add(encoderStack);

  Sequential<>* decoderPE = new Sequential<>();

  // Pull out the sequences of target language which is stacked below in the
  // input matrix. Here 'lastRow = -1' and 'lastCol = -1' denotes upto last
  // row and last batch of the input matrix respectively.
  decoderPE->Add<Subview<>>(1, srcSeqLen, -1, 0, -1);
  decoderPE->Add<Lookup<>>(tgtVocabSize, dModel);
  decoderPE->Add<PositionalEncoding<>>(dModel, tgtSeqLen);

  Concat<>* encoderDecoderConcat = new Concat<>();
  encoderDecoderConcat->Add(encoder);
  encoderDecoderConcat->Add(decoderPE);

  Sequential<>* decoder = new Sequential<>();
  decoder->Add(encoderDecoderConcat);

  Sequential<>* decoderStack = mlpack::ann::TransformerDecoder<
    ActivationFunction, RegularizerType, InputDataType, OutputDataType>(
      numLayers,
      tgtSeqLen,
      srcSeqLen,
      dModel,
      numHeads,
      dimFFN,
      dropout,
      attentionMask,
      keyPaddingMask,
  ).Model();

  decoder->Add(decoderStack);
  transformer->Add(decoder);
}

template <typename ActivationFunction, typename RegularizerType,
    typename InputDataType, typename OutputDataType>
void Transformer<ActivationFunction, RegularizerType, InputDataType,
OutputDataType>::LoadModel(const std::string& filePath)
{
  data::Load(filePath, "Transformer", transformer.Model());
  std::cout << "Loaded model" << std::endl;
}

template <typename ActivationFunction, typename RegularizerType,
    typename InputDataType, typename OutputDataType>
void Transformer<ActivationFunction, RegularizerType, InputDataType,
OutputDataType>::SaveModel(const std::string& filePath)
{
  std::cout << "Saving model" << std::endl;
  data::Save(filePath, "Transformer", transformer.Model());
  std::cout << "Model saved in " << filePath << std::endl;
}

} // namespace ann
} // namespace mlpack

#endif
