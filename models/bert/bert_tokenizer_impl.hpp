/**
 * @file models/bert/bert_tokenizer_impl.hpp
 * @author Mrityunjay Tripathi
 *
 * Implementation of the BERT Tokenizer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MODELS_BERT_BERT_TOKENIZER_IMPL_HPP
#define MODELS_BERT_BERT_TOKENIZER_IMPL_HPP

#include "bert_tokenizer.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

BertTokenizer::BertTokenizer() :
    vocabFile(""),
    lowerCase(true),
    basicTokenize(true),
    unkToken("[UNK]"),
    sepToken("[SEP]"),
    padToken("[PAD]"),
    clsToken("[CLS]"),
    maskToken("[MASK]")
{
  // Nothing to do here.
}

BertTokenizer::BertTokenizer(
    const std::string vocabFile,
    const bool lowerCase,
    const bool basicTokenize,
    const std::vector<std::string> neverSplit,
    const std::string unkToken,
    const std::string sepToken,
    const std::string padToken,
    const std::string clsToken,
    const std::string maskToken) :
    vocabFile(vocabFile),
    lowerCase(lowerCase),
    basicTokenize(basicTokenize),
    neverSplit(neverSplit),
    unkToken(unkToken),
    sepToken(sepToken),
    padToken(padToken),
    clsToken(clsToken),
    maskToken(maskToken)
{
  // code here.
}

} // namespace ann
} // namespace mlpack

#endif
