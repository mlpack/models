/**
 * @file models/bert/bert_tokenizer.hpp
 * @author Mrityunjay Tripathi
 *
 * Definition of the BERT Tokenizer.
 *
 * @code
 * @article{Wolf2019HuggingFacesTS,
 * title = {HuggingFace's Transformers: State-of-the-art Natural Language
 *          Processing},
 * author = {Thomas Wolf and Lysandre Debut and Victor Sanh and Julien Chaumond
 *           and Clement Delangue and Anthony Moi and Pierric Cistac and
 *           Tim Rault and R'emi Louf and Morgan Funtowicz and Jamie Brew},
 * journal = {ArXiv},
 * year = {2019},
 * volume = {abs/1910.03771}
 * }
 * @endcode
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MODELS_BERT_BERT_TOKENIZER_HPP
#define MODELS_BERT_BERT_TOKENIZER_HPP

#include <mlpack/prereqs.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * @tparam InputDataType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputDataType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template <
  typename InputDataType = arma::mat,
  typename OutputDataType = arma::mat
>
class BertTokenizer
{
 public:
  BertTokenizer();

  /**
   * Create the TransformerDecoder object using the specified parameters.
   *
   * @param vocabFile Location of file containing the vocabulary.
   * @param lowerCase Whether to turn each token to lower case.
   * @param basicTokenize Whether to do basic tokenization before WordPiece.
   * @param neverSplit Tokens which will never be split during tokenization.
   *        Only has an effect when basicTokenize = true.
   * @param unkSplit The unknown token. A token that is not in the vocabulary
   *        cannot be converted to an ID and is set to be this token instead.
   * @param sepToken The separator token. It is used when building a sequence
   *        from multiple sequences, e.g. two sequences for sequence
   *        classification or for a text and a question for question answering.
   *        It is also used as the last token of a sequence built with special
   *        tokens.
   * @param padToken The token used for padding, for example when batching
   *        sequences of different lengths.
   * @param clsToken The classifier token which is used when doing sequence
   *        classification (classification of the whole sequence instead of
   *        per-token classification). It is the first token of the sequence
   *        when built with special tokens.
   * @param maskToken The token used for masking values. This is the token used
   *        when training this model with masked language modeling. This is the
   *        token which the model will try to predict.
   */
  BertTokenizer(const std::string vocabFile,
       const bool lowerCase = true,
       const bool basicTokenize = true,
       const std::vector<std::string> neverSplit = std::vector<std::string>(),
       const std::string unkToken = "[UNK]",
       const std::string sepToken = "[SEP]",
       const std::string padToken = "[PAD]",
       const std::string clsToken = "[CLS]",
       const std::string maskToken = "[MASK]");

 private:
  //! Location of vocabulary.
  std::string vocabFile;

  //! Locally-stored vocabulary.
  std::vector<std::string> vocabulary;

  //! Whether to turn each token to lower case.
  bool lowerCase;

  //! Whether to do basic tokenization before WordPiece.
  bool basicTokenize;

  //! Tokens which will never be split during tokenization. Only has an effect
  //! when basicTokenize = true.
  std::vector<std::string> neverSplit;

  //! The unknown token. A token that is not in the vocabulary cannot be
  //! converted to an ID and is set to be this token instead.
  std::string unkToken;

  //! The separator token. It is used when building a sequence from multiple
  //! sequences, e.g. two sequences for sequence classification or for a text
  //! and a question for question answering. It is also used as the last token
  //! of a sequence built with special tokens.
  std::string sepToken;

  //! The token used for padding, for example when batching sequences of
  //! different lengths.
  std::string padToken;

  //! The classifier token which is used when doing sequence classification
  //! (classification of the whole sequence instead of per-token
  //! classification). It is the first token of the sequence when built with
  //! special tokens.
  std::string clsToken;

  //! The token used for masking values. This is the token used when training
  //! this model with masked language modeling. This is the token which the
  //! model will try to predict.
  std::string maskToken;
}; // class BertTokenizer

} // namespace ann
} // namespace mlpack

#endif
