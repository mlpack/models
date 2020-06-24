/**
 * @file object_classification.hpp
 * @author Kartik Dutt
 *
 * Contains implementation of object classification suite. It can be used
 * to select object classification model, it's parameter dataset and
 * other training parameters.
 *
 * NOTE: This code needs to be adapted as this implementation doesn't support
 *       Command Line Arguments.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core.hpp>
#include <dataloader/dataloader.hpp>
#include <models/darknet/darknet.hpp>
#include <utils/utils.hpp>
#include <ensmallen_utils/print_metric.hpp>
#include <ensmallen_utils/periodic_save.hpp>
#include <ensmallen.hpp>
#include <mlpack/methods/ann/layer_names.hpp>

using namespace mlpack;
using namespace mlpack::ann;
using namespace arma;
using namespace std;
using namespace ens;

struct output : public boost::static_visitor<>
{
  template <class T>
  void operator()(T t) const
  {
    // We could use this as a summary.
    LayerNameVisitor a1;
    t.Parameters();
    std::cout << a1.LayerString(&t) << std::endl;
    return;
  }
};

/**
 * std::cout << boost::get<Sequential<> *>(
 *  darknetModel.GetModel().Model()[3])->Parameters().n_elem << std::endl;
 */
int main()
{
  #if defined(_OPENMP)
    std::cout << "Compiled with OpenMP!" << std::endl;
  #endif
  DarkNet<> darknetModel(3, 224, 224, 1000, "none", true);
  std::cout << "Model Compiled" << std::endl;

  // std::cout << darknetModel.GetModel().Parameters().n_rows << " " <<
  //    darknetModel.GetModel().Parameters().n_cols << std::endl;

  size_t outSize = 32;
  Convolution<> *layer = new Convolution<>(3, 32, 3, 3, 1, 1, 1, 1, 224, 224);
  // std::cout << layer->Parameters().n_elem << std::endl;
  size_t layerElement = layer->Parameters().n_elem - outSize;

  // This will be equal to output size for conv layers.
  // In darknet model bias term is removed.
  size_t offset = 0;
  size_t biasOffset = outSize;

  arma::mat ConvWeights;
  mlpack::data::Load("../../PyTorch-mlpack-DarkNet-Weight-Converter/conv_1_1.csv",
      ConvWeights);
  // Transpose weights to match FFN Class.
  ConvWeights = ConvWeights.t();
  std::cout << ConvWeights.n_rows << " " << ConvWeights.n_cols << std::endl;

  darknetModel.GetModel().Parameters()(arma::span(offset, offset + layerElement - 1),
      arma::span()) = ConvWeights;
  

  bool weightsEqual = true;
  for (size_t i = offset; i < offset + layerElement; i++)
  {
    if (darknetModel.GetModel().Parameters()(i) != ConvWeights(i - offset))
    {
      weightsEqual = false;
      break;
    }
  }

  if (weightsEqual)
  {
    std::cout << "Yay!, Transferred weights" << std::endl;
  }
  else
  {
    std::cout << "Hmm, Looks like you missed something!" << std::endl;
  }

  offset = offset + layerElement + biasOffset;
  biasOffset = 0;

  BatchNorm<>* layer2 = new BatchNorm<>(32);
  std::cout << layer2->Parameters().n_elem << std::endl;
  return 0;
}
