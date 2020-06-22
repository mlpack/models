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

using namespace mlpack;
using namespace mlpack::ann;
using namespace arma;
using namespace std;
using namespace ens;

class Accuracy
{
 public:
  template<typename InputType, typename OutputType>
  static double Evaluate(InputType& input, OutputType& output)
  {
    arma::Row<size_t> predLabels(input.n_cols);
    for (arma::uword i = 0; i < input.n_cols; ++i)
    {
      predLabels(i) = input.col(i).index_max() + 1;
    }
    return arma::accu(predLabels == output) / (double)output.n_elem * 100;
  }
};

struct output : public boost::static_visitor<>
{
  template <class T>
  void operator()(T t) const { std::cout<< t->Parameters().n_cols << '\n'; }
};

int main()
{
  #if defined(_OPENMP)
    std::cout << "Compiled with OpenMP!" << std::endl;
  #endif

  DarkNet<> darknetModel(3, 32, 32, 10);
  std::cout << "Model Compiled" << std::endl;

  boost::apply_visitor(output{}, darknetModel.GetModel().Model()[0]);
  //std::cout << boost::get<Sequential<> *>(darknetModel.GetModel().Model()[0])->Parameters().n_rows << std::endl;

  return 0;
}
