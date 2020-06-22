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
    std::cout << a1.LayerString(&t) << std::endl;
    return;
  }
};

int main()
{
  #if defined(_OPENMP)
    std::cout << "Compiled with OpenMP!" << std::endl;
  #endif

  DarkNet<> darknetModel(3, 32, 32, 10, "none", true);
  std::cout << "Model Compiled" << std::endl;
  for (int i = 0; i < darknetModel.GetModel().Model().size(); i++)
  {
    cout << i << " : ";
    boost::apply_visitor(output{}, darknetModel.GetModel().Model()[i]);
  }

  //std::cout << boost::get<Sequential<> *>(darknetModel.GetModel().Model()[0])->Parameters().n_rows << std::endl;
  return 0;
}
