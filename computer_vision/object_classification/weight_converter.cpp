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
#include <models/models.hpp>
#include <utils/utils.hpp>
#include <ensmallen_utils/print_metric.hpp>
#include <ensmallen_utils/periodic_save.hpp>
#include <ensmallen.hpp>
#include <mlpack/methods/ann/layer/layer_types.hpp>
#include <mlpack/methods/ann/layer_names.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/foreach.hpp>

using namespace mlpack;
using namespace mlpack::ann;
using namespace arma;
using namespace std;
using namespace ens;

std::queue<std::string> batchNormRunningMean;
std::queue<std::string> batchNormRunningVar;
template <
    typename OutputLayer = mlpack::ann::NegativeLogLikelihood<>,
    typename InitializationRule = mlpack::ann::RandomInitialization>
void LoadWeights(mlpack::ann::FFN<OutputLayer, InitializationRule> &model,
                 std::string modelConfigPath)
{
  std::cout << "Loading Weights\n";
  size_t currentOffset = 0;
  boost::property_tree::ptree xmlFile;
  boost::property_tree::read_xml(modelConfigPath, xmlFile);
  boost::property_tree::ptree modelConfig = xmlFile.get_child("model");
  BOOST_FOREACH (boost::property_tree::ptree::value_type const &layer, modelConfig)
  {
    std::string progressBar(81, '-');
    size_t filled = std::ceil(currentOffset * 80.0 / model.Parameters().n_elem);
    progressBar[0] = '[';
    std::fill(progressBar.begin() + 1, progressBar.begin() + filled + 1, '=');
    std::cout << progressBar << "] " << filled * 100.0 / 80.0 << "%\r";
    std::cout.flush();
    // Load Weights.
    if (layer.second.get_child("has_weights").data() != "0")
    {
      arma::mat weights;
      mlpack::data::Load("./../../../" + layer.second.get_child("weight_csv").data(), weights);
      model.Parameters()(arma::span(currentOffset, currentOffset + weights.n_elem - 1),
                         arma::span()) = weights.t();
      currentOffset += weights.n_elem;
    }
    else
    {
      currentOffset += std::stoi(layer.second.get_child("weight_offset").data());
    }

    // Load Biases.
    if (layer.second.get_child("has_bias").data() != "0")
    {
      arma::mat bias;
      mlpack::data::Load("./../../../" + layer.second.get_child("bias_csv").data(), bias);
      model.Parameters()(arma::span(currentOffset, currentOffset + bias.n_elem - 1),
                         arma::span()) = bias.t();
      currentOffset += bias.n_elem;
    }
    else
    {
      currentOffset += std::stoi(layer.second.get_child("bias_offset").data());
    }

    if (layer.second.get_child("has_running_mean").data() != "0")
    {
      batchNormRunningMean.push("./../../../" + layer.second.get_child("running_mean_csv").data());
    }

    if (layer.second.get_child("has_running_var").data() != "0")
    {
      batchNormRunningMean.push("./../../../" + layer.second.get_child("running_var_csv").data());
    }
  }
  std::cout << std::endl;
}

template<typename LayerType = mlpack::ann::FFN<>>
LayerType LoadRunningMeanAndVariance(LayerType&& baseLayer, size_t i = 0)
{
  while (i < baseLayer.Model().size() && !batchNormRunningMean.empty())
  {
    if (baseLayer.Model()[i].type() == typeid(new mlpack::ann::Sequential<>()))
    {
      std::cout << "Sequential Layer. " << i << std::endl;
      LoadRunningMeanAndVariance<mlpack::ann::Sequential<>>(&(baseLayer.Model()[i]));
    }

    if (!batchNormRunningMean.empty() &&
        baseLayer.Model()[i].type() == typeid(new mlpack::ann::BatchNorm<>()))
    {
      std::cout << "BATCHNORM Layer " << i << std::endl;
      arma::mat runningMean;
      mlpack::data::Load(batchNormRunningMean.front(), runningMean);
      batchNormRunningMean.pop();
     // baseLayer.Model()[i].TrainingMean() = runningMean;
     // baseLayer.Model()[i].TrainingMean().print();
    }

    i++;
  }
  return baseLayer;
}


int main()
{
  DarkNet<> darknet(3, 224, 224, 1000);
  batchNormRunningMean.push("as");
  // LoadWeights<>(darknet.GetModel(), "./../../../cfg/darknet19.xml");
  LoadRunningMeanAndVariance<>(darknet.GetModel());
  boost::get<Sequential<>*>(darknet.GetModel().Model()[1])->Model().size();
  std::cout << boost::get<Sequential<>*>(darknet.GetModel().Model()[1])->Model().size() << std::endl;
  std::cout << (boost::get<Sequential<> *>(darknet.GetModel().Model()[1])->Model()[1].type() == typeid(new mlpack::ann::BatchNorm<>())) << std::endl;

  return 0;
}
