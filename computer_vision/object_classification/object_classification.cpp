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

int main()
{
  #if defined(_OPENMP)
    std::cout << "Compiled with OpenMP!" << std::endl;
  #endif

  DataLoader<> dataloader;

  // Utils::DownloadFile("/datasets/cifar10.tar.gz",
  //    "./../data/cifar10.tar.gz", "", false, true,
  //    "www.mlpack.org", true);
  std::cout << "Loading Dataset!" << std::endl;
  dataloader.LoadImageDatasetFromDirectory("./../data/cifar10-small/",
      32, 32, 3, true, 0.2, true);

  std::cout << "Dataset Loaded!" << std::endl;
  dataloader.TrainLabels() = dataloader.TrainLabels() + 1;
  DarkNet<> darknetModel(3, 32, 32, 10);
  std::cout << "Model Compiled" << std::endl;

  constexpr double RATIO = 0.1;
  constexpr size_t EPOCHS = 5;
  constexpr double STEP_SIZE = 0.001;
  constexpr int BATCH_SIZE = 1;

  mlpack::data::MinMaxScaler scaler;
  scaler.Fit(dataloader.TrainFeatures());
  scaler.Transform(dataloader.TrainFeatures(), dataloader.TrainFeatures());
  scaler.Transform(dataloader.ValidFeatures(), dataloader.ValidFeatures());

  ens::Adam optimizer(STEP_SIZE, BATCH_SIZE, 0.9, 0.995, 1e-8,
      dataloader.TrainFeatures().n_cols * EPOCHS);

  std::cout << "Optimizer Created, Starting Training!" << std::endl;

  darknetModel.GetModel().Train(dataloader.TrainFeatures(),
      dataloader.TrainLabels(),
      optimizer,
      ens::PrintLoss(),
      ens::ProgressBar(),
      ens::EarlyStopAtMinLoss(),
      ens::PrintMetric<FFN<NegativeLogLikelihood<>, RandomInitialization>,
          Accuracy>(
            darknetModel.GetModel(),
            dataloader.TrainFeatures(),
            dataloader.TrainLabels(),
            "accuracy",
            true),
      ens::PrintMetric<FFN<NegativeLogLikelihood<>, RandomInitialization>,
          Accuracy>(
              darknetModel.GetModel(),
              dataloader.ValidFeatures(),
              dataloader.ValidLabels(),
              "accuracy",
              false),
      ens::PeriodicSave<FFN<NegativeLogLikelihood<>, RandomInitialization>>(
          darknetModel.GetModel(),
          "./../weights/",
          "darknet19", 1));

  mlpack::data::Save("darknet19.bin", "darknet",
      darknetModel.GetModel(), false);
  return 0;
}
