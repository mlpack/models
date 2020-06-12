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
 #include <dataloader/dataloader.hpp>
 #include <models/darknet/darknet.hpp>
 #include <utils/utils.hpp>
 #include <ensmallen.hpp>

 using namespace mlpack;
 using namespace mlpack::ann;
 using namespace arma;
 using namespace std;
 using namespace ens;

 int main()
 {
  DataLoader<> dataloader;
  std::cout << "Loading Dataset!" << std::endl;
  dataloader.LoadImageDatasetFromDirectory("./../data/cifar10", 32,
      32, 3, true, 0.2, true, {"resize : 56"});

  std::cout << "Dataset Loaded!" << std::endl;

  DarkNet<> darknetModel(3, 56, 56, 10);
  std::cout << "Model Compiled" << std::endl;

  constexpr double RATIO = 0.1;
  constexpr size_t EPOCHS = 3;
  constexpr double STEP_SIZE = 1.2e-3;
  constexpr int BATCH_SIZE = 50;

  ens::Adam optimizer(STEP_SIZE, BATCH_SIZE, 0.9, 0.998, 1e-8,
      dataloader.TrainFeatures().n_cols * EPOCHS);
  std::cout << "Optimizer Created, Starting Training!" << std::endl;

  darknetModel.GetModel().Train(dataloader.TrainFeatures(),
      dataloader.TrainLabels(), optimizer, ens::PrintLoss(),
      ens::ProgressBar(), ens::EarlyStopAtMinLoss());
  mlpack::data::Save("darknet19.bin", "darknet",
      darknetModel.GetModel(), false);
  return 0;
}
