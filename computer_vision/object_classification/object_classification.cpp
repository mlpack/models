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
#include <models/lenet/lenet.hpp>
#include <ensmallen.hpp>

using namespace mlpack;
using namespace mlpack::ann;
using namespace arma;
using namespace std;
using namespace ens;

int main()
{
  const int EPOCHS = 2;
  const double STEP_SIZE = 5e-3;
  const int BATCH_SIZE = 32;
  const double RATIO = 0.2;

  DataLoader<> dataloader("mnist", true, RATIO);

  LeNet<> module1(1, 28, 28, 10);
  cout << "Training." << endl;

  SGD<AdamUpdate> optimizer(STEP_SIZE, BATCH_SIZE,
                            EPOCHS * dataloader.TrainLabels().n_cols,
                            1e-8,
                            true,
                            AdamUpdate(1e-8, 0.9, 0.999));

  module1.GetModel().Train(dataloader.TrainFeatures(),
                           dataloader.TrainLabels(),
                           optimizer,
                           ens::PrintLoss(),
                           ens::ProgressBar(),
                           ens::EarlyStopAtMinLoss());

  module1.SaveModel("./../weights/lenet/lenet" + to_string(1)+"_mnist.bin");

  return 0;
}