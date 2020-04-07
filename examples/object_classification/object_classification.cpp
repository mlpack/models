#include <mlpack/core.hpp>
#include <mlpack/prereqs.hpp>
#include <mlpack/methods/ann/layer/layer.hpp>
#include <mlpack/methods/ann/ffn.hpp>
#include <mlpack/methods/ann/layer/layer_types.hpp>
#include <mlpack/core/data/split_data.hpp>
#include <mlpack/methods/ann/init_rules/random_init.hpp>
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
  const int EPOCHS = 10;
  const double STEP_SIZE = 1.2e-3;
  const int BATCH_SIZE = 32;
  const double RATIO = 0.2;

  DataLoader<> dataloader("mnist", true, RATIO);

  FFN<NegativeLogLikelihood<>, RandomInitialization> model;
  LeNet<> module1(1, 28, 28, 10);
  module1.LoadModel("./../weights/lenet.bin");
  Sequential<> *layer = module1.GetModel();
  model.Add<IdentityLayer<>>();
  model.Add(layer);
  cout << "Training." << endl;
  SGD<AdamUpdate> optimizer(STEP_SIZE, BATCH_SIZE,
      EPOCHS * dataloader.TrainY().n_cols, 1e-8, true, AdamUpdate(1e-8, 0.9, 0.999));

  model.Train(dataloader.TrainX(),
              dataloader.TrainY(),
              optimizer,
              ens::PrintLoss(),
              ens::ProgressBar(),
              ens::EarlyStopAtMinLoss());

  module1.SaveModel("./../weights/lenet.bin");
  module1.LoadModel("./../weights/lenet.bin");
}