/**
 * @file model_tests.cpp
 * @author Kartik Dutt
 *
 * Tests for various functionalities and performance of models.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BOOST_TEST_DYN_LINK
#include <utils/utils.hpp>
#include <ensmallen.hpp>
#include <dataloader/dataloader.hpp>
#include <models/darknet/darknet.hpp>
#include <boost/test/unit_test.hpp>

// Use namespaces for convenience.
using namespace boost::unit_test;

BOOST_AUTO_TEST_SUITE(FFNModelsTests);

template<
    typename OptimizerType,
    typename OutputLayerType = mlpack::ann::NegativeLogLikelihood<>,
    typename InitializationRuleType = mlpack::ann::RandomInitialization,
    class MetricType = mlpack::metric::SquaredEuclideanDistance,
    typename InputType = arma::mat,
    typename OutputType = arma::mat
>
void CheckFFNClassificationWeights(mlpack::ann::FFN<OutputLayerType,
    InitializationRuleType>& model, const std::string& datasetName,
    const double threshold, const bool takeMean,
    OptimizerType& optimizer)
{
  DataLoader<InputType, OutputType> dataloader(datasetName, true);

  // Train the model. Note: Callbacks such as progress bar and loss aren't
  // used in testing. Training the model for few epochs ensures that a
  // user can use the pretrained model on any other dataset.
  model.Train(dataloader.TrainFeatures(), dataloader.TrainLabels(), optimizer);

  // Verify viability of model on validation datset.
  OutputType predictions;
  model.Predict(dataloader.ValidFeatures(), predictions);

  // Since this checks weights for classification problem, we need to convert
  // predictions into labels.
  arma::Row<size_t> predLabels(predictions.n_cols);
  for (arma::uword i = 0; i < predictions.n_cols; ++i)
  {
    predLabels(i) = predictions.col(i).index_max() + 1;
  }

  double error = MetricType::Evaluate(predLabels, dataloader.ValidLabels());

  if (takeMean)
  {
    error = error / predictions.n_elem;
  }

  BOOST_REQUIRE_LE(error, threshold);
}

/**
 * Simple test for Darknet model.
 */
BOOST_AUTO_TEST_CASE(DarknetModelTest)
{
  mlpack::ann::DarkNet<> darknetModel(3, 56, 56, 10);

  // Create an optimizer object for tests.
  ens::SGD<ens::AdamUpdate> optimizer(1e-4, 16, 50,
      1e-8, true, ens::AdamUpdate(1e-8, 0.9, 0.999));

  // Check whether FFN model performs well.
  // CheckFFNClassificationWeights<ens::SGD<ens::AdamUpdate>>(
  //    darknetModel1.GetModel(), "mnist", 1e-1, true, optimizer);
}

BOOST_AUTO_TEST_SUITE_END();
