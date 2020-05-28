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
#include <dataloader/dataloader.hpp>
#include <models/lenet/lenet.hpp>
#include <ensmallen.hpp>
#include <boost/test/unit_test.hpp>

// Use namespaces for convenience.
using namespace boost::unit_test;
using namespace mlpack;
using namespace mlpack::ann;

BOOST_AUTO_TEST_SUITE(ModelTest);

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
 * Test for sequential model.
 *
 * @param layer Sequential layer that contains the model.
 * @param datasetName Dataset which will be used for training and
 *                    validation.
 * @param threshold Maximum error for given metric.
 * @param takeMean Determines whether or not to take mean in error.
 * @param optimizer Optimizer that will be used for training.
 *
 * @tparam OptimizerType Optimizer type from ensmallen.
 * @tparam OutputLayerType The output layer type used to evaluate the network.
 * @tparam InitializationRuleType Rule used to initialize the weight matrix.
 * @tparam MetricType Metric that will be used 
 */
template<
    typename OptimizerType,
    typename OutputLayerType = mlpack::ann::NegativeLogLikelihood<>,
    typename InitializationRuleType = mlpack::ann::RandomInitialization,
    class MetricType = mlpack::metric::SquaredEuclideanDistance,
    typename InputType = arma::mat,
    typename OutputType = arma::mat
>
void CheckSequentialModel(mlpack::ann::Sequential<>* layer,
    const std::string& datasetName, const double threshold,
    const bool takeMean, OptimizerType& optimizer)
{
  // We can run two tests for sequential layer.
  // 1. Can it be attached to other models.
  // 2. Used as an FFN for training / inference.
  FFN<OutputLayerType, InitializationRuleType> model;
  model.Add(new IdentityLayer<>());
  model.Add(layer);
  CheckFFNClassificationWeights<OptimizerType, OutputLayerType, InitializationRuleType,
      MetricType, InputType, OutputType>(model, datasetName, threshold,
      takeMean, optimizer);
  // Using Layer in another model.
}

/**
 * Simple test for Le-Net model.
 */
BOOST_AUTO_TEST_CASE(LeNetModelTest)
{
  LeNet<> lenetModel1(1, 28, 28, 10, "mnist");

  // Create an optimizer object for tests.
  ens::SGD<ens::AdamUpdate> optimizer(1e-4, 16, 50,
      1e-8, true, ens::AdamUpdate(1e-8, 0.9, 0.999));

  // Check whether FFN model performs well.
  CheckFFNClassificationWeights<ens::SGD<ens::AdamUpdate>>(
      lenetModel1.GetModel(), "mnist", 1e-1, true, optimizer);

  LeNet<
      mlpack::ann::NegativeLogLikelihood<>,
      mlpack::ann::RandomInitialization,
      4
      >lenetModel4(1, 28, 28, 10, "mnist");

  // Check whether FFN model performs well.
  CheckFFNClassificationWeights<ens::SGD<ens::AdamUpdate>>(
      lenetModel4.GetModel(), "mnist", 1e-1, true, optimizer);

  LeNet<
      mlpack::ann::NegativeLogLikelihood<>,
      mlpack::ann::RandomInitialization,
      5
      >lenetModel5(1, 28, 28, 10, "mnist");

  // Check whether FFN model performs well.
  /**
   * CheckFFNClassificationWeights<ens::SGD<ens::AdamUpdate>>(
   *    lenetModel5.GetModel(), "mnist", 1e-1, true, optimizer);
   */
}

BOOST_AUTO_TEST_SUITE_END();
