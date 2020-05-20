/**
 * @file dataloader_tests.cpp
 * @author Kartik Dutt
 *
 * Tests for various functionalities of dataloader.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN // Do not define this anywhere else.
#include <dataloader/dataloader.hpp>
#include <boost/test/unit_test.hpp>

using namespace boost::unit_test;

BOOST_AUTO_TEST_SUITE(DataLoadersTest);

/**
 * Simple test for Dataloader.
 */
BOOST_AUTO_TEST_CASE(CSVDataLoaderTest)
{
  // Check that dataloader loads only known datasets.
  BOOST_REQUIRE_THROW(DataLoader<>("no-dataset", true), std::runtime_error);

  // Check Load CSV Function for Dataloader.
  Utils::DownloadFile("/datasets/iris.csv", "./../data/iris.csv");
  // Check the file has been downloaded.
  BOOST_REQUIRE(Utils::PathExists("./../data/iris.csv"));

  DataLoader<> irisDataloader;
  irisDataloader.LoadCSV("./../data/iris.csv", true, true, 0.5, false, false,
      0, -1, 1, -1);

  // Check for length and columns of training dataset.
  BOOST_REQUIRE_EQUAL(irisDataloader.TrainLabels().n_cols, 75);
  BOOST_REQUIRE_EQUAL(irisDataloader.TrainLabels().n_rows, 3);

  // Check for validation data as well.
  BOOST_REQUIRE_EQUAL(irisDataloader.ValidFeatures().n_cols, 75);
  BOOST_REQUIRE_EQUAL(irisDataloader.ValidFeatures().n_rows, 4);

  // Check for validation dataset using tuples.
  BOOST_REQUIRE_EQUAL(std::get<1>(irisDataloader.ValidSet()).n_cols, 75);
  BOOST_REQUIRE_EQUAL(std::get<1>(irisDataloader.ValidSet()).n_rows, 3);

  // Check for training dataset using tuples.
  BOOST_REQUIRE_EQUAL(std::get<0>(irisDataloader.TrainSet()).n_cols, 75);
  BOOST_REQUIRE_EQUAL(std::get<0>(irisDataloader.TrainSet()).n_rows, 4);

  Utils::RemoveFile("./../data/iris.csv");
}

/**
 * Simple test for MNIST Dataloader.
 */
BOOST_AUTO_TEST_CASE(MNISTDataLoaderTest)
{
  /**
  DataLoader<> dataloader("mnist", true, 0.80);

  // Check for correct dimensions.
  BOOST_REQUIRE_EQUAL(dataloader.TrainFeatures().n_rows, 784);
  BOOST_REQUIRE_EQUAL(dataloader.TestFeatures().n_rows, 784);
  BOOST_REQUIRE_EQUAL(dataloader.ValidFeatures().n_rows, 784);

  // Check for correct dimensions.
<<<<<<< HEAD
  BOOST_REQUIRE_EQUAL(dataloader.TrainFeatures().n_cols, 8400);
  BOOST_REQUIRE_EQUAL(dataloader.ValidFeatures().n_cols, 33600);
  BOOST_REQUIRE_EQUAL(dataloader.TestFeatures().n_cols, 28000);

  // Check if we can access both features and labels using
  // TrainSet tuple and ValidSet tuple.
  BOOST_REQUIRE_EQUAL(std::get<0>(dataloader.TrainSet()).n_cols, 8400);
  BOOST_REQUIRE_EQUAL(std::get<1>(dataloader.TrainSet()).n_rows, 1);
  BOOST_REQUIRE_EQUAL(std::get<0>(dataloader.ValidSet()).n_cols, 33600);
  BOOST_REQUIRE_EQUAL(std::get<1>(dataloader.ValidSet()).n_rows, 1);

  // Clean up.
  Utils::RemoveFile("./../data/mnist-dataset/mnist_all.csv");
  Utils::RemoveFile("./../data/mnist-dataset/mnist_all_centroids.csv");
  Utils::RemoveFile("./../data/mnist-dataset/mnist_train.csv");
  Utils::RemoveFile("./../data/mnist-dataset/mnist_test.csv");
  Utils::RemoveFile("./../data/mnist.tar.gz");
=======
  BOOST_REQUIRE_EQUAL(dataloader.TrainFeatures().n_cols, 784);
  BOOST_REQUIRE_EQUAL(dataloader.TestFeatures().n_cols, 784);
  BOOST_REQUIRE_EQUAL(dataloader.ValidFeatures().n_cols, 784);
  BOOST_REQUIRE_EQUAL(dataloader.TrainFeatures().n_rows, 33600);
  */
}

/**
 * Simple Test for object detection dataloader.
 */
BOOST_AUTO_TEST_CASE(ObjectDetectionDataLoader)
{
  DataLoader<> dataloader;
  dataloader.LoadObjectDetectionDataset("./../data/annotations/", "./../data");
>>>>>>> 3353e2e... Add basic definition of models, Needs to be trained and tested
}

BOOST_AUTO_TEST_SUITE_END();
