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
BOOST_AUTO_TEST_CASE(DataLoaderTest)
{
  // Check that dataloader loads only known datasets.
  BOOST_REQUIRE_THROW(DataLoader<>("no-dataset", true), std::runtime_error);

  // Check Load CSV Function for Dataloader.
  Utils::DownloadFile("iris.csv", "./../data/iris.csv");
  DataLoader<> irisDataloader;
  irisDataloader.LoadCSV("./../data/iris.csv", true, true, 0.5, false, false,
      0, -1, 1, -1);
  // Check for length and columns.
  BOOST_REQUIRE_EQUAL(irisDataloader.TrainX().n_cols, 4);
  BOOST_REQUIRE_EQUAL(irisDataloader.TrainX().n_rows, 75);
  BOOST_REQUIRE_EQUAL(irisDataloader.TrainY().n_cols, 3);
  BOOST_REQUIRE_EQUAL(irisDataloader.TrainY().n_rows, 75);

  // Check for validation data as well.
  BOOST_REQUIRE_EQUAL(irisDataloader.ValidX().n_cols, 4);
  BOOST_REQUIRE_EQUAL(irisDataloader.ValidX().n_rows, 75);
  BOOST_REQUIRE_EQUAL(irisDataloader.ValidY().n_cols, 3);
  BOOST_REQUIRE_EQUAL(irisDataloader.ValidY().n_rows, 75);
}

/**
 * Simple test for MNIST Dataloader.
 */
BOOST_AUTO_TEST_CASE(MNISTDataLoaderTest)
{
  DataLoader<> dataloader("mnist", true, 0.80);
  // Check for correct dimensions.
  BOOST_REQUIRE_EQUAL(dataloader.TrainX().n_cols, 784);
  BOOST_REQUIRE_EQUAL(dataloader.TestX().n_cols, 784);
  BOOST_REQUIRE_EQUAL(dataloader.ValidX().n_cols, 784);
  BOOST_REQUIRE_EQUAL(dataloader.TrainX().n_rows, 33600);
}

BOOST_AUTO_TEST_SUITE_END();
