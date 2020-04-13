/**
 * @file utils_tests.cpp
 * @author Kartik Dutt
 *
 * Tests for various functionalities of utils.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#define BOOST_TEST_DYN_LINK
#include <utils/utils.hpp>
#include <dataloader/datasets.hpp>
#include <boost/test/unit_test.hpp>

BOOST_AUTO_TEST_SUITE(UtilsTest);

/**
 * Simple test for Data Downloader.
 */
BOOST_AUTO_TEST_CASE(DownloadFileTest)
{
  Utils::DownloadFile(Datasets::MNIST().trainDownloadUrl,
      "./../data/mnist_train.csv", "", false);
  BOOST_REQUIRE(Utils::PathExists("./../data/mnist_train.csv"));

  BOOST_REQUIRE(Utils::CompareSHA256("./../data/mnist.tar.gz",
      Datasets::MNIST().trainHash));

  Utils::DownloadFile(Datasets::MNIST().testDownloadUrl,
      "./../data/mnist_test.csv", "", false);
  BOOST_REQUIRE(Utils::PathExists("./../data/mnist_test.csv"));

  BOOST_REQUIRE(Utils::CompareSHA256("./../data/mnist.tar.gz",
      Datasets::MNIST().testHash));
}

/**
 * Simple test for CompareSHA256.
 */
BOOST_AUTO_TEST_CASE(CheckSumTest)
{
 // BOOST_REQUIRE(Utils::CompareSHA256("./.gitignore",
   //   "d1ceb335f6fb27209271c893fcdac809c7ff0381d00ffd28a9fdbe09e6dda9e2"));
}

/**
 * Simple test for PathExists.
 */
BOOST_AUTO_TEST_CASE(PathExistsTest)
{
  BOOST_REQUIRE(Utils::PathExists("./../models/lenet/lenet.hpp"));
  BOOST_REQUIRE(Utils::PathExists("./../models/lenet/lenet_impl.hpp"));
}

BOOST_AUTO_TEST_SUITE_END();
