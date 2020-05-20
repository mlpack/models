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
using namespace boost::unit_test;

BOOST_AUTO_TEST_SUITE(UtilsTest);

/**
 * Simple test for Data Downloader.
 */
BOOST_AUTO_TEST_CASE(DownloadFileTest)
{
  // To check downloader, perform the following :
  // 1. Download the file.
  // 2. Check for it's existence.
  // 3. Match checksum for given file.
  Utils::DownloadFile("/datasets/iris.csv", "./../data/iris.csv");
  BOOST_REQUIRE(Utils::PathExists("./../data/iris.csv"));
  BOOST_REQUIRE(Utils::CompareCRC32("./../data/iris.csv",
      "7c30e225"));
  // Clean up.
  Utils::RemoveFile("./../data/iris.csv");
}

/**
 * Simple test for CompareCRC32.
 */
BOOST_AUTO_TEST_CASE(CheckSumTest)
{
  // Download the file and verify it's checksum.
  Utils::DownloadFile("/datasets/iris_test.csv", "./../data/iris_test.csv");
  BOOST_REQUIRE(Utils::CompareCRC32("./../data/iris_test.csv",
      "3be1f79e"));

  // Clean up.
  Utils::RemoveFile("./../data/iris_test.csv");
}

/**
 * Simple test for PathExists.
 */
BOOST_AUTO_TEST_CASE(PathExistsTest)
{
  // Check for files that exist.
  BOOST_REQUIRE(Utils::PathExists("./../tests/CMakeLists.txt"));
  BOOST_REQUIRE(Utils::PathExists("./../CMakeLists.txt"));
}

/**
 * Simple test for RemoveFile.
 */
BOOST_AUTO_TEST_CASE(RemoveFileTest)
{
  // Check for files that exist.
  bool file = static_cast<bool>(std::ofstream("./../data/file.txt").put('!'));
  if (!file)
  {
    mlpack::Log::Warn << "Unable to create file for testing." << std::endl;
  }

  Utils::RemoveFile("./../data/file.txt");
  BOOST_REQUIRE_EQUAL(Utils::PathExists("./../data/file.txt"), 0);
}

BOOST_AUTO_TEST_CASE(ExtractFilesTest)
{
<<<<<<< HEAD
  std::vector<boost::filesystem::path> vec;

  Utils::DownloadFile("/datasets/USCensus1990.tar.gz",
      "./../data/USCensus1990.tar.gz", "", false, true,
      "www.mlpack.org", true, "./../data/");

  BOOST_REQUIRE(Utils::PathExists("./../data/USCensus1990.csv"));
  BOOST_REQUIRE(Utils::PathExists("./../data/USCensus1990_centroids.csv"));

  // Clean up.
  Utils::RemoveFile("./../data/USCensus1990.csv");
  Utils::RemoveFile("./../data/USCensus1990_centroids.csv");
  Utils::RemoveFile("./../data/USCensus1990.tar.gz");
=======
  Utils::DownloadFile("/datasets/mnist.tar.gz", "./../data/mnist.tar.gz", "",
      false, true, "www.mlpack.org", true, "./../data/");

  BOOST_REQUIRE(Utils::PathExists("./../data/mnist_all.csv"));
  BOOST_REQUIRE(Utils::PathExists("./../data/mnist.tar.gz"));

  // Clean up.
  Utils::RemoveFile("./../data/mnist_all.csv");
  Utils::RemoveFile("./../data/mnist_all_centroids.csv");
>>>>>>> 3353e2e... Add basic definition of models, Needs to be trained and tested
}

BOOST_AUTO_TEST_SUITE_END();
