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

#include <utils/utils.hpp>
#include "catch.hpp"

/**
 * Simple test for Data Downloader.
 */
TEST_CASE("DownloadFileTest", "[UtilsTest]")
{
  // To check downloader, perform the following :
  // 1. Download the file.
  // 2. Check for it's existence.
  // 3. Match checksum for given file.
  Utils::DownloadFile("/datasets/iris.csv", "./../data/iris.csv");
  REQUIRE(Utils::PathExists("./../data/iris.csv") == true);
  REQUIRE(Utils::CompareCRC32("./../data/iris.csv",
      "7c30e225") == true);
  // Clean up.
  Utils::RemoveFile("./../data/iris.csv");
}

/**
 * Simple test for CompareCRC32.
 */
TEST_CASE("CheckSumTest", "[UtilsTest]")
{
  // Download the file and verify it's checksum.
  Utils::DownloadFile("/datasets/iris_test.csv", "./../data/iris_test.csv");
  REQUIRE(Utils::CompareCRC32("./../data/iris_test.csv",
      "3be1f79e") == true);

  // Clean up.
  Utils::RemoveFile("./../data/iris_test.csv");
}

/**
 * Simple test for PathExists.
 */
TEST_CASE("PathExistsTest", "[UtilsTest]")
{
  // Check for files that exist.
  REQUIRE(Utils::PathExists("./../../tests/CMakeLists.txt") == true);
  REQUIRE(Utils::PathExists("./../../CMakeLists.txt") == true);
}

/**
 * Simple test for RemoveFile.
 */
TEST_CASE("RemoveFileTest", "[UtilsTest]")
{
  // Check for files that exist.
  bool file = static_cast<bool>(std::ofstream("./../data/file.txt").put('!'));
  if (!file)
  {
    mlpack::Log::Warn << "Unable to create file for testing." << std::endl;
  }

  Utils::RemoveFile("./../data/file.txt");
  REQUIRE(Utils::PathExists("./../data/file.txt") == 0);
}

TEST_CASE("ExtractFilesTest", "[UtilsTest]")
{
  std::vector<boost::filesystem::path> vec;

  Utils::DownloadFile("/datasets/USCensus1990.tar.gz",
      "./../data/USCensus1990.tar.gz", "", false, true,
      "www.mlpack.org", true, "./../data/");

  REQUIRE(Utils::PathExists("./../data/USCensus1990.csv") == true);
  REQUIRE(Utils::PathExists("./../data/USCensus1990_centroids.csv") == true);

  // Clean up.
  Utils::RemoveFile("./../data/USCensus1990.csv");
  Utils::RemoveFile("./../data/USCensus1990_centroids.csv");
  Utils::RemoveFile("./../data/USCensus1990.tar.gz");
}

/**
 * Simple test for downloading using curl.
 */
TEST_CASE("CurlDownloadTest", "[UtilsTest]")
{
  std::string serverName = "https://raw.githubusercontent.com/mlpack/";
  std::string path =
      "mlpack/master/src/mlpack/tests/data/test_image.png";

  // Download file from an https server.
  Utils::DownloadFile(path, "./../data/test_image.jpg", "", false, true,
      serverName);

  // Check whether or not the image was downloaded. If yes, perform a checksum.
  REQUIRE(Utils::PathExists("./../data/test_image.jpg") == true);
  REQUIRE(Utils::CompareCRC32("./../data/test_image.jpg", "59721bac") == true);

  // Clean up.
  Utils::RemoveFile("./../data/test_image.jpg");
}
