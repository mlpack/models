/**
 * @file datasets.hpp
 * @author Kartik Dutt
 * 
 * File containing details for every datasets.
 * 
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#ifndef MODELS_DATASETS_HPP
#define MODELS_DATASETS_HPP

#include <iostream>

/**
 * Structure used to provide details about the dataset.
 */
struct DatasetDetails
{
  std::string datasetName;
  std::string trainDownloadUrl;
  std::string testDownloadUrl;
  std::string trainHash;
  std::string testHash;

  DatasetDetails(std::string datasetName,
                 std::string trainDownloadUrl,
                 std::string testDownloadUrl,
                 std::string trainHash,
                 std::string testHash) :
                 datasetName(datasetName),
                 trainDownloadUrl(trainDownloadUrl),
                 testDownloadUrl(testDownloadUrl),
                 trainHash(trainHash),
                 testHash(testHash)
  {
   // Nothing to do here.
  }
};

class Datasets
{
  public:
  const static DatasetDetails MNIST()
  {
    return DatasetDetails("mnist",
      "https://raw.githubusercontent.com/kartikdutt18/mlpack-models-weights-and-datasets/master/mnist-dataset/mnist_train.csv",
      "https://raw.githubusercontent.com/kartikdutt18/mlpack-models-weights-and-datasets/master/mnist-dataset/mnist_test.csv",
      "cc10cd2dcac4fa2b67b8e9b7c90019cb8669668ed11a1ea71d980418785b5b11",
      "1e4c6240156c2316c012a655ff268e8b7f37b4a4dabe49853ab3a60af0ed1bca");
  }


};

#endif