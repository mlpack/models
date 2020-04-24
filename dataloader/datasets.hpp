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

#include <dataloader/preprocessor.hpp>

/**
 * Structure used to provide details about the dataset.
 *
 * @tparam DatasetX Datatype for loading input features for the dataset.
 * @tparam DatasetY Datatype for prediction features for the dataset.
 */
template<
    typename DatasetX = arma::mat,
    typename DatasetY = arma::mat
>
struct DatasetDetails
{
  std::string datasetName;
  std::string trainDownloadUrl;
  std::string testDownloadUrl;
  std::string trainHash;
  std::string testHash;
  bool loadCSV;
  std::string trainPath;
  std::string testPath;

  // Pre-Process functor.
  std::function<void(DatasetX&, DatasetY&,
      DatasetX&, DatasetY&, DatasetX&)> PreProcess;

  // The following parameters are for CSVs only.
  //! First Index which will be fed into the model as input.
  size_t startTrainingInputFeatures;
  //! Last Index which will be fed into the model as input.
  size_t endTrainingInputFeatures;

  //! First Index which be predicted by the model as output.
  size_t startTrainingPredictionFeatures;
  //! Last Index which be predicted by the model as output.
  size_t endTrainingPredictionFeatures;

  //! First Index which will be fed into the model as input for testing.
  size_t startTestingInputFeatures;
  //! Last Index which will be fed into the model as input for testing.
  size_t endTestingInputFeatures;

  //! Whether or not to drop the first row from CSV.
  bool dropHeader;

  // Default constructor.
  DatasetDetails() :
      datasetName(""),
      trainDownloadUrl(""),
      testDownloadUrl(""),
      trainHash(""),
      testHash(""),
      loadCSV(false),
      trainPath(""),
      testPath(""),
      startTrainingInputFeatures(0),
      endTrainingInputFeatures(0),
      startTrainingPredictionFeatures(0),
      endTrainingPredictionFeatures(0),
      startTestingInputFeatures(0),
      endTestingInputFeatures(0),
      dropHeader(false)
  {/* Nothing to do here. */}

  // Constructor for initializing object.
  DatasetDetails(const std::string& datasetName,
                 const std::string& trainDownloadUrl,
                 const std::string& testDownloadUrl,
                 const std::string& trainHash,
                 const std::string& testHash,
                 const bool loadCSV,
                 const std::string& trainPath,
                 const std::string& testPath) :
                 datasetName(datasetName),
                 trainDownloadUrl(trainDownloadUrl),
                 testDownloadUrl(testDownloadUrl),
                 trainHash(trainHash),
                 testHash(testHash),
                 loadCSV(loadCSV),
                 trainPath(trainPath),
                 testPath(testPath),
                 startTrainingInputFeatures(0),
                 endTrainingInputFeatures(0),
                 startTrainingPredictionFeatures(0),
                 endTrainingPredictionFeatures(0),
                 startTestingInputFeatures(0),
                 endTestingInputFeatures(0),
                 dropHeader(false)
  {
    // Nothing to do here.
  }
};

/**
 * Class used to get details about the dataset.
 *
 * @tparam DatasetX Datatype for loading input features for the dataset.
 * @tparam DatasetY Datatype for prediction features for the dataset.
 */
template<
    typename DatasetX = arma::mat,
    typename DatasetY = arma::mat
>
class Datasets
{
 public:
  const static DatasetDetails<DatasetX, DatasetY> MNIST()
  {
    DatasetDetails<DatasetX, DatasetY> mnistDetails(
        "mnist",
        "/datasets/mnist_train.csv",
        "/datasets/mnist_test.csv",
        "772495e3",
        "8bcdb7e1",
        true,
        "./../data/mnist_train.csv",
        "./../data/mnist_test.csv");

    // Set the Pre-Processor Function.
    mnistDetails.PreProcess = PreProcessor<DatasetX, DatasetY>::MNIST;

    // Set Parameters for CSV file.
    mnistDetails.startTestingInputFeatures = 0;
    mnistDetails.endTestingInputFeatures = -1;
    mnistDetails.startTrainingInputFeatures = 1;
    mnistDetails.endTrainingInputFeatures = -1;
    mnistDetails.startTrainingPredictionFeatures = 0;
    mnistDetails.endTrainingPredictionFeatures = 0;
    mnistDetails.dropHeader = true;
    return mnistDetails;
  }
};

#endif
