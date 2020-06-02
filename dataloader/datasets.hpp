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
  //! Locally stored name of dataset used for identification
  //! during dataloader call.
  std::string datasetName;

  //! Locally stored URL for downloading training data.
  std::string trainDownloadURL;

  //! Locally stored URL for downloading testing data.
  std::string testDownloadURL;

  //! CRC-32 checksum for training data file.
  std::string trainHash;

  //! CRC-32 checksum for testing data file.
  std::string testHash;

  //! Locally stored boolean to determine if dataset is of CSV or similar
  //! format.
  bool loadCSV;

  //! Locally stored path to file / directory for training data.
  std::string trainPath;

  //! Locally stored path to file / directory for testing data.
  std::string testPath;

  //! Locally held boolean to determine whether dataset will be in zip format.
  bool zipFile;

  //! Locally stored URL for downloading dataset.
  std::string datasetURL;

  //! Locally stored CRC-32 checksum for the dataset.
  std::string datasetHash;

  //! Locally stored path for saving the archived / zip dataset.
  std::string datasetPath;

  //! Locally stored server name for download file.
  std::string serverName;

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
      trainDownloadURL(""),
      testDownloadURL(""),
      trainHash(""),
      testHash(""),
      loadCSV(false),
      trainPath(""),
      testPath(""),
      zipFile(false),
      datasetURL(""),
      datasetPath(""),
      datasetHash(""),
      serverName("www.mlpack.org"),
      startTrainingInputFeatures(0),
      endTrainingInputFeatures(0),
      startTrainingPredictionFeatures(0),
      endTrainingPredictionFeatures(0),
      startTestingInputFeatures(0),
      endTestingInputFeatures(0),
      dropHeader(false)
  {/* Nothing to do here. */}

  /**
   * Constructor for initializing object for seperate
   * train and test download URL.
   *
   * @param datasetName Name of dataset used for identification during
   *                    dataloader call.
   * @param trainDownloadURL URL for downloading training data.
   * @param testDownloadURL  URL for downloading testing data.
   * @param trainHash CRC-32 checksum for training data.
   * @param testHash CRC-32 checksum for testing data.
   * @param loadCSV Determines if the format of dataset is similar to CSV.
   * @param trainPath Path for training dataset.
   * @param testPath Path for testing dataset.
   */
  DatasetDetails(const std::string& datasetName,
                 const std::string& trainDownloadURL,
                 const std::string& testDownloadURL,
                 const std::string& trainHash,
                 const std::string& testHash,
                 const bool loadCSV,
                 const std::string& trainPath,
                 const std::string& testPath) :
                 datasetName(datasetName),
                 trainDownloadURL(trainDownloadURL),
                 testDownloadURL(testDownloadURL),
                 trainHash(trainHash),
                 testHash(testHash),
                 loadCSV(loadCSV),
                 trainPath(trainPath),
                 testPath(testPath),
                 zipFile(false),
                 datasetURL(""),
                 datasetHash(""),
                 serverName("www.mlpack.org"),
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

  /**
   * Constructor for initializing paths for zip files.
   *
   * @param datasetName Name of dataset used for identification during
   *                    dataloader call.
   * @param zipFile Boolean to determine if dataset is stored in zip format.
   * @param datasetURL  URL for downloading dataset.
   * @param datasetPath Path where the dataset will be downloaded.
   * @param datasetHash CRC-32 checksum for dataset.
   * @param loadCSV Determines if the format of dataset is similar to CSV.
   * @param trainPath Path for training dataset.
   * @param testPath Path for testing dataset.
   */
  DatasetDetails(const std::string& datasetName,
                 const bool zipFile,
                 const std::string& datasetURL,
                 const std::string& datasetPath,
                 const std::string& datasetHash,
                 const bool loadCSV,
                 const std::string& trainPath,
                 const std::string& testPath) :
                 datasetName(datasetName),
                 zipFile(zipFile),
                 datasetURL(datasetURL),
                 datasetHash(datasetHash),
                 datasetPath(datasetPath),
                 loadCSV(loadCSV),
                 trainPath(trainPath),
                 testPath(testPath),
                 trainDownloadURL(""),
                 testDownloadURL(""),
                 trainHash(""),
                 testHash(""),
                 serverName("www.mlpack.org"),
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
  //! Get details of MNIST Dataset.
  const static DatasetDetails<DatasetX, DatasetY> MNIST()
  {
    DatasetDetails<DatasetX, DatasetY> mnistDetails(
        "mnist",
        true,
        "/datasets/mnist.tar.gz",
        "./../data/mnist.tar.gz",
        "33470ca3",
        true,
        "./../data/mnist-dataset/mnist_train.csv",
        "./../data/mnist-dataset/mnist_test.csv");

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
