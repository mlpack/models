/**
 * @file dataloader.hpp
 * @author Kartik Dutt
 * 
 * Definition of Dataloader for popular datasets.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */
#include <mlpack/core/data/scaler_methods/min_max_scaler.hpp>
#include <mlpack/core/math/shuffle_data.hpp>
#include <mlpack/core/data/split_data.hpp>
#include <mlpack/prereqs.hpp>
#include <dataloader/datasets.hpp>
#include <mlpack/core.hpp>
#include <utils/utils.hpp>


#ifndef MODELS_DATALOADER_HPP
#define MODELS_DATALOADER_HPP

/**
 * Dataloader class to load popular datasets.
 *
 * @code
 * // Create a dataloader for any popular dataset.
 * // Set parameters for dataset.
 * const string datasetName = "mnist";
 * bool shuffleData = true;
 * double ratioForTrainTestSplit = 0.75;
 * 
 * // Create the DataLoader object.
 * DataLoader<> dataloader(datasetName, shuffleData,
 *    ratioForTrainTestSplit);
 *
 * // Use the dataloader for training.
 * model.Train(dataloader.TrainFeatures(), dataloader.TrainLabels());
 *
 * // Use the dataloader for prediction.
 * model.Predict(dataloader.TestFeatures(), dataloader.TestLabels());
 * @endcode
 * 
 * @tparam DatasetX Datatype for loading input features.
 * @tparam DatasetY Datatype for prediction features.
 * @tparam ScalerType mlpack's Scaler Object for scaling features.
 */
template<
  typename DatasetX = arma::mat,
  typename DatasetY = arma::mat,
  class ScalerType = mlpack::data::MinMaxScaler
>
class DataLoader
{
 public:
  //! Create DataLoader object.
  DataLoader();

  /**
   * Constructor for DataLoader. This is used for loading popular Datasets such as
   * MNIST, ImageNet, Pascal VOC and many more.
   * 
   * @param datasetPath Path or name of dataset.
   * @param shuffle whether or not to shuffle the data.
   * @param ratio Ratio for train-test split.
   * @param useScaler Use feature scaler for pre-processing the dataset.
   * @param augmentation Adds augmentation to training data only.
   * @param augmentationProbability Probability of applying augmentation on dataset.
   */
  DataLoader(const std::string& dataset,
             const bool shuffle,
             const double ratio = 0.75,
             const bool useScaler = true,
             const std::vector<std::string> augmentation =
                 std::vector<std::string>(),
             const double augmentationProbability = 0.2);

  /**
   * Function to load and preprocess train or test data stored in CSV files.
   * 
   * @param datasetPath Path to the dataset.
   * @param loadTrainData Boolean to determine whether data will be stored for
   *                      training or testing. If true, data will be loaded for training.
   *                      Note: This option augmentation to NULL, set ratio to 1 and
   *                      scaler will be used to only transform the test data.
   * @param shuffle Boolean to determine whether or not to shuffle the data.
   * @param ratio Ratio for train-test split.
   * @param useScaler Fits the scaler on training data and transforms dataset.
   * @param dropHeader Drops the first row from CSV.
   * @param startInputFeatures First Index which will be fed into the model as input.
   *                           Note: Indicies are wrapped and -1 implies last
   *                           column.
   * @param endInputFeature Last Index which will be fed into the model as input.
   *                        Note: Indicies are wrapped and -1 implies last
   *                        column.
   * @param startPredictionFeatures First Index which be predicted by the model as output.
   *                                Note: Indicies are wrapped and -1 implies last
   *                                column.
   * @param endPredictionFeatures Last Index which be predicted by the model as output.
   *                              Note: Indicies are wrapped and -1 implies last
   *                              column.
   * @param augmentation Vector strings of augmentations supported by mlpack.
   * @param augmentationProbability Probability of applying augmentation to a particular cell.
   */
  void LoadCSV(const std::string& datasetPath,
               const bool loadTrainData = true,
               const bool shuffle = true,
               const double ratio = 0.75,
               const bool useScaler = false,
               const bool dropHeader = false,
               const int startInputFeatures = -1,
               const int endInputFeatures = -1,
               const int startPredictionFeatures = -1,
               const int endPredictionFeatures = -1,
               const std::vector<std::string> augmentation =
                   std::vector<std::string>(),
               const double augmentationProbability = 0.2);

  //! Get the training dataset features.
  DatasetX TrainFeatures() const { return trainFeatures; }

  //! Modify the training dataset features.
  DatasetX& TrainFeatures() { return trainFeatures; }

  //! Get the training dataset labels.
  DatasetY TrainLabels() const { return trainLabels; }
  //! Modify the training dataset labels.
  DatasetY& TrainLabels() { return trainLabels; }

  //! Get the test dataset features.
  DatasetX TestFeatures() const { return testFeatures; }
  //! Modify the test dataset features.
  DatasetX& TestFeatures() { return testFeatures; }

  //! Get the test dataset labels.
  DatasetY TestLabels() const { return testLabels; }
  //! Modify the test dataset labels.
  DatasetY& TestLabels() { return testLabels; }

  //! Get the validation dataset features.
  DatasetX ValidFeatures() const { return validFeatures; }
  //! Modify the validation dataset features.
  DatasetX& ValidFeatures() { return validFeatures; }

  //! Get the validation dataset labels.
  DatasetY ValidLabels() const { return validLabels; }
  //! Modify the validation dataset labels.
  DatasetY& ValidLabels() { return validLabels; }

  //! Get the training dataset.
  std::tuple<DatasetX, DatasetY> TrainSet() const
  {
    return std::tuple<DatasetX, DatasetY>(trainFeatures, trainLabels);
  }

  //! Get the validation dataset.
  std::tuple<DatasetX, DatasetY> ValidSet() const
  {
    return std::tuple<DatasetX, DatasetY>(validFeatures, validLabels);
  }

  //! Get the testing dataset.
  std::tuple<DatasetX, DatasetY> TestSet() const
  {
    return std::tuple<DatasetX, DatasetY>(testFeatures, testLabels);
  }

  //! Get the Scaler.
  ScalerType Scaler() const { return scaler; }
  //! Modify the Scaler.
  ScalerType& Scaler() { return scaler; }

 private:
  /**
   * Downloads and checks hash for given dataset.
   *
   * @param dataset Name of the data set which will be downloaded.
   */
  void DownloadDataset(const std::string& dataset)
  {
    if (datasetMap[dataset].zipFile && (!Utils::PathExists(
        datasetMap[dataset].trainPath) ||
        !Utils::PathExists(datasetMap[dataset].testPath)))
    {
      Utils::DownloadFile(datasetMap[dataset].datasetURL,
          datasetMap[dataset].datasetPath, dataset + "_training_data.",
          false, false, datasetMap[dataset].serverName,
          datasetMap[dataset].zipFile);

      if (!Utils::CompareCRC32(datasetMap[dataset].datasetPath,
          datasetMap[dataset].datasetHash))
      {
        mlpack::Log::Fatal << "Corrupted Data for " << dataset <<
            " downloaded." << std::endl;
      }

      return;
    }

    if (!Utils::PathExists(datasetMap[dataset].trainPath))
    {
      Utils::DownloadFile(datasetMap[dataset].trainDownloadURL,
          datasetMap[dataset].trainPath, dataset + "_training_data.",
          false, false, datasetMap[dataset].serverName);

      if (!Utils::CompareCRC32(datasetMap[dataset].trainPath,
          datasetMap[dataset].trainHash))
      {
        mlpack::Log::Fatal << "Corrupted Training Data for " <<
            dataset << " downloaded." << std::endl;
      }
    }

    if (!Utils::PathExists(datasetMap[dataset].testPath))
    {
      Utils::DownloadFile(datasetMap[dataset].trainDownloadURL,
          datasetMap[dataset].testPath, dataset + "_testing_data.",
          false, false, datasetMap[dataset].serverName);

      if (!Utils::CompareCRC32(datasetMap[dataset].testPath,
          datasetMap[dataset].testHash))
        {
          mlpack::Log::Fatal << "Corrupted Testing Data for " <<
            dataset << " downloaded." << std::endl;
        }
    }
  }

  /**
   * Intializes dataset map to provide access to dataset details.
   */
  void InitializeDatasets()
  {
    datasetMap.insert({"mnist", Datasets<DatasetX, DatasetY>::MNIST()});
  }

  /**
   * Utility Function to wrap indices.
   *
   * @param index Index that will wraped over the length.
   * @param length Size / length of array.
   */

  size_t WrapIndex(int index, size_t length)
  {
    if (index < 0)
      return length - size_t(std::abs(index));

    return index;
  }

  //! Locally stored mappings for some well known datasets.
  std::unordered_map<std::string,
      DatasetDetails<DatasetX, DatasetY>> datasetMap;

  //! Locally stored input features for training.
  DatasetX trainFeatures;
  //! Locally stored input features for testing.
  DatasetX validFeatures;
  //! Locally stored input features for validation.
  DatasetX testFeatures;

  //! Locally stored labels for training.
  DatasetY trainLabels;
  //! Locally stored labels for validation.
  DatasetY validLabels;
  //! Locally stored labels for testing.
  DatasetY testLabels;

  //! Locally Stored scaler.
  ScalerType scaler;

  //! Locally stored path of dataset.
  std::string trainDatasetPath;

  //! Locally stored path of dataset.
  std::string testDatasetPath;

  //! Locally stored ratio for train-test split.
  double ratio;

  //! Locally stored augmentation.
  std::vector<std::string> augmentation;

  //! Locally stored augmentation probability.
  double augmentationProbability;
};

#include "dataloader_impl.hpp" // Include implementation.

#endif
