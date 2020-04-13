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
   * @param augmentation Adds augmentation to training data only.
   * @param augmentationProbability Probability of applying augmentation on dataset.
   */
  DataLoader(const std::string &dataset,
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
  void LoadCSV(const std::string &datasetPath,
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

  //! Get the Training Dataset.
  DatasetX TrainX() const { return trainX; }

  //! Modify the Training Dataset.
  DatasetX& TrainX() { return trainX; }

  //! Get the Training Dataset.
  DatasetY TrainY() const { return trainY; }
  //! Modify the Training Dataset.
  DatasetY& TrainY() { return trainY; }

  //! Get the Test Dataset.
  DatasetX TestX() const { return testX; }
  //! Modify the Test Dataset.
  DatasetX& TestX() { return testX; }

  //! Get the Test Dataset.
  DatasetY TestY() const { return testY; }
  //! Modify the Training Dataset.
  DatasetY& TestY() { return testY; }

  //! Get the Validation Dataset.
  DatasetX ValidX() const { return validX; }
  //! Modify the Validation Dataset.
  DatasetX& ValidX() { return validX; }

  //! Get the Validation Dataset.
  DatasetY ValidY() const { return validY; }
  //! Modify the Validation Dataset.
  DatasetY& ValidY() { return validY; }

  //! Get the Scaler.
  ScalerType Scaler() const { return scaler; }
  //! Modify the Sclaer.
  ScalerType& Scaler() { return scaler; }

 private:
  /**
   * Downloads and checks hash for given dataset.
   */
  void DownloadDataset(const std::string& dataset)
  {
    if (!Utils::PathExists(datasetMap[dataset].trainPath))
    {
      Utils::DownloadFile(datasetMap[dataset].trainDownloadUrl,
          datasetMap[dataset].trainPath, dataset + "_training_data.");

      if (!Utils::CompareSHA256(datasetMap[dataset].trainPath,
          datasetMap[dataset].trainHash))
      {
        mlpack::Log::Fatal << "Corrupted Training Data for " <<
            dataset << " Downloaded." << std::endl;
      }
    }
    if (!Utils::PathExists(datasetMap[dataset].testPath))
    {
      Utils::DownloadFile(datasetMap[dataset].trainDownloadUrl,
          datasetMap[dataset].testPath, dataset + "_testing_data.");

      if (!Utils::CompareSHA256(datasetMap[dataset].testPath,
          datasetMap[dataset].testHash))
        mlpack::Log::Fatal << "Corrupted Testing Data for " <<
            dataset << " Downloaded." << std::endl;
    }
  }

  /**
   * Intializes dataset map to provide access to dataset details.
   */
  void InitializeDatasets()
  {
    datasetMap.insert({"mnist", Datasets::MNIST()});
  }

  // Utility Function to wrap indices.
  size_t WrapIndex(int index, size_t length)
  {
    if (index < 0)
      return length - size_t(std::abs(index));

    return index;
  }

  //! Locally stored mappings for some well known datasets.
  std::unordered_map<std::string, DatasetDetails> datasetMap;

  //! Locally stored input for training.
  DatasetX trainX;
  //! Locally stored input for testing.
  DatasetX validX;
  //! Locally stored input for validation.
  DatasetX testX;

  //! Locally stored labels for training.
  DatasetY trainY;
  //! Locally stored labels for validation.
  DatasetY validY;
  //! Locally stored labels for testing.
  DatasetY testY;

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

  //! Locally stored augmented probability.
  double augmentationProbability;
};

#include "dataloader_impl.hpp" // Include implementation.

#endif
