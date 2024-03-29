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
#ifndef MODELS_DATALOADER_DATALOADER_HPP
#define MODELS_DATALOADER_DATALOADER_HPP

#include <mlpack.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/ptree.hpp>
#include <augmentation/augmentation.hpp>
#include <dataloader/datasets.hpp>
#include <boost/foreach.hpp>
#include <utils/utils.hpp>
#include <set>

namespace mlpack {
namespace models {

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
   * @param validRatio Ratio of dataset to be used for validation set.
   * @param useScaler Use feature scaler for pre-processing the dataset.
   * @param augmentation Adds augmentation to training data only.
   * @param augmentationProbability Probability of applying augmentation on dataset.
   */
  DataLoader(const std::string& dataset,
             const bool shuffle,
             const double validRatio = 0.25,
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
   * @param validRatio Ratio of dataset to be used for validation set.
   * @param useScaler Fits the scaler on training data and transforms dataset.
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
               const double validRatio = 0.25,
               const bool useScaler = false,
               const int startInputFeatures = -1,
               const int endInputFeatures = -1,
               const int startPredictionFeatures = -1,
               const int endPredictionFeatures = -1,
               const std::vector<std::string> augmentation =
                   std::vector<std::string>(),
               const double augmentationProbability = 0.2);

  /**
   * Loads object detection dataset. It requires a single annotation file in XML format.
   * Each XML file should correspond to a single image in images folder.
   *
   * XML file should containg the following :
   * 1. Each XML file should be wrapped in annotation tag.
   * 2. Filename of image in images folder will be depicted by filename tag.
   * 3. Object tag depicting characteristics of bounding box.
   * 4. Each object tag should contain name tag i.e. class of the object.
   * 5. Each object tag should contain bndbox tag containing xmin, ymin, xmax, ymax.
   *
   * NOTE : Labels are assigned using classes vector. Set verbose to 1 to print labels
   * and their corresponding class. The labels type should be field type here.
   *
   * @param pathToAnnotations Path to the folder containing XML type annotation files.
   * @param pathToImages Path to folder containing images corresponding to annotations.
   * @param classes Vector of strings containing list of classes. Labels are assigned
   *                according to this vector.
   * @param validRatio Ratio of dataset that will be used for validation.
   * @param shuffle Boolean to determine whether the dataset is shuffled.
   * @param augmentation Vector strings of augmentations supported by mlpack.
   * @param augmentationProbability Probability of applying augmentation
   *                                to a particular image.
   * @param absolutePath Boolean to determine if absolute path is used. Defaults to false.
   * @param baseXMLTag XML tag name which wraps around the annotation file.
   * @param imageNameXMLTag XML tag name which holds the value of image filename.
   * @param objectXMLTag XML tag name which holds details of bounding box i.e. class and
   *                     coordinates of bounding box.
   * @param bndboxXMLTag XML tag name which holds coordinates of bounding box.
   * @param classNameXMLTag XML tag name inside objectXMLTag which holds the name of the
   *                        class of bounding box.
   * @param x1XMLTag XML tag name inside bndboxXMLTag which hold value of lower most
   *                 x coordinate of bounding box.
   * @param y1XMLTag XML tag name inside bndboxXMLTag which hold value of lower most
   *                 y coordinate of bounding box.
   * @param x2XMLTag XML tag name inside bndboxXMLTag which hold value of upper most
   *                 x coordinate of bounding box.
   * @param y2XMLTag XML tag name inside bndboxXMLTag which hold value of upper most
   *                 y coordinate of bounding box.
   */
  void LoadObjectDetectionDataset(const std::string& pathToAnnotations,
                                  const std::string& pathToImages,
                                  const std::vector<std::string>& classes,
                                  const double validRatio = 0.2,
                                  const bool shuffle = true,
                                  const std::vector<std::string>& augmentation =
                                      std::vector<std::string>(),
                                  const double augmentationProbability = 0.2,
                                  const bool absolutePath = false,
                                  const std::string& baseXMLTag = "annotation",
                                  const std::string& imageNameXMLTag =
                                      "filename",
                                  const std::string& sizeXMLTag = "size",
                                  const std::string& objectXMLTag = "object",
                                  const std::string& bndboxXMLTag = "bndbox",
                                  const std::string& classNameXMLTag = "name",
                                  const std::string& x1XMLTag = "xmin",
                                  const std::string& y1XMLTag = "ymin",
                                  const std::string& x2XMLTag = "xmax",
                                  const std::string& y2XMLTag = "ymax");

  /**
   * Load all images from directory.
   *
   * @param imagesPath Path to all images.
   * @param dataset Armadillo type where images will be loaded.
   * @param labels Armadillo type where labels will be loaded.
   * @param imageWidth Width of images in dataset.
   * @param imageHeight Height of images in dataset.
   * @param imageDepth Depth of images in dataset.
   * @param label Label which will be assigned to image.
   * @param augmentation Vector strings of augmentations supported by mlpack.
   * @param augmentationProbability Probability of applying augmentation
   *                                to a particular image.
   */
  void LoadAllImagesFromDirectory(const std::string& imagesPath,
                                  DatasetX& dataset,
                                  DatasetY& labels,
                                  const size_t imageWidth,
                                  const size_t imageHeight,
                                  const size_t imageDepth,
                                  const size_t label = 0);

  /**
   * Load all images from directory.
   *
   * @param pathToDataset Path to all folders containing all images.
   * @param imageWidth Width of images in dataset.
   * @param imageHeight Height of images in dataset.
   * @param imageDepth Depth of images in dataset.
   * @param trainData Determines whether data is training set or test set.
   * @param shuffle Boolean to determine whether or not to shuffle the data.
   * @param validRatio Ratio of dataset to be used for validation set.
   * @param augmentation Vector strings of augmentations supported by mlpack.
   * @param augmentationProbability Probability of applying augmentation
   *                                to a particular image.
   */
  void LoadImageDatasetFromDirectory(const std::string& pathToDataset,
                                     const size_t imageWidth,
                                     const size_t imageHeight,
                                     const size_t imageDepth,
                                     const bool trainData = true,
                                     const double validRatio = 0.2,
                                     const bool shuffle = true,
                                     const std::vector<std::string>&
                                      augmentation = std::vector<std::string>(),
                                     const double augmentationProbability =
                                        0.2);

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
        !Utils::PathExists(datasetMap[dataset].testPath) ||
        !Utils::PathExists(datasetMap[dataset].trainingImagesPath) ||
        !Utils::PathExists(datasetMap[dataset].trainingAnnotationPath) ||
        !Utils::PathExists(datasetMap[dataset].testingImagesPath)))
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
    datasetMap.insert({"voc-detection",
        Datasets<DatasetX, DatasetY>::VOCDetection()});
    datasetMap.insert({"cifar10", Datasets<DatasetX, DatasetY>::CIFAR10()});
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

  /**
   * Performs train test split.
   *
   * @tparam DatasetType Type of input dataset.
   * @tparam LabelsType Type of input labels.
   *
   * @param dataset Features of dataset.
   * @param labels Labels of the dataset.
   * @param validRatio Ratio for train-test split.
   * @param shuffle Boolean to determine shuffling of dataset.
   */
  void TrainTestSplit(DatasetX& dataset,
                      std::deque<arma::vec>& labels,
                      DatasetX& /* trainFeatures */,
                      arma::field<arma::vec>& /* trainLabels */,
                      DatasetX& /* validFeatures */,
                      arma::field<arma::vec>& /* validLabels */,
                      const double validRatio,
                      const bool shuffle)
  {
    const size_t validSize = static_cast<size_t>(dataset.n_cols * validRatio);
    const size_t trainSize = dataset.n_cols - validSize;

    arma::uvec order = arma::linspace<arma::uvec>(0, dataset.n_cols - 1,
        dataset.n_cols);
    if (shuffle)
      order = arma::shuffle(order);

    if (trainSize > 0)
    {
      trainLabels.set_size(1, trainSize);
      trainFeatures = dataset.cols(order.subvec(0, trainSize - 1));

      // Field type has fixed size so we can't use span and assignment
      // operator.
      for (size_t i = 0; i < trainSize; i++)
        trainLabels(0, i) = labels[i];
    }

    if (validSize <= dataset.n_cols)
    {
      validFeatures = dataset.cols(order.subvec(trainSize,
          dataset.n_cols - 1));
      validLabels.set_size(1, validSize);
      for (size_t i = trainSize; i < dataset.n_cols; i++)
        validLabels(0, i - trainSize) = labels[i];
    }
    return;
  }

  /**
   * Performs train/test split.
   *
   * @tparam DatasetType Type of input dataset.
   * @tparam LabelsType Type of input labels.
   *
   * @param dataset Features of dataset.
   * @param labels Labels of the dataset.
   * @param validRatio Ratio for train-test split.
   * @param shuffle Boolean to determine shuffling of dataset.
   */
  void TrainTestSplit(DatasetX& dataset,
                      std::deque<arma::vec>& labels,
                      DatasetX& /* trainFeatures */,
                      arma::mat& /* trainLabels */,
                      DatasetX& /* validFeatures */,
                      arma::mat& /* validLabels */,
                      const double validRatio,
                      const bool shuffle)
  {
    // Calculate number of objects in the image.
    size_t numberOfObjects = labels[0].n_rows;
    DatasetY labelsTemp(numberOfObjects, labels.size());

    for (size_t i = 0; i < labels.size(); i++)
      labelsTemp.col(i) = labels[i];

    DatasetX completeDataset = arma::join_cols(dataset, labelsTemp);
    mlpack::data::Split(completeDataset, trainFeatures, validFeatures,
        validRatio, shuffle);

    // Features are all rows except the last 5 rows which correspond
    // to bounding box.
    trainLabels = trainFeatures.rows(trainFeatures.n_rows -
        numberOfObjects, trainFeatures.n_rows - 1);
    trainFeatures = trainFeatures.rows(0, trainFeatures.n_rows -
        numberOfObjects - 1);

    validLabels = validFeatures.rows(validFeatures.n_rows -
        numberOfObjects, validFeatures.n_rows - 1);
    validFeatures = validFeatures.rows(0, validFeatures.n_rows -
        numberOfObjects - 1);
    return;
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

} // namespace models
} // namespace mlpack

#include "dataloader_impl.hpp" // Include implementation.

#endif
