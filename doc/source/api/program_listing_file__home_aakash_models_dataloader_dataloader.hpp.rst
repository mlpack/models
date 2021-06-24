
.. _program_listing_file__home_aakash_models_dataloader_dataloader.hpp:

Program Listing for File dataloader.hpp
=======================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_models_dataloader_dataloader.hpp>` (``/home/aakash/models/dataloader/dataloader.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MODELS_DATALOADER_DATALOADER_HPP
   #define MODELS_DATALOADER_DATALOADER_HPP
   
   #include <mlpack/core/data/scaler_methods/min_max_scaler.hpp>
   #include <boost/property_tree/xml_parser.hpp>
   #include <mlpack/core/math/shuffle_data.hpp>
   #include <mlpack/core/data/split_data.hpp>
   #include <boost/property_tree/ptree.hpp>
   #include <augmentation/augmentation.hpp>
   #include <dataloader/datasets.hpp>
   #include <mlpack/prereqs.hpp>
   #include <boost/foreach.hpp>
   #include <mlpack/core.hpp>
   #include <utils/utils.hpp>
   #include <set>
   
   namespace mlpack {
   namespace models {
   
   template<
     typename DatasetX = arma::mat,
     typename DatasetY = arma::mat,
     class ScalerType = mlpack::data::MinMaxScaler
   >
   class DataLoader
   {
    public:
     DataLoader();
   
     DataLoader(const std::string& dataset,
                const bool shuffle,
                const double validRatio = 0.25,
                const bool useScaler = true,
                const std::vector<std::string> augmentation =
                    std::vector<std::string>(),
                const double augmentationProbability = 0.2);
   
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
   
     void LoadAllImagesFromDirectory(const std::string& imagesPath,
                                     DatasetX& dataset,
                                     DatasetY& labels,
                                     const size_t imageWidth,
                                     const size_t imageHeight,
                                     const size_t imageDepth,
                                     const size_t label = 0);
   
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
   
     DatasetX TrainFeatures() const { return trainFeatures; }
   
     DatasetX& TrainFeatures() { return trainFeatures; }
   
     DatasetY TrainLabels() const { return trainLabels; }
     DatasetY& TrainLabels() { return trainLabels; }
   
     DatasetX TestFeatures() const { return testFeatures; }
     DatasetX& TestFeatures() { return testFeatures; }
   
     DatasetY TestLabels() const { return testLabels; }
     DatasetY& TestLabels() { return testLabels; }
   
     DatasetX ValidFeatures() const { return validFeatures; }
     DatasetX& ValidFeatures() { return validFeatures; }
   
     DatasetY ValidLabels() const { return validLabels; }
     DatasetY& ValidLabels() { return validLabels; }
   
     std::tuple<DatasetX, DatasetY> TrainSet() const
     {
       return std::tuple<DatasetX, DatasetY>(trainFeatures, trainLabels);
     }
   
     std::tuple<DatasetX, DatasetY> ValidSet() const
     {
       return std::tuple<DatasetX, DatasetY>(validFeatures, validLabels);
     }
   
     std::tuple<DatasetX, DatasetY> TestSet() const
     {
       return std::tuple<DatasetX, DatasetY>(testFeatures, testLabels);
     }
   
     ScalerType Scaler() const { return scaler; }
     ScalerType& Scaler() { return scaler; }
   
    private:
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
   
     void InitializeDatasets()
     {
       datasetMap.insert({"mnist", Datasets<DatasetX, DatasetY>::MNIST()});
       datasetMap.insert({"voc-detection",
           Datasets<DatasetX, DatasetY>::VOCDetection()});
       datasetMap.insert({"cifar10", Datasets<DatasetX, DatasetY>::CIFAR10()});
     }
   
     size_t WrapIndex(int index, size_t length)
     {
       if (index < 0)
         return length - size_t(std::abs(index));
   
       return index;
     }
   
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
   
     std::unordered_map<std::string,
         DatasetDetails<DatasetX, DatasetY>> datasetMap;
   
     DatasetX trainFeatures;
     DatasetX validFeatures;
     DatasetX testFeatures;
   
     DatasetY trainLabels;
     DatasetY validLabels;
     DatasetY testLabels;
   
     ScalerType scaler;
   
     std::string trainDatasetPath;
   
     std::string testDatasetPath;
   
     double ratio;
   
     std::vector<std::string> augmentation;
   
     double augmentationProbability;
   };
   
   } // namespace models
   } // namespace mlpack
   
   #include "dataloader_impl.hpp" // Include implementation.
   
   #endif
