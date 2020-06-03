/**
 * @file dataloader_impl.hpp
 * @author Kartik Dutt
 * 
 * Implementation of DataLoader.
 * 
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MODELS_DATALOADER_IMPL_HPP
#define MODELS_DATALOADER_IMPL_HPP

#include "dataloader.hpp"

using namespace mlpack;

template<
  typename DatasetX,
  typename DatasetY,
  class ScalerType
>DataLoader<
    DatasetX, DatasetY, ScalerType
>::DataLoader()
{
  // Nothing to do here.
}

template<
  typename DatasetX,
  typename DatasetY,
  class ScalerType
>DataLoader<
    DatasetX, DatasetY, ScalerType
>::DataLoader(const std::string& dataset,
              const bool shuffle,
              const double validRatio,
              const bool useScaler,
              const std::vector<std::string> augmentation,
              const double augmentationProbability)
{
  InitializeDatasets();
  if (datasetMap.count(dataset))
  {
    // Use utility functions to download the dataset.
    DownloadDataset(dataset);

    if (datasetMap[dataset].datasetType == "csv")
    {
      LoadCSV(datasetMap[dataset].trainPath, true, shuffle, validRatio,
              useScaler, datasetMap[dataset].dropHeader,
              datasetMap[dataset].startTrainingInputFeatures,
              datasetMap[dataset].endTrainingInputFeatures,
              datasetMap[dataset].endTrainingPredictionFeatures,
              datasetMap[dataset].endTrainingPredictionFeatures);

      LoadCSV(datasetMap[dataset].testPath, false, false, validRatio, useScaler,
              datasetMap[dataset].dropHeader,
              datasetMap[dataset].startTestingInputFeatures,
              datasetMap[dataset].endTestingInputFeatures);
    }
    else if (datasetMap[dataset].datasetType == "image-detection")
    {
      std::vector<std::string> augmentations = augmentation;
      // If user doesn't set size for images, set size of images to {64, 64}.
      if (augmentations.size() == 0)
      {
        augmentations.push_back("resize = {64, 64}");
      }

      LoadObjectDetectionDataset(datasetMap[dataset].trainingAnnotationPath,
          datasetMap[dataset].trainingImagesPath, validRatio,
          datasetMap[dataset].classes, augmentations, augmentationProbability);
    }

    // Preprocess the dataset.
    datasetMap[dataset].PreProcess(trainFeatures, trainLabels,
        validFeatures, validLabels, testFeatures);
  }
  else
  {
    mlpack::Log::Fatal << "Unknown Dataset. " << dataset <<
        " For other datasets try loading data using" <<
        " generic dataloader functions such as LoadCSV." <<
        " Refer to the documentation for more info." << std::endl;
  }
}


template<
  typename DatasetX,
  typename DatasetY,
  class ScalerType
> void DataLoader<
    DatasetX, DatasetY, ScalerType
>::LoadCSV(const std::string& datasetPath,
           const bool loadTrainData,
           const bool shuffle,
           const double validRatio,
           const bool useScaler,
           const bool dropHeader,
           const int startInputFeatures,
           const int endInputFeatures,
           const int startPredictionFeatures,
           const int endPredictionFeatures,
           const std::vector<std::string> augmentation,
           const double augmentationProbability)
{
  arma::mat dataset;
  data::Load(datasetPath, dataset, true);

  dataset = dataset.submat(0, size_t(dropHeader), dataset.n_rows - 1,
      dataset.n_cols - 1);

  if (loadTrainData)
  {
    arma::mat trainDataset, validDataset;
    data::Split(dataset, trainDataset, validDataset, validRatio, shuffle);

    trainFeatures = trainDataset.rows(WrapIndex(startInputFeatures,
        trainDataset.n_rows), WrapIndex(endInputFeatures,
        trainDataset.n_rows));

    trainLabels = trainDataset.rows(WrapIndex(startPredictionFeatures,
        trainDataset.n_rows), WrapIndex(endPredictionFeatures,
        trainDataset.n_rows));

    validFeatures = validDataset.rows(WrapIndex(startInputFeatures,
        validDataset.n_rows), WrapIndex(endInputFeatures,
        validDataset.n_rows));

    validLabels = validDataset.rows(WrapIndex(startPredictionFeatures,
        validDataset.n_rows), WrapIndex(endPredictionFeatures,
        validDataset.n_rows));

    if (useScaler)
    {
      scaler.Fit(trainFeatures);
      scaler.Transform(trainFeatures, trainFeatures);
      scaler.Transform(validFeatures, validFeatures);
    }

    Augmentation<DatasetX> augmentations(augmentation,
        augmentationProbability);
    augmentations.Transform(trainFeatures, 1, dataset.n_rows, 1);

    mlpack::Log::Info << "Training Dataset Loaded." << std::endl;
  }
  else
  {
    if (useScaler)
    {
      scaler.Transform(dataset, dataset);
    }

    testFeatures = dataset.rows(WrapIndex(startInputFeatures, dataset.n_rows),
        WrapIndex(endInputFeatures, dataset.n_rows));

    mlpack::Log::Info << "Testing Dataset Loaded." << std::endl;
  }
}

template<
  typename DatasetX,
  typename DatasetY,
  class ScalerType
> void DataLoader<
    DatasetX, DatasetY, ScalerType
>::LoadObjectDetectionDataset(const std::string& pathToAnnotations,
                              const std::string& pathToImages,
                              const double validRatio,
                              const std::vector<std::string>& classes,
                              const std::vector<std::string>& augmentations,
                              const double augmentationProbability,
                              const bool absolutePath,
                              const std::string& baseXMLTag,
                              const std::string& imageNameXMLTag,
                              const std::string& sizeXMLTag,
                              const std::string& objectXMLTag,
                              const std::string& bndboxXMLTag,
                              const std::string& classNameXMLTag,
                              const std::string& x1XMLTag,
                              const std::string& y1XMLTag,
                              const std::string& x2XMLTag,
                              const std::string& y2XMLTag)
{
  Augmentation<DatasetX> augmentation(augmentations, augmentationProbability);

  std::vector<boost::filesystem::path> annotationsDirectory, imagesDirectory;

  // Fill the directory.
  Utils::ListDir(pathToAnnotations, annotationsDirectory, absolutePath);
  Utils::ListDir(pathToImages, imagesDirectory, absolutePath);

  // Create a map for labels and corresponding class name.
  // This provides faster access to class labels.
  std::unordered_map<std::string, size_t> classMap;
  for (size_t i = 0; i < classes.size(); i++)
    classMap.insert(std::make_pair(classes[i], i));


  // Map to insert values in a column vector.
  std::unordered_map<std::string, size_t> indexMap;
  indexMap.insert(std::make_pair(classNameXMLTag, 0));
  indexMap.insert(std::make_pair(x1XMLTag, 1));
  indexMap.insert(std::make_pair(y1XMLTag, 2));
  indexMap.insert(std::make_pair(x2XMLTag, 3));
  indexMap.insert(std::make_pair(y2XMLTag, 4));

  // Keep track of files loaded.
  size_t totalFiles = annotationsDirectory.size(), loadedFiles = 0;

  // Read the XML file.
  for (boost::filesystem::path annotationFile : annotationsDirectory)
  {
    if (annotationFile.string().length() <= 3 ||
        annotationFile.string().substr(
            annotationFile.string().length() - 3) != "xml")
    {
      continue;
    }

    loadedFiles++;
    Log::Info << "Files Loaded : " << loadedFiles << " / " << totalFiles <<
        std::endl;
    // Read the XML file.
    boost::property_tree::ptree xmlFile;
    boost::property_tree::read_xml(annotationFile.string(), xmlFile);

    // Get annotation from XML file.
    boost::property_tree::ptree annotation = xmlFile.get_child(baseXMLTag);

    // Read properties inside annotation file.
    // Get image name.
    std::string imgName = annotation.get_child(imageNameXMLTag).data();

    // If image doesn't exist then skip the current XML file.
    if (!Utils::PathExists(pathToImages + imgName, absolutePath))
    {
      mlpack::Log::Warn << "Image not found! Tried finding " <<
          pathToImages + imgName << std::endl;
      continue;
    }

    // Get the size of image to create image info required
    // by mlpack::data::Load function.
    boost::property_tree::ptree sizeInfo = annotation.get_child(sizeXMLTag);
    size_t imageWidth = std::stoi(sizeInfo.get_child("width").data());
    size_t imageHeight = std::stoi(sizeInfo.get_child("height").data());
    size_t imageDepth = std::stoi(sizeInfo.get_child("depth").data());
    mlpack::data::ImageInfo imageInfo(imageWidth, imageHeight, imageDepth);

    // Load the image.
    // The image loaded here will be in column format i.e. Output will
    // be matrix with the following shape {1, cols * rows * slices} in
    // column major format.
    DatasetX image;
    mlpack::data::Load(pathToImages + imgName, image, imageInfo);

    if (augmentation.HasResizeParam())
    {
      augmentation.ResizeTransform(image, imageWidth, imageHeight, imageDepth,
          augmentation.augmentations[0]);
    }

    // Iterate over all object in annotation.
    BOOST_FOREACH(boost::property_tree::ptree::value_type const& object,
        annotation)
    {
      arma::vec predictions(5);
      // Iterate over property of the object to get class label and
      // bounding box coordinates.
      if (object.first == objectXMLTag)
      {
        if (classMap.count(object.second.get_child(classNameXMLTag).data()))
        {
          predictions(indexMap[classNameXMLTag]) = classMap[
              object.second.get_child(classNameXMLTag).data()];
          boost::property_tree::ptree const &boundingBox =
              object.second.get_child(bndboxXMLTag);

          BOOST_FOREACH(boost::property_tree::ptree::value_type
              const& coordinate, boundingBox)
          {
            if (indexMap.count(coordinate.first))
            {
              predictions(indexMap[coordinate.first]) =
                  std::stoi(coordinate.second.data());
            }
          }

          // Add object to training set.
          trainFeatures.insert_cols(0, image);
          trainLabels.insert_cols(0, predictions);
        }
      }
    }
    // Add augmentation and split here.
  }
}

#endif
