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
              const double ratio,
              const bool useScaler,
              const std::vector<std::string> augmentation,
              const double augmentationProbability)
{
  InitializeDatasets();
  if (datasetMap.count(dataset))
  {
    // Use utility functions to download the dataset.
    DownloadDataset(dataset);

    if (datasetMap[dataset].loadCSV)
    {
      LoadCSV(datasetMap[dataset].trainPath, true, shuffle, ratio, useScaler,
              datasetMap[dataset].dropHeader,
              datasetMap[dataset].startTrainingInputFeatures,
              datasetMap[dataset].endTrainingInputFeatures,
              datasetMap[dataset].endTrainingPredictionFeatures,
              datasetMap[dataset].endTrainingPredictionFeatures);

      LoadCSV(datasetMap[dataset].testPath, false, false, useScaler,
              datasetMap[dataset].dropHeader,
              datasetMap[dataset].startTestingInputFeatures,
              datasetMap[dataset].endTestingInputFeatures);
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
           const double ratio,
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
    data::Split(dataset, trainDataset, validDataset, ratio, shuffle);

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
    // TODO : Add support for augmentation here.
    mlpack::Log::Info << "Training Dataset Loaded." << std::endl;
  }
  else
  {
    if (useScaler)
    {
      scaler.Transform(dataset, dataset);
    }

    testFeatures = dataset.submat(WrapIndex(startInputFeatures, dataset.n_rows),
        0, WrapIndex(endInputFeatures, dataset.n_rows), dataset.n_cols - 1);
    mlpack::Log::Info << "Testing Dataset Loaded." << std::endl;
  }
}

#endif
