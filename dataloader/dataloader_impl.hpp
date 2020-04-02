/**
 * @file dataloader_impl.hpp
 * @author Eugene Freyman
 * @author Mehul Kumar Nirala.
 * @author Zoltan Somogyi
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
  typename DataSetX,
  typename DataSetY,
  class ScalerType
>DataLoader<
    DataSetX, DataSetY, ScalerType
>::DataLoader()
{
  // Nothing to do here.
}

template<
  typename DataSetX,
  typename DataSetY,
  class ScalerType
>DataLoader<
    DataSetX, DataSetY, ScalerType
>::DataLoader(const std::string &dataset,
              const bool shuffle,
              const double ratio,
              const bool useScaler,
              const std::vector<std::string> augmentation,
              const double augmentationProbability)
{
  if (dataset == "mnist")
  {
    if (!Utils::PathExists("./../data/mnist_train.csv"))
    {
      Utils::DownloadFile(Datasets::MNIST().trainDownloadUrl, "./../data/mnist_train.csv",
          "mnist_train.csv");

      if (!Utils::CompareSHA256("./../data/mnist_train.csv", Datasets::MNIST().trainHash))
        std::cout << "Corrupted Train Data Downloaded." << std::endl;
    }

    LoadCSV("./../data/mnist_train.csv", true, true, ratio, useScaler, true,
        1, -1, 0, 0);
    trainY = trainY + 1;
    validY = validY + 1;

    if (!Utils::PathExists("./../data/mnist_test.csv"))
    {
      Utils::DownloadFile(Datasets::MNIST().testDownloadUrl, "./../data/mnist_test.csv",
          "mnist_test.csv");

      if (!Utils::CompareSHA256("./../data/mnist_test.csv", Datasets::MNIST().testHash))
        std::cout << "Corrupted Test Data Downloaded." << std::endl;
    }

    LoadCSV("./../data/mnist_test.csv", false, false, useScaler, true, 0, -1);
  }
}


template<
  typename DataSetX,
  typename DataSetY,
  class ScalerType
> void DataLoader<
    DataSetX, DataSetY, ScalerType
>::LoadCSV(const std::string &datasetPath,
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
    data::Split(dataset, trainDataset, validDataset, ratio /* Add option for shuffle here.*/);

    if (useScaler)
    {
      scaler.Fit(trainDataset);
      scaler.Transform(trainDataset, trainDataset);
      scaler.Transform(validDataset, validDataset);
    }

    trainX = trainDataset.rows(wrapIndex(startInputFeatures, trainDataset.n_rows),
        wrapIndex(endInputFeatures, trainDataset.n_rows));

    trainY = trainDataset.rows(wrapIndex(startPredictionFeatures, trainDataset.n_rows),
        wrapIndex(endPredictionFeatures, trainDataset.n_rows));

    validX = validDataset.rows(wrapIndex(startInputFeatures, validDataset.n_rows),
        wrapIndex(endInputFeatures, validDataset.n_rows));

    validY = trainDataset.rows(wrapIndex(startPredictionFeatures, validDataset.n_rows),
        wrapIndex(endPredictionFeatures, validDataset.n_rows));

    // Add support for augmentation here.
    std::cout << "Training Dataset Loaded." << std::endl;
  }
  else
  {
    if (useScaler)
    {
      scaler.Transform(dataset, dataset);
    }

    testX = dataset.submat(wrapIndex(startInputFeatures, dataset.n_rows),
      0, wrapIndex(endInputFeatures, dataset.n_rows), dataset.n_cols - 1);
    std::cout << "Testing Dataset Loaded." << std::endl;
  }
}

#endif