/**
 * @file preprocessor.hpp
 * @author Kartik Dutt
 * 
 * Definition of PreProcessor class for popular datasets.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MODELS_DATALOADER_PREPROCESSOR_HPP
#define MODELS_DATALOADER_PREPROCESSOR_HPP

#include <mlpack.hpp>

namespace mlpack {
namespace models {

/**
 * Contains standatd pre-process functions for popular datasets.
 *
 * @tparam DatasetX Datatype for loading input features.
 * @tparam DatasetY Datatype for prediction features.
 */
template<
  typename DatasetX = arma::mat,
  typename DatasetY = arma::mat
>
class PreProcessor
{
 public:
  static void MNIST(DatasetX& /* trainX */,
                    DatasetY& trainY,
                    DatasetY& /* validX */,
                    DatasetY& validY,
                    DatasetX& /* testX */)
  {
    trainY = trainY + 1;
    validY = validY + 1;
  }

  static void PascalVOC(DatasetX& /* trainX */,
                        DatasetY& /* trainY */,
                        DatasetY& /* validX */,
                        DatasetY& /* validY */,
                        DatasetX& /* testX */)
  {
    // Nothing to do here. Added to match the rest of the codebase.
  }

  static void CIFAR10(DatasetX & /* trainX */,
                      DatasetY & /* trainY */,
                      DatasetY & /* validX */,
                      DatasetY & /* validY */,
                      DatasetX & /* testX */)
  {
    // Nothing to do here. Added to match the rest of the codebase.
  }

  /**
   * Converts image to channel first format used in PyTorch. Performs the same function
   * as torch.transforms.ToTensor().
   *
   * @param trainFeatures Input features that will be converted into channel first format.
   * @param imageWidth Width of the image in dataset.
   * @param imageHeight Height of the image in dataset.
   * @param imageDepth Depth / Number of channels of the image in dataset.
   */
  static void ChannelFirstImages(DatasetX& trainFeatures,
      const size_t imageWidth,
      const size_t imageHeight,
      const size_t imageDepth,
      bool normalize = true)
  {
    for (size_t idx = 0; idx < trainFeatures.n_cols; idx++)
    {
        // Create a copy of the current image so that the image isn't affected.
        arma::cube inputTemp(trainFeatures.col(idx).memptr(), 3, 224, 224);

        size_t currentOffset = 0;
        for (size_t i = 0; i < inputTemp.n_slices; i++)
        {
          trainFeatures.col(idx)(arma::span(currentOffset, currentOffset +
              inputTemp.slice(i).n_elem - 1), arma::span()) =
              arma::vectorise(inputTemp.slice(i).t());
          currentOffset += inputTemp.slice(i).n_elem;
        }
    }

    if (normalize)
    {
      // Convert each element to uint8 first and then divide by 255.
      for (size_t i = 0; i < trainFeatures.n_elem; i++)
          trainFeatures(i) = ((uint8_t)(trainFeatures(i)) / 255.0);
    }
  }

  /**
   * PreProcessor for YOLO model. Converts arma::field type annotations to
   * arma::mat type for training YOLO model. Each column in target matrix has
   * the size : gridWidth * gridHeight * (5 * numBoxes + classes).
   *
   * @param annotations Field object created using model's dataloader containing
   *     annotation for images.
   * @param output Output matrix where output will be stored.
   * @param imageWidth Width of image used for training YOLO model.
   * @param imageHeight Height of image used for training YOLO model.
   * @param gridWidth Width of output feature map of YOLO model.
   * @param gridHeight Height of output feature map of YOLO model.
   * @param numBoxes Number of bounding boxes per grid.
   * @param numClasses Number of classes in training set.
   * @param normalize Boolean to determine whether coordinates are to
   *    to be normalized or not. Defaults to true.
   *
   * Note : This function must be called manually before model is used.
   */
  template<typename eT>
  static void YOLOPreProcessor(const DatasetY& annotations,
                               arma::Mat<eT>& output,
                               const size_t version = 1,
                               const size_t imageWidth = 224,
                               const size_t imageHeight = 224,
                               const size_t gridWidth = 7,
                               const size_t gridHeight = 7,
                               const size_t numBoxes = 2,
                               const size_t numClasses = 20,
                               const bool normalize = true)
  {
    // See if we can change this to v4 / v5.
    mlpack::Log::Assert(version >= 1 && version <= 3, "Supported YOLO versions \
        are version 1 to version 3.");

    mlpack::Log::Assert(typeid(annotations) == typeid(arma::field<arma::vec>),
        "Use Field type to represent annotations.");

    size_t batchSize = annotations.n_cols;
    size_t numPredictions = 5 * numBoxes + numClasses;
    if (version > 1)
    {
      // Each bounding boxes has a corresponding class.
      numPredictions = numBoxes * (5 + numClasses);
    }

    double cellSizeHeight = (double) 1.0 / gridHeight;
    double cellSizeWidth = (double) 1.0 / gridWidth;

    // Set size of output and use cubes convenience.
    output.set_size(gridWidth * gridHeight * numPredictions, batchSize);
    output.zeros();

    // Use offset to create a cube for a particular column / batch.
    size_t offset = 0;
    for (size_t boxIdx = 0; boxIdx < batchSize; boxIdx++)
    {
      arma::cube outputTemp(const_cast<arma::Mat<eT> &>(output).memptr() +
          offset, gridHeight, gridWidth, numPredictions, false, false);
      offset += gridWidth * gridHeight * numPredictions;

      // Get the bounding box and labels corresponding to current image.
      arma::mat labels(1, annotations(0, boxIdx).n_elem / 5);
      arma::mat boundingBoxes(4, annotations(0, boxIdx).n_elem / 5);
      for (size_t i = 0; i < boundingBoxes.n_cols; i++)
      {
        labels.col(i)(0) = annotations(0, boxIdx)(i * 5);
        boundingBoxes.col(i) = annotations(0, boxIdx)(arma::span(i * 5 + 1,
            (i + 1) * 5 - 1));
      }

      // For YOLOv2 or higher, each bounding box can represent a class
      // so we don't repeat labels as done for YOLOv1. We will use map
      // to store last inserted bounding box.
      std::map<std::pair<size_t, size_t>, size_t> boundingBoxOffset;

      // Normalize the coordinates.
      boundingBoxes.row(0) /= imageWidth;
      boundingBoxes.row(2) /= imageWidth;
      boundingBoxes.row(1) /= imageHeight;
      boundingBoxes.row(3) /= imageHeight;

      // Get width and height as well as centres for the bounding box.
      arma::mat widthAndHeight(2, boundingBoxes.n_cols);
      widthAndHeight.row(0) = (boundingBoxes.row(2) - boundingBoxes.row(0));
      widthAndHeight.row(1) = (boundingBoxes.row(3) - boundingBoxes.row(1));

      arma::mat centres(2, boundingBoxes.n_cols);
      centres.row(0) = (boundingBoxes.row(2) + boundingBoxes.row(0)) / 2.0;
      centres.row(1) = (boundingBoxes.row(3) + boundingBoxes.row(1)) / 2.0;

      // Assign bounding boxes to the grid.
      for (size_t i = 0; i < boundingBoxes.n_cols; i++)
      {
        // Index for representing bounding box on grid.
        arma::vec gridCoordinates = centres.col(i);
        arma::vec centreCoordinates = centres.col(i);

        if (normalize)
        {
          gridCoordinates(0) = std::ceil(gridCoordinates(0) /
              cellSizeWidth) - 1;
          gridCoordinates(1) = std::ceil(gridCoordinates(1) /
              cellSizeHeight) - 1;
        }
        else
        {
          gridCoordinates(0) = std::ceil((gridCoordinates(0) /
              imageWidth) / cellSizeWidth) - 1;
          gridCoordinates(1) = std::ceil((gridCoordinates(1) /
              imageHeight) / cellSizeHeight) - 1;
        }

        size_t gridX = gridCoordinates(0);
        size_t gridY = gridCoordinates(1);
        gridCoordinates(0) = gridCoordinates(0) * cellSizeWidth;
        gridCoordinates(1) = gridCoordinates(1) * cellSizeHeight;

        // Normalize to 1.0.
        gridCoordinates = centres.col(i) - gridCoordinates;
        gridCoordinates(0) /= cellSizeWidth;
        gridCoordinates(1) /= cellSizeHeight;

        if (normalize)
          centreCoordinates = gridCoordinates;

        if (version == 1)
        {
          // Fill elements in the grid.
          for (size_t k = 0; k < numBoxes; k++)
          {
            size_t s = 5 * k;
            outputTemp(arma::span(gridX), arma::span(gridY),
                arma::span(s, s + 1)) = centreCoordinates;
            outputTemp(arma::span(gridX), arma::span(gridY),
                arma::span(s + 2, s + 3)) = widthAndHeight.col(i);
            outputTemp(gridX, gridY, s + 4) = 1.0;
          }
          outputTemp(gridX, gridY, 5 * numBoxes + labels.col(i)(0)) = 1;
        }
        else
        {
          size_t s = 0;
          if (boundingBoxOffset.count({gridX, gridY}))
          {
            s = boundingBoxOffset[{gridX, gridY}] + 1;
            boundingBoxOffset[{gridX, gridY}]++;
          }
          else
            boundingBoxOffset.insert({{gridX, gridY}, s});

          if (s > numBoxes)
            continue;

          size_t bBoxOffset = (5 + numClasses) * s;
          outputTemp(arma::span(gridX), arma::span(gridY),
              arma::span(bBoxOffset, bBoxOffset + 1)) = centreCoordinates;
          outputTemp(arma::span(gridX), arma::span(gridY),
              arma::span(bBoxOffset + 2,
                  bBoxOffset + 3)) = widthAndHeight.col(i);
          outputTemp(gridX, gridY, bBoxOffset + 4) = 1.0;
          outputTemp(gridX, gridY, bBoxOffset + 5 + labels.col(i)(0)) = 1;
        }
      }
    }
  }
};

} // namespace models
} // namespace mlpack

#endif
