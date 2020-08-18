/**
 * @file visualization_tools.hpp
 * @author Kartik Dutt
 * 
 * Visualization tools to visualize detection and segmentation algorithms.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MODELS_VISUALIZATION_TOOLS_HPP
#define MODELS_VISUALIZATION_TOOLS_HPP

#include <mlpack/prereqs.hpp>
#include <opencv2/opencv.hpp>

/**
 * Tools to visualize data and predictions.
 */
class VisualizationTools
{
 public:
  /**
   * Save and plot bounding boxes on images.
   */
  template<typename ImageType>
  static void VisualizeBoundingBoxes(
      ImageType& images,
      arma::field<arma::vec>& boundingBoxes,
      const size_t imageWidth = 224,
      const size_t imageHeight = 224,
      const size_t imageDepth = 3,
      const bool cornerRepresentation = true,
      const bool plot = false,
      const bool saveImages = false,
      const std::vector<std::string> imagePath = std::vector<std::string>())
  {
    if (saveImages)
    {
      mlpack::Log::Assert(images.n_cols == imagePath.size(),
          "Mismatch between number of images," + std::to_string(image.n_cols) +\
          " and image file paths" + to_string(imagePath.size()) + ".");
    }
    for (size_t i = 0; i < image.n_cols; i++)
    {
      arma::vec bBoxes = boundingBoxes(0, i);
      arma::cube image(imageWidth * imageHeight * imageDepth, 1, 1);
      image.slice(0).col(0) = image.col(i);
      image.reshape(imageWidth, imageHeight, imageDepth);
      cv::mat img = CubeToOpenCV(image);
      for (size_t boxIdx = 0; boxIdx < bBoxes.n_elem; boxIdx += 4)
      {
        cv::Point upperRightPoint, lowerLeftPoint;
        if (cornerRepresentation)
        {
          upperRightPoint = cv::Point(bBoxes(boxIdx * 4), bBoxes(boxIdx * 4 + 1));
          lowerLeftPoint = cv::Point(bBoxes(boxIdx * 4 + 2), bBoxes(boxIdx * 4 + 3));
        }
        else
        {
          upperRightPoint = cv::Point(bBoxes(boxIdx * 4), bBoxes(boxIdx * 4 + 1));
          lowerLeftPoint = cv::Point(bBoxes(boxIdx * 4) + bBoxes(boxIdx * 4 + 2),
              bBoxes(boxIdx * 4 + 1) + bBoxes(boxIdx * 4 + 3));
        }

        cv::rectangle(img, lowerRightPoint, upperRightPoint,
            cv::Scalar(rand() % 255, rand() % 255, rand() % 255));
      }
    }

    if (plot)
    {
      cv::imshow("Image", img);
    }

    if (saveImages)
    {
      cv::imwrite(imagePath[i], img);
    }
  }

 private:
  /**
   * Convert armadillo cube to opencv matrix.
   */
  template<typename eT>
  cv::Mat<eT> CubeToOpenCVMat(const Mat<eT>& input)
  {
    vector<cv::Mat_<T>> channels;
    for (size_t c = 0; c < input.n_slices; ++c)
    {
      auto* data = const_cast<T *>(input.slice(c).memptr());
      channels.push_back({int(input.n_cols), int(input.n_rows), data});
    }

    cv::Mat dst;
    cv::merge(channels, dst);
    return dst;
  }
};

#endif