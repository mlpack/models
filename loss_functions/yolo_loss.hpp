/**
 * @file yolo_loss.cpp
 * @author Kartik Dutt
 *
 * Loss function for training YOLO model.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MODELS_LOSS_FUNCTIONS_YOLO_LOSS_HPP
#define MODELS_LOSS_FUNCTIONS_YOLO_LOSS_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/core/metrics/iou_metric.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * The YOLO loss function is used to decode output of YOLO model and train
 * it.
 *
 * @tparam InputDataType Type of the input data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 * @tparam OutputDataType Type of the output data (arma::colvec, arma::mat,
 *         arma::sp_mat or arma::cube).
 */
template <
    typename InputDataType = arma::mat,
    typename OutputDataType = arma::mat
>
class YOLOLoss
{
 public:
  /**
   * Create the YOLOLoss object.
   *
   * @param version Version of YOLO model used in training.
   * @param gridWidth Width of output feature map of YOLO model.
   * @param gridHeight Height of output feature map of YOLO model.
   * @param numBoxes Number of bounding boxes per grid.
   * @param numClasses Number of classes in training set.
   * @param lambdaCoordinates Multiplicative factor for loss obtained from
   *    coordinates.
   * @param lambdaObjectness Multiplicative factor for loss obtained from
   *    misclassification.
   */
  YOLOLoss(const size_t version = 1,
           const size_t gridWidth = 7,
           const size_t gridHeight = 7,
           const size_t numBoxes = 2,
           const size_t numClasses = 20,
           const double lambdaCoordinates = 5.0,
           const double lambdaObjectness = 0.5);

  /**
   * Computes the YOLO loss function.
   *
   * @param input Input data used for evaluating the specified function.
   * @param target The target vector.
   */
  template<typename InputType, typename TargetType>
  typename InputType::elem_type Forward(const InputType &input,
                                        const TargetType &target);

  /**
   * Ordinary feed backward pass of a neural network.
   *
   * @param input The propagated input activation.
   * @param target The target vector.
   * @param output The calculated error.
   */
  template<typename InputType, typename TargetType, typename OutputType>
  void Backward(const InputType& input,
                const TargetType& target,
                OutputType& output);

  //! Get the output parameter.
  OutputDataType& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  OutputDataType& OutputParameter() { return outputParameter; }

  //! Get the version.
  size_t Version() const { return version; }
  //! Modify the version.
  size_t& Version() { return version; }

  //! Get the Grid Width.
  size_t GridWidth() const { return gridWidth; }
  //! Modify the Grid Width.
  size_t& GridWidth() { return gridWidth; }

  //! Get the Grid Height.
  size_t GridHeight() const { return gridHeight; }
  //! Modify the Grid Height.
  size_t& GridHeight() { return gridHeight; }

  //! Get the Number of boxes.
  size_t NumBoxes() const { return numBoxes; }
  //! Modify the Number of boxes.
  size_t& NumBoxes() { return numBoxes; }

  //! Get the Number of classes.
  size_t NumClasses() const { return numClasses; }
  //! Modify the Number of classes.
  size_t& NumClasses() { return numClasses; }

  //! Get the lambdaCoordinates.
  double LambdaCoordinates() const { return lambdaCoordinates; }
  //! Modify the lambdaCoordinates.
  double& LambdaCoordinates() { return lambdaCoordinates; }

  //! Get the lambdaObjectness.
  double LambdaObjectness() const { return lambdaObjectness; }
  //! Modify the lambdaObjectness.
  double& LambdaObjectness() { return lambdaObjectness; }

  /**
   * Serialize the layer.
   */
  template <typename Archive>
  void serialize(Archive &ar, const unsigned int /* version */);

 private:
  //! Version of YOLO model used in training.
  size_t version;

  //! Width of output feature map of YOLO model.
  size_t gridWidth;

  //! Height of output feature map of YOLO model.
  size_t gridHeight;

  //! Number of bounding boxes per grid.
  size_t numBoxes;

  //! Number of classes in training set.
  size_t numClasses;

  //! Multiplicative factor for loss obtained from coordinates.
  double lambdaCoordinates;

  //! Multiplicative factor for loss obtained from misclassification.
  double lambdaObjectness;

  //! Locally-stored output parameter object.
  OutputDataType outputParameter;
};

} // namespace ann
} // namespace mlpack

#include "yolo_loss_impl.hpp"

#endif
