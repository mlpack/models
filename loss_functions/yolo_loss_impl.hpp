/**
 * @file yolo_loss_impl.cpp
 * @author Kartik Dutt
 *
 * Loss function for training YOLO model.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#include "yolo_loss.hpp"

#ifndef MODELS_LOSS_FUNCTIONS_YOLO_LOSS_HPP
#define MODELS_LOSS_FUNCTIONS_YOLO_LOSS_HPP

template<typename InputDataType, OutputDataType>
YOLOLoss<InputDataType, OutputDataType>::YOLOLoss(
    const size_t version,
    const size_t gridWidth,
    const size_t gridHeight,
    const size_t numBoxes,
    const size_t numClasses,
    const double lambdaCoordinates,
    const double lambdaObjectness) :
    version(version),
    gridWidth(gridWidth),
    gridHeight(gridHeight),
    numBoxes(numBoxes),
    numClasses(numClasses),
    lambdaCoordinates(lambdaCoordinates),
    lambdaObjectness(lambdaObjectness)
{
  // Nothing to do here.
}

template<typename InputType, typename TargetType>
template<typename InputDataType, OutputDataType>
void YOLOLoss<InputDataType, OutputDataType>::Forward(
    const InputType& input,
    const TargetType& target)
{
  // Seperate Predictions.
  InputDataType noObjectMatrix, classificationMatrix;
  InputDataType coodMatrix, heightWidthMatrix;

  // Seperate Targets.
  InputDataType noObjectTargetMatrix, classificationTargetMatrix;
  InputDataType coodTargetMatrix, heightWidthTargetMatrix;

  noObjectMatrix.zeros(arma::size(input));
  classificationMatrix.zeros(arma::size(input))
  coodMatrix.zeros(arma::size(input));
  heightWidthMatrix.zeros(arma::size(input));

  for (size_t i = 0; i < input.n_cols; i++)
  {
    // Fill the matrtix.
  }

  heightWidthMatrix = arma::sqrt(heightWidthMatrix);
  heightWidthTargetMatrix = arma::sqrt(heightWidthTargetMatrix);

  double noObjLoss = lambdaObjectness * arma::accu(arma::square(
      noObjectMatrix - noObjectTargetMatrix)) /
      noObjectTargetMatrix.n_cols;

  double classificationLoss = arma::accu(arma::square(
      classificationMatrix - classificationTargetMatrix)) /
      classificationTargetMatrix.n_cols;

  double coodLoss = lambdaCoordinates * arma::accu(arma::square(
      coodTargetMatrix - coodMatrix))
      / coodTargetMatrix.n_cols;

  double heightWidthLoss = lambdaCoordinates * arma::accu(arma::square(
      heightWidthTargetMatrix - heightWidthMatrix))
      / heightWidthTargetMatrixtMatrix.n_cols;

  return heightWidthLoss + coodLoss + classificationLoss + noObjLoss;
}

template<typename InputType, typename TargetType>
template<typename InputDataType, OutputDataType>
void YOLOLoss<InputDataType, OutputDataType>::Backward(
    const InputType& input,
    const TargetType& target,
    OutputType& output)
{
  // Seperate Predictions.
  InputDataType noObjectMatrix, classificationMatrix;
  InputDataType coodMatrix, heightWidthMatrix;

  // Seperate Targets.
  InputDataType noObjectTargetMatrix, classificationTargetMatrix;
  InputDataType coodTargetMatrix, heightWidthTargetMatrix;

  noObjectMatrix.zeros(arma::size(input));
  classificationMatrix.zeros(arma::size(input))
  coodMatrix.zeros(arma::size(input));
  heightWidthMatrix.zeros(arma::size(input));

  for (size_t i = 0; i < input.n_cols; i++)
  {
    // Fill the matrtix.
  }

  heightWidthMatrix = arma::sqrt(heightWidthMatrix);
  heightWidthTargetMatrix = arma::sqrt(heightWidthTargetMatrix);

  output = -2 * (noObjectMatrix + classificationMatrix + coodMatrix +
      heightWidthMatrix - noObjectTargetMatrix - classificationTargetMatrix
      - coodTargetMatrix - heightWidthTargetMatrix) / input.n_cols;
}

template<typename InputDataType, typename OutputDataType>
template<typename Archive>
void YOLOLoss<InputDataType, OutputDataType>::serialize(
    Archive& ar,
    const unsigned int /* version */)
{
  ar& BOOST_SERIALIZATION_NVP(version);
  ar& BOOST_SERIALIZATION_NVP(gridWidth);
  ar& BOOST_SERIALIZATION_NVP(gridHeight);
  ar& BOOST_SERIALIZATION_NVP(numBoxes);
  ar& BOOST_SERIALIZATION_NVP(numClasses);
  ar& BOOST_SERIALIZATION_NVP(lambdaCoordinates);
  ar& BOOST_SERIALIZATION_NVP(lambdaObjectness);
}

#endif