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

#ifndef MODELS_LOSS_FUNCTIONS_YOLO_LOSS_IMPL_HPP
#define MODELS_LOSS_FUNCTIONS_YOLO_LOSS_IMPL_HPP

#include "yolo_loss.hpp"

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

template<typename InputDataType, typename OutputDataType>
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

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename TargetType>
typename InputType::elem_type
YOLOLoss<InputDataType, OutputDataType>::Forward(
    const InputType& input,
    const TargetType& target)
{
  InputType lossMatrix;
  lossMatrix.zeros(arma::size(input));

  size_t numPredictions = 5 * numBoxes + numClasses;
  for (size_t i = 0; i < input.n_cols; i++)
  {
    arma::cube inputTemp(
        const_cast<arma::mat&>(input).memptr(),
        gridWidth, gridHeight, numPredictions, false, false);
    arma::cube outputTemp(
        const_cast<arma::mat&>(target).memptr(),
        gridWidth, gridHeight, numPredictions, false, false);
    arma::cube lossTemp(
        const_cast<arma::mat&>(lossMatrix).memptr(),
        gridHeight, gridWidth, numPredictions, false, false);

    for (size_t gridX = 0; gridX < gridWidth; gridX++)
    {
      for (size_t gridY = 0; gridY < gridHeight; gridY++)
      {
        for (size_t k = 0; k < numBoxes; k++)
        {
          size_t s = 5 * k;

          // Coordinate Loss.
          // MSE Loss on coordinates.
          lossTemp(arma::span(gridX), arma::span(gridY),
              arma::span(s, s + 1)) = lambdaCoordinates *
              (arma::square(inputTemp(arma::span(gridX), arma::span(gridY),
              arma::span(s, s + 1)) - outputTemp(arma::span(gridX),
              arma::span(gridY), arma::span(s, s + 1))));

          // MSE Loss on square root of width and height.
          lossTemp(arma::span(gridX), arma::span(gridY),
              arma::span(s + 2, s + 3)) = lambdaCoordinates *
              (arma::square(arma::sqrt(
              inputTemp(arma::span(gridX), arma::span(gridY),
              arma::span(s + 2, s + 3))) - arma::sqrt(
              outputTemp(arma::span(gridX),
              arma::span(gridY), arma::span(s + 2, s + 3)))));

          arma::vec predBBox = inputTemp(arma::span(gridX),
              arma::span(gridY), arma::span(s, s + 3));
          arma::vec targetBBox = outputTemp(arma::span(gridX),
              arma::span(gridY), arma::span(s, s + 3));
          
          inputTemp(gridX, gridY, s + 4) = metric::IoU<false>::Evaluate(
              predBBox, targetBBox);

          // MSE loss on objectness score.
          lossTemp(gridX, gridY, s + 4) = lambdaObjectness *
              (std::pow(inputTemp(gridX, gridY, s + 4) -
              outputTemp(gridX, gridY, s + 4), 2));
        }

        // Classification loss.
        lossTemp(arma::span(gridX), arma::span(gridY),
            arma::span()) = arma::square(
            inputTemp(arma::span(gridX), arma::span(gridY),
            arma::span()) - outputTemp(arma::span(gridX),
            arma::span(gridY), arma::span()));
      }
    }
  }

  return arma::accu(lossMatrix) / lossMatrix.n_cols;
}

template<typename InputDataType, typename OutputDataType>
template<typename InputType, typename TargetType, typename OutputType>
void YOLOLoss<InputDataType, OutputDataType>::Backward(
    const InputType& input,
    const TargetType& target,
    OutputType& output)
{
  output.zeros(arma::size(input));

  size_t numPredictions = 5 * numBoxes + numClasses;
  for (size_t i = 0; i < input.n_cols; i++)
  {
    arma::cube inputTemp(
        const_cast<arma::mat&>(input).memptr(),
        gridWidth, gridHeight, numPredictions, false, false);
    arma::cube targetTemp(
        const_cast<arma::mat&>(target).memptr(),
        gridWidth, gridHeight, numPredictions, false, false);
    arma::cube outputTemp(
        const_cast<arma::mat&>(output).memptr(),
        gridHeight, gridWidth, numPredictions, false, false);

    for (size_t gridX = 0; gridX < gridWidth; gridX++)
    {
      for (size_t gridY = 0; gridY < gridHeight; gridY++)
      {
        for (size_t k = 0; k < numBoxes; k++)
        {
          size_t s = 5 * k;

          // Coordinate Loss.
          // MSE Loss on coordinates.
          outputTemp(arma::span(gridX), arma::span(gridY),
              arma::span(s, s + 1)) = -2 * lambdaCoordinates *
              (inputTemp(arma::span(gridX), arma::span(gridY),
              arma::span(s, s + 1)) - targetTemp(arma::span(gridX),
              arma::span(gridY), arma::span(s, s + 1)));

          // MSE Loss on square root of width and height.
          outputTemp(arma::span(gridX), arma::span(gridY),
              arma::span(s + 2, s + 3)) = -2 * lambdaCoordinates *
              (arma::sqrt(inputTemp(arma::span(gridX),
              arma::span(gridY), arma::span(s + 2, s + 3))) -
              arma::sqrt(targetTemp(arma::span(gridX),
              arma::span(gridY), arma::span(s + 2, s + 3))));

          // MSE loss on objectness score.
          outputTemp(gridX, gridY, s + 4) = lambdaObjectness *
              (std::pow(inputTemp(gridX, gridY, s + 4) -
              targetTemp(gridX, gridY, s + 4), 2));
        }

        // Classification loss.
        outputTemp(arma::span(gridX), arma::span(gridY),
            arma::span()) = inputTemp(arma::span(gridX),
            arma::span(gridY), arma::span()) -
            targetTemp(arma::span(gridX), arma::span(gridY),
            arma::span());
      }
    }
  }
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

} // namespace ann
} // namespace mlpack

#endif
