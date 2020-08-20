/**
 * @file yolov3_layer.hpp
 * @author Kartik Dutt
 *
 * Definition of YOLOv3 layer.
 *
 * mlpack is free software; you may redistribute it and/or modify it under the
 * terms of the 3-clause BSD license.  You should have received a copy of the
 * 3-clause BSD license along with mlpack.  If not, see
 * http://www.opensource.org/licenses/BSD-3-Clause for more information.
 */

#ifndef MODELS_YOLOV3_LAYER_HPP
#define MODELS_YOLOV3_LAYER_HPP

#include <mlpack/prereqs.hpp>
#include <mlpack/methods/ann/layer/layer_types.hpp>
#include <mlpack/methods/ann/layer/layer_traits.hpp>

namespace mlpack {
namespace ann /** Artificial Neural Network. */ {

/**
 * Implementation of the YOLOv3 layer. This class implements the forward
 * propogation for the same.
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
class YOLOv3
{
 public:
  /**
   * @param inputWidth Input width of the model.
   * @param inputHeight Input height of the model.
   * @param inSize Input depth of the model.
   */
  YOLOv3(const size_t inputWidth,
         const size_t inputHeight,
         const size_t inSize);

  /**
   * Ordinary feed forward pass of a neural network, evaluating the function
   * f(x) by propagating the activity forward through f.
   *
   * @param input Input data used for evaluating the specified function.
   * @param output Resulting output activation.
   */
  template<typename eT>
  void Forward(const arma::Mat<eT>& input, arma::Mat<eT>& output);

  /**
   * Ordinary feed backward pass of a neural network, calculating the function
   * f(x) by propagating x backwards trough f. Using the results from the feed
   * forward pass.
   *
   * @param * (input) The propagated input activation.
   * @param gy The backpropagated error.
   * @param g The calculated gradient.
   */
  template<typename eT>
  void Backward(const arma::Mat<eT>& /* input */,
                const arma::Mat<eT>& gy,
                arma::Mat<eT>& g);

  /**
   * Calculate the gradient using the output delta and the input activation.
   *
   * @param * (input) The propagated input.
   * @param error The calculated error.
   * @param gradient The calculated gradient.
   */
  template <typename eT>
  void Gradient(const arma::Mat<eT> & /* input */,
                const arma::Mat<eT> &error,
                arma::Mat<eT> &gradient);

  /**
   * Serialize the layer
   */
  template <typename Archive>
  void serialize(Archive &ar, const unsigned int /* version */);

  /**
   * Add a new module to the model.
   *
   * @param layer The Layer to be added to the model.
   */
  template<typename LayerType>
  void Add(const LayerType& layer) { network.push_back(new LayerType(layer)); }

  /**
   * Add a new module to the model.
   *
   * @param args The layer parameter.
   */
  template <class LayerType, class... Args>
  void Add(Args... args) { network.push_back(new LayerType(args...)); }

  /**
   * Add a new module to the model.
   *
   * @param layer The Layer to be added to the model.
   */
  void Add(LayerTypes layer) { network.push_back(layer); }

  //! Return the model modules.
  std::vector<LayerTypes>& Model()
  {
    if (model)
      return network;

    return empty;
  }

  //! Return the initial point for the optimization.
  const arma::mat& Parameters() const { return parameters; }
  //! Modify the initial point for the optimization.
  arma::mat& Parameters() { return parameters; }

  arma::mat const& InputParameter() const { return inputParameter; }
  //! Modify the input parameter.
  arma::mat& InputParameter() { return inputParameter; }

  //! Get the output parameter.
  arma::mat const& OutputParameter() const { return outputParameter; }
  //! Modify the output parameter.
  arma::mat& OutputParameter() { return outputParameter; }

  //! Get the delta.e
  arma::mat const& Delta() const { return delta; }
  //! Modify the delta.
  arma::mat& Delta() { return delta; }

  //! Get the gradient.
  arma::mat const& Gradient() const { return gradient; }
  //! Modify the gradient.
  arma::mat& Gradient() { return gradient; }

  //! Get the input width.
  size_t const& InputWidth() const { return inputWidth; }
  //! Modify input the width.
  size_t& InputWidth() { return inputWidth; }

  //! Get the input height.
  size_t const& InputHeight() const { return inputHeight; }
  //! Modify the input height.
  size_t& InputHeight() { return inputHeight; }

 private:
  //! Locally-stored network modules.
  std::vector<LayerTypes> network;

  //! Locally-stored skip - indices.
  std::vector<std::vector<size_t>> skipIndices;

  //! Locally-stored output.


  //! Locally-stored number of input units.
  size_t inSize;

  //! Locally-stored number of output units.
  size_t outSize;

  //! Locally-stored model parameters.
  arma::mat parameters;

  //! Locally-stored delta visitor.
  DeltaVisitor deltaVisitor;

  //! Locally-stored output parameter visitor.
  OutputParameterVisitor outputParameterVisitor;

  //! Locally-stored delete visitor.
  DeleteVisitor deleteVisitor;

  //! Locally-stored delta object.
  arma::mat delta;

  //! Locally-stored input parameter object.
  arma::mat inputParameter;

  //! Locally-stored output parameter object.
  arma::mat outputParameter;

  //! Locally-stored gradient object.
  arma::mat gradient;

  //! Locally-stored transformed output parameter.
  arma::cube outputTemp;
};

} // namespace ann
} // namespace mlpack

#endif