#ifndef MODELS_MODELS_YOLOV3_TINY_YOLOV3_TINY_HPP
#define MODELS_MODELS_YOLOV3_TINY_YOLOV3_TINY_HPP

#include <mlpack/methods/ann/layer/adaptive_max_pooling.hpp>
#include <mlpack/methods/ann/layer/add.hpp>
#include <mlpack/methods/ann/layer/batch_norm.hpp>
#include <mlpack/methods/ann/layer/max_pooling.hpp>
#include <mlpack/methods/ann/layer/multi_layer.hpp>

#define MLPACK_ENABLE_ANN_SERIALIZATION
#include <mlpack.hpp>

namespace mlpack {
namespace models {

template<typename MatType>
class YoloV3Tiny {
public:
  YoloV3Tiny(const size_t inputWidth,
	     const size_t inputHeight,
	     const size_t inputChannels,
	     const size_t numClasses = 80,
	     const float ignoreThresh = 0.7
	     ) :
	inputWidth(inputWidth),
	inputHeight(inputHeight),
	inputChannels(inputChannels),
	numClasses(numClasses),
	ignoreThresh(ignoreThresh)
	{}

  ~YoloV3Tiny() {}

  void LoadModel(const std::string &filePath);
  void SaveModel(const std::string &filePath);

private:
  MultiLayer<MatType>* MaxPoolBlock(const size_t size) {
    return new MaxPooling(inputWidth / (double)size, inputHeight / (double)size);
  }

  void Darknet19() {}

  size_t inputWidth;
  size_t inputHeight;
  size_t inputChannels;
  size_t numClasses;
  float ignoreThresh;
};

} // namespace models
} // namespace mlpack

#include "yolov3_tiny_impl.hpp"

#endif
