
.. _program_listing_file__home_aakash_models_models_yolo_yolo.hpp:

Program Listing for File yolo.hpp
=================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_models_models_yolo_yolo.hpp>` (``/home/aakash/models/models/yolo/yolo.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

    author = {Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi},
    title = {You Only Look Once : Unified, Real-Time Object Detection},
    year = {2016},
    url = {https://arxiv.org/pdf/1506.02640.pdf}
   }
   
   #ifndef MODELS_MODELS_YOLO_YOLO_HPP
   #define MODELS_MODELS_YOLO_YOLO_HPP
   
   #include <mlpack/core.hpp>
   #include <mlpack/methods/ann/layer/layer.hpp>
   #include <mlpack/methods/ann/ffn.hpp>
   #include <mlpack/methods/ann/layer/layer_types.hpp>
   #include <mlpack/methods/ann/init_rules/random_init.hpp>
   
   
   namespace mlpack {
   namespace models {
   
   template<
     typename OutputLayerType = ann::NegativeLogLikelihood<>,
     typename InitializationRuleType = ann::RandomInitialization
   >
   class YOLO
   {
    public:
     YOLO();
   
     YOLO(const size_t inputChannel,
          const size_t inputWidth,
          const size_t inputHeight,
          const std::string yoloVersion = "v1-tiny",
          const size_t numClasses = 20,
          const size_t numBoxes = 2,
          const size_t featureSizeWidth = 7,
          const size_t featureSizeHeight = 7,
          const std::string& weights = "none",
          const bool includeTop = true);
   
     YOLO(const std::tuple<size_t, size_t, size_t> inputShape,
          const std::string yoloVersion = "v1-tiny",
          const size_t numClasses = 1000,
          const size_t numBoxes = 2,
          const std::tuple<size_t, size_t> featureShape = {7, 7},
          const std::string& weights = "none",
          const bool includeTop = true);
   
     ann::FFN<OutputLayerType, InitializationRuleType>& GetModel() { return yolo; }
   
     void LoadModel(const std::string& filePath);
   
     void SaveModel(const std::string& filePath);
   
    private:
     template<typename SequentialType = ann::Sequential<>>
     void ConvolutionBlock(const size_t inSize,
                           const size_t outSize,
                           const size_t kernelWidth,
                           const size_t kernelHeight,
                           const size_t strideWidth = 1,
                           const size_t strideHeight = 1,
                           const size_t padW = 0,
                           const size_t padH = 0,
                           const bool batchNorm = false,
                           SequentialType* baseLayer = NULL)
     {
       ann::Sequential<>* bottleNeck = new ann::Sequential<>();
       bottleNeck->Add(new ann::Convolution<>(inSize, outSize, kernelWidth,
           kernelHeight, strideWidth, strideHeight, padW, padH, inputWidth,
           inputHeight));
   
       mlpack::Log::Info << "Conv Layer.  ";
       mlpack::Log::Info << "(" << inputWidth << ", " << inputHeight <<
           ", " << inSize << ") ----> ";
   
       inputWidth = ConvOutSize(inputWidth, kernelWidth, strideWidth, padW);
       inputHeight = ConvOutSize(inputHeight, kernelHeight, strideHeight, padH);
       mlpack::Log::Info << "(" << inputWidth << ", " << inputHeight <<
           ", " << outSize << ")" << std::endl;
   
       if (batchNorm)
         bottleNeck->Add(new ann::BatchNorm<>(outSize, 1e-8, false));
   
       bottleNeck->Add(new ann::LeakyReLU<>(0.01));
   
       if (baseLayer != NULL)
         baseLayer->Add(bottleNeck);
       else
         yolo.Add(bottleNeck);
     }
   
     void PoolingBlock(const size_t factor = 2,
                       const std::string type = "max")
     {
       if (type == "max")
       {
         yolo.Add(new ann::AdaptiveMaxPooling<>(
             std::ceil(inputWidth * 1.0 / factor),
             std::ceil(inputHeight * 1.0 / factor)));
       }
       else
       {
         yolo.Add(new ann::AdaptiveMeanPooling<>(std::ceil(inputWidth * 1.0 /
             factor), std::ceil(inputHeight * 1.0 / factor)));
       }
   
       mlpack::Log::Info << "Pooling Layer.  ";
       mlpack::Log::Info << "(" << inputWidth << ", " << inputHeight <<
           ") ----> ";
       // Update inputWidth and inputHeight.
       inputWidth = std::ceil(inputWidth * 1.0 / factor);
       inputHeight = std::ceil(inputHeight * 1.0 / factor);
   
       mlpack::Log::Info << "(" << inputWidth << ", " << inputHeight <<
           ")" << std::endl;
     }
   
     size_t ConvOutSize(const size_t size,
                        const size_t k,
                        const size_t s,
                        const size_t padding)
     {
       return std::floor(size + 2 * padding - k) / s + 1;
     }
   
     ann::FFN<OutputLayerType, InitializationRuleType> yolo;
   
     size_t inputChannel;
   
     size_t inputWidth;
   
     size_t inputHeight;
   
     size_t numClasses;
   
     size_t numBoxes;
   
     size_t featureWidth;
   
     size_t featureHeight;
   
     std::string weights;
   
     std::string yoloVersion;
   }; // YOLO class.
   
   } // namespace models
   } // namespace mlpack
   
   # include "yolo_impl.hpp"
   
   #endif
