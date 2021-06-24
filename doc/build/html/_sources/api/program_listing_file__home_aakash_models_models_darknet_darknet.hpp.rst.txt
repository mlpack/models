
.. _program_listing_file__home_aakash_models_models_darknet_darknet.hpp:

Program Listing for File darknet.hpp
====================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_models_models_darknet_darknet.hpp>` (``/home/aakash/models/models/darknet/darknet.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

    author = {Joseph Redmon, Ali Farhadi},
    title = {YOLO9000 : Better, Faster, Stronger},
    year = {2016},
    url = {https://pjreddie.com/media/files/papers/YOLO9000.pdf}
   }
    author = {Joseph Redmon, Ali Farhadi},
    title = {YOLOv3 :  An Incremental Improvement},
    year = {2019},
    url = {https://pjreddie.com/media/files/papers/YOLOv3.pdf}
   }
   
   #ifndef MODELS_MODELS_DARKNET_DARKNET_HPP
   #define MODELS_MODELS_DARKNET_DARKNET_HPP
   
   #include <mlpack/core.hpp>
   #include <mlpack/methods/ann/layer/layer.hpp>
   #include <mlpack/methods/ann/ffn.hpp>
   #include <mlpack/methods/ann/layer/layer_types.hpp>
   #include <mlpack/methods/ann/init_rules/random_init.hpp>
   #include <mlpack/methods/ann/loss_functions/binary_cross_entropy_loss.hpp>
   #include <mlpack/methods/ann/init_rules/he_init.hpp>
   #include <mlpack/methods/ann/init_rules/glorot_init.hpp>
   
   namespace mlpack {
   namespace models {
   
   template<
     typename OutputLayerType = ann::CrossEntropyError<>,
     typename InitializationRuleType = ann::RandomInitialization,
     size_t DarkNetVersion = 19
   >
   class DarkNet
   {
    public:
     DarkNet();
   
     DarkNet(const size_t inputChannel,
             const size_t inputWidth,
             const size_t inputHeight,
             const size_t numClasses = 1000,
             const std::string& weights = "none",
             const bool includeTop = true);
   
     DarkNet(const std::tuple<size_t, size_t, size_t> inputShape,
             const size_t numClasses = 1000,
             const std::string& weights = "none",
             const bool includeTop = true);
   
     ann::FFN<OutputLayerType, InitializationRuleType>& GetModel()
     {
       return darkNet;
     }
   
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
                           const bool batchNorm = true,
                           const double negativeSlope = 1e-1,
                           SequentialType* baseLayer = NULL)
     {
       ann::Sequential<>* bottleNeck = new mlpack::ann::Sequential<>();
       bottleNeck->Add(new ann::Convolution<>(inSize, outSize, kernelWidth,
           kernelHeight, strideWidth, strideHeight, padW, padH, inputWidth,
           inputHeight));
   
       // Update inputWidth and input Height.
       mlpack::Log::Info << "Conv Layer.  ";
       mlpack::Log::Info << "(" << inputWidth << ", " << inputHeight <<
           ", " << inSize << ") ----> ";
   
       inputWidth = ConvOutSize(inputWidth, kernelWidth, strideWidth, padW);
       inputHeight = ConvOutSize(inputHeight, kernelHeight, strideHeight, padH);
       mlpack::Log::Info << "(" << inputWidth << ", " << inputHeight <<
           ", " << outSize << ")" << std::endl;
   
       if (batchNorm)
         bottleNeck->Add(new ann::BatchNorm<>(outSize, 1e-5, false));
   
       bottleNeck->Add(new ann::LeakyReLU<>(negativeSlope));
   
       if (baseLayer != NULL)
         baseLayer->Add(bottleNeck);
       else
         darkNet.Add(bottleNeck);
     }
   
     void PoolingBlock(const size_t factor = 2,
                       const std::string type = "max")
     {
       if (type == "max")
       {
         darkNet.Add(new ann::AdaptiveMaxPooling<>(
             std::ceil(inputWidth * 1.0 / factor),
             std::ceil(inputHeight * 1.0 / factor)));
       }
       else
       {
         darkNet.Add(new ann::AdaptiveMeanPooling<>(std::ceil(inputWidth * 1.0 /
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
   
     void DarkNet19SequentialBlock(const size_t inputChannel,
                                   const size_t kernelWidth,
                                   const size_t kernelHeight,
                                   const size_t padWidth,
                                   const size_t padHeight)
     {
       ConvolutionBlock(inputChannel, inputChannel * 2,
           kernelWidth, kernelHeight, 1, 1, padWidth, padHeight, true);
       ConvolutionBlock(inputChannel * 2, inputChannel,
           1, 1, 1, 1, 0, 0, true);
       ConvolutionBlock(inputChannel, inputChannel * 2,
           kernelWidth, kernelHeight, 1, 1, padWidth, padHeight, true);
     }
   
     void DarkNet53ResidualBlock(const size_t inputChannel,
                                 const size_t kernelWidth = 3,
                                 const size_t kernelHeight = 3,
                                 const size_t padWidth = 1,
                                 const size_t padHeight = 1)
     {
       mlpack::Log::Info << "Residual Block Begin." << std::endl;
       ann::Residual<>* residualBlock = new ann::Residual<>();
       ConvolutionBlock(inputChannel, inputChannel / 2,
           1, 1, 1, 1, 0, 0, true, 1e-2, residualBlock);
       ConvolutionBlock(inputChannel / 2, inputChannel, kernelWidth,
           kernelHeight, 1, 1, padWidth, padHeight, true, 1e-2, residualBlock);
       darkNet.Add(residualBlock);
       mlpack::Log::Info << "Residual Block end." << std::endl;
     }
   
     size_t ConvOutSize(const size_t size,
                        const size_t k,
                        const size_t s,
                        const size_t padding)
     {
       return std::floor(size + 2 * padding - k) / s + 1;
     }
   
     ann::FFN<OutputLayerType, InitializationRuleType> darkNet;
   
     size_t inputWidth;
   
     size_t inputHeight;
   
     size_t inputChannel;
   
     size_t numClasses;
   
     std::string weights;
   }; // DarkNet class.
   
   } // namespace models
   } // namespace mlpack
   
   # include "darknet_impl.hpp"
   
   #endif
