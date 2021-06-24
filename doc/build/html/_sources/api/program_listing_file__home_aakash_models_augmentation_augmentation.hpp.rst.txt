
.. _program_listing_file__home_aakash_models_augmentation_augmentation.hpp:

Program Listing for File augmentation.hpp
=========================================

|exhale_lsh| :ref:`Return to documentation for file <file__home_aakash_models_augmentation_augmentation.hpp>` (``/home/aakash/models/augmentation/augmentation.hpp``)

.. |exhale_lsh| unicode:: U+021B0 .. UPWARDS ARROW WITH TIP LEFTWARDS

.. code-block:: cpp

   
   #ifndef MODELS_AUGMENTATION_AUGMENTATION_HPP
   #define MODELS_AUGMENTATION_AUGMENTATION_HPP
   
   #include <mlpack/methods/ann/layer/bilinear_interpolation.hpp>
   #include <mlpack/core/util/to_lower.hpp>
   #include <boost/regex.hpp>
   
   namespace mlpack {
   namespace models {
   
   class Augmentation
   {
    public:
     Augmentation() :
         augmentations(std::vector<std::string>()),
         augmentationProbability(0.2)
     {
       // Nothing to do here.
     }
   
     Augmentation(const std::vector<std::string>& augmentations,
                  const double augmentationProbability) :
                  augmentations(augmentations),
                  augmentationProbability(augmentationProbability)
     {
       // Convert strings to lower case.
       for (size_t i = 0; i < augmentations.size(); i++)
         this->augmentations[i] = mlpack::util::ToLower(augmentations[i]);
   
       // Sort the vector to place resize parameter to the front of the string.
       // This prevents constant lookups for resize.
       sort(this->augmentations.begin(), this->augmentations.end(), [](
           std::string& str1, std::string& str2)
           {
             return str1.find("resize") != std::string::npos;
           });
     }
   
     template<typename DatasetType>
     void Transform(DatasetType& dataset,
                    const size_t datapointWidth,
                    const size_t datapointHeight,
                    const size_t datapointDepth = 1);
   
     template<typename DatasetType>
     void ResizeTransform(DatasetType& dataset,
                          const size_t datapointWidth,
                          const size_t datapointHeight,
                          const size_t datapointDepth,
                          const std::string& augmentation);
   
    private:
     bool HasResizeParam(const std::string& augmentation = "")
     {
       if (augmentation.length())
         return augmentation.find("resize") != std::string::npos;
   
   
       // Search in augmentation vector.
       return augmentations.size() <= 0 ? false :
           augmentations[0].find("resize") != std::string::npos;
     }
   
     void GetResizeParam(size_t& outWidth,
                         size_t& outHeight,
                         const std::string& augmentation)
     {
       if (!HasResizeParam())
         return;
   
       outWidth = 0;
       outHeight = 0;
   
       // Use regex to find one or two numbers. If only one provided
       // set output width equal to output height.
       boost::regex regex{"[0-9]+"};
   
       // Create an iterator to find matches.
       boost::sregex_token_iterator matches(augmentation.begin(),
           augmentation.end(), regex, 0), end;
   
       size_t matchesCount = std::distance(matches, end);
   
       if (matchesCount == 0)
       {
         mlpack::Log::Fatal << "Invalid size / shape in " <<
             augmentation << std::endl;
       }
   
       if (matchesCount == 1)
       {
         outWidth = std::stoi(*matches);
         outHeight = outWidth;
       }
       else
       {
         outWidth = std::stoi(*matches);
         matches++;
         outHeight = std::stoi(*matches);
       }
     }
   
     std::vector<std::string> augmentations;
   
     double augmentationProbability;
   
     // The dataloader class should have access to internal functions of
     // the augmentation class.
     template<typename DatasetX, typename DatasetY, class ScalerType>
     friend class DataLoader;
   };
   
   } // namespace models
   } // namespace mlpack
   
   #include "augmentation_impl.hpp" // Include implementation.
   
   #endif
