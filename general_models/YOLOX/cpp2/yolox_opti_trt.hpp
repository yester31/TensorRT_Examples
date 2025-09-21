#pragma once
#include "base_opti_trt.hpp"
#include <filesystem>
#include <fstream>
#include <NvOnnxParser.h>
#include <NvInferPlugin.h>

class yolox_opti_trt : public base_opti_trt
{
public:
    /**
     * @brief yolox model inference constructor
     * @details yolox_opti_trt class constructor and generate engine using ONNX or load engine file and inference.
     * @param[in] batch_size_        input image batch size
     * @param[in] input_h_           input image height
     * @param[in] input_w_           input image width
     * @param[in] input_c_           input image channel
     * @param[in] class_count_       the number of class
     * @param[in] precision_mode_    precision mode for trt model config (fp32 mode : 32, fp16 mode : 16)
     * @param[in] serialize_         force serialize flag (IF true, recreate the engine file unconditionally)
     * @param[in] gpu_device_        gpu device index (default = 0)
     * @param[in] engine_dir_path_   engine directory path (engine file will be generated in this location)
     * @param[in] engine_file_name_  engine file name (engine file will be generated uisng this name)
     * @param[in] weight_file_path_  weight file path
     * @return
     */

    yolox_opti_trt(
        int batch_size_, 
        int input_h_, 
        int input_w_, 
        int input_c_, 
        int class_count_, 
        int precision_mode_, 
        bool serialize_,
        int gpu_device_,
        std::string engine_dir_path_, 
        std::string engine_file_name_, 
        std::string weight_file_path_);

    /**
     * @brief send input date to model (host -> device)
     * @details There are two preprocess mode (cuda version : 0, cpp version : 1)
     * @param[in] inputs input data (host)
     * @return void
     */

    void input_data(const void* inputs) override;

    /**
     * @brief run inference model
     * @details run inference model
     * @return void
     */

    void run_model() override;

    /**
     * @brief get output date from model (device -> host)
     * @details get output from model and then postprocess
     * @param[out] outputs output data (host)
     * @return void
     */

    void output_data(void* outputs) override;

    /**
     * @brief infer destructor
     * @details delete device memory space
     */

    ~yolox_opti_trt();

private:
    // Creat the engine using onnx.
    void createEngineFromOnnx(std::unique_ptr<nvinfer1::IBuilder>& builder, std::unique_ptr<nvinfer1::IBuilderConfig>& config);

    std::vector<int> output_post0;      //!< output0 for cpu 
    std::vector<float> output_post1;    //!< output1 for cpu 
    std::vector<float> output_post2;    //!< output2 for cpu 
    std::vector<int> output_post3;      //!< output3 for cpu 

    int INPUT_SIZE0;
    int OUTPUT_SIZE0 = 1;
    int OUTPUT_SIZE1 = 300 * 4;
    int OUTPUT_SIZE2 = 300;
    int OUTPUT_SIZE3 = 300;

    std::string INPUT_NAME0 = "input";
    std::string OUTPUT_NAME0 = "num_dets";
    std::string OUTPUT_NAME1 = "det_boxes";
    std::string OUTPUT_NAME2 = "det_scores";
    std::string OUTPUT_NAME3 = "det_classes";
};
