#pragma once
#include <iostream>
#include <vector>
#include <memory>
#include <NvInfer.h>

// CUDA RUNTIME API for cuda error check
#define CHECK(status)                                          \
    do                                                         \
    {                                                          \
        auto ret = (status);                                   \
        if (ret != 0)                                          \
        {                                                      \
            std::cerr << "Cuda failure: " << ret << std::endl; \
            abort();                                           \
        }                                                      \
    } while (0)


// Logger for TensorRT info/warning/errors
class Logger : public nvinfer1::ILogger
{
public:
    void log(Severity severity, const char *msg) noexcept override
    {
        // Filter out info-level messages
        // if (severity != Severity::kERROR) // 1 
        // if (severity != Severity::kWARNING) // 2 
        // if (severity != Severity::kINFO) // 3
        if (severity != Severity::kVERBOSE) // 4
        {
            std::cout << msg << std::endl;
        }
    }
};


/**
 * @class base_opti_trt
 * @brief base inference class using by TensorRT
 * This base class is for inference optimization.
 */
class base_opti_trt
{
public:
    base_opti_trt(
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
        std::string weight_file_path_)
        : 
        batch_size(batch_size_), 
        input_h(input_h_), 
        input_w(input_w_), 
        input_c(input_c_), 
        class_count(class_count_), 
        precision_mode(precision_mode_), 
        serialize(serialize_),
        gpu_device(gpu_device_), 
        engine_dir_path(engine_dir_path_), 
        engine_file_name(engine_file_name_),
        weight_file_path(weight_file_path_)
    {}

    virtual void input_data(const void* inputs) = 0;

    virtual void run_model() = 0;

    virtual void output_data(void* outputs) = 0;

    virtual ~base_opti_trt();

protected:

     /**
     * @brief gpu device count
     * @return int
     */
    int get_device_count();

    /**
     * @brief check device information print
     * @return void
     */
    void check_device();

    /**
     * @brief set specific gpu device
     * @details If there is no GPU corresponding to the received gpu_index, it is treated as index 0.
     * @param[in] gpu_index Index of GPU you want to allocate
     * @return void
     */
    void set_device(int& gpu_index);

    Logger trtLogger;   //!< trt log

    int batch_size;
    int input_h;
    int input_w;
    int input_c;
    int class_count;

    int precision_mode; //!< fp32 : 32, fp16 : 16, int8(ptq) : 8
    bool serialize;     //!< force serialize flag (IF true, recreate the engine file unconditionally)
    int gpu_device;     //!< gpu device index (default = 0)

    std::string engine_dir_path;  //!< Engine directory path
    std::string engine_file_name; //!< model name
    std::string engine_file_path; //!< Engine full path
    std::string weight_file_path; //!< weight file path

    std::vector<void*> buffers;

    std::unique_ptr<nvinfer1::IRuntime> runtime{ nullptr };
    std::shared_ptr<nvinfer1::ICudaEngine> engine{ nullptr };
    std::unique_ptr<nvinfer1::IExecutionContext> context{ nullptr };
    cudaStream_t stream;
};
