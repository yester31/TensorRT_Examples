#include "base_opti_trt.hpp"

base_opti_trt::~base_opti_trt()
{
    for (auto buf : buffers) {
        CHECK(cudaFree(buf));
    }
    cudaStreamDestroy(stream);
}

// gpu device count
int base_opti_trt::get_device_count()
{
    // This function call returns 0 if there are no CUDA capable devices.
    int deviceCount = 0;
    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

    if (error_id != cudaSuccess)
    {
        trtLogger.log(Logger::Severity::kWARNING, ("cudaGetDeviceCount returned " + 
            std::to_string(static_cast<int>(error_id)) + " -> " + cudaGetErrorString(error_id)).c_str());
        exit(EXIT_FAILURE);
    }

    return deviceCount;
}

// check device information print
void base_opti_trt::check_device()
{
    // 1. Get the number of GPU device
    int deviceCount = get_device_count();

    if (deviceCount == 0)
    {
        trtLogger.log(Logger::Severity::kWARNING, "There are no available device(s) that support CUDA");
        exit(EXIT_FAILURE);
    }
    else
    {
        trtLogger.log(Logger::Severity::kINFO, ("Detected " + std::to_string(deviceCount) + " CUDA Capable device(s)").c_str());
    }

    // 2. Check each deivce information
    int dev, driverVersion = 0, runtimeVersion = 0;
    for (dev = 0; dev < deviceCount; ++dev)
    {
        cudaSetDevice(dev);
        cudaDeviceProp deviceProp;
        cudaGetDeviceProperties(&deviceProp, dev);
        trtLogger.log(Logger::Severity::kINFO, ("Device [" + std::to_string(dev) + "] " + deviceProp.name).c_str());

        // Cuda Version & Compute Capability
        cudaDriverGetVersion(&driverVersion);
        cudaRuntimeGetVersion(&runtimeVersion);
        trtLogger.log(Logger::Severity::kINFO, ("CUDA Driver Version / Runtime Version          " + std::to_string(driverVersion / 1000) + "." + std::to_string((driverVersion % 100) / 10) 
            + " / " + std::to_string(runtimeVersion / 1000) + "." + std::to_string((runtimeVersion % 100) / 10)).c_str());
        trtLogger.log(Logger::Severity::kINFO, ("CUDA Capability Major/Minor version number:    " + std::to_string(deviceProp.major) + "." + std::to_string(deviceProp.minor)).c_str());
    }

    // 3. Set default deivce index (0)
    cudaSetDevice(0);
}

// set specific gpu device
void base_opti_trt::set_device(int& gpu_index)
{
    // 1. Get the number of GPU device
    int deviceCount = get_device_count();

    if (deviceCount == 0)
    {
        trtLogger.log(Logger::Severity::kWARNING, "There are no available device(s) that support CUDA");
        exit(EXIT_FAILURE);
    }
    else
    {
        trtLogger.log(Logger::Severity::kINFO, ("Selectable index range : [0, " + std::to_string(deviceCount) + ")").c_str());
    }

    // 2. Check given gpu index
    trtLogger.log(Logger::Severity::kINFO, ("Given GPU index parameter: " + std::to_string(gpu_index)).c_str());
    if (gpu_index < deviceCount)
    {
        cudaSetDevice(gpu_index);
        trtLogger.log(Logger::Severity::kINFO, ("Selected GPU index : " + std::to_string(gpu_index)).c_str());
    }
    else
    {
        trtLogger.log(Logger::Severity::kINFO, "There is no corresponding gpu index device. Check GPU device count.");
        gpu_index = 0;
        cudaSetDevice(gpu_index);
        trtLogger.log(Logger::Severity::kINFO, ("Selected GPU index : " + std::to_string(gpu_index)).c_str());
    }
}
