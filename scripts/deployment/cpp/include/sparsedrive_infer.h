#ifndef SPARSEDRIVE_INFER_H
#define SPARSEDRIVE_INFER_H

#include <torch/script.h>
#include <torch/torch.h>
#include <NvInfer.h>
#include <memory>
#include <vector>
#include <string>
#include <map>

namespace sparsedrive {

struct InferenceInput {
    torch::Tensor imgs;
    torch::Tensor status_feature;
    torch::Tensor lidar2img;
    torch::Tensor lidar2cam;
    torch::Tensor cam2lidar;
    torch::Tensor cam_intrinsic;
};

struct InferenceOutput {
    torch::Tensor trajectory;
};

class Logger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override;
};

class SparseDriveInfer {
public:
    SparseDriveInfer();
    ~SparseDriveInfer();

    bool load_model(const std::string& model_path, bool use_gpu = true);
    
    bool infer(const InferenceInput& input, InferenceOutput& output);
    
    void set_device(torch::DeviceType device_type);
    
    torch::Device get_device() const;
    
    bool is_loaded() const;
    
    std::string get_model_info() const;

private:
    bool load_torchscript_model(const std::string& model_path, bool use_gpu);
    
    bool load_tensorrt_engine(const std::string& engine_path, bool use_gpu);
    
    void allocate_buffers();
    
    bool copy_inputs(const InferenceInput& input);
    
    bool copy_outputs(InferenceOutput& output);

    std::shared_ptr<torch::jit::script::Module> ts_model_;
    std::unique_ptr<nvinfer1::ICudaEngine> trt_engine_;
    std::unique_ptr<nvinfer1::IExecutionContext> trt_context_;
    std::unique_ptr<Logger> logger_;
    
    torch::Device device_;
    bool model_loaded_;
    bool use_trt_;
    
    std::vector<torch::Tensor> input_tensors_;
    std::vector<torch::Tensor> output_tensors_;
    std::map<std::string, int> input_indices_;
    std::map<std::string, int> output_indices_;
};

} 

#endif