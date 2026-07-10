#ifndef SPARSEDRIVE_INFER_H
#define SPARSEDRIVE_INFER_H

#include <torch/script.h>
#include <torch/torch.h>
#include <memory>
#include <vector>
#include <string>

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
    std::shared_ptr<torch::jit::script::Module> model_;
    torch::Device device_;
    bool model_loaded_;
};

} 

#endif