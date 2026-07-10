#include "sparsedrive_infer.h"

namespace sparsedrive {

SparseDriveInfer::SparseDriveInfer() : model_loaded_(false), device_(torch::kCPU) {}

SparseDriveInfer::~SparseDriveInfer() {}

bool SparseDriveInfer::load_model(const std::string& model_path, bool use_gpu) {
    try {
        model_ = std::make_shared<torch::jit::script::Module>(
            torch::jit::load(model_path)
        );
        
        if (use_gpu && torch::cuda::is_available()) {
            device_ = torch::kCUDA;
            model_->to(device_);
            std::cout << "Using CUDA device" << std::endl;
        } else {
            device_ = torch::kCPU;
            model_->to(device_);
            std::cout << "Using CPU device" << std::endl;
        }
        
        model_->eval();
        model_loaded_ = true;
        
        std::cout << "Model loaded successfully: " << model_path << std::endl;
        
        std::string filename = model_path.substr(model_path.find_last_of("/\\") + 1);
        if (filename.find(".ts") != std::string::npos) {
            std::cout << "Note: This appears to be a TensorRT optimized model (.ts)" << std::endl;
        }
        
        return true;
    } catch (const c10::Error& e) {
        std::cerr << "Failed to load model: " << e.what() << std::endl;
        std::cerr << "Possible reasons:" << std::endl;
        std::cerr << "  - Model was compiled with different PyTorch/TensorRT version" << std::endl;
        std::cerr << "  - TensorRT library not found" << std::endl;
        std::cerr << "  - CUDA driver version mismatch" << std::endl;
        return false;
    }
}

bool SparseDriveInfer::infer(const InferenceInput& input, InferenceOutput& output) {
    if (!model_loaded_) {
        std::cerr << "Model not loaded" << std::endl;
        return false;
    }
    
    try {
        torch::NoGradGuard no_grad;
        
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input.imgs.to(device_));
        inputs.push_back(input.status_feature.to(device_));
        inputs.push_back(input.lidar2img.to(device_));
        inputs.push_back(input.lidar2cam.to(device_));
        inputs.push_back(input.cam2lidar.to(device_));
        inputs.push_back(input.cam_intrinsic.to(device_));
        
        torch::Tensor result = model_->forward(inputs).toTensor();
        output.trajectory = result;
        
        return true;
    } catch (const c10::Error& e) {
        std::cerr << "Inference failed: " << e.what() << std::endl;
        return false;
    }
}

void SparseDriveInfer::set_device(torch::DeviceType device_type) {
    device_ = torch::Device(device_type);
    if (model_loaded_) {
        model_->to(device_);
    }
}

torch::Device SparseDriveInfer::get_device() const {
    return device_;
}

bool SparseDriveInfer::is_loaded() const {
    return model_loaded_;
}

std::string SparseDriveInfer::get_model_info() const {
    if (!model_loaded_) {
        return "Model not loaded";
    }
    
    std::ostringstream oss;
    oss << "Model: SparseDriveV2" << std::endl;
    oss << "Device: " << (device_.is_cuda() ? "CUDA" : "CPU") << std::endl;
    
    if (torch::cuda::is_available()) {
        int device_count = torch::cuda::device_count();
        if (device_count > 0) {
            oss << "GPU count: " << device_count << std::endl;
        }
    }
    
    return oss.str();
}

} 