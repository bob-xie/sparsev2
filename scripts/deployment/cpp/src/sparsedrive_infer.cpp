#include "sparsedrive_infer.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <cstring>
#include <NvInferRuntimeCommon.h>
#include <cuda_runtime.h>

namespace sparsedrive {

void Logger::log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept {
    if (severity <= nvinfer1::ILogger::Severity::kINFO) {
        std::cout << "[TRT] " << msg << std::endl;
    }
}

SparseDriveInfer::SparseDriveInfer() 
    : model_loaded_(false), 
      device_(torch::kCPU),
      use_trt_(false),
      logger_(std::make_unique<Logger>()) {}

SparseDriveInfer::~SparseDriveInfer() {
    input_tensors_.clear();
    output_tensors_.clear();
    input_indices_.clear();
    output_indices_.clear();
}

bool SparseDriveInfer::load_model(const std::string& model_path, bool use_gpu) {
    size_t dot_pos = model_path.find_last_of('.');
    std::string extension = (dot_pos != std::string::npos) ? model_path.substr(dot_pos) : "";
    
    if (extension == ".engine") {
        return load_tensorrt_engine(model_path, use_gpu);
    } else {
        return load_torchscript_model(model_path, use_gpu);
    }
}

bool SparseDriveInfer::load_torchscript_model(const std::string& model_path, bool use_gpu) {
    try {
        ts_model_ = std::make_shared<torch::jit::script::Module>(
            torch::jit::load(model_path)
        );
        
        if (use_gpu && torch::cuda::is_available()) {
            device_ = torch::kCUDA;
            ts_model_->to(device_);
            std::cout << "Using CUDA device" << std::endl;
        } else {
            device_ = torch::kCPU;
            ts_model_->to(device_);
            std::cout << "Using CPU device" << std::endl;
        }
        
        ts_model_->eval();
        model_loaded_ = true;
        use_trt_ = false;
        
        std::cout << "Model loaded successfully: " << model_path << std::endl;
        return true;
    } catch (const c10::Error& e) {
        std::cerr << "Failed to load model: " << e.what() << std::endl;
        return false;
    }
}

bool SparseDriveInfer::load_tensorrt_engine(const std::string& engine_path, bool use_gpu) {
    try {
        std::ifstream engine_file(engine_path, std::ios::binary);
        if (!engine_file) {
            std::cerr << "Failed to open engine file: " << engine_path << std::endl;
            return false;
        }
        
        engine_file.seekg(0, std::ios::end);
        size_t size = engine_file.tellg();
        engine_file.seekg(0, std::ios::beg);
        
        std::vector<char> buffer(size);
        engine_file.read(buffer.data(), size);
        
        nvinfer1::IRuntime* runtime = nvinfer1::createInferRuntime(*logger_);
        
        trt_engine_ = std::unique_ptr<nvinfer1::ICudaEngine>(
            runtime->deserializeCudaEngine(buffer.data(), size)
        );
        
        if (!trt_engine_) {
            std::cerr << "Failed to deserialize TensorRT engine" << std::endl;
            return false;
        }
        
        trt_context_ = std::unique_ptr<nvinfer1::IExecutionContext>(
            trt_engine_->createExecutionContext()
        );
        
        if (!trt_context_) {
            std::cerr << "Failed to create TensorRT execution context" << std::endl;
            return false;
        }
        
        if (use_gpu && torch::cuda::is_available()) {
            device_ = torch::kCUDA;
            std::cout << "Using CUDA device for TensorRT" << std::endl;
        } else {
            device_ = torch::kCPU;
            std::cout << "Using CPU device" << std::endl;
        }
        
        allocate_buffers();
        
        model_loaded_ = true;
        use_trt_ = true;
        
        std::cout << "TensorRT engine loaded successfully: " << engine_path << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "Failed to load TensorRT engine: " << e.what() << std::endl;
        return false;
    }
}

void SparseDriveInfer::allocate_buffers() {
    if (!trt_engine_) return;
    
    int num_io_tensors = trt_engine_->getNbIOTensors();
    
    std::vector<std::string> input_names = {"imgs", "status_feature", "lidar2img", 
                                            "lidar2cam", "cam2lidar", "cam_intrinsic"};
    std::vector<std::string> output_names = {"trajectory"};
    
    for (int i = 0; i < num_io_tensors; ++i) {
        const char* name = trt_engine_->getIOTensorName(i);
        std::string name_str(name);
        
        nvinfer1::DataType dtype = trt_engine_->getTensorDataType(name_str.c_str());
        
        size_t element_size = sizeof(float);
        torch::ScalarType torch_dtype = torch::kFloat32;
        
        if (dtype == nvinfer1::DataType::kFLOAT) {
            element_size = sizeof(float);
            torch_dtype = torch::kFloat32;
        } else if (dtype == nvinfer1::DataType::kHALF) {
            element_size = 2;
            torch_dtype = torch::kFloat16;
        }
        
        torch::TensorOptions options = torch::TensorOptions()
            .dtype(torch_dtype)
            .device(device_);
        
        torch::Tensor tensor = torch::empty({1024}, options);
        
        trt_context_->setTensorAddress(name, tensor.data_ptr());
        
        bool is_input = false;
        for (const auto& in_name : input_names) {
            if (name_str == in_name) {
                is_input = true;
                break;
            }
        }
        
        if (is_input) {
            input_tensors_.push_back(tensor);
            input_indices_[name_str] = input_tensors_.size() - 1;
        } else {
            output_tensors_.push_back(tensor);
            output_indices_[name_str] = output_tensors_.size() - 1;
        }
    }
}

bool SparseDriveInfer::copy_inputs(const InferenceInput& input) {
    if (input_indices_.empty()) return false;
    
    auto copy_tensor = [&](const std::string& name, const torch::Tensor& src) {
        if (input_indices_.find(name) == input_indices_.end()) return false;
        int idx = input_indices_[name];
        input_tensors_[idx].copy_(src.to(device_));
        return true;
    };
    
    copy_tensor("imgs", input.imgs);
    copy_tensor("status_feature", input.status_feature);
    copy_tensor("lidar2img", input.lidar2img);
    copy_tensor("lidar2cam", input.lidar2cam);
    copy_tensor("cam2lidar", input.cam2lidar);
    copy_tensor("cam_intrinsic", input.cam_intrinsic);
    
    return true;
}

bool SparseDriveInfer::copy_outputs(InferenceOutput& output) {
    if (output_indices_.find("trajectory") == output_indices_.end()) return false;
    
    int idx = output_indices_["trajectory"];
    output.trajectory = output_tensors_[idx].clone().to(torch::kFloat32);
    
    return true;
}

bool SparseDriveInfer::infer(const InferenceInput& input, InferenceOutput& output) {
    if (!model_loaded_) {
        std::cerr << "Model not loaded" << std::endl;
        return false;
    }
    
    try {
        torch::NoGradGuard no_grad;
        
        if (use_trt_) {
            copy_inputs(input);
            
            cudaStream_t stream = nullptr;
            
            bool success = trt_context_->enqueueV3(stream);
            if (!success) {
                std::cerr << "TensorRT inference failed" << std::endl;
                return false;
            }
            
            if (device_.is_cuda()) {
                cudaStreamSynchronize(stream);
            }
            
            copy_outputs(output);
        } else {
            std::vector<torch::jit::IValue> inputs;
            inputs.push_back(input.imgs.to(device_));
            inputs.push_back(input.status_feature.to(device_));
            inputs.push_back(input.lidar2img.to(device_));
            inputs.push_back(input.lidar2cam.to(device_));
            inputs.push_back(input.cam2lidar.to(device_));
            inputs.push_back(input.cam_intrinsic.to(device_));
            
            torch::Tensor result = ts_model_->forward(inputs).toTensor();
            output.trajectory = result;
        }
        
        return true;
    } catch (const c10::Error& e) {
        std::cerr << "Inference failed: " << e.what() << std::endl;
        return false;
    }
}

void SparseDriveInfer::set_device(torch::DeviceType device_type) {
    device_ = torch::Device(device_type);
    if (model_loaded_ && !use_trt_) {
        ts_model_->to(device_);
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
    oss << "Type: " << (use_trt_ ? "TensorRT Engine" : "TorchScript") << std::endl;
    oss << "Device: " << (device_.is_cuda() ? "CUDA" : "CPU") << std::endl;
    
    if (torch::cuda::is_available()) {
        int device_count = torch::cuda::device_count();
        if (device_count > 0) {
            oss << "GPU count: " << device_count << std::endl;
        }
    }
    
    if (use_trt_ && trt_engine_) {
        int num_io_tensors = trt_engine_->getNbIOTensors();
        oss << "TRT IO tensors: " << num_io_tensors << std::endl;
        oss << "TRT inputs: " << input_indices_.size() << std::endl;
        oss << "TRT outputs: " << output_indices_.size() << std::endl;
    }
    
    return oss.str();
}

} 
