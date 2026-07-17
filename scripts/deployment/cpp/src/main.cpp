#include "sparsedrive_infer.h"
#include <iostream>
#include <chrono>
#include <iomanip>
#include <string>

int main(int argc, char* argv[]) {
    std::cout << "============================================================" << std::endl;
    std::cout << "SparseDriveV2 C++ Deployment Test" << std::endl;
    std::cout << "============================================================" << std::endl;
    
    std::string model_path = "";
    bool use_gpu = true;
    bool test_mode = false;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--engine" && i + 1 < argc) {
            model_path = argv[i + 1];
            i++;
        } else if (arg == "--model" && i + 1 < argc) {
            model_path = argv[i + 1];
            i++;
        } else if (arg == "--cpu") {
            use_gpu = false;
        } else if (arg == "--gpu") {
            use_gpu = true;
        } else if (arg == "--test") {
            test_mode = true;
        } else if (model_path.empty()) {
            model_path = arg;
        }
    }
    
    if (model_path.empty()) {
        std::cerr << "Usage: " << argv[0] << " --engine <model_path> [--cpu|--gpu]" << std::endl;
        return 1;
    }
    
    std::cout << "\nModel path: " << model_path << std::endl;
    std::cout << "Use GPU: " << (use_gpu ? "Yes" : "No") << std::endl;
    
    sparsedrive::SparseDriveInfer infer;
    
    auto load_start = std::chrono::high_resolution_clock::now();
    bool loaded = infer.load_model(model_path, use_gpu);
    auto load_end = std::chrono::high_resolution_clock::now();
    
    if (!loaded) {
        std::cerr << "Failed to load model" << std::endl;
        return 1;
    }
    
    double load_time = std::chrono::duration<double, std::milli>(load_end - load_start).count() / 1000.0;
    std::cout << "Model load time: " << std::fixed << std::setprecision(2) << load_time << " seconds" << std::endl;
    
    std::cout << "\n" << infer.get_model_info() << std::endl;
    
    int B = 1;
    int num_cams = 3;
    int H = 256;
    int W = 512;
    
    std::cout << "\nTest input config:" << std::endl;
    std::cout << "  Batch size: " << B << std::endl;
    std::cout << "  Camera count: " << num_cams << std::endl;
    std::cout << "  Image size: " << H << " x " << W << std::endl;
    
    torch::Tensor imgs = torch::randn({B, num_cams, 3, H, W}, torch::kFloat32);
    torch::Tensor status = torch::randn({B, 8}, torch::kFloat32);
    torch::Tensor lidar2img = torch::randn({B, num_cams, 4, 4}, torch::kFloat32);
    torch::Tensor lidar2cam = torch::randn({B, num_cams, 4, 4}, torch::kFloat32);
    torch::Tensor cam2lidar = torch::randn({B, num_cams, 4, 4}, torch::kFloat32);
    torch::Tensor cam_intrinsic = torch::randn({B, num_cams, 3, 3}, torch::kFloat32);
    
    sparsedrive::InferenceInput input;
    input.imgs = imgs;
    input.status_feature = status;
    input.lidar2img = lidar2img;
    input.lidar2cam = lidar2cam;
    input.cam2lidar = cam2lidar;
    input.cam_intrinsic = cam_intrinsic;
    
    sparsedrive::InferenceOutput output;
    
    for (int i = 0; i < 3; ++i) {
        infer.infer(input, output);
        if (infer.get_device().is_cuda()) {
            torch::cuda::synchronize();
        }
    }
    
    const int num_runs = 10;
    std::vector<double> times;
    
    for (int i = 0; i < num_runs; ++i) {
        auto start = std::chrono::high_resolution_clock::now();
        infer.infer(input, output);
        if (infer.get_device().is_cuda()) {
            torch::cuda::synchronize();
        }
        auto end = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double, std::milli>(end - start).count();
        times.push_back(elapsed);
    }
    
    double avg_time_ms = 0;
    double min_time_ms = times[0];
    double max_time_ms = times[0];
    
    for (double t : times) {
        avg_time_ms += t;
        min_time_ms = std::min(min_time_ms, t);
        max_time_ms = std::max(max_time_ms, t);
    }
    avg_time_ms /= num_runs;
    
    std::cout << "\n============================================================" << std::endl;
    std::cout << "Inference Test Results" << std::endl;
    std::cout << "============================================================" << std::endl;
    std::cout << "Device: " << (infer.get_device().is_cuda() ? "CUDA" : "CPU") << std::endl;
    std::cout << "Average inference time: " << std::fixed << std::setprecision(4) << avg_time_ms << " ms" << std::endl;
    std::cout << "Minimum inference time: " << std::fixed << std::setprecision(4) << min_time_ms << " ms" << std::endl;
    std::cout << "Maximum inference time: " << std::fixed << std::setprecision(4) << max_time_ms << " ms" << std::endl;
    std::cout << "FPS: " << std::fixed << std::setprecision(2) << 1000.0 / avg_time_ms << std::endl;
    
    torch::Tensor traj = output.trajectory.cpu();
    std::cout << "\nOutput shape: [" << traj.size(0) << ", " << traj.size(1) << ", " << traj.size(2) << "]" << std::endl;
    std::cout << "\nOutput example:" << std::endl;
    
    for (int i = 0; i < std::min(3, static_cast<int>(traj.size(1))); ++i) {
        std::cout << "  [" << std::fixed << std::setprecision(4) 
                  << traj[0][i][0].item<float>() << ", " 
                  << traj[0][i][1].item<float>() << ", " 
                  << traj[0][i][2].item<float>() << "]" << std::endl;
    }
    
    std::cout << "\n============================================================" << std::endl;
    std::cout << "Test completed!" << std::endl;
    std::cout << "============================================================" << std::endl;
    
    return 0;
}
