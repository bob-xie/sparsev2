"""
从 TorchScript (.pt) 模型导出 ONNX 格式

功能：将已导出的 .pt 文件转换为 ONNX 格式，用于后续 TensorRT 编译

使用方法：
    python export_onnx_from_pt.py --pt model.pt --output model.onnx

输入格式：
    imgs: (B, num_cams, 3, H, W)  # B=1, num_cams=3, H=256, W=512
    status_feature: (B, 8)
    lidar2img: (B, num_cams, 4, 4)
    lidar2cam: (B, num_cams, 4, 4)
    cam2lidar: (B, num_cams, 4, 4)
    cam_intrinsic: (B, num_cams, 3, 3)

输出格式：
    trajectory: (B, 9, 3)  # 9个轨迹点 (x, y, heading)
"""

import argparse
import os
import sys

import torch


def export_onnx(pt_path: str, output_path: str, opset_version: int = 17):
    print("=" * 60)
    print("SparseDriveV2 ONNX Export from TorchScript")
    print("=" * 60)
    print(f"\nLoading TorchScript model from: {pt_path}")

    if not os.path.exists(pt_path):
        print(f"Error: Model file not found - {pt_path}")
        sys.exit(1)

    model = torch.jit.load(pt_path)
    model.eval()
    print("✓ Model loaded successfully!")

    B = 1
    num_cams = 3
    H, W = 256, 512

    example_imgs = torch.randn(B, num_cams, 3, H, W)
    example_status = torch.randn(B, 8)
    example_lidar2img = torch.randn(B, num_cams, 4, 4)
    example_lidar2cam = torch.randn(B, num_cams, 4, 4)
    example_cam2lidar = torch.randn(B, num_cams, 4, 4)
    example_cam_intrinsic = torch.randn(B, num_cams, 3, 3)

    print("\nRunning forward pass for trace...")
    with torch.no_grad():
        model(example_imgs, example_status, example_lidar2img,
              example_lidar2cam, example_cam2lidar, example_cam_intrinsic)
    print("✓ Forward pass completed!")

    print(f"\nExporting to ONNX (opset version: {opset_version})...")
    torch.onnx.export(
        model,
        (example_imgs, example_status, example_lidar2img,
         example_lidar2cam, example_cam2lidar, example_cam_intrinsic),
        output_path,
        opset_version=opset_version,
        input_names=["imgs", "status_feature", "lidar2img",
                     "lidar2cam", "cam2lidar", "cam_intrinsic"],
        output_names=["trajectory"],
        dynamic_axes={
            "imgs": {0: "batch_size"},
            "status_feature": {0: "batch_size"},
            "lidar2img": {0: "batch_size"},
            "lidar2cam": {0: "batch_size"},
            "cam2lidar": {0: "batch_size"},
            "cam_intrinsic": {0: "batch_size"},
            "trajectory": {0: "batch_size"},
        },
        verbose=False,
        do_constant_folding=True,
        export_params=True,
    )

    print(f"\n✓ ONNX model exported to: {output_path}")

    print("\nVerifying ONNX model...")
    try:
        import onnx
        import onnxruntime as ort

        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("✓ ONNX model is valid!")

        sess = ort.InferenceSession(output_path, providers=["CPUExecutionProvider"])

        input_dict = {
            "imgs": example_imgs.numpy(),
            "status_feature": example_status.numpy(),
            "lidar2img": example_lidar2img.numpy(),
            "lidar2cam": example_lidar2cam.numpy(),
            "cam2lidar": example_cam2lidar.numpy(),
            "cam_intrinsic": example_cam_intrinsic.numpy(),
        }

        onnx_output = sess.run(["trajectory"], input_dict)[0]

        with torch.no_grad():
            pt_output = model(example_imgs, example_status, example_lidar2img,
                              example_lidar2cam, example_cam2lidar, example_cam_intrinsic).numpy()

        max_diff = abs(onnx_output - pt_output).max()
        print(f"✓ ONNX vs PT output max difference: {max_diff:.6f}")

        if max_diff < 1e-3:
            print("✓ Output consistency check passed!")
        else:
            print(f"⚠ Output difference is larger than expected: {max_diff}")

        print(f"\nONNX model info:")
        print(f"  Inputs: {[input.name for input in sess.get_inputs()]}")
        print(f"  Outputs: {[output.name for output in sess.get_outputs()]}")
        for input in sess.get_inputs():
            print(f"    {input.name}: {input.shape}")
        for output in sess.get_outputs():
            print(f"    {output.name}: {output.shape}")

    except ImportError:
        print("⚠ ONNX/ONNX Runtime not installed, skipping verification")

    print("\n" + "=" * 60)
    print("ONNX export completed!")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Export SparseDriveV2 .pt model to ONNX")
    parser.add_argument(
        "--pt",
        type=str,
        required=True,
        help="Path to the TorchScript model (.pt)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="model.onnx",
        help="Output path for the ONNX model"
    )
    parser.add_argument(
        "--opset",
        type=int,
        default=17,
        help="ONNX opset version (default: 17)"
    )

    args = parser.parse_args()

    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    export_onnx(args.pt, args.output, args.opset)


if __name__ == "__main__":
    main()