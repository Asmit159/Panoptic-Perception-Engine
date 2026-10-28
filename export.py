import torch
import onnx
from onnxsim import simplify
from model import PanopticPerceptionEngine

def generate_optimized_onnx():
    model = PanopticPerceptionEngine().eval()
    
    checkpoint = torch.load('panoptic_best_v3.pth', map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])

    dummy_input = torch.randn(1, 3, 384, 640)
    onnx_path = "panoptic_edge.onnx"

    torch.onnx.export(
        model, 
        dummy_input, 
        onnx_path,
        export_params=True, 
        opset_version=13, 
        do_constant_folding=True,
        input_names=['input_image'], 
        output_names=['seg_mask', 'yolo_cls', 'yolo_reg'],
        dynamic_axes={'input_image': {0: 'batch_size'}}
    )

    onnx_model = onnx.load(onnx_path)
    model_simp, check = simplify(onnx_model)
    
    if check:
        onnx.save(model_simp, onnx_path)

if __name__ == "__main__":
    generate_optimized_onnx()
