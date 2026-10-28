import onnxruntime as ort
import numpy as np
import cv2
import time

class EdgeInferencer:
    def __init__(self, onnx_path):
        trt_providers = [
            ('TensorrtExecutionProvider', {
                'trt_fp16_enable': True,
                'trt_engine_cache_enable': True,
                'trt_engine_cache_path': './trt_cache'
            }),
            'CUDAExecutionProvider'
        ]
        
        self.session = ort.InferenceSession(onnx_path, providers=trt_providers)
        self.input_name = self.session.get_inputs()[0].name
        
        dummy_input = np.random.randn(1, 3, 384, 640).astype(np.float32)
        for _ in range(5):
            self.session.run(None, {self.input_name: dummy_input})

    def preprocess(self, image_path):
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (640, 384))
        img_tensor = np.transpose(img_resized, (2, 0, 1)) 
        img_tensor = np.expand_dims(img_tensor, axis=0).astype(np.float32) / 255.0
        return img, img_tensor

    def infer(self, image_path):
        orig_img, input_tensor = self.preprocess(image_path)

        start_time = time.perf_counter()
        
        outputs = self.session.run(None, {self.input_name: input_tensor})
        
        latency = (time.perf_counter() - start_time) * 1000
        seg_mask, yolo_cls, yolo_reg = outputs

        print(f"TensorRT Inference Latency: {latency:.2f} ms")
        return seg_mask, yolo_cls, yolo_reg

if __name__ == "__main__":
    inferencer = EdgeInferencer("panoptic_edge.onnx")
