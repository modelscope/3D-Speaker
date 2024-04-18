import cv2
import os
import numpy as np
import onnxruntime

class FaceQualityAssess:
    """
    Face quality assessment with pretrained model. 
    Reference:
    - https://modelscope.cn/models/iic/cv_manual_face-quality-assessment_fqa
    """
    def __init__(self, onnx_dir, device='cpu', device_id=0):
        onnx_file_name = os.path.join(onnx_dir, 'fqa.onnx')
        assert os.path.exists(onnx_file_name), '%s does not exist. Please check if it has been downloaded accurately.' % onnx_file_name
        self.ort_net = self.create_net(onnx_file_name, device, device_id)

    def __call__(self, img):
        img = img[:, :, ::-1] # bgr to rgb
        img = cv2.resize(img, (112, 112))
        img = np.transpose(img, axes=(2, 0, 1))
        img = (img / 255. - 0.5) / 0.5
        img = np.expand_dims(img.astype(np.float32), 0)
        ort_inputs = {self.ort_net.get_inputs()[0].name:img}
        result = self.ort_net.run(None, ort_inputs)[0]
        score = np.mean(result)
        return score

    def create_net(self, onnx_file_name, device='cpu', device_id=0):
        options = onnxruntime.SessionOptions()
        # set op_num_threads
        options.intra_op_num_threads = 8
        options.inter_op_num_threads = 8
        # set providers
        providers = ['CPUExecutionProvider']
        if device == 'cuda':
            providers.insert(0, ('CUDAExecutionProvider', {'device_id': device_id}))
        ort_session = onnxruntime.InferenceSession(onnx_file_name, options, providers=providers) 
        return ort_session


if __name__ == '__main__':
    predictor = FaceQualityAssess('pretrained_models', 'cuda', '0')
    input = np.random.randn(221, 196, 3).astype('float32')
    output = predictor(input)
    print(output)
