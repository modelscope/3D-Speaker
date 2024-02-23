import os
import numpy as np
import onnxruntime

class ASDTalknet:
    """
    Active speaker detection with TalkNet pretrained model. 
    Reference:
    - https://github.com/TaoRuijie/TalkNet-ASD
    """
    def __init__(self, onnx_dir, device='cpu', device_id=0):
        onnx_file_name = os.path.join(onnx_dir, 'asd.onnx')
        assert os.path.exists(onnx_file_name), '%s does not exist. Please check if it has been downloaded accurately.' % onnx_file_name
        self.ort_net = self.create_net(onnx_file_name, device, device_id)

    def __call__(self, inputA, inputV):
        ort_inputs = {self.ort_net.get_inputs()[0].name:inputA, self.ort_net.get_inputs()[1].name:inputV}
        scores = self.ort_net.run(None, ort_inputs)[0]
        return scores

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
    predictor = ASDTalknet('pretrained_models', 'cuda', 0)
    inputA = np.random.randn(1, 100, 13).astype('float32')
    inputV = np.random.randn(1, 25, 112, 112).astype('float32')
    scores = predictor(inputA, inputV)
    assert scores.shape == (25,)
    