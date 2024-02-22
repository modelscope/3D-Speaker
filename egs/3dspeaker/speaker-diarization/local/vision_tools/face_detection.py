import torch
import cv2
import os
import numpy as np
import onnxruntime


def area_of(left_top, right_bottom) -> torch.Tensor:
    """Compute the areas of rectangles given two corners.

    Args:
        left_top (N, 2): left top corner.
        right_bottom (N, 2): right bottom corner.

    Returns:
        area (N): return the area.
    """
    hw = torch.clamp(right_bottom - left_top, min=0.0)
    return hw[..., 0] * hw[..., 1]

def iou_of(boxes0, boxes1, eps=1e-5):
    """Return intersection-over-union (Jaccard index) of boxes.

    Args:
        boxes0 (N, 4): ground truth boxes.
        boxes1 (N or 1, 4): predicted boxes.
        eps: a small number to avoid 0 as denominator.
    Returns:
        iou (N): IoU values.
    """
    overlap_left_top = torch.max(boxes0[..., :2], boxes1[..., :2])
    overlap_right_bottom = torch.min(boxes0[..., 2:], boxes1[..., 2:])

    overlap_area = area_of(overlap_left_top, overlap_right_bottom)
    area0 = area_of(boxes0[..., :2], boxes0[..., 2:])
    area1 = area_of(boxes1[..., :2], boxes1[..., 2:])
    return overlap_area / (area0 + area1 - overlap_area + eps)

def hard_nms(box_scores, iou_threshold, top_k=-1, candidate_size=200):
    """
    Args:
        box_scores (N, 5): boxes in corner-form and probabilities.
        iou_threshold: intersection over union threshold.
        top_k: keep top_k results. If k <= 0, keep all the results.
        candidate_size: only consider the candidates with the highest scores.
    Returns:
         picked: a list of indexes of the kept boxes
    """
    scores = box_scores[:, -1]
    boxes = box_scores[:, :-1]
    picked = []
    _, indexes = scores.sort(descending=True)
    indexes = indexes[:candidate_size]
    while len(indexes) > 0:
        current = indexes[0]
        picked.append(current.item())
        if 0 < top_k == len(picked) or len(indexes) == 1:
            break
        current_box = boxes[current, :]
        indexes = indexes[1:]
        rest_boxes = boxes[indexes, :]
        iou = iou_of(
            rest_boxes,
            current_box.unsqueeze(0),
        )
        indexes = indexes[iou <= iou_threshold]

    return box_scores[picked, :]


class Resize(object):
    def __init__(self, size=(300, 300)):
        self.size = size

    def __call__(self, image, boxes=None, labels=None):
        image = cv2.resize(image, (self.size[0],
                                   self.size[1]))
        return image, boxes, labels


class SubtractMeans(object):
    def __init__(self, mean):
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image, boxes=None, labels=None):
        image = image.astype(np.float32)
        image -= self.mean
        return image.astype(np.float32), boxes, labels


class ToTensor(object):
    def __call__(self, cvimage, boxes=None, labels=None):
        return torch.from_numpy(cvimage.astype(np.float32)).permute(2, 0, 1), boxes, labels


class Compose(object):
    """Composes several augmentations together.
    Args:
        transforms (List[Transform]): list of transforms to compose.
    Example:
        >>> augmentations.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, boxes=None, labels=None):
        for t in self.transforms:
            img, boxes, labels = t(img, boxes, labels)
        return img, boxes, labels


class PredictionTransform:
    def __init__(self, size, mean=0.0, std=1.0):
        self.transform = Compose([
            Resize(size),
            SubtractMeans(mean),
            lambda img, boxes=None, labels=None: (img / std, boxes, labels),
            ToTensor()
        ])

    def __call__(self, image):
        image, _, _ = self.transform(image)
        return image


class Config:
    image_size = [320, 240]
    image_mean_test = np.array([127, 127, 127])
    image_std = 128.0


class Predictor:
    """
    Face detection with pretrained model. 
    Reference:
    - https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB
    """
    def __init__(
        self, 
        onnx_dir,
        device='cpu',
        device_id=0,
        iou_threshold=0.3, 
        filter_threshold=0.01, 
        candidate_size=200, 
        ):
        onnx_file_name = os.path.join(onnx_dir, 'version-RFB-320.onnx')
        assert os.path.exists(onnx_file_name), \
            '%s does not exist. Please check if it has been downloaded accurately.' % onnx_file_name
        self.ort_net = self.create_net(onnx_file_name, device, device_id)
        self.transform = PredictionTransform(
            Config.image_size, Config.image_mean_test, Config.image_std)
        self.iou_threshold = iou_threshold
        self.filter_threshold = filter_threshold
        self.candidate_size = candidate_size
        self.device = device

    def __call__(self, image, top_k=-1, prob_threshold=None):
        height, width, _ = image.shape
        image = self.transform(image)
        images = image.unsqueeze(0).numpy()
        # net inference
        inputs = {self.ort_net.get_inputs()[0].name:images}
        scores, boxes = self.ort_net.run(None, inputs)
        boxes = torch.from_numpy(boxes[0])
        scores = torch.from_numpy(scores[0])
        if not prob_threshold:
            prob_threshold = self.filter_threshold
        picked_box_probs = []
        picked_labels = []
        for class_index in range(1, scores.size(1)):
            probs = scores[:, class_index]
            mask = probs > prob_threshold
            probs = probs[mask]
            if probs.size(0) == 0:
                continue
            subset_boxes = boxes[mask, :]
            box_probs = torch.cat([subset_boxes, probs.reshape(-1, 1)], dim=1)
            box_probs = hard_nms(box_probs, self.iou_threshold, top_k, self.candidate_size)
            picked_box_probs.append(box_probs)
            picked_labels.extend([class_index] * box_probs.size(0))
        if not picked_box_probs:
            return torch.tensor([]), torch.tensor([]), torch.tensor([])
        picked_box_probs = torch.cat(picked_box_probs)
        picked_box_probs[:, 0] *= width
        picked_box_probs[:, 1] *= height
        picked_box_probs[:, 2] *= width
        picked_box_probs[:, 3] *= height
        return picked_box_probs[:, :4], torch.tensor(picked_labels), picked_box_probs[:, 4]

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
    predictor_det = Predictor('pretrained_models', 'cuda', '0')
    image_input = np.random.randn(1920, 1080, 3).astype('float32')
    bboxes, _, probs = predictor_det(image_input, top_k=10, prob_threshold=0.9)
    