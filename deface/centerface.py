import datetime
import os

import numpy as np
import cv2
from onnxruntime import InferenceSession, SessionOptions, get_all_providers


# Find file relative to the location of this code files
default_onnx_path = f'{os.path.dirname(__file__)}/centerface.onnx'


def ensure_rgb(img: np.ndarray) -> np.ndarray:
    """Convert input image to RGB if it is in RGBA or L format"""
    if img.ndim == 2:  # 1-channel grayscale -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:  # 4-channel RGBA -> RGB
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
    return img


class CenterFace:
    def __init__(self, onnx_path=None, in_shape=None, backend='auto'):
        self.in_shape = in_shape
        self.onnx_input_name = 'input.1'
        self.onnx_output_names = ['537', '538', '539', '540']

        if onnx_path is None:
            onnx_path = default_onnx_path

        if backend == 'auto':
            try:
                providers = get_all_providers()
                if 'openvino' in providers:
                    options = SessionOptions()
                    options.intra_op_num_threads = 1
                    options.execution_mode = SessionOptions.ExecutionMode.ORT_SEQUENTIAL
                    options.graph_optimization_level = SessionOptions.GraphOptimizationLevel.ORT_ENABLE_ALL
                    options.providers = ['OpenVINOExecutionProvider']
                    self.sess = InferenceSession(onnx_path, options)
                else:
                    self.sess = InferenceSession(onnx_path)
            except:
                print('Failed to import onnxruntime or OpenVINO. Falling back to slower OpenCV backend.')
                backend = 'opencv'
        self.backend = backend

        if self.backend == 'opencv':
            self.net = cv2.dnn.readNetFromONNX(onnx_path)

    def __call__(self, img, threshold=0.5):
        img = ensure_rgb(img)
        self.orig_shape = img.shape[:2]
        if self.in_shape is None:
            self.in_shape = self.orig_shape[::-1]
        if not hasattr(self, 'h_new'):  # First call, need to compute sizes
            self.w_new, self.h_new, self.scale_w, self.scale_h = self.transform(self.in_shape)

        blob = cv2.dnn.blobFromImage(
            img, scalefactor=1.0, size=(self.w_new, self.h_new),
            mean=(0, 0, 0), swapRB=False, crop=False
        )

        if self.backend == 'opencv':
            self.net.setInput(blob)
            heatmap, scale, offset, lms = self.net.forward(self.onnx_output_names)
        elif self.backend == 'onnxrt':
            input_name = self.sess.get_inputs()[0].name
            outputs = self.sess.run(None, {input_name: blob})
            heatmap, scale, offset, lms = outputs
        else:
            raise RuntimeError(f'Unknown backend {self.backend}')

        dets, lms = self.decode(heatmap, scale, offset, lms, (self.h_new, self.w_new), threshold=threshold)
        if len(dets) > 0:
            dets[:, 0:4:2], dets[:, 1:4:2] = dets[:, 0:4:2] / self.scale_w, dets[:, 1:4:2] / self.scale_h
            lms[:, :, 0], lms[:, :, 1] = lms[:, :, 0] / self.scale_w, lms[:, :, 1] / self.scale_h

        return dets, lms

    @staticmethod
    def decode(heatmap, scale, offset, lms, target_shape, threshold=0.5):
        h, w = target_shape
        mask = np.logical_and(heatmap > threshold, heatmap == np.max(heatmap, axis=(1, 2), keepdims=True))
        indices = np.where(mask)
        dets = []
        for y, x, z in zip(*indices):
            dx, dy = offset[y, x, z]
            cx = (x + dx + 0.5) * w / heatmap.shape[2]
            cy = (y + dy + 0.5) * h / heatmap.shape[1]
            s = scale[y, x, z]
            w_box = w * s
            h_box = h * s
            x1 = cx - w_box / 2
            y1 = cy - h_box / 2
            x2 = x1 + w_box
            y2 = y1 + h_box
            dets.append([x1, y1, x2, y2])
        dets = np.array(dets)
        lms = lms[indices[0], indices[1]]
        return dets, lms

    def transform(self, in_shape):
        in_h, in_w = in_shape
        stride = 32
        self.h_new = (in_h | stride - 1) + 1
        self.w_new = (in_w | stride - 1) + 1
        scale_h = self.h_new / in_h
        scale_w = self.w_new / in_w
        return self.w_new, self.h_new, scale_w, scale_h


def main():
    # Example usage
    centerface = CenterFace()
    image_path = 'path_to_image.jpg'
    image = cv2.imread(image_path)
    dets, lms = centerface(image)
    print('Face detections:', dets)
    print('Landmarks:', lms)


if __name__ == '__main__':
    main()
