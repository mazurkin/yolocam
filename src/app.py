import argh
import pathlib
import logging
import logging.config
import typing as t

import numpy as np
import yaml
import sys
import json
import warnings

import torch
import torch.cuda

import lib.cameras
import lib.depth
import lib.detector
import lib.segmentation
import lib.viewer


# noinspection DuplicatedCode,PyMethodMayBeStatic
class Application:
    """
    application runner
    """

    PATH_APPLICATION: t.Final[pathlib.Path] = pathlib.Path(__file__)

    PATH_DIR_SOURCES: t.Final[pathlib.Path] = PATH_APPLICATION.parent.resolve()

    PATH_DIR_PACKAGE: t.Final[pathlib.Path] = PATH_DIR_SOURCES.parent.resolve()

    PATH_DIR_WORK: t.Final[pathlib.Path] = PATH_DIR_PACKAGE / 'work'

    def __init__(self):
        # initialize logging
        logging_config_path: pathlib.Path = self.PATH_DIR_SOURCES / 'app.yaml'
        logging_config = self.load_yaml(logging_config_path, yaml.SafeLoader)
        logging.config.dictConfig(logging_config)

        # local logger
        self.logger = logging.getLogger('application')
        self.logger.info('command line :\n%s', json.dumps(sys.argv[1:], default=str, indent=2, sort_keys=False))

        # avoid the warning:
        # TensorFloat32 tensor cores for float32 matrix multiplication available but not enabled.
        # Consider setting `torch.set_float32_matmul_precision('high')` for better performance.
        torch.set_float32_matmul_precision('high')

        # default device is CPU
        torch.set_default_device('cpu')

        # default dtype
        torch.set_default_dtype(torch.float32)

        # torch multi-threading setup
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)

        # determinism
        torch.use_deterministic_algorithms(mode=False)

        # avoid the CheckPoint warning
        warnings.filterwarnings(
            action='ignore',
            message=r'Checkpoint directory .* exists and is not empty\.',
            category=UserWarning,
        )
        warnings.filterwarnings(
            action='ignore',
            message=r'.* is set, but there is no last checkpoint available\. No checkpoint will be loaded\. .*',
            category=UserWarning,
        )

        # avoid the LitLogger warning
        warnings.filterwarnings(
            action='ignore',
            message=r'LitLogger does not support `log_graph`',
            category=UserWarning,
        )

    @argh.arg('--camera', type=int, help='index of web camera')
    def detect(self, camera: int = 0) -> None:
        """
        capture frames from the given web camera, run YOLO object detection on each frame and
        display the annotated frames in a window until the user quits.

        :param camera: index of the web camera as used by `cv2.VideoCapture`
        :return: nothing, the annotated frames are shown in a window
        """
        # load the YOLO detector once, it picks the GPU automatically when available and downloads
        # the weights into the work directory instead of the current working directory
        detector: lib.detector.Detector = lib.detector.Detector(model_dir=self.PATH_DIR_WORK)

        # build the pipeline: capture frames -> annotate with detections -> show in the window
        capturer: lib.cameras.Camera = lib.cameras.Camera(camera)

        frames: t.Iterator[np.ndarray] = capturer.capture()
        annotated: t.Iterator[np.ndarray] = (detector.annotate(frame) for frame in frames)

        lib.viewer.Viewer().show(annotated)

    @argh.arg('--camera', type=int, help='index of web camera')
    def depth(self, camera: int = 0) -> None:
        """
        capture frames from the given web camera, run YOLO monocular depth estimation on each frame
        and display the colorized depth maps in a window until the user quits.

        :param camera: index of the web camera as used by `cv2.VideoCapture`
        :return: nothing, the colorized depth maps are shown in a window
        """
        # load the YOLO depth estimator once, it picks the GPU automatically when available and
        # downloads the weights into the work directory instead of the current working directory
        estimator: lib.depth.DepthEstimator = lib.depth.DepthEstimator(model_dir=self.PATH_DIR_WORK)

        # build the pipeline: capture frames -> estimate depth -> show in the window
        capturer: lib.cameras.Camera = lib.cameras.Camera(camera)

        frames: t.Iterator[np.ndarray] = capturer.capture()
        annotated: t.Iterator[np.ndarray] = (estimator.annotate(frame) for frame in frames)

        lib.viewer.Viewer().show(annotated)

    @argh.arg('--camera', type=int, help='index of web camera')
    def segmentation(self, camera: int = 0) -> None:
        """
        capture frames from the given web camera, run YOLO instance segmentation on each frame and
        display the annotated frames with masks in a window until the user quits.

        :param camera: index of the web camera as used by `cv2.VideoCapture`
        :return: nothing, the annotated frames are shown in a window
        """
        # load the YOLO segmenter once, it picks the GPU automatically when available and downloads
        # the weights into the work directory instead of the current working directory
        segmenter: lib.segmentation.InstanceSegmentation = lib.segmentation.InstanceSegmentation(
            model_dir=self.PATH_DIR_WORK,
        )

        # build the pipeline: capture frames -> segment with masks -> show in the window
        capturer: lib.cameras.Camera = lib.cameras.Camera(camera)

        frames: t.Iterator[np.ndarray] = capturer.capture()
        annotated: t.Iterator[np.ndarray] = (segmenter.annotate(frame) for frame in frames)

        lib.viewer.Viewer().show(annotated)

    def cameras(self) -> None:
        """
        scan the system for available web cameras and print the details of every camera found.

        :return: nothing, the discovered cameras are written to the log
        """
        found_cameras: list[lib.cameras.CameraInfo] = lib.cameras.Cameras.detect()

        if found_cameras:
            self.logger.info('detected %d web camera(s):', len(found_cameras))
            for camera in found_cameras:
                self.logger.info('%s', camera)
        else:
            self.logger.info('no web cameras were detected')

    @staticmethod
    def load_yaml(path: pathlib.Path, yaml_loader_class: t.Type) -> t.Dict:
        with path.open('rt') as file:
            yaml_text = file.read()

        # noinspection PyTypeChecker
        yaml_dict = yaml.load(yaml_text, yaml_loader_class)

        return yaml_dict


if __name__ == '__main__':
    application = Application()

    parser = argh.ArghParser()
    argh.add_commands(parser, [application.detect])
    argh.add_commands(parser, [application.depth])
    argh.add_commands(parser, [application.segmentation])
    argh.add_commands(parser, [application.cameras])

    try:
        argh.dispatch(parser)
    finally:
        logging.info('the work is finished')
        logging.shutdown()
