import logging
import pathlib
import typing as t

import numpy
import torch
import torch.cuda
import ultralytics
import ultralytics.engine.results


class Detector:
    """
    thin wrapper around the ultralytics YOLO model that runs object detection on single frames
    """

    # default YOLO26 detection weights; the file is downloaded automatically on first use.
    # YOLO26 is natively end-to-end (NMS-free) by default which keeps live inference fast.
    # the large 'l' scale is the accuracy sweet spot that still runs real-time on a 12GB GPU
    DEFAULT_MODEL: t.Final[str] = 'yolo26l.pt'

    # only keep detections whose confidence is above this threshold
    CONFIDENCE_THRESHOLD: t.Final[float] = 0.25

    def __init__(
        self,
        model_dir: pathlib.Path,
        model: str = DEFAULT_MODEL,
        device: t.Optional[str] = None,
    ):
        """
        load the YOLO model and place it on the requested device.

        the weights file is downloaded into `model_dir` on first use and reused afterwards, so the
        model never ends up in the current working directory.

        :param model_dir: directory where the weights file is downloaded to and cached
        :param model: file name of the YOLO weights (downloaded automatically if missing)
        :param device: torch device to run on ('cuda', 'cpu', ...); auto-detected when omitted
        """

        # module level logger
        self.logger: logging.Logger = logging.getLogger('detector')

        # pick the best available device unless the caller forces a specific one
        self.device: t.Final[str] = device if device is not None else self.select_device()

        # report the resolved device so it is always obvious whether inference runs on GPU or CPU
        if torch.cuda.is_available():
            self.logger.info(
                'CUDA is available, using GPU device [%s]: %s',
                self.device,
                torch.cuda.get_device_name(0),
            )
        else:
            self.logger.warning('CUDA is not available, falling back to CPU device [%s]', self.device)

        # make sure the target directory exists so ultralytics can download the weights into it
        model_dir.mkdir(parents=True, exist_ok=True)

        # absolute path of the weights file; ultralytics downloads the known asset to this exact
        # path when the file does not exist yet, keeping it out of the current working directory
        model_path: pathlib.Path = model_dir / model

        self.logger.info('loading YOLO model [%s] on device [%s]', model_path, self.device)

        # load the ultralytics YOLO model and move it to the target device
        self.model: t.Final[ultralytics.YOLO] = ultralytics.YOLO(model_path)
        self.model.to(self.device)

        # confirm where the model weights physically reside after moving them to the device
        weights_device: torch.device = next(self.model.model.parameters()).device
        self.logger.info('YOLO model weights loaded on device [%s]', weights_device)

    @staticmethod
    def select_device() -> str:
        """
        choose the best torch device available on this machine.

        :return: 'cuda' when a GPU is present, otherwise 'cpu'
        """
        return 'cuda' if torch.cuda.is_available() else 'cpu'

    def detect(self, frame: numpy.ndarray) -> ultralytics.engine.results.Results:
        """
        run object detection on a single BGR frame.

        :param frame: a BGR image as a numpy array with shape (height, width, 3)
        :return: the ultralytics detection result for the frame
        """

        # run inference on the single frame; verbose is disabled to keep the console clean
        # noinspection bad-assignment
        predictions: list[ultralytics.engine.results.Results] = self.model.predict(
            source=frame,
            conf=self.CONFIDENCE_THRESHOLD,
            device=self.device,
            verbose=False,
        )
        assert len(predictions) == 1

        return predictions[0]

    def annotate(self, frame: numpy.ndarray) -> numpy.ndarray:
        """
        detect objects in a frame and draw the bounding boxes with class labels onto a copy of it.

        :param frame: a BGR image as a numpy array with shape (height, width, 3)
        :return: a new BGR image with the detected boxes and class names rendered on top
        """
        result: ultralytics.engine.results.Results = self.detect(frame)

        # ultralytics renders the boxes and class labels for us and returns a BGR numpy image
        annotated: numpy.ndarray = result.plot()
        assert annotated.shape == frame.shape

        return annotated
