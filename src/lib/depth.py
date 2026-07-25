import logging
import pathlib
import typing as t

import numpy
import torch
import torch.cuda
import ultralytics
import ultralytics.engine.results


# module level logger
logger: logging.Logger = logging.getLogger('depth')


class DepthEstimator:
    """
    thin wrapper around the ultralytics YOLO26 depth model that estimates a per-pixel depth map
    for single frames
    """

    # default YOLO26 monocular depth weights; the file is downloaded automatically on first use.
    # the large 'l' scale is the accuracy sweet spot that still runs real-time on a 12GB GPU
    DEFAULT_MODEL: t.Final[str] = 'yolo26l-depth.pt'

    def __init__(
        self,
        model_dir: pathlib.Path,
        model: str = DEFAULT_MODEL,
        device: t.Optional[str] = None,
    ):
        """
        load the YOLO26 depth model and place it on the requested device.

        the weights file is downloaded into `model_dir` on first use and reused afterwards, so the
        model never ends up in the current working directory.

        :param model_dir: directory where the weights file is downloaded to and cached
        :param model: file name of the YOLO depth weights (downloaded automatically if missing)
        :param device: torch device to run on ('cuda', 'cpu', ...); auto-detected when omitted
        """
        # pick the best available device unless the caller forces a specific one
        self.device: t.Final[str] = device if device is not None else self.select_device()

        # report the resolved device so it is always obvious whether inference runs on GPU or CPU
        if torch.cuda.is_available():
            logger.info(
                'CUDA is available, using GPU device [%s]: %s',
                self.device,
                torch.cuda.get_device_name(0),
            )
        else:
            logger.warning('CUDA is not available, falling back to CPU device [%s]', self.device)

        # make sure the target directory exists so ultralytics can download the weights into it
        model_dir.mkdir(parents=True, exist_ok=True)

        # absolute path of the weights file; ultralytics downloads the known asset to this exact
        # path when the file does not exist yet, keeping it out of the current working directory
        model_path: pathlib.Path = model_dir / model

        logger.info('loading YOLO depth model [%s] on device [%s]', model_path, self.device)

        # load the ultralytics YOLO model and move it to the target device
        self.model: t.Final[ultralytics.YOLO] = ultralytics.YOLO(model_path)
        self.model.to(self.device)

        # confirm where the model weights physically reside after moving them to the device
        weights_device: torch.device = next(self.model.model.parameters()).device
        logger.info('YOLO depth model weights loaded on device [%s]', weights_device)

    @staticmethod
    def select_device() -> str:
        """
        choose the best torch device available on this machine.

        :return: 'cuda' when a GPU is present, otherwise 'cpu'
        """
        return 'cuda' if torch.cuda.is_available() else 'cpu'

    def estimate(self, frame: numpy.ndarray) -> ultralytics.engine.results.Results:
        """
        run monocular depth estimation on a single BGR frame.

        :param frame: a BGR image as a numpy array with shape (height, width, 3)
        :return: the ultralytics depth result for the frame
        """
        # run inference on the single frame; verbose is disabled to keep the console clean
        predictions: list[ultralytics.engine.results.Results] = self.model.predict(
            source=frame,
            device=self.device,
            verbose=False,
        )
        assert len(predictions) == 1

        return predictions[0]

    def annotate(self, frame: numpy.ndarray) -> numpy.ndarray:
        """
        estimate the depth of a frame and render the colorized depth map for display.

        :param frame: a BGR image as a numpy array with shape (height, width, 3)
        :return: a new BGR image with the colorized depth map
        """
        result: ultralytics.engine.results.Results = self.estimate(frame)

        # ultralytics renders the depth map into a colorized BGR image for us
        annotated: numpy.ndarray = result.plot()
        assert annotated.shape == frame.shape

        return annotated
