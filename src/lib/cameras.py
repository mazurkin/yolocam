import dataclasses
import logging
import pathlib
import typing as t

import cv2
import cv2.utils.logging
import numpy


# module level logger
logger: logging.Logger = logging.getLogger('cameras')


@dataclasses.dataclass(frozen=True)
class CameraInfo:
    """description of a single web camera detected in the system"""

    # index of the camera as used by `cv2.VideoCapture`
    index: int

    # human readable device name reported by the driver (empty if unknown)
    name: str

    # default width of the captured frame in pixels
    width: int

    # default height of the captured frame in pixels
    height: int

    # frames per second reported by the camera driver for the default mode
    fps: float

    # maximum width in pixels the camera negotiates when a very high resolution is requested
    max_width: int

    # maximum height in pixels the camera negotiates when a very high resolution is requested
    max_height: int


class Cameras:
    """
    helper for discovering and capturing frames from the system web cameras
    """

    # the highest camera index probed when scanning for available web cameras
    CAMERA_SCAN_LIMIT: t.Final[int] = 10

    # capture backend used to open web cameras; V4L2 is the native Linux backend and avoids the
    # noisy FFMPEG fallback that raises errors when probing non-existing camera indices
    CAMERA_BACKEND: t.Final[int] = cv2.CAP_V4L2

    # intentionally oversized resolution requested to discover the maximum mode; the driver clamps
    # this request down to the nearest supported resolution which we then read back
    PROBE_MAX_WIDTH: t.Final[int] = 8192

    PROBE_MAX_HEIGHT: t.Final[int] = 8192

    # default capture resolution requested from the camera; the driver clamps it to the nearest
    # supported mode, so the actually negotiated size may differ slightly
    DEFAULT_CAPTURE_WIDTH: t.Final[int] = 640

    DEFAULT_CAPTURE_HEIGHT: t.Final[int] = 480

    # OpenCV flip code for a horizontal (left-right) mirror
    FLIP_HORIZONTAL: t.Final[int] = 1

    # sysfs directory holding the human readable name of each video4linux device on Linux
    SYSFS_VIDEO_PATH: t.Final[pathlib.Path] = pathlib.Path('/sys/class/video4linux')

    @classmethod
    def report(cls) -> None:
        """
        scan the system for available web cameras and print the details of every camera found.

        :return: nothing, the discovered cameras are written to the log
        """
        found_cameras: list[CameraInfo] = cls.detect()

        if not found_cameras:
            logger.info('no web cameras were detected')
            return

        logger.info('detected %d web camera(s):', len(found_cameras))
        for camera in found_cameras:
            logger.info(
                'camera #%d [%s] : default %dx%d @ %.1f fps, max %dx%d',
                camera.index,
                camera.name or 'unknown',
                camera.width,
                camera.height,
                camera.fps,
                camera.max_width,
                camera.max_height,
            )

    @classmethod
    def detect(cls) -> list[CameraInfo]:
        """
        probe camera indices from 0 up to `CAMERA_SCAN_LIMIT` and collect the ones that can be opened.

        :return: the list of detected cameras described by `CameraInfo`
        """
        found_cameras: list[CameraInfo] = []

        # probing non-existing camera indices makes the native OpenCV logger print noisy warnings
        # like "can't open camera by index"; silence the native logger for the duration of the scan
        previous_log_level: int = cv2.utils.logging.getLogLevel()
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)

        try:
            for index in range(cls.CAMERA_SCAN_LIMIT):
                # try to open the camera by its index using the native Linux backend
                capture: cv2.VideoCapture = cv2.VideoCapture(index, cls.CAMERA_BACKEND)
                try:
                    if not capture.isOpened():
                        # nothing is connected at this index, skip it
                        continue

                    # read a single frame to make sure the camera really delivers data
                    success, _ = capture.read()
                    if not success:
                        continue

                    # remember the default capture mode before probing the maximum resolution
                    default_width: int = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                    default_height: int = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    default_fps: float = float(capture.get(cv2.CAP_PROP_FPS))

                    # discover the maximum resolution the driver is willing to negotiate
                    max_width, max_height = cls.probe_max_resolution(capture)

                    camera: CameraInfo = CameraInfo(
                        index=index,
                        name=cls.read_device_name(index),
                        width=default_width,
                        height=default_height,
                        fps=default_fps,
                        max_width=max_width,
                        max_height=max_height,
                    )

                    found_cameras.append(camera)
                finally:
                    # always release the device handle
                    capture.release()
        finally:
            # restore the native OpenCV log level
            cv2.utils.logging.setLogLevel(previous_log_level)

        return found_cameras

    @classmethod
    def probe_max_resolution(cls, capture: cv2.VideoCapture) -> tuple[int, int]:
        """
        request an oversized resolution so the driver clamps it to the nearest supported mode.

        :param capture: an already opened `cv2.VideoCapture` handle
        :return: a tuple of (max_width, max_height) the camera actually accepted
        """
        # ask for an intentionally huge frame; the driver clamps it to the largest supported mode
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, cls.PROBE_MAX_WIDTH)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, cls.PROBE_MAX_HEIGHT)

        # read back the resolution the driver settled on
        max_width: int = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        max_height: int = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        return max_width, max_height

    @classmethod
    def read_device_name(cls, index: int) -> str:
        """
        read the human readable device name of a video4linux camera from sysfs on Linux.

        :param index: index of the camera which maps to the `/dev/videoN` device node
        :return: the device name reported by the driver, or an empty string if it is unavailable
        """
        # the name is exposed by the kernel at /sys/class/video4linux/videoN/name
        name_path: pathlib.Path = cls.SYSFS_VIDEO_PATH / f'video{index}' / 'name'
        try:
            return name_path.read_text(encoding='utf-8').strip()
        except OSError:
            # sysfs is not available (non-linux) or the node disappeared, the name is simply unknown
            return ''

    @classmethod
    def capture(
        cls,
        camera: int,
        width: int = DEFAULT_CAPTURE_WIDTH,
        height: int = DEFAULT_CAPTURE_HEIGHT,
        mirror: bool = True,
    ) -> t.Iterator[numpy.ndarray]:
        """
        open the given web camera and continuously yield captured frames until interrupted.

        the generator keeps the camera open and yields frames one by one; the caller can pass every
        frame further to any processing method (for example a YOLO classifier). the loop runs until
        the user presses CTRL+C or the generator is closed.

        when mirroring is enabled the raw frame is flipped horizontally before it is yielded, so any
        downstream detector draws boxes and labels on the already mirrored image; this gives the
        natural "selfie" view while keeping the label text readable (not mirrored).

        :param camera: index of the web camera as used by `cv2.VideoCapture`
        :param width: requested capture width in pixels; the driver clamps it to the nearest mode
        :param height: requested capture height in pixels; the driver clamps it to the nearest mode
        :param mirror: when True each frame is mirrored horizontally before being yielded
        :return: an iterator producing BGR frames as numpy arrays with shape (height, width, 3)
        """
        # open the requested camera using the native Linux backend
        capture: cv2.VideoCapture = cv2.VideoCapture(camera, cls.CAMERA_BACKEND)
        if not capture.isOpened():
            raise RuntimeError(f'unable to open web camera with index {camera}')

        # explicitly request the desired capture resolution; the driver picks the nearest supported mode
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # read back the resolution the driver actually negotiated
        actual_width: int = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height: int = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(
            'capturing from web camera #%d at %dx%d (requested %dx%d), press CTRL+C to stop',
            camera,
            actual_width,
            actual_height,
            width,
            height,
        )
        try:
            while True:
                # grab the next frame from the camera
                success, frame = capture.read()
                if not success:
                    # the camera failed to deliver a frame, stop the loop
                    logger.warning('failed to read a frame from web camera #%d', camera)
                    break

                # mirror the raw frame before yielding so detection and label drawing happen on the
                # mirrored image, keeping the boxes aligned and the text readable
                if mirror:
                    frame = cv2.flip(frame, cls.FLIP_HORIZONTAL)

                yield frame
        except KeyboardInterrupt:
            # the user requested to stop the capturing loop
            logger.info('capturing from web camera #%d is interrupted by the user', camera)
        finally:
            # always release the device handle
            capture.release()
