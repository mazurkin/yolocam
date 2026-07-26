import dataclasses
import logging
import pathlib
import typing as t

import cv2
import cv2.utils.logging
import numpy


@dataclasses.dataclass(frozen=True)
class CameraInfo:
    """description of a single web camera detected in the system"""

    # index of the camera as used by `cv2.VideoCapture`
    index: int

    # human-readable device name reported by the driver (empty if unknown)
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

    def __str__(self) -> str:
        return '#{index} [{name}] : default {width}x{height}@{fps:.1f} fps, max {max_width}x{max_height}'.format(
            index=self.index,
            name=self.name or 'unknown',
            width=self.width,
            height=self.height,
            fps=self.fps,
            max_width=self.max_width,
            max_height=self.max_height,
        )


class Cameras:
    """
    helper for discovering and capturing frames from the system web cameras
    """

    # capture backend used to open web cameras; V4L2 is the native Linux backend and avoids the
    # noisy FFMPEG fallback that raises errors when probing non-existing camera indices
    CAMERA_BACKEND: t.Final[int] = cv2.CAP_V4L2

    # the highest camera index probed when scanning for available web cameras
    CAMERA_SCAN_LIMIT: t.Final[int] = 10

    # intentionally oversized resolution requested to discover the maximum mode; the driver clamps
    # this request down to the nearest supported resolution which we then read back
    PROBE_MAX_WIDTH: t.Final[int] = 8192
    PROBE_MAX_HEIGHT: t.Final[int] = 8192

    # sysfs directory holding the human-readable name of each 'video4linux' device on Linux
    SYSFS_VIDEO_PATH: t.Final[pathlib.Path] = pathlib.Path('/sys/class/video4linux')

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
                camera: CameraInfo | None = cls.get_camera_info(index)

                if camera is not None:
                    found_cameras.append(camera)
        finally:
            # restore the native OpenCV log level
            cv2.utils.logging.setLogLevel(previous_log_level)

        return found_cameras

    @classmethod
    def get_camera_info(cls, index: int) -> CameraInfo | None:
        """
        open a single camera by its index and collect its details, if it can be opened.

        :param index: index of the camera as used by `cv2.VideoCapture`
        :return: the `CameraInfo` for the camera, or None when it cannot be opened or read
        """
        # try to open the camera by its index using the native Linux backend
        capture: cv2.VideoCapture = cv2.VideoCapture(index, cls.CAMERA_BACKEND)

        try:
            if not capture.isOpened():
                # nothing is connected at this index, skip it
                return None

            # read a single frame to make sure the camera really delivers data
            success, _ = capture.read()
            if not success:
                return None

            # remember the default capture mode before probing the maximum resolution
            default_width: int = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
            default_height: int = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
            default_fps: float = float(capture.get(cv2.CAP_PROP_FPS))

            # discover the maximum resolution the driver is willing to negotiate
            max_width, max_height = cls._probe_max_resolution(capture)

            # device name
            name = cls._read_device_name(index)

            camera: CameraInfo = CameraInfo(
                index=index,
                name=name,
                width=default_width,
                height=default_height,
                fps=default_fps,
                max_width=max_width,
                max_height=max_height,
            )

            return camera
        finally:
            # always release the device handle
            capture.release()

    @classmethod
    def _probe_max_resolution(cls, capture: cv2.VideoCapture) -> tuple[int, int]:
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
    def _read_device_name(cls, index: int) -> str:
        """
        read the human-readable device name of a video4linux camera from sysfs on Linux.

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


class Camera:

    # OpenCV flip code for a horizontal (left-right) mirror
    FLIP_HORIZONTAL: t.Final[int] = 1

    # default capture resolution requested from the camera; the driver clamps it to the nearest
    # supported mode, so the actually negotiated size may differ slightly
    DEFAULT_CAPTURE_WIDTH: t.Final[int] = 640
    DEFAULT_CAPTURE_HEIGHT: t.Final[int] = 480

    def __init__(
        self,
        index: int,
        width: int = DEFAULT_CAPTURE_WIDTH,
        height: int = DEFAULT_CAPTURE_HEIGHT,
        mirror: bool = True,
    ) -> None:
        # module level logger
        self.logger: logging.Logger = logging.getLogger(self.__class__.__name__)

        self.index: int = index

        self.width: int = width
        self.height: int = height

        self.mirror: bool = mirror

    def capture(self) -> t.Generator[numpy.ndarray, None, None]:
        """
        open the given web camera and continuously yield captured frames until interrupted.

        the generator keeps the camera open and yields frames one by one; the caller can pass every
        frame further to any processing method (for example a YOLO classifier). the loop runs until
        the user presses CTRL+C or the generator is closed.

        when mirroring is enabled the raw frame is flipped horizontally before it is yielded, so any
        downstream detector draws boxes and labels on the already mirrored image; this gives the
        natural "selfie" view while keeping the label text readable (not mirrored).

        :return: an iterator producing BGR frames as numpy arrays with shape (height, width, 3)
        """
        # open the requested camera using the native Linux backend
        capture: cv2.VideoCapture = cv2.VideoCapture(self.index, Cameras.CAMERA_BACKEND)
        if not capture.isOpened():
            raise RuntimeError(f'unable to open web camera with index {self.index}')

        # explicitly request the desired capture resolution; the driver picks the nearest supported mode
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # read back the resolution the driver actually negotiated
        actual_width: int = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height: int = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.logger.debug(
            'capturing from web camera #%d at %dx%d (requested %dx%d)',
            self.index,
            actual_width,
            actual_height,
            self.width,
            self.height,
        )

        try:
            while True:
                # grab the next frame from the camera
                success, frame = capture.read()

                if not success:
                    # the camera failed to deliver a frame, stop the loop
                    self.logger.warning('failed to read a frame from web camera #%d', self.index)
                    break

                # mirror the raw frame before yielding so detection and label drawing happen on the
                # mirrored image, keeping the boxes aligned and the text readable
                if self.mirror:
                    frame = cv2.flip(frame, self.FLIP_HORIZONTAL)

                yield frame
        finally:
            # always release the device handle
            capture.release()
