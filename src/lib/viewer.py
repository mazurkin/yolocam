import logging
import typing as t

import cv2
import numpy


# module level logger
logger: logging.Logger = logging.getLogger('viewer')


class Viewer:
    """
    simple OpenCV window that displays a stream of frames until the user quits
    """

    # title of the OpenCV window
    WINDOW_TITLE: t.Final[str] = 'yolocam'

    # initial size of the window in pixels; the frames are scaled to fit this window
    WINDOW_WIDTH: t.Final[int] = 1280

    WINDOW_HEIGHT: t.Final[int] = 960

    # how long `cv2.waitKey` waits for a key press between frames, in milliseconds
    WAIT_KEY_DELAY_MS: t.Final[int] = 1

    # key codes that stop the viewer; ESC and the letter 'q'
    KEY_ESCAPE: t.Final[int] = 27
    KEY_QUIT: t.Final[int] = ord('q')

    def __init__(self):
        """
        create the OpenCV window in normal (resizable) mode and set its initial size.
        """
        cv2.namedWindow(self.WINDOW_TITLE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW_TITLE, self.WINDOW_WIDTH, self.WINDOW_HEIGHT)

    def show(self, frames: t.Iterable[numpy.ndarray]) -> None:
        """
        display every frame from the given iterable in the window until the stream ends or the
        user presses 'q' / ESC / CTRL+C.

        :param frames: an iterable producing BGR frames as numpy arrays
        :return: nothing, the frames are drawn to the window
        """
        logger.info('showing camera window, press [q] or [ESC] or close the window to quit')

        try:
            for frame in frames:
                # draw the frame in the window
                cv2.imshow(self.WINDOW_TITLE, frame)

                # pump the GUI event loop and check whether the user asked to quit
                key: int = cv2.waitKey(self.WAIT_KEY_DELAY_MS) & 0xFF
                if key in (self.KEY_QUIT, self.KEY_ESCAPE):
                    logger.info('window closed by the user via keyboard')
                    break

                # OpenCV has no close callback, so detect the window's X button by checking whether
                # the window is still visible; the visible property drops below 1 once it is closed
                if not self.is_window_open():
                    logger.info('window closed by the user via the close button')
                    break
        except KeyboardInterrupt:
            # the user pressed CTRL+C in the terminal
            logger.info('window interrupted by the user')
        finally:
            # always destroy the window
            self.close()

    def is_window_open(self) -> bool:
        """
        check whether the OpenCV window is still open, i.e. the user has not closed it.

        :return: True while the window is visible, False once the user closed it
        """
        try:
            # WND_PROP_VISIBLE is >= 1 while the window exists and drops to 0 after it is closed;
            # a closed/destroyed window may also report a negative value
            return cv2.getWindowProperty(self.WINDOW_TITLE, cv2.WND_PROP_VISIBLE) >= 1
        except cv2.error:
            # with the Qt backend querying an already destroyed window raises instead of returning
            # a value ("NULL guiReceiver"), which simply means the window is no longer open
            return False

    def close(self) -> None:
        """
        destroy the OpenCV window and release its GUI resources.

        :return: nothing
        """
        
        # the window may already be gone (the user clicked the close button), in which case
        # cv2.destroyWindow raises; only destroy it while it is still open
        if not self.is_window_open():
            return

        cv2.destroyWindow(self.WINDOW_TITLE)
        # process pending GUI events so the window actually disappears
        cv2.waitKey(self.WAIT_KEY_DELAY_MS)
