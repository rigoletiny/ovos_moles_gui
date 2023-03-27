import imutils

try:
    import cv2
except ImportError:
    raise ImportError('Can\'t find OpenCV Python module. If you\'ve built it from sources without installation, '
                      'configure environemnt variable PYTHONPATH to "opencv_build_dir/lib" directory (with "python3" '
                      'subdirectory if required)')


class WebCam:
    def __init__(self):
        self.cap = None
        self.connected = False

    def start(self):
        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        self.connected = True

    def get_image(self):
        ret, frame = self.cap.read()
        frame = imutils.resize(frame, height=960, width=1280)
        return frame

    def stop(self):
        self.cap.release()
        cv2.destroyAllWindows()
        self.connected = False

