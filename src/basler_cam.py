import cv2
import time
import numpy
from pypylon import pylon
from pypylon_opencv_viewer import BaslerOpenCVViewer

class BaslerCam:
    def __init__(self, serial, width, height):
        self.serial = serial
        self.camera = None
        self.converter = None
        self.connected = False
        self.empty_img = cv2.imread(r"workspace\logo_basler.png")
        self.width = width
        self.height = height

    def start(self):
        info = None
        for i in pylon.TlFactory.GetInstance().EnumerateDevices():
            if i.GetSerialNumber() == self.serial:
                info = i
                self.connected = True
                break
        else:
            print('Camera with {} serial number not found'.format(self.serial))
        if info is not None:
            self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(info))
            self.camera.Open()
            self.camera.Width = self.width
            self.camera.Height = self.height
            print("Camera model: %s" % self.camera.DeviceModelName.GetValue())
        if self.connected:
            # Grabing Continusely (video) with minimal delay
            self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
            self.converter = pylon.ImageFormatConverter()

            # Converting to opencv bgr format
            self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    def get_image(self):
        try:
            if self.connected:
                grabResult = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                if grabResult.GrabSucceeded():
                    image = self.converter.Convert(grabResult)
                    img = image.GetArray()
                return img
            else:
                self.start()
            return self.empty_img
        except Exception as x:
            print(x)
            self.connected = False
            return self.empty_img

    def stop(self):
        if self.connected:
            self.camera.StopGrabbing()
