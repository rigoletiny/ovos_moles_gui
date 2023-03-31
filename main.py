import sys
import time
import cv2
import imutils
from threading import *
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtGui import QImage
from PyQt5.QtWidgets import QDialog, QStackedWidget
from src.pythonsnap7.snap7.types import *
from PyQt5.uic import loadUi
from src import WebCam, PLCSiemens, DefectsDetector, Settings, BaslerCam, init_logger
import traceback
from src.simulation import simulationScreen
from PyQt5 import QtCore


class uiMainWindow(QDialog):
    def __init__(self):
        super(uiMainWindow, self).__init__()
        loadUi("PrincipalScreen.ui", self)
        self.started = False
        self.manual_trigger = False
        self.cam_1 = None
        self.cam_2 = None
        self.image_processor_1 = None
        self.image_processor_2 = None
        self.plc_handler = None
        self.start_button.clicked.connect(self.start_main_loop)
        self.manual_trigger_btn.clicked.connect(self.set_manual_trigger)
        self.simButton.clicked.connect(self.go_to_simulation)
        self.SaveBox.toggled.connect(self.set_saving_images)


    def closeConnections(self):
        if self.cam_1 is not None:
            self.cam_1.stop()
        if self.cam_2 is not None:
            self.cam_2.stop()
        if self.plc_handler is not None:
            self.plc_handler.stop()

    def set_saving_images(self):
        if self.image_processor_1 is not None and self.image_processor_2 is not None:
            if self.SaveBox.isChecked():
                self.image_processor_1.save = True
                self.image_processor_2.save = True
            else:
                self.image_processor_1.save = False
                self.image_processor_2.save = False

    def go_to_simulation(self):
        simulation_handler = simulationScreen(widget)
        widget.addWidget(simulation_handler)
        widget.setCurrentIndex(widget.currentIndex()+1)

    def set_manual_trigger(self):
        self.manual_trigger = True

    def start_main_loop(self):
        #
        if self.started:
            self.started = False
            self.start_button.setText('Start')
            if self.cam_1 is not None:
                self.cam_1.stop()
            if self.cam_2 is not None:
                self.cam_2.stop()
            if self.plc_handler is not None:
                self.plc_handler.stop()
            return 0
        else:
            self.started = True
            self.start_button.setText('Stop')

        # Start logger
        logger = init_logger()
        # Init section
        logger.info("Init program")
        # Setting up components
        Param = Settings("general_settings.json")
        # Start Camera 1

        self.cam_1 = BaslerCam(Param.data.Camera.Serial1, Param.data.Camera.Width, Param.data.Camera.Height)
        self.cam_1.start()
        # Start Camera 2
        self.cam_2 = BaslerCam(Param.data.Camera.Serial2, Param.data.Camera.Width, Param.data.Camera.Height)
        self.cam_2.start()
        # Start PLC
        self.plc_handler = PLCSiemens(Param.data.Plc.Ip, Param.data.Plc.Rack, Param.data.Plc.Slot)
        self.plc_handler.start()
        # Starting loop for reading conditions on PLC to trigger
        t1 = Thread(target=self.plc_handler.StartGrabbing)
        t1.start()
        # Starting loop for reading conditions on PLC to trigger
        self.image_processor_1 = DefectsDetector("Cam1", Param.data.ROIs.Cam1, Param.data.ColorThreshold1,
                                            Param.data.ColorThreshold2, Param.data.Offset_Camera.ref_middle_point_cam1,
                                            Param.data.Offset_Camera.roi_offset_cam1, Param.data.hmi_width,
                                            Param.data.hmi_height, logger)
        self.image_processor_2 = DefectsDetector("Cam2", Param.data.ROIs.Cam2, Param.data.ColorThreshold1,
                                            Param.data.ColorThreshold2, Param.data.Offset_Camera.ref_middle_point_cam2,
                                                     Param.data.Offset_Camera.roi_offset_cam2, Param.data.hmi_width,
                                                     Param.data.hmi_height, logger)

        while self.started:
            try:
                raw_cam1 = self.cam_1.get_image()
                raw_cam2 = self.cam_2.get_image()

                # pre-process Image
                frame_cam1 = self.image_processor_1.preprocess(raw_cam1, 0, 0)
                frame_cam2 = self.image_processor_1.preprocess(raw_cam2, 0, 0)

                plc_ready = self.plc_handler.auto_mode and self.plc_handler.artificial_vision and self.plc_handler.trigger
                cam_ready = self.cam_1.connected and self.cam_2.connected
                manual_trigger = self.manual_trigger
                if manual_trigger:
                    self.manual_trigger = False

                if (plc_ready or manual_trigger) and cam_ready:
                    start_time = time.time()
                    defects_cam_1, frame_painted_defect_cam1 = self.image_processor_1.process(frame_cam1)
                    defects_cam_2, frame_painted_defect_cam2 = self.image_processor_2.process(frame_cam2)

                    # Joining both outputs into one
                    defects = self.image_processor_1.join_defects(defects_cam_1, defects_cam_2)

                    # Writing output to PLC
                    self.plc_handler.write_defects(defects)

                    # Writing end of image_processing
                    self.plc_handler.write_flag(True)

                    # Displaying processed images
                    frame_painted_defect_cam1_small = self.image_processor_1.resize(frame_painted_defect_cam1)
                    frame_painted_defect_cam2_small = self.image_processor_2.resize(frame_painted_defect_cam2)
                    self.setImages(frame_painted_defect_cam1_small, frame_painted_defect_cam2_small)
                    key = cv2.waitKey(1)
                    self.timelabel.setText("Time: %s" %(time.time() - start_time))
                    time.sleep(3)
                else:
                    frame_cam1 = self.image_processor_1.resize(frame_cam1)
                    frame_cam2 = self.image_processor_2.resize(frame_cam2)
                    self.setImages(frame_cam1, frame_cam2)
                    key = cv2.waitKey(1)

            except Exception as x:
                logger.error('Error getting data:' + str(x))
                traceback_info = traceback.format_exc()
                logger.error('Traceback:' + traceback_info)
                defects = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                self.plc_handler.write_defects(defects)

    def setImages(self, image1, image2):
        """ This function will take image input and resize it
            only for display purpose and convert it to QImage
            to set at the label.
        """

        frame_1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        frame_2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

        image_1 = QImage(frame_1, frame_1.shape[1], frame_1.shape[0], frame_1.strides[0], QImage.Format_RGB888)
        image_2 = QImage(frame_2, frame_2.shape[1], frame_2.shape[0], frame_2.strides[0], QImage.Format_RGB888)

        self.img_box1.setPixmap(QtGui.QPixmap.fromImage(image_1))
        self.img_box2.setPixmap(QtGui.QPixmap.fromImage(image_2))



if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = uiMainWindow()
    widget = QStackedWidget()
    widget.addWidget(ui)
    widget.setFixedHeight(723)
    widget.setFixedWidth(1024)
    widget.show()
    ui.start_main_loop()
    app.exec_()
    ui.closeConnections()



