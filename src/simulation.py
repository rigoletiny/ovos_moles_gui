from PyQt5.QtWidgets import QDialog
from PyQt5.uic import loadUi
from PyQt5.QtWidgets import QFileDialog
from src import Settings, DefectsDetector
import cv2, imutils
from PyQt5.QtGui import QImage
from PyQt5 import QtGui
from src import init_logger


class simulationScreen(QDialog):
    def __init__(self):
        super(simulationScreen, self).__init__()
        loadUi("Simulation.ui", self)
        self.param = Settings("general_settings.json")
        self.image1 = None
        self.image2 = None

        self.read_settings_from_json()

        self.openButton.clicked.connect(self.loadImage)
        self.runButton.clicked.connect(self.runSim)
        self.saveButton.clicked.connect(self.saveParam)

    def update_param(self):
        # TODO find a better way to read this
        self.param.data.ROIs.Cam1[0][0]  = self.cam1_roi_x_1.value()
        self.param.data.ROIs.Cam1[1][0]  = self.cam1_roi_x_2.value()
        self.param.data.ROIs.Cam1[2][0]  = self.cam1_roi_x_3.value()
        self.param.data.ROIs.Cam1[3][0]  = self.cam1_roi_x_4.value()
        self.param.data.ROIs.Cam1[4][0]  = self.cam1_roi_x_5.value()
        self.param.data.ROIs.Cam1[5][0]  = self.cam1_roi_x_6.value()
        self.param.data.ROIs.Cam1[6][0]  = self.cam1_roi_x_7.value()
        self.param.data.ROIs.Cam1[7][0]  = self.cam1_roi_x_8.value()
        self.param.data.ROIs.Cam1[8][0]  = self.cam1_roi_x_9.value()
        self.param.data.ROIs.Cam1[9][0]  = self.cam1_roi_x_10.value()
        self.param.data.ROIs.Cam1[10][0] = self.cam1_roi_x_11.value()
        self.param.data.ROIs.Cam1[0][1]  = self.cam1_roi_y_1.value()
        self.param.data.ROIs.Cam1[1][1]  = self.cam1_roi_y_2.value()
        self.param.data.ROIs.Cam1[2][1]  = self.cam1_roi_y_3.value()
        self.param.data.ROIs.Cam1[3][1]  = self.cam1_roi_y_4.value()
        self.param.data.ROIs.Cam1[4][1]  = self.cam1_roi_y_5.value()
        self.param.data.ROIs.Cam1[5][1]  = self.cam1_roi_y_6.value()
        self.param.data.ROIs.Cam1[6][1]  = self.cam1_roi_y_7.value()
        self.param.data.ROIs.Cam1[7][1]  = self.cam1_roi_y_8.value()
        self.param.data.ROIs.Cam1[8][1]  = self.cam1_roi_y_9.value()
        self.param.data.ROIs.Cam1[9][1]  = self.cam1_roi_y_10.value()
        self.param.data.ROIs.Cam1[10][1] = self.cam1_roi_y_11.value()
        self.param.data.ROIs.Cam2[0][0]  = self.cam2_roi_x_1.value()
        self.param.data.ROIs.Cam2[1][0]  = self.cam2_roi_x_2.value()
        self.param.data.ROIs.Cam2[2][0]  = self.cam2_roi_x_3.value()
        self.param.data.ROIs.Cam2[3][0]  = self.cam2_roi_x_4.value()
        self.param.data.ROIs.Cam2[4][0]  = self.cam2_roi_x_5.value()
        self.param.data.ROIs.Cam2[5][0]  = self.cam2_roi_x_6.value()
        self.param.data.ROIs.Cam2[6][0]  = self.cam2_roi_x_7.value()
        self.param.data.ROIs.Cam2[7][0]  = self.cam2_roi_x_8.value()
        self.param.data.ROIs.Cam2[8][0]  = self.cam2_roi_x_9.value()
        self.param.data.ROIs.Cam2[9][0]  = self.cam2_roi_x_10.value()
        self.param.data.ROIs.Cam2[10][0] = self.cam2_roi_x_11.value()

        self.param.data.ROIs.Cam2[0][1]  = self.cam2_roi_y_1.value()
        self.param.data.ROIs.Cam2[1][1]  = self.cam2_roi_y_2.value()
        self.param.data.ROIs.Cam2[2][1]  = self.cam2_roi_y_3.value()
        self.param.data.ROIs.Cam2[3][1]  = self.cam2_roi_y_4.value()
        self.param.data.ROIs.Cam2[4][1]  = self.cam2_roi_y_5.value()
        self.param.data.ROIs.Cam2[5][1]  = self.cam2_roi_y_6.value()
        self.param.data.ROIs.Cam2[6][1]  = self.cam2_roi_y_7.value()
        self.param.data.ROIs.Cam2[7][1]  = self.cam2_roi_y_8.value()
        self.param.data.ROIs.Cam2[8][1]  = self.cam2_roi_y_9.value()
        self.param.data.ROIs.Cam2[9][1]  = self.cam2_roi_y_10.value()
        self.param.data.ROIs.Cam2[10][1] = self.cam2_roi_y_11.value()

        self.param.data.ColorThreshold1.threshold = self.tresh_1.value()
        self.param.data.ColorThreshold1.H_upper = self.tresh_1_H_upper.value()
        self.param.data.ColorThreshold1.S_upper = self.tresh_1_S_upper.value()
        self.param.data.ColorThreshold1.V_upper = self.tresh_1_V_upper.value()
        self.param.data.ColorThreshold1.H_lower = self.tresh_1_H_lower.value()
        self.param.data.ColorThreshold1.S_lower = self.tresh_1_S_lower.value()
        self.param.data.ColorThreshold1.V_lower = self.tresh_1_V_lower.value()
        self.param.data.ColorThreshold1.H_upper = self.tresh_2_H_upper.value()
        self.param.data.ColorThreshold1.S_upper = self.tresh_2_S_upper.value()
        self.param.data.ColorThreshold1.V_upper = self.tresh_2_V_upper.value()
        self.param.data.ColorThreshold2.threshold = self.tresh_2.value()
        self.param.data.ColorThreshold2.H_lower = self.tresh_2_H_lower.value()
        self.param.data.ColorThreshold2.S_lower = self.tresh_2_S_lower.value()
        self.param.data.ColorThreshold2.V_lower = self.tresh_2_V_lower.value()

        self.param.data.Offset_Camera.ref_middle_point_cam1 = self.cam1_offset_middle_point.value()
        self.param.data.Offset_Camera.ref_middle_point_cam2 = self.cam2_offset_middle_point.value()

        self.param.data.Offset_Camera.roi_offset_cam1[0] = self.cam1_offset_1.value()
        self.param.data.Offset_Camera.roi_offset_cam1[1] = self.cam1_offset_2.value()
        self.param.data.Offset_Camera.roi_offset_cam1[2] = self.cam1_offset_3.value()
        self.param.data.Offset_Camera.roi_offset_cam1[3] = self.cam1_offset_4.value()

        self.param.data.Offset_Camera.roi_offset_cam2[0] = self.cam2_offset_1.value()
        self.param.data.Offset_Camera.roi_offset_cam2[1] = self.cam2_offset_2.value()
        self.param.data.Offset_Camera.roi_offset_cam2[2] = self.cam2_offset_3.value()
        self.param.data.Offset_Camera.roi_offset_cam2[3] = self.cam2_offset_4.value()

    def saveParam(self):
        self.param.write_settings()

    def runSim(self):
        self.update_param()
        # Start logger
        logger = init_logger()
        # Init section
        logger.info("Init program")

        # Starting loop for reading conditions on PLC to trigger
        image_processor_1 = DefectsDetector("Cam1", self.param.data.ROIs.Cam1, self.param.data.ColorThreshold1,
                                            self.param.data.ColorThreshold2,
                                            self.param.data.Offset_Camera.ref_middle_point_cam1,
                                            self.param.data.Offset_Camera.roi_offset_cam1, self.param.data.hmi_width,
                                            self.param.data.hmi_height, logger)
        image_processor_2 = DefectsDetector("Cam2", self.param.data.ROIs.Cam2, self.param.data.ColorThreshold1,
                                            self.param.data.ColorThreshold2,
                                            self.param.data.Offset_Camera.ref_middle_point_cam2,
                                            self.param.data.Offset_Camera.roi_offset_cam2, self.param.data.hmi_width,
                                            self.param.data.hmi_height, logger)

        self.update_param()
        if self.image1 is None or self.image2 is None:
            return
        else:
            raw_cam1 = self.image1
            raw_cam2 = self.image2
            # pre-process Image
            #frame_cam1 = image_processor_1.preprocess(raw_cam1, 0, 0)
            #frame_cam2 = image_processor_1.preprocess(raw_cam2, 0, 0)

            defects_cam_1, frame_painted_defect_cam1 = image_processor_1.process(raw_cam1)
            defects_cam_2, frame_painted_defect_cam2 = image_processor_2.process(raw_cam2)

            self.setPhoto(frame_painted_defect_cam1, frame_painted_defect_cam2)

    def read_settings_from_json(self):
        # TODO find a better way to read this
        self.cam1_roi_x_1.setValue(self.param.data.ROIs.Cam1[0][0])
        self.cam1_roi_x_2.setValue(self.param.data.ROIs.Cam1[1][0])
        self.cam1_roi_x_3.setValue(self.param.data.ROIs.Cam1[2][0])
        self.cam1_roi_x_4.setValue(self.param.data.ROIs.Cam1[3][0])
        self.cam1_roi_x_5.setValue(self.param.data.ROIs.Cam1[4][0])
        self.cam1_roi_x_6.setValue(self.param.data.ROIs.Cam1[5][0])
        self.cam1_roi_x_7.setValue(self.param.data.ROIs.Cam1[6][0])
        self.cam1_roi_x_8.setValue(self.param.data.ROIs.Cam1[7][0])
        self.cam1_roi_x_9.setValue(self.param.data.ROIs.Cam1[8][0])
        self.cam1_roi_x_10.setValue(self.param.data.ROIs.Cam1[9][0])
        self.cam1_roi_x_11.setValue(self.param.data.ROIs.Cam1[10][0])

        self.cam1_roi_y_1.setValue(self.param.data.ROIs.Cam1[0][1])
        self.cam1_roi_y_2.setValue(self.param.data.ROIs.Cam1[1][1])
        self.cam1_roi_y_3.setValue(self.param.data.ROIs.Cam1[2][1])
        self.cam1_roi_y_4.setValue(self.param.data.ROIs.Cam1[3][1])
        self.cam1_roi_y_5.setValue(self.param.data.ROIs.Cam1[4][1])
        self.cam1_roi_y_6.setValue(self.param.data.ROIs.Cam1[5][1])
        self.cam1_roi_y_7.setValue(self.param.data.ROIs.Cam1[6][1])
        self.cam1_roi_y_8.setValue(self.param.data.ROIs.Cam1[7][1])
        self.cam1_roi_y_9.setValue(self.param.data.ROIs.Cam1[8][1])
        self.cam1_roi_y_10.setValue(self.param.data.ROIs.Cam1[9][1])
        self.cam1_roi_y_11.setValue(self.param.data.ROIs.Cam1[10][1])

        self.cam2_roi_x_1.setValue(self.param.data.ROIs.Cam2[0][0])
        self.cam2_roi_x_2.setValue(self.param.data.ROIs.Cam2[1][0])
        self.cam2_roi_x_3.setValue(self.param.data.ROIs.Cam2[2][0])
        self.cam2_roi_x_4.setValue(self.param.data.ROIs.Cam2[3][0])
        self.cam2_roi_x_5.setValue(self.param.data.ROIs.Cam2[4][0])
        self.cam2_roi_x_6.setValue(self.param.data.ROIs.Cam2[5][0])
        self.cam2_roi_x_7.setValue(self.param.data.ROIs.Cam2[6][0])
        self.cam2_roi_x_8.setValue(self.param.data.ROIs.Cam2[7][0])
        self.cam2_roi_x_9.setValue(self.param.data.ROIs.Cam2[8][0])
        self.cam2_roi_x_10.setValue(self.param.data.ROIs.Cam2[9][0])
        self.cam2_roi_x_11.setValue(self.param.data.ROIs.Cam2[10][0])

        self.cam2_roi_y_1.setValue(self.param.data.ROIs.Cam2[0][1])
        self.cam2_roi_y_2.setValue(self.param.data.ROIs.Cam2[1][1])
        self.cam2_roi_y_3.setValue(self.param.data.ROIs.Cam2[2][1])
        self.cam2_roi_y_4.setValue(self.param.data.ROIs.Cam2[3][1])
        self.cam2_roi_y_5.setValue(self.param.data.ROIs.Cam2[4][1])
        self.cam2_roi_y_6.setValue(self.param.data.ROIs.Cam2[5][1])
        self.cam2_roi_y_7.setValue(self.param.data.ROIs.Cam2[6][1])
        self.cam2_roi_y_8.setValue(self.param.data.ROIs.Cam2[7][1])
        self.cam2_roi_y_9.setValue(self.param.data.ROIs.Cam2[8][1])
        self.cam2_roi_y_10.setValue(self.param.data.ROIs.Cam2[9][1])
        self.cam2_roi_y_11.setValue(self.param.data.ROIs.Cam2[10][1])

        self.tresh_1.setValue(self.param.data.ColorThreshold1.threshold)

        self.tresh_1_H_upper.setValue(self.param.data.ColorThreshold1.H_upper)
        self.tresh_1_S_upper.setValue(self.param.data.ColorThreshold1.S_upper)
        self.tresh_1_V_upper.setValue(self.param.data.ColorThreshold1.V_upper)

        self.tresh_1_H_lower.setValue(self.param.data.ColorThreshold1.H_lower)
        self.tresh_1_S_lower.setValue(self.param.data.ColorThreshold1.S_lower)
        self.tresh_1_V_lower.setValue(self.param.data.ColorThreshold1.V_lower)

        self.tresh_2.setValue(self.param.data.ColorThreshold2.threshold)

        self.tresh_2_H_upper.setValue(self.param.data.ColorThreshold1.H_upper)
        self.tresh_2_S_upper.setValue(self.param.data.ColorThreshold1.S_upper)
        self.tresh_2_V_upper.setValue(self.param.data.ColorThreshold1.V_upper)

        self.tresh_2_H_lower.setValue(self.param.data.ColorThreshold2.H_lower)
        self.tresh_2_S_lower.setValue(self.param.data.ColorThreshold2.S_lower)
        self.tresh_2_V_lower.setValue(self.param.data.ColorThreshold2.V_lower)

        self.cam1_offset_middle_point.setValue(self.param.data.Offset_Camera.ref_middle_point_cam1)
        self.cam2_offset_middle_point.setValue(self.param.data.Offset_Camera.ref_middle_point_cam2)

        self.cam1_offset_1.setValue(self.param.data.Offset_Camera.roi_offset_cam1[0])
        self.cam1_offset_2.setValue(self.param.data.Offset_Camera.roi_offset_cam1[1])
        self.cam1_offset_3.setValue(self.param.data.Offset_Camera.roi_offset_cam1[2])
        self.cam1_offset_4.setValue(self.param.data.Offset_Camera.roi_offset_cam1[3])

        self.cam2_offset_1.setValue(self.param.data.Offset_Camera.roi_offset_cam2[0])
        self.cam2_offset_2.setValue(self.param.data.Offset_Camera.roi_offset_cam2[1])
        self.cam2_offset_3.setValue(self.param.data.Offset_Camera.roi_offset_cam2[2])
        self.cam2_offset_4.setValue(self.param.data.Offset_Camera.roi_offset_cam2[3])

    def loadImage(self):
        """ This function will load the user selected image
            and set it to label using the setPhoto function
        """
        self.filename = QFileDialog.getOpenFileName(filter="Image (*.*)", directory='./workspace/raw_img')[0]
        if "Cam1" in self.filename:
            file_cam1 = self.filename
            file_cam2 = self.filename.replace("Cam1", "Cam2")
        elif "Cam2" in self.filename:
            file_cam2 = self.filename
            file_cam1 = self.filename.replace("Cam2", "Cam1")
        else:
            None

        self.image1 = cv2.imread(file_cam1)
        self.image2 = cv2.imread(file_cam2)

        self.setPhoto(self.image1, self.image2)

    def setPhoto(self, image1, image2):
        """ This function will take image input and resize it
            only for display purpose and convert it to QImage
            to set at the label.
        """
        # Setting Image 1
        image1 = imutils.resize(image1, width=512, height=384)
        frame1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image1 = QImage(frame1, frame1.shape[1], frame1.shape[0], frame1.strides[0], QImage.Format_RGB888)
        self.picture_cam1.setPixmap(QtGui.QPixmap.fromImage(image1))
        # Setting Image 2
        image2 = imutils.resize(image2, width=512, height=384)
        frame2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        image2 = QImage(frame2, frame2.shape[1], frame2.shape[0], frame2.strides[0], QImage.Format_RGB888)
        self.picture_cam2.setPixmap(QtGui.QPixmap.fromImage(image2))
