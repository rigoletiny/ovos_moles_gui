from PyQt5.QtWidgets import QDialog
from PyQt5.uic import loadUi

from src import Settings


class simulationScreen(QDialog):
    def __init__(self):
        super(simulationScreen, self).__init__()
        loadUi("Simulation.ui", self)
        self.param = Settings("general_settings.json")
        self.read_settings_from_json()

        #self.param.data.hmi_height = self.RoisCam1_1.value()
        self.run_simulation_btn.clicked.connect(self.go_to_simulation)

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
        self.param = Settings("general_settings.json")
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


    def go_to_simulation(self):
        print(self.param.data.hmi_height)
        None
