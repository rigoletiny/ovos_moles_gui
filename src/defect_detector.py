import os

import imutils
import numpy as np
from scipy import stats
from datetime import datetime
from pandas import DataFrame
from sklearn.linear_model import LinearRegression
import traceback

try:
    import cv2 as cv
except ImportError:
    raise ImportError('Can\'t find OpenCV Python module. If you\'ve built it from sources without installation, '
                      'configure environemnt variable PYTHONPATH to "opencv_build_dir/lib" directory (with "python3" '
                      'subdirectory if required)')


class DefectsDetector:
    def __init__(self, name, rois, threshold1, threshold2, roi_offset_cam, ref_middle_point_cam, hmi_width, hmi_height,
                 logger):
        self.rois = self.parse_rois(rois.copy())

        self.name = name
        self.dict_defects = {"hole": 1,
                             "crack": 2,
                             "presence": 3,
                             "marker": 4}
        self.show = True
        self.save = False
        self.save_path_img = "workspace/raw_img"
        self.limit_img = 800
        self.debug = False
        self.threshold_presence = 7000
        self.offset = 0
        self.logger = logger
        self.mask_images = []
        self.load_images()
        self.blue_lower = np.array([threshold1.H_lower, threshold1.S_lower, threshold1.V_lower], np.uint8)
        self.blue_upper = np.array([threshold1.H_upper, threshold1.S_upper, threshold1.V_upper], np.uint8)
        self.blue2_lower = np.array([threshold2.H_lower, threshold2.S_lower, threshold2.V_lower], np.uint8)
        self.blue2_upper = np.array([threshold2.H_upper, threshold2.S_upper, threshold2.V_upper], np.uint8)

        self.roi_offset_cam = tuple(ref_middle_point_cam)
        self.ref_middle_point_cam = roi_offset_cam

        self.hmi_width = hmi_width
        self.hmi_height = hmi_height

        self.full_mode = False

    def parse_rois(self, rois):
        rois_tupple_list = []
        for roi in rois:
            rois_tupple_list.append(tuple(roi))
        return rois_tupple_list

    def set_save(self):
        self.save = not self.save

    def load_images(self):
        for counter in range(1, 12):
            # Reading Mask
            name_mask = self.name + "_img_" + str(counter) + ".bmp"
            mask_cookie = cv.imread(os.path.join("workspace", "mask", name_mask))
            self.mask_images.append(mask_cookie)

    def preprocess(self, camera_frame, offset_x, offset_y):
        # Get rectified image
        frame_cam1 = self.barrel_distortion(camera_frame)
        # Reset roi and add roi offset x and y
        self.reset_roi(self.rois, offset_x, offset_y)
        return frame_cam1

    def process(self, frame_cam):
        offset_cam1 = self.get_offset(frame_cam, self.roi_offset_cam, self.ref_middle_point_cam)
        # Processing image cam 1
        defects_holes_cam_1 = self.hole_detection(frame_cam)
        defect_presence_cam_1 = self.presence_detection(frame_cam, self.blue_lower, self.blue_upper, 5000)
        defect_presence_2_cam_1 = self.presence_detection(frame_cam, self.blue2_lower, self.blue2_upper, 500)
        # Joining all from camera 1 into 1 output
        defects_cam1 = self.join_defects(defects_holes_cam_1, defect_presence_cam_1)
        defects_cam1 = self.join_defects(defects_cam1, defect_presence_2_cam_1)
        # Full mode include cracks detection
        if self.full_mode:
            defect_cracks_cam_1 = self.crack_detection(frame_cam)
            defects_cam1 = self.join_defects(defects_cam1, defect_cracks_cam_1)
        # Getting defect image
        frame_painted_defect = self.paint_defects(frame_cam, defects_cam1)
        self.save_img(frame_cam, frame_painted_defect)

        return defects_cam1, frame_painted_defect

    def join_defects(self, defects_1, defects_2):
        joined_defects = []
        for i in range(0, len(defects_1)):
            joined_defects.append(defects_1[i] and defects_2[i])

        return joined_defects

    def barrel_distortion(self, src):
        width = src.shape[1]
        height = src.shape[0]

        distCoeff = np.zeros((4, 1), np.float64)

        k1 = -1.7e-5;  # negative to remove barrel distortion
        k2 = 0.0;
        p1 = 0.0;
        p2 = 0.0;

        distCoeff[0, 0] = k1;
        distCoeff[1, 0] = k2;
        distCoeff[2, 0] = p1;
        distCoeff[3, 0] = p2;

        # assume unit matrix for camera
        cam = np.eye(3, dtype=np.float32)

        cam[0, 2] = width / 2.0  # define center x
        cam[1, 2] = height / 2.0  # define center y
        cam[0, 0] = 10.  # define focal length x
        cam[1, 1] = 10.  # define focal length y

        # here the undistortion will be computed
        dst = cv.undistort(src, cam, distCoeff)

        return dst

    def resize(self, img):
        frame = imutils.resize(img, height=self.hmi_height, width=self.hmi_width)
        return frame

    def paint_defects(self, raw_img, defects):
        defect_image = raw_img.copy()
        for i in range(0, len(self.rois)):
            eval = defects[i]
            if eval:
                color = (0, 255, 0)
            else:
                color = (0, 0, 255)
            defect_image = self.draw_roi(defect_image, self.rois[i], color)
            defect_image = self.draw_number(defect_image, self.rois[i], (0, 0, 0), i + 1)
        return defect_image

    def save_img(self, raw_img, defect_img):
        if self.save:
            list_dir_cam2 = os.listdir(self.save_path_img)
            if len(list_dir_cam2) > self.limit_img:
                os.remove(os.path.join(self.save_path_img, list_dir_cam2[0]))
                os.remove(os.path.join(self.save_path_img, list_dir_cam2[1]))

            dt_string = self.get_date()

            path_img_raw = os.path.join(self.save_path_img, "Image_" + self.name + "_" + dt_string + "_raw.bmp")
            path_img_defect = os.path.join(self.save_path_img, "Image_" + self.name + "_" + dt_string + "_defect.bmp")

            cv.imwrite(path_img_raw, raw_img)
            cv.imwrite(path_img_defect, defect_img)

    def get_date(self):
        # Getting date
        now = datetime.now()
        dt_string = now.strftime("%Y-%m-%d__%H-%M-%S")
        return dt_string

    def draw_number(self, raw_img, roi, color, roi_number):
        cv.putText(raw_img, str(roi_number), (roi[0] + 100, roi[1] + 100),
                   cv.FONT_HERSHEY_SIMPLEX, 2, color)
        return raw_img

    def draw_roi(self, cv_image, roi, color):
        # Line thickness of 2 px
        thickness = 2

        # Draw a rectangle with color lines provided on parameters, borders of thickness of 2 px
        image = cv.rectangle(cv_image, roi, color, thickness)

        return image

    def hole_detection(self, cv_image):
        list_defects_by_roi = []
        counter = 1
        for r in self.rois:
            # Crop image for each one of the defined rois
            img = cv_image[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
            # Transforming to grey color
            img = self.apply_mask(counter, img)
            counter += 1

            # take the gaussian blur image
            gauss = cv.GaussianBlur(img, (5, 5), 100)
            if self.debug:
                cv.imshow("Gauss", gauss);
                cv.waitKey(1)

            # find threshold to convert into pure black white image
            ret, thresh = cv.threshold(gauss, 110, 255, 0)

            if self.debug:
                cv.imshow("Thresh", thresh);
                cv.waitKey(1)
            # Setup SimpleBlobDetector parameters.

            # Setup SimpleBlobDetector parameters.
            params = cv.SimpleBlobDetector_Params()

            # Change thresholds
            params.minThreshold = 10;
            params.maxThreshold = 250;

            # Filter by Area.
            params.filterByArea = True
            params.minArea = 10

            # Filter by Circularity
            params.filterByCircularity = True
            params.minCircularity = 0.2

            # Filter by Convexity
            params.filterByConvexity = True
            params.minConvexity = 0.67

            # Filter by Inertia
            params.filterByInertia = True
            params.minInertiaRatio = 0.1

            # detect holes using blob detector
            detector = cv.SimpleBlobDetector_create(params)

            keypoint = detector.detect(thresh);
            if keypoint:
                list_defects_by_roi.append(False)
                if self.debug:
                    cv_image = self.draw_roi(cv_image, r, (0, 0, 255))
            else:
                list_defects_by_roi.append(True)
                if self.debug:
                    cv_image = self.draw_roi(cv_image, r, (0, 255, 0))
            imgkeypoint = cv.drawKeypoints(img, keypoint, np.array([]), (0, 255, 0),
                                           cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            if self.debug:
                cv.imshow("img_key", imgkeypoint);
                cv.waitKey(1)
        if self.debug:
            cv.destroyAllWindows()
        return list_defects_by_roi

    def crack_detection(self, cv_image):
        threshold = 20
        # CRACK DETECTION
        list_defects_by_roi = []
        # Start counter
        counter = 1
        for r in self.rois:
            img = cv_image[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
            # Apply mask to filter border
            img = self.apply_mask(counter, img)

            # Displaying image masked
            if self.debug:
                cv.imshow('result', img)
                cv.waitKey(1)
            # Crop image for each one of the defined rois
            # Filtering just blue color
            blue_lower = np.array([90, 57, 43], np.uint8)
            blue_upper = np.array([155, 115, 120], np.uint8)
            if counter == 4 or counter == 8 or counter == 7 or counter == 11:
                blue_lower = np.array([90, 57, 43], np.uint8)
                blue_upper = np.array([155, 105, 112], np.uint8)
            # boundaries
            lower = np.array(blue_lower, dtype="uint8")
            upper = np.array(blue_upper, dtype="uint8")

            # find the colors within the specified boundaries and apply
            # the mask
            mask = cv.inRange(img, lower, upper)
            output = cv.bitwise_and(img, img, mask=mask)
            if self.debug:
                cv.imshow(self.name + "Crack", self.resize(output));
                cv.waitKey(1)
            # Splitting components
            B, G, R = cv.split(output)
            gray = cv.cvtColor(output, cv.COLOR_BGR2GRAY)
            if self.debug:
                cv.imshow("Grey", self.resize(gray));
                cv.waitKey(1)
            # show the images
            ret, thresh2 = cv.threshold(gray, 127, 255, cv.THRESH_TRUNC)

            # Counting blue pixels
            count_blue_pixels = cv.countNonZero(gray)

            # Using threshold to determine if there is a threshold
            defect_present = count_blue_pixels > threshold

            if defect_present:
                list_defects_by_roi.append(False)
                if self.debug:
                    cv_image = self.draw_roi(cv_image, r, (0, 0, 255))
            else:
                list_defects_by_roi.append(True)
                if self.debug:
                    cv_image = self.draw_roi(cv_image, r, (0, 255, 0))
            # Create an output image
            if self.show:
                cv.imshow(self.name + "Crack", self.resize(cv_image));
                if cv.waitKey(1) >= 0:
                    break
            # Increment counter
            counter += 1
        if self.debug:
            cv.destroyAllWindows()
        return list_defects_by_roi

    def apply_mask(self, counter, img):
        try:
            # Reading Mask
            mask_cookie = self.mask_images[counter-1]
            # Using image into cookie image
            if img.shape == mask_cookie.shape:
                mask_gray = cv.cvtColor(mask_cookie, cv.COLOR_BGR2GRAY)
                # mask the image
                masked_image = np.copy(img)
                masked_image[mask_gray == 0] = [255, 255, 255]

                return masked_image
            else:
                self.logger.info('Cam_name: ' + self.name + ' Counter: '+str(counter))
                self.logger.info('Img Shape: ' + str(img.shape) + ' Mask Shape: ' + str(mask_cookie.shape))
                return img
        except Exception as x:
            self.logger.error('Cam_name: ' + self.name + ' Counter: '+str(counter))
            self.logger.error('Error getting data:' + str(x))
            traceback_info = traceback.format_exc()
            self.logger.error('Traceback:' + traceback_info)
            return img

    def presence_detection(self, cv_image, blue_lower, blue_upper, threshold):
        # CRACK DETECTION
        self.debug = True
        list_defects_by_roi = []
        counter = 1
        # Convert from RGB to HSV
        cv_image = cv.cvtColor(cv_image, cv.COLOR_BGR2RGB)
        for r in self.rois:
            # Crop image for each one of the defined rois
            img = cv_image[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]

            # Apply mask to filter border
            img = self.apply_mask(counter, img)
            # Increment counter
            counter += 1
            # boundaries
            lower = np.array(blue_lower, dtype="uint8")
            upper = np.array(blue_upper, dtype="uint8")

            # find the colors within the specified boundaries and apply
            # the mask
            mask = cv.inRange(img, lower, upper)
            output = cv.bitwise_and(img, img, mask=mask)

            B, G, R = cv.split(output)
            # show the images
            ret, thresh2 = cv.threshold(B, 127, 255, cv.THRESH_BINARY)

            count_blue_pixels = cv.countNonZero(mask)
            is_present = count_blue_pixels < threshold

            # Show images for debugging
            if self.debug:
                cv.imshow("mask", mask)
                if cv.waitKey(1) >= 0:
                    break
            if self.debug:
                cv.imshow("legend", img);
                if cv.waitKey(1) >= 0:
                    break

            if not is_present:
                list_defects_by_roi.append(False)
                if self.debug:
                    cv_image = self.draw_roi(cv_image, r, (0, 0, 255))
            else:
                list_defects_by_roi.append(True)
                if self.debug:
                    cv_image = self.draw_roi(cv_image, r, (0, 255, 0))

        if self.debug:
            cv.destroyAllWindows()
        self.debug = False

        return list_defects_by_roi

    def get_offset(self, cv_image, r, ref_middle_point):

        blue_lower = np.array([106, 7, 4], np.uint8)
        blue_upper = np.array([250, 128, 65], np.uint8)

        img = cv_image[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]

        # boundaries
        lower = np.array(blue_lower, dtype="uint8")
        upper = np.array(blue_upper, dtype="uint8")

        # find the colors within the specified boundaries and apply
        # the mask
        mask = cv.inRange(img, lower, upper)
        output = cv.bitwise_and(img, img, mask=mask)

        B, G, R = cv.split(output)
        # show the images
        ret, thresh2 = cv.threshold(B, 127, 255, cv.THRESH_BINARY)

        count_blue_pixels = cv.countNonZero(thresh2)
        is_present = count_blue_pixels < self.threshold_presence

        length_roi = B.shape[1]

        # Getting
        help_array = np.argmax(B>0, axis=0 )
        x_ = np.where(help_array>0)[0]
        y_ = help_array[help_array>0]
        np.argmax(B[:,0]>0)
        horizontal_line = DataFrame({'x': x_.tolist(), 'y': y_.tolist()})

        if len(horizontal_line) > 10:
            # Deleting outliers
            horizontal_line_filter = horizontal_line[(np.abs(stats.zscore(horizontal_line)) < 1).all(axis=1)]

            x = np.array(horizontal_line_filter['x']).reshape((-1, 1))
            y = np.array(horizontal_line_filter['y'])

            # Predicting missing values
            model = LinearRegression()
            model.fit(x, y)
            x_pred = np.linspace(0, length_roi, length_roi + 1).reshape((-1, 1))
            y_pred = model.predict(x_pred)

            middle_pixel = int(length_roi/2)
            point_middle_section = int(y_pred[800])#int(y_pred[middle_pixel])
            offset_middle_section = point_middle_section - ref_middle_point
            # Modifying rois with offset
            for i in range(0, len(self.rois)):
                rois_as_list = list(self.rois[i])
                rois_as_list[1] = rois_as_list[1] + int(offset_middle_section)
                self.rois[i] = tuple(rois_as_list)
            # Draw line
            line_thickness = 1
            cv.line(output, (int(x_pred[0]), int(y_pred[0])), (int(x_pred[length_roi]), int(y_pred[length_roi])),
                    (0, 255, 0), thickness=line_thickness)
        else:
            offset_middle_section = 0
        #print(offset_middle_section)
        # Show images for debugging
        if self.debug:
            cv.imshow("images", output)
            cv.waitKey(1)
        # Return offset
        return offset_middle_section

    def reset_roi(self, rois_cam1, x, y):
        self.rois = rois_cam1.copy()
        # Modifying rois with offset
        for i in range(0, len(self.rois)):
            rois_as_list = list(self.rois[i])
            rois_as_list[1] = rois_as_list[1] + int(y)
            rois_as_list[0] = rois_as_list[0] + int(x)
            self.rois[i] = tuple(rois_as_list)

