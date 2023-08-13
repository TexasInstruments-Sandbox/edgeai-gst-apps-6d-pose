#  Copyright (C) 2021 Texas Instruments Incorporated - http://www.ti.com/
#
#  Redistribution and use in source and binary forms, with or without
#  modification, are permitted provided that the following conditions
#  are met:
#
#    Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#
#    Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the
#    distribution.
#
#    Neither the name of Texas Instruments Incorporated nor the names of
#    its contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
#  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
#  A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
#  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
#  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
#  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
#  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
#  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
#  (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#  OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import cv2
import numpy as np
import copy
import debug

np.set_printoptions(threshold=np.inf, linewidth=np.inf)


def create_title_frame(title, width, height):
    frame = np.zeros((height, width, 3), np.uint8)
    if title != None:
        frame = cv2.putText(
            frame,
            "Texas Instruments - Edge Analytics",
            (40, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 0, 0),
            2,
        )
        frame = cv2.putText(
            frame, title, (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2
        )
    return frame


def overlay_model_name(frame, model_name, start_x, start_y, width, height):
    row_size = 40 * width // 1280
    font_size = width / 1280
    cv2.putText(
        frame,
        "Model : " + model_name,
        (start_x + 5, start_y - row_size // 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_size,
        (255, 255, 255),
        2,
    )
    return frame


class PostProcess:
    """
    Class to create a post process context
    """

    def __init__(self, flow):
        self.flow = flow
        self.model = flow.model
        self.debug = None
        self.debug_str = ""
        if flow.debug_config and flow.debug_config.post_proc:
            self.debug = debug.Debug(flow.debug_config, "post")

    def get(flow):
        """
        Create a object of a subclass based on the task type
        """
        if flow.model.task_type == "classification":
            return PostProcessClassification(flow)
        elif flow.model.task_type == "detection":
            return PostProcessDetection(flow)
        elif flow.model.task_type == "segmentation":
            return PostProcessSegmentation(flow)
        elif flow.model.task_type == "object_6d_pose_estimation":
            return PostProcessObject6DPoseEstimation(flow)


class PostProcessClassification(PostProcess):
    def __init__(self, flow):
        super().__init__(flow)

    def __call__(self, img, results):
        """
        Post process function for classification
        Args:
            img: Input frame
            results: output of inference
        """
        results = np.squeeze(results)
        img = self.overlay_topN_classnames(img, results)

        if self.debug:
            self.debug.log(self.debug_str)
            self.debug_str = ""

        return img

    def overlay_topN_classnames(self, frame, results):
        """
        Process the results of the image classification model and draw text
        describing top 5 detected objects on the image.

        Args:
            frame (numpy array): Input image in BGR format where the overlay should
        be drawn
            results (numpy array): Output of the model run
        """
        orig_width = frame.shape[1]
        orig_height = frame.shape[0]
        row_size = 40 * orig_width // 1280
        font_size = orig_width / 1280
        N = self.model.topN
        topN_classes = np.argsort(results)[: (-1 * N) - 1 : -1]
        title_text = "Recognized Classes (Top %d):" % N
        font = cv2.FONT_HERSHEY_SIMPLEX

        text_size, _ = cv2.getTextSize(title_text, font, font_size, 2)

        bg_top_left = (0, (2 * row_size) - text_size[1] - 5)
        bg_bottom_right = (text_size[0] + 10, (2 * row_size) + 3 + 5)
        font_coord = (5 , 2 * row_size)

        cv2.rectangle(frame,
                      bg_top_left,
                      bg_bottom_right,
                      (5, 11, 120),
                      -1)

        cv2.putText(
            frame,
            title_text,
            font_coord,
            font,
            font_size,
            (0, 255, 0),
            2,
        )
        row = 3
        for idx in topN_classes:
            class_name = self.model.classnames.get(idx + self.model.label_offset)

            text_size, _ = cv2.getTextSize(class_name, font, font_size, 2)

            bg_top_left = (0, (row_size * row) - text_size[1] - 5)
            bg_bottom_right = (text_size[0] + 10, (row_size * row) + 3 + 5)
            font_coord = (5, row_size * row)

            cv2.rectangle(frame,
                         bg_top_left,
                         bg_bottom_right,
                         (5, 11, 120),
                         -1)
            cv2.putText(
                frame,
                class_name,
                font_coord,
                font,
                font_size,
                (255, 255, 0),
                2,
            )
            row = row + 1
            if self.debug:
                self.debug_str += class_name + "\n"

        return frame


class PostProcessDetection(PostProcess):
    def __init__(self, flow):
        super().__init__(flow)

    def __call__(self, img, results):
        """
        Post process function for detection
        Args:
            img: Input frame
            results: output of inference
        """
        for i, r in enumerate(results):
            r = np.squeeze(r)
            if r.ndim == 1:
                r = np.expand_dims(r, 1)
            results[i] = r

        if self.model.shuffle_indices:
            results_reordered = []
            for i in self.model.shuffle_indices:
                results_reordered.append(results[i])
            results = results_reordered

        if results[-1].ndim < 2:
            results = results[:-1]

        bbox = np.concatenate(results, axis=-1)

        if self.model.formatter:
            if self.model.ignore_index == None:
                bbox_copy = copy.deepcopy(bbox)
            else:
                bbox_copy = copy.deepcopy(np.delete(bbox, self.model.ignore_index, 1))
            bbox[..., self.model.formatter["dst_indices"]] = bbox_copy[
                ..., self.model.formatter["src_indices"]
            ]

        if not self.model.normalized_detections:
            bbox[..., (0, 2)] /= self.model.resize[0]
            bbox[..., (1, 3)] /= self.model.resize[1]

        for b in bbox:
            if b[5] > self.model.viz_threshold:
                if type(self.model.label_offset) == dict:
                    class_name = self.model.classnames[self.model.label_offset[int(b[4])]]
                else:
                    class_name = self.model.classnames[self.model.label_offset + int(b[4])]
                img = self.overlay_bounding_box(img, b, class_name)

        if self.debug:
            self.debug.log(self.debug_str)
            self.debug_str = ""

        return img

    def overlay_bounding_box(self, frame, box, class_name):
        """
        draw bounding box at given co-ordinates.

        Args:
            frame (numpy array): Input image where the overlay should be drawn
            bbox : Bounding box co-ordinates in format [X1 Y1 X2 Y2]
            class_name : Name of the class to overlay
        """
        box = [
            int(box[0] * frame.shape[1]),
            int(box[1] * frame.shape[0]),
            int(box[2] * frame.shape[1]),
            int(box[3] * frame.shape[0]),
        ]
        box_color = (20, 220, 20)
        text_color = (0, 0, 0)
        cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), box_color, 2)
        cv2.rectangle(
            frame,
            (int((box[2] + box[0]) / 2) - 5, int((box[3] + box[1]) / 2) + 5),
            (int((box[2] + box[0]) / 2) + 160, int((box[3] + box[1]) / 2) - 15),
            box_color,
            -1,
        )
        cv2.putText(
            frame,
            class_name,
            (int((box[2] + box[0]) / 2), int((box[3] + box[1]) / 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            text_color,
        )

        if self.debug:
            self.debug_str += class_name
            self.debug_str += str(box) + "\n"

        return frame


class PostProcessSegmentation(PostProcess):
    def __call__(self, img, results):
        """
        Post process function for segmentation
        Args:
            img: Input frame
            results: output of inference
        """
        img = self.blend_segmentation_mask(img, results[0])

        return img

    def blend_segmentation_mask(self, frame, results):
        """
        Process the result of the semantic segmentation model and return
        an image color blended with the mask representing different color
        for each class

        Args:
            frame (numpy array): Input image in BGR format which should be blended
            results (numpy array): Results of the model run
        """

        mask = np.squeeze(results)

        if len(mask.shape) > 2:
            mask = mask[0]

        if self.debug:
            self.debug_str += str(mask.flatten()) + "\n"
            self.debug.log(self.debug_str)
            self.debug_str = ""

        # Resize the mask to the original image for blending
        org_image_rgb = frame
        org_width = frame.shape[1]
        org_height = frame.shape[0]

        mask_image_rgb = self.gen_segment_mask(mask)
        mask_image_rgb = cv2.resize(
            mask_image_rgb, (org_width, org_height), interpolation=cv2.INTER_LINEAR
        )

        blend_image = cv2.addWeighted(
            mask_image_rgb, 1 - self.model.alpha, org_image_rgb, self.model.alpha, 0
        )

        return blend_image

    def gen_segment_mask(self, inp):
        """
        Generate the segmentation mask from the result of semantic segmentation
        model. Creates an RGB image with different colors for each class.

        Args:
            inp (numpy array): Result of the model run
        """

        r_map = (inp * 10).astype(np.uint8)
        g_map = (inp * 20).astype(np.uint8)
        b_map = (inp * 30).astype(np.uint8)

        return cv2.merge((r_map, g_map, b_map))

class PostProcessObject6DPoseEstimation(PostProcess):
    def __init__(self, flow):
        super().__init__(flow)
        self.x_offset = flow.width / flow.model.resize[0]
        self.y_offset = flow.height / flow.model.resize[1]
        self.dataset = flow.model.dataset

        # camera_matrix for test-split of YCB dataset
        self.YCBV_CAMERA_MATRIX = np.array(
            [[1066.778, 0, 312.9869], [0.0, 1067.487, 241.3109], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

        # camera_matrix for linemod dataset
        self.LM_CAMERA_MATRIX = np.array(
            [[572.4114, 0.0, 325.2611], [0.0, 573.57043, 242.04899], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )

        # vertices for YCB21 objects
        self.YCBV_VERTICES = np.array(
            [
                [51.1445, 51.223, 70.072],
                [35.865, 81.9885, 106.743],
                [24.772, 47.024, 88.0075],
                [33.927, 33.875, 51.0185],
                [48.575, 33.31, 95.704],
                [42.755, 42.807, 16.7555],
                [68.924, 64.3955, 19.414],
                [44.6775, 50.5545, 15.06],
                [51.0615, 30.161, 41.8185],
                [54.444, 89.206, 18.335],
                [74.4985, 72.3845, 121.32],
                [51.203, 33.856, 125.32],
                [80.722, 80.5565, 27.485],
                [58.483, 46.5375, 40.692],
                [92.1205, 93.717, 28.6585],
                [51.9755, 51.774, 102.945],
                [48.04, 100.772, 7.858],
                [10.5195, 60.4225, 9.4385],
                [59.978, 85.639, 19.575],
                [104.897, 82.18, 18.1665],
                [26.315, 38.921, 25.5655]
            ],
            dtype=np.float32,
        )

        # vertices for Linemod objects
        self.LM_VERTICES = np.array(
            [
                [-37.93430000, 38.79960000, 45.88450000],
                [107.83500000, 60.92790000, 109.70500000],
                [83.21620000, 82.65910000, 37.23640000],
                [68.32970000, 71.51510000, 50.24850000],
                [50.39580000, 90.89790000, 96.86700000],
                [33.50540000, 63.81650000, 58.72830000],
                [58.78990000, 45.75560000, 47.31120000],
                [114.73800000, 37.73570000, 104.00100000],
                [52.21460000, 38.70380000, 42.84850000],
                [75.09230000, 53.53750000, 34.62070000],
                [18.36050000, 38.93300000, 86.40790000],
                [50.44390000, 54.24850000, 45.40000000],
                [129.11300000, 59.24100000, 70.56620000],
                [101.57300000, 58.87630000, 106.55800000],
                [46.95910000, 73.71670000, 92.37370000],
            ],
            dtype=np.float32,
        )

        # order of vertices to draw cuboid
        vertices_order = np.array(
            [
                [-1, -1, -1],
                [-1, -1, 1],
                [-1, 1, 1],
                [-1, 1, -1],
                [1, -1, -1],
                [1, -1, 1],
                [1, 1, 1],
                [1, 1, -1],
            ],
            dtype=np.float32,
        )

        if self.dataset == "ycbv":
            self.camera_matrix = self.YCBV_CAMERA_MATRIX.reshape((3, 3)).T
            self.vertices = self.YCBV_VERTICES[:, None, :] * vertices_order
        else:
            self.camera_matrix = self.LM_CAMERA_MATRIX.reshape((3, 3)).T
            self.vertices = self.LM_VERTICES[:, None, :] * vertices_order

        self.COLORS = [
            (0, 113, 188),
            (216, 82, 24),
            (236, 176, 31),
            (125, 46, 141),
            (118, 171, 47),
            (76, 189, 237),
            (161, 19, 46),
            (76, 76, 76),
            (153, 153, 153),
            (255, 0, 0),
            (255, 127, 0),
            (190, 190, 0),
            (0, 255, 0),
            (0, 0, 255),
            (170, 0, 255),
            (84, 84, 0),
            (84, 170, 0),
            (84, 255, 0),
            (170, 84, 0),
            (170, 170, 0),
            (170, 255, 0),
            (255, 84, 0),
            (255, 170, 0),
            (255, 255, 0),
            (0, 84, 127),
            (0, 170, 127),
            (0, 255, 127),
            (84, 0, 127),
            (84, 84, 127),
            (84, 170, 127),
            (84, 255, 127),
            (170, 0, 127),
            (170, 84, 127),
            (170, 170, 127),
            (170, 255, 127),
            (255, 0, 127),
            (255, 84, 127),
            (255, 170, 127),
            (255, 255, 127),
            (0, 84, 255),
            (0, 170, 255),
            (0, 255, 255),
            (84, 0, 255),
            (84, 84, 255),
            (84, 170, 255),
            (84, 255, 255),
            (170, 0, 255),
            (170, 84, 255),
            (170, 170, 255),
            (170, 255, 255),
            (255, 0, 255),
            (255, 84, 255),
            (255, 170, 255),
            (84, 0, 0),
            (127, 0, 0),
            (170, 0, 0),
            (212, 0, 0),
            (255, 0, 0),
            (0, 42, 0),
            (0, 84, 0),
            (0, 127, 0),
            (0, 170, 0),
            (0, 212, 0),
            (0, 255, 0),
            (0, 0, 42),
            (0, 0, 84),
            (0, 0, 127),
            (0, 0, 170),
            (0, 0, 212),
            (0, 0, 255),
            (0, 0, 0),
            (36, 36, 36),
            (72, 72, 72),
            (109, 109, 109),
            (145, 145, 145),
            (182, 182, 182),
            (218, 218, 218),
            (0, 113, 188),
            (80, 182, 188),
            (127, 127, 0)
        ]

    def __call__(self, img, result):
        result = np.squeeze(result[0])
        for det in result:
            box, score, cls = det[:4], det[4], int(det[5])
            if score > self.model.viz_threshold:
                r1, r2 = det[6:9, None], det[9:12, None]
                r3 = np.cross(r1, r2, axis=0)
                rotation_mat = np.concatenate((r1, r2, r3), axis=1)
                translation_vec = det[12:15]
                cuboid_corners_2d = self.project_3d_2d(
                    self.vertices[int(cls)], rotation_mat, translation_vec
                )
                self.draw_cuboid_2d(img, cuboid_corners_2d, self.COLORS[cls][::-1], 2)

                # Labels on cuboid
                text = "{}:{:.1f}%".format(self.model.classnames[cls], score * 100)
                txt_color = (
                    (0, 0, 0) if np.mean(self.COLORS[cls]) > 127 else (255, 255, 255)
                )
                font = cv2.FONT_HERSHEY_SIMPLEX
                txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
                x0, y0 = int(box[0] * self.x_offset), int(box[1] * self.y_offset)
                cv2.rectangle(
                    img,
                    (x0, y0 + 1),
                    (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
                    self.COLORS[cls][::-1],
                    -1,
                )
                cv2.putText(
                    img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1
                )
        return img

    def project_3d_2d(self, pts_3d, rotation_mat, translation_vec):
        xformed_3d = np.matmul(pts_3d, rotation_mat.T) + translation_vec
        xformed_3d[:, :3] = xformed_3d[:, :3] / xformed_3d[:, 2:3]
        projected_2d = np.matmul(xformed_3d, self.camera_matrix)[:, :2]
        return projected_2d

    def draw_cuboid_2d(self, img, cuboid_corners, color, thickness):
        box = np.copy(cuboid_corners).astype(np.int32)
        box = [
            (int(kpt[0] * self.x_offset), int(kpt[1] * self.y_offset)) for kpt in box
        ]
        # Draw circles on the edges
        for i in box:
            cv2.circle(img, i, thickness + 3, color, -1)
        # back
        cv2.line(img, box[4], box[5], color, thickness)
        cv2.line(img, box[5], box[6], color, thickness)
        cv2.line(img, box[6], box[7], color, thickness)
        cv2.line(img, box[7], box[4], color, thickness)
        # sides
        cv2.line(img, box[0], box[4], color, thickness)
        cv2.line(img, box[1], box[5], color, thickness)
        cv2.line(img, box[2], box[6], color, thickness)
        cv2.line(img, box[3], box[7], color, thickness)
        # front
        cv2.line(img, box[0], box[1], color, thickness)
        cv2.line(img, box[1], box[2], color, thickness)
        cv2.line(img, box[2], box[3], color, thickness)
        cv2.line(img, box[3], box[0], color, thickness)
        return img