# -*- coding:utf-8 -*-
import sys

import os
from pathlib import Path

from eigenimages import (load_images_from_folder, 
                         calculate_pca,
                         calculate_scores)

import cv2
import numpy as np

from PySide6.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel,
    QFileDialog, QVBoxLayout, QMessageBox,
    QCheckBox
)
from PySide6.QtGui import QKeyEvent
from PySide6.QtCore import Qt


class BatchWindow(QWidget):
    def __init__(self):
        super().__init__()

        self._input_dir_path = None
        self._output_dir_path = None

        self.setWindowTitle("Create eigenimages (batch version)")

        # Path labels
        self.input_dir_label = QLabel("Image dir: None selected")

        # Buttons
        self.select_input_dir_btn = QPushButton("Select input folder")
        self.calculate_btn = QPushButton("Calculate")
        self.checkbox_btn = QCheckBox("Scale images statically (100, 100)")
        self.checkbox_btn.setChecked(True)

        # Deactivate button until path is ready
        self.calculate_btn.setEnabled(False)

        # Connections
        self.select_input_dir_btn.clicked.connect(self.select_input_directory)
        self.calculate_btn.clicked.connect(self.calculate)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.select_input_dir_btn)
        layout.addWidget(self.input_dir_label)
        layout.addWidget(self.checkbox_btn)
        layout.addWidget(self.calculate_btn)

        self.setLayout(layout)

    # Exit on Escape
    def keyPressEvent(self, event: QKeyEvent):
        if event.key() == Qt.Key_Escape:
            QApplication.quit() 
        else:
            super().keyPressEvent(event)

    def select_input_directory(self):
        dir_path = QFileDialog.getExistingDirectory(self, 
                                                    "Select input folder", 
                                                    dir=os.path.expanduser("~"))
        if dir_path:
            self.input_dir_label.setText(f"Image dir: {dir_path}")
            self._input_dir_path = dir_path
            # Create a results folder in the parent path
            self._output_dir_path = os.path.join(dir_path, "Results")
            try:
                os.makedirs(self._output_dir_path)
            except:
                print(f"Warning: {self._output_dir_path} already exists")
            self.calculate_btn.setEnabled(True)

    def calculate(self):
        '''Calculate for all subfolders in a folder'''
        
        # Get all folder paths
        subfolders = [subfolder for subfolder in os.listdir(
            self._input_dir_path) if os.path.isdir(
                os.path.join(self._input_dir_path, subfolder))]

        # Don't look at hidden files nor the Results folder
        folder_names = [subfolder for subfolder in subfolders if not\
                        subfolder.startswith(".") and subfolder != "Results"]
        subfolders = [os.path.join(self._input_dir_path, subfolder) for\
                      subfolder in folder_names]

        self.calculate_btn.setEnabled(False)
        for fname, subdir in zip(folder_names, subfolders):
            print(f"Processing folder {subdir}...")
            try:
                img_array, original_shape, small_shape = \
                        load_images_from_folder(
                            subdir,
                            static_resizing=self.checkbox_btn.isChecked())
                single_img = calculate_pca(img_array, small_shape)
                scores = calculate_scores(single_img)
                
                # Save outputs
                with open(os.path.join(self._output_dir_path, f"{fname}.csv"), 
                          "w") as fp:
                    fp.write(f"All values,No Negatives,\n{scores[0]},{scores[1]}")
                
                # Save images (w/wo negatives)
                non_negative_single_img = ((single_img > 0) * single_img)
                
                single_img_255 = cv2.normalize(
                    single_img, 
                    None, 
                    alpha=0, 
                    beta=255, 
                    norm_type=cv2.NORM_MINMAX).reshape(
                        small_shape).astype(np.uint8)
                single_img_255 = cv2.resize(single_img_255, original_shape[::-1])
                
                non_negative_single_img_255 = cv2.normalize(
                    non_negative_single_img, 
                    None,
                    alpha=0, 
                    beta=255, 
                    norm_type=cv2.NORM_MINMAX).reshape(
                        small_shape).astype(np.uint8)
                non_negative_single_img_255 = cv2.resize(
                    non_negative_single_img_255, 
                    original_shape[::-1])

                prefix_path = os.path.join(self._output_dir_path, f"{fname}")
                cv2.imwrite(prefix_path + ".png", single_img_255)
                cv2.imwrite(prefix_path + "_non_neg.png", non_negative_single_img_255)

            except Exception as e:
                self.show_exception_dialog(e)
        QMessageBox.information(
            self,
            "Finished!",
            f"Finished processing folder:\n{self._input_dir_path}",
            QMessageBox.Ok
        )
        self.calculate_btn.setEnabled(True)

    def show_exception_dialog(self, exception):
        QMessageBox.critical(
            self,
            "An Error Occurred",
            f"Exception: {str(exception)}",
            QMessageBox.Ok
        )

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = BatchWindow()
    window.resize(600, 200)
    window.show()
    sys.exit(app.exec())
