# -*- coding:utf-8 -*-
import sys

from os.path import expanduser
from pathlib import Path

from eigenimages import (load_images_from_folder, 
                         calculate_pca,
                         calculate_scores)

import cv2
import numpy as np

from PySide6.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel,
    QFileDialog, QVBoxLayout, QMessageBox
)
from PySide6.QtGui import QKeyEvent
from PySide6.QtCore import Qt


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self._input_dir_ready = False
        self._output_file_ready = False

        self._input_dir_path = None
        self._output_file_path = None

        self.setWindowTitle("Create eigenimages")

        # Path labels
        self.input_dir_label = QLabel("Image dir: None selected")
        self.output_file_label = QLabel("Output file: None selected")

        # Buttons
        self.select_input_dir_btn = QPushButton("Select input folder")
        self.select_output_file_btn = QPushButton("Select output image path")
        self.calculate_btn = QPushButton("Calculate")

        # Deactivate button until paths ready
        self.calculate_btn.setEnabled(False)

        # Connections
        self.select_input_dir_btn.clicked.connect(self.select_input_directory)
        self.select_output_file_btn.clicked.connect(self.select_output_file)
        self.calculate_btn.clicked.connect(self.calculate)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.select_input_dir_btn)
        layout.addWidget(self.input_dir_label)
        layout.addWidget(self.select_output_file_btn)
        layout.addWidget(self.output_file_label)
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
                                                    dir=expanduser("~"))
        if dir_path:
            self.input_dir_label.setText(f"Image dir: {dir_path}")
            self._input_dir_path = dir_path
            self._input_dir_ready = True
            self.enable_calculate_btn()

    def select_output_file(self):
        if self._input_dir_ready:
            input_dir = self._input_dir_path
        else:
            input_dir = expanduser("~")

        file_path, _ = QFileDialog.getSaveFileName(self, 
                                                   "Select output file",
                                                   str(Path(input_dir).parent), 
                                                   "CSV files(*.csv);;All files (*)")
        if file_path:
            if not file_path.endswith(".csv"):
                file_path += ".csv"
            self.output_file_label.setText(f"Output file: {file_path}")
            self._output_file_path = file_path
            self._output_file_ready = True
            self.enable_calculate_btn()

    def calculate(self):
        try:
            self.calculate_btn.setEnabled(False)
            img_array, original_shape = load_images_from_folder(self._input_dir_path)
            single_img = calculate_pca(img_array, original_shape)
            scores = calculate_scores(single_img)
            QMessageBox.information(
                self,
                "Finished!",
                f"Finished processing folder:\n{self._input_dir_path}",
                QMessageBox.Ok
            )
            # Save outputs
            with open(self._output_file_path, "w") as fp:
                fp.write(f"All values,No Negatives,\n{scores[0]},{scores[1]}")
            
            # Save images (w/wo negatives)
            non_negative_single_img = ((single_img > 0) * single_img)
            
            single_img_255 = cv2.normalize(
                single_img, 
                None, 
                alpha=0, 
                beta=255, 
                norm_type=cv2.NORM_MINMAX).reshape(
                    original_shape).astype(np.uint8)
            
            non_negative_single_img_255 = cv2.normalize(
                non_negative_single_img, 
                None,
                alpha=0, 
                beta=255, 
                norm_type=cv2.NORM_MINMAX).reshape(
                    original_shape).astype(np.uint8)

            prefix_path = self._output_file_path[:-4]
            cv2.imwrite(prefix_path + ".png", single_img_255)
            cv2.imwrite(prefix_path + "_non_neg.png", non_negative_single_img_255)

            self.calculate_btn.setEnabled(True)
        except Exception as e:
            self.calculate_btn.setEnabled(True)
            self.show_exception_dialog(e)

    def enable_calculate_btn(self):
        if self._input_dir_ready and self._output_file_ready:
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
    window = MainWindow()
    window.resize(600, 200)
    window.show()
    sys.exit(app.exec())
