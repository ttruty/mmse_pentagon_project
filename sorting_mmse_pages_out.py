# Import the required modules
import corner_dets_methods
import os
import numpy as np
import tkinter.filedialog
from tkinter import *
import shutil
import corner_dets_methods
import datetime
import tkinter as tk
from time import sleep
from math import trunc
from tkinter import ttk


def detection_funct(path):
    detections, found_dets = corner_dets_methods.find_contours(path, 20, 0.1, 4, 0)
    return detections, found_dets


cwd = os.getcwd()
path = os.path.join(cwd, 'training_mmse_pentagons')
model_path = os.path.join(path, "models", "svm.model")

testing = r'C:\Users\KinectProcessing\Desktop\testing_classify'

file_list = [os.path.join(d, x)
            for d, dirs, files in os.walk(testing)
            for x in files]

for file in file_list:

    _, dets = detection_funct(file)
    if dets != []:
        p = os.path.dirname(file).split('\\')[-1]
        f = os.path.basename(file)
        new_file = p+f
        print(p+f)
        print(dets)
        new_dir =  os.path.join(testing, 'mmse_data')
        shutil.copy(file, os.path.join(file, os.path.join(new_dir, new_file)))