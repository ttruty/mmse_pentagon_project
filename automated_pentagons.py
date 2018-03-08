import cv2
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

good_total = 0

class AutoApp:
    def __init__(self, master):
        self.master = master
        master.wm_title("Automation")

        self.corner_num = 500
        self.quality = 0.1
        self.distance = 10
        self.detection_threshold = 0.00
        self.line_threshold = .95
        self.corner_total = 0
        self.line_total = 0
        self.det_core = 0
        self.count = 0

        self.shi_c = []
        self.shi_lines = []
        self.distilled_lines = []

        self.path = tkinter.filedialog.askopenfilename()
        self.directory = os.path.dirname(self.path)

        self.label = Label(master, text="Automating Pentagon output")
        self.label.grid(row=0, column=0, sticky="N")

        self.progress_var = DoubleVar()

        self.progress = ttk.Progressbar(master, length=100)
        self.progress.grid(row=1, column=0, sticky="NE")
        self.progress.after(1, self.select_pentagon())



    def detection_funct(self, path):
        detections, found_dets = corner_dets_methods.find_contours(path, self.corner_num, self.quality, self.distance,
                                                                   self.detection_threshold)
        return detections, found_dets

    def select_pentagon(self):
        global good_total
        file_list = self.make_file_list()
        print(len(file_list))
        step = (100 / len(file_list))
        for file in file_list:
            self.path = os.path.join(self.directory, file)
            im = cv2.imread(self.path)
            gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
            det, found_details = self.detection_funct(self.path)
            if found_details != []:
                self.corner_total, self.line_total = self.corner_details(gray, found_details[0])
                if self.corner_total == 12:
                    good_total += 1
            self.progress.step(step)
            self.progress.update()
            #print(good_total)

    def make_file_list(self):
        f = []
        for (_, _, file_names) in os.walk(self.directory):
            f.append(file_names)
            break
        print(f[0])
        return f[0]

    def corner_details(self,image, detections):
        self.shi_c, self.shi_lines, self.distilled_lines = corner_dets_methods.cornerMeths(image,
            [detections],
            self.corner_num,
            self.quality,
            self.distance)
        line_count = len(self.shi_lines)
        corner_count = len(self.shi_c)
        return corner_count, line_count

    def save_cmd(self):
        save_file = os.path.join(self.directory, (os.path.basename(self.path)+".txt"))
        with open(save_file, "w") as text_file:
            text_file.write("MMSE Pentagon GUI Version=1\n")
            text_file.write("Process date = {0}\n".format(datetime.datetime.now()))
            text_file.write("corner count =  {0}\n".format(self.corner_total))
            text_file.write("line count =  {0}\n".format(self.line_total))
            text_file.write("min corner count =  {0}\n".format(self.corner_num))
            text_file.write("corner list =  {0}\n".format(self.shi_c))
            text_file.write("lines list =  {0}\n".format(self.shi_lines))


if __name__ == '__main__':
    root = Tk()
    my_gui = AutoApp(root)
    root.mainloop()

print("Final Total=", good_total)
