from tkinter import *
from PIL import Image
from PIL import ImageTk
import tkinter.filedialog
import cv2, os, datetime
import numpy as np
import corner_dets_methods
import naming_file as name
import glob



class PentGui:
    def __init__(self, master):
        self.master = master
        master.wm_title("MMSE Pentagon")

        self.corner_num = 30
        self.quality = 0.1
        self.distance = 4
        self.detection_threshold = 0.00
        self.line_threshold= .95
        self.corner_total = 0
        self.line_total = 0
        self.det_core = 0

        self.shi_c = []
        self.shi_lines = []
        self.distilled_lines = []

        self.panelA = None
        self.panelB = None
        self.panelC = None
        self.panelD = None
        self.pacelE = None

        self.btn = Button(master, text="Select an image", command=self.select_image)
        self.btn.grid(row=22, column=0, sticky="WS")

        self.prev_btn = Button(master, text="Previous image", command=self.prev_image)
        self.prev_btn.grid(row=21, column=0, sticky="WS")

        self.next_btn = Button(master, text="Next image", command=self.next_image)
        self.next_btn.grid(row=21, column=1, sticky="WS")

        self.image_label = Label(master, text="Orginal Image")
        self.image_label.grid(row=0, column=3)

        self.det_label = Label(master, text="Detection Image")
        self.det_label.grid(row=0, column=4)

        self.corner_label = Label(master, text="Corner Image")
        self.corner_label.grid(row=0, column=5)

        self.corner_label = Label(master, text="Line Image")
        self.corner_label.grid(row=0, column=6)

        self.corner_label = Label(master, text="Gap Image")
        self.corner_label.grid(row=21, column=3)

        self.c_label = Label(master, text="Corner count")
        self.c_label.grid(row=1, column=0, sticky="W")
        self.corner_count = DoubleVar()
        self.c_scale = Scale(orient='horizontal', from_=0, to=99, variable = self.corner_count )
        self.c_scale.grid(row=1, column=1 , sticky="NW")
        self.c_scale.set(self.corner_num)

        self.q_label = Label(master, text="Min Quality")
        self.q_label.grid(row=2, column=0, sticky="W")
        self.quality_count = DoubleVar()
        self.q_scale = Scale(orient='horizontal', from_=0.01, to=1.00, resolution=0.01, variable = self.quality_count)
        self.q_scale.grid(row=2, column=1, sticky="NW")
        self.q_scale.set(self.quality)

        self.d_label = Label(master, text="Min distance")
        self.d_label.grid(row=3, column=0, sticky="W")
        self.distance_count = DoubleVar()
        self.d_scale = Scale(orient='horizontal', from_=1, to=50, variable = self.distance_count)
        self.d_scale.grid(row=3, column=1, sticky="NW")
        self.d_scale.set(self.distance)

        self.detection_label = Label(master, text="Detection Threshold")
        self.detection_label.grid(row=4, column=0, sticky="W")
        self.det_var = DoubleVar()
        self.det_scale = Scale(orient='horizontal', from_=-3.000, to=3.000, resolution=0.001, variable = self.det_var)
        self.det_scale.grid(row=4, column=1, sticky=N+W)
        self.det_scale.set(self.detection_threshold)

        self.line_label = Label(master, text="Line Threshold")
        self.line_label.grid(row=5, column=0, sticky="W")
        self.line_var = DoubleVar()
        self.line_scale = Scale(orient='horizontal', from_=0.01, to=1.00, resolution=0.01, variable = self.line_var)
        self.line_scale.grid(row=5, column=1, sticky="NW")
        self.line_scale.set(self.line_threshold)

        self.export_btn = Button(master, text="Save Text File", command=self.save_cmd)
        self.export_btn.grid(row=10, column=0, columnspan=2)

        self.flag_var = IntVar()
        self.flag_check = Checkbutton(master, text="Flag Page", variable=self.flag_var)
        self.flag_check.grid(row=11, column=0, sticky="w")

        self.apply_btn = Button(master, text="Apply", command=self.apply_cmd)
        self.apply_btn.grid(row=6, column=0, columnspan=2)

        self.multipent = IntVar()
        self.multi_check = Checkbutton(master, text="Multiple pentagons", variable=self.multipent)
        self.multi_check.grid(row=13, column=0, sticky="w")

        self.file_label = Label(master, text= "Choose a file")
        self.file_label.grid(row=0, column=0, columnspan=2)

        self.dat_id_label = Label(master, text= "Project ID")
        self.dat_id_label.grid(row=23, column=0)

        self.dat_date_label = Label(master, text= "Project Date")
        self.dat_date_label.grid(row=24, column=0)

        self.dat_fu_label = Label(master, text= "Project FU Year")
        self.dat_fu_label.grid(row=25, column=0)

        self.dat_score_label = Label(master, text= "Pentagon Scored Correct")
        self.dat_score_label.grid(row=26, column=0)

        self.det_label = Label(master, text="Top Detection Score:  No Detection")
        self.det_label.grid(row=9, column=0)

        self.corner_count_label = Label(master, text='Corner Count: No Detection')
        self.corner_count_label.grid(row=7, column=0, sticky="w")
        
        self.line_count_label = Label(master, text='Line Count: No Detection')
        self.line_count_label.grid(row=8, column=0, sticky="w")

    def apply_cmd(self):
        self.corner_num = self.c_scale.get()
        self.quality = self.q_scale.get()
        self.distance = self.d_scale.get()
        self.detection_threshold = self.det_scale.get()
        self.line_threshold = self.line_scale.get()  
        self.detections, self.found_dets = self.detection_funct(self.path)
        self.load_image(self.image,self.detections, self.found_dets)

    def select_image(self):
        self.path = tkinter.filedialog.askopenfilename()
        
        # ensure a file path was selected
        if len(self.path) > 0:
            self.file_name = os.path.basename(self.path)
            
            self.file_label['text'] = self.file_name
            self.reset_scales()
            # load the image from disk, convert it to grayscale, and detect
            # edges in it
            self.image = cv2.imread(self.path)
            self.clone = self.image.copy()
            self.line_img = self.image.copy()
            self.gap_img = self.image.copy()
            self.detections, self.found_dets = self.detection_funct(self.path)
            self.load_image(self.image,self.detections, self.found_dets)

    def next_image(self):
        self.reset_scales()
        
        filename = os.path.basename(self.path)
        #print(filename)
        dir_name = (os.path.dirname(self.path))
        file_list = os.listdir(dir_name)
        file_list.sort()
        next_index = file_list.index(filename) + 1
        #print(str(next_index))
        if next_index == 0 or next_index == len(file_list):
            return 0
        #print(str(file_list[next_index]))
        next_image = os.path.join(dir_name, file_list[next_index])
        #print(next_image)
        self.file_label['text'] = file_list[next_index]
        # load the image from disk, convert it to grayscale, and detect
        # edges in it
        self.image = cv2.imread(next_image)
        self.clone = self.image.copy()
        self.line_img = self.image.copy()
        self.gap_img = self.image.copy()
        self.detections, self.found_dets = self.detection_funct(next_image)
        self.load_image(self.image,self.detections, self.found_dets)
        self.path = next_image

    def prev_image(self):
        
        self.reset_scales()
        
        filename = os.path.basename(self.path)
        #print(filename)
        dir_name = (os.path.dirname(self.path))
        file_list = os.listdir(dir_name)
        file_list.sort()
        prev_index = file_list.index(filename) - 1
        #print(str(next_index))
        #print(str(file_list[next_index]))
        prev_image = os.path.join(dir_name, file_list[prev_index])
        #print(next_image)
        self.file_label['text'] = file_list[prev_index]
        # load the image from disk, convert it to grayscale, and detect
        # edges in it
        self.image = cv2.imread(prev_image)
        self.clone = self.image.copy()
        self.line_img = self.image.copy()
        self.gap_img = self.image.copy()
        self.detections, self.found_dets = self.detection_funct(prev_image)
        self.load_image(self.image,self.detections, self.found_dets)
        self.path = prev_image
        
    def save_cmd(self):
        with open( os.path.basename(self.path)+".txt", "w") as text_file:
            text_file.write("MMSE Pentagon GUI Version=1\n")
            text_file.write("Process date = {0}\n".format(datetime.datetime.now()))
            text_file.write("corner count =  {0}\n".format(self.corner_total))
            text_file.write("line count =  {0}\n".format(self.line_total))
            text_file.write("min corner count =  {0}\n".format(self.corner_num))
            text_file.write("min quality =  {0}\n".format(self.quality))
            text_file.write("distance =  {0}\n".format(self.distance))
            text_file.write("detection_threshold =  {0}\n".format(self.detection_threshold))
            text_file.write("line_threshold =  {0}\n".format(self.line_threshold))
            text_file.write("corner list =  {0}\n".format(self.shi_c))
            text_file.write("lines list =  {0}\n".format(self.shi_lines))
            text_file.write("multipent =  {0}\n".format(self.multipent.get()))
            text_file.write("flag =  {0}\n".format(self.flag_var.get()))
            text_file.write("detections = {0}\n".format(self.found_dets))
        self.file_label['text'] = "TEXT FILE SAVED" + os.path.basename(self.path)+".txt"
            
    def apply_corners(self, image, corner_list):
        for i in corner_list:
            cv2.circle(image,(i[0],i[1]),5,255,-1)

    def apply_lines(self, image, lines, distilled_lines):
        for line in lines:
            start, end = line
            color = np.random.uniform(0,255,3)
            cv2.line(image, start, end, color=color, thickness=2)

    def apply_gaps(self, image, corner_list):
        gaps_list = corner_dets_methods.gapCorners(image,corner_list)
        for i in gaps_list:
            p1, p2 = i
            cv2.circle(image, (p1[0], p1[1]), 5, 255, -1)
            cv2.circle(image, (p2[0], p2[1]), 5, 255, -1)
            cv2.line(image, p1, p2, color=(0,0,255), thickness=1)

    def detection_funct(self, path):
        detections, found_dets = corner_dets_methods.find_contours(path,
                                                                   self.corner_num,
                                                                   self.quality,
                                                                   self.distance,
                                                                   self.detection_threshold)
        return detections, found_dets

    def corner_details(self,image, corner_image, line_image, gap_image, detections):
        self.shi_c, self.shi_lines, self.distilled_lines = corner_dets_methods.cornerMeths(image,
                                                                                            [detections],
                                                                                            self.corner_num,
                                                                                            self.quality,
                                                                                            self.distance)
        line_count = len(self.shi_lines)
        self.apply_corners(corner_image,self.shi_c)
        self.apply_lines(line_image, self.shi_lines, self.distilled_lines)
        self.apply_gaps(gap_image, self.shi_c)
        corner_count = len(self.shi_c)
        return corner_count, line_count

    def load_image(self, image, detections, found_dets):
        if name.parse_filename(self.file_label['text']) is not None:
            dat_id, dat_date, dat_fuyear, dat_score = name.parse_filename(self.file_label['text'])
        else:
            dat_id, dat_date, dat_fuyear, dat_score = 66, 66, 66, 66
        self.dat_id_label["text"] = dat_id
        self.dat_date_label["text"] = dat_date
        self.dat_fu_label["text"] = dat_fuyear
        self.dat_score_label["text"] = dat_score
        
        clone = image.copy()
        gap_img = image.copy()
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        line_img = np.zeros_like(clone)
        line_img.fill(255)

        if found_dets != []:
            if len(found_dets) > 1 and self.multipent.get() == 0:
                found_pent = max(found_dets, key=lambda x: x[2]) # best detetion
                self.corner_total, self.line_total = self.corner_details(gray, clone, line_img, gap_img, found_pent)
            elif len(found_dets) > 1 and self.multipent.get() == 1:
                self.corner_total = 0
                self.line_total = 0
                for i in found_dets:
                    self.corner_count, self.line_count = self.corner_details(gray, clone, line_img, gap_img, i)
                    self.corner_total += self.corner_count
                    self.line_total += self.line_count
                found_pent = max(found_dets, key=lambda x: x[2]) # best detetion
            elif len(found_dets) == 1:
                found_pent = found_dets[0]
                self.corner_total, self.line_total = self.corner_details(gray, clone, line_img, gap_img, found_pent)
            #print(found_pent)
            self.det_label['text'] = "Top Detection Score: " + str(found_pent[2])
            self.corner_count_label['text'] = 'Corner Count: ' + str(self.corner_total)
            self.line_count_label['text'] = 'Line Count: ' + str(self.line_total)
 
        else:
            self.det_label['text'] = "Top Detection Score: No Detection" 
            self.corner_count_label['text'] = 'Corner Count: No Detection'
            self.line_count_label['text'] = 'Line Count: No Detection'

        # OpenCV represents images in BGR order; however PIL represents
        # images in RGB order, so we need to swap the channels
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        clone = cv2.cvtColor(clone, cv2.COLOR_BGR2RGB)
        line_img = cv2.cvtColor(line_img, cv2.COLOR_BGR2RGB)
        gap_img = cv2.cvtColor(gap_img, cv2.COLOR_BGR2RGB)
        dets = cv2.cvtColor(detections, cv2.COLOR_BGR2RGB)

        # convert the images to PIL format...
        image = Image.fromarray(image)
        image = image.resize((250, 400), Image.ANTIALIAS)
        clone = Image.fromarray(clone)
        clone = clone.resize((250, 400), Image.ANTIALIAS)
        dets = Image.fromarray(dets)
        dets = dets.resize((250, 400), Image.ANTIALIAS)
        line_img = Image.fromarray(line_img)
        line_img = line_img.resize((250, 400), Image.ANTIALIAS)
        gap_img = Image.fromarray(gap_img)
        gap_img = gap_img.resize((250, 400), Image.ANTIALIAS)
        
        # ...and then to ImageTk format
        image = ImageTk.PhotoImage(image)
        clone = ImageTk.PhotoImage(clone)
        dets = ImageTk.PhotoImage(dets)
        line_img = ImageTk.PhotoImage(line_img)
        gap_img = ImageTk.PhotoImage(gap_img)

        # if the panels are None, initialize them
        if self.panelA is None or self.panelB is None:
            # the first panel will store our original image
            self.panelA = Label(image=image)
            self.panelA.image = image
            self.panelA.grid(row=1, column=3, rowspan = 20, padx=5)

            # while the second panel will store the detection map
            self.panelB = Label(image=dets)
            self.panelB.image = dets
            self.panelB.grid(row=1, column=4, rowspan = 20, padx=5)

            self.panelC = Label(image=clone)
            self.panelC.image = clone
            self.panelC.grid(row=1, column=5, rowspan = 20, padx=5)

            self.panelD = Label(image=line_img)
            self.panelD.image = line_img
            self.panelD.grid(row=1, column=6, rowspan = 20, padx=5)

            self.panelE = Label(image=gap_img)
            self.panelE.image = gap_img
            self.panelE.grid(row=22, column=3, rowspan=20, padx=5)



        # otherwise, update the image panels
        else:
            # update the pannels
            self.panelA.configure(image=image)
            self.panelB.configure(image=dets)
            self.panelC.configure(image=clone)
            self.panelD.configure(image=line_img)
            self.panelE.configure(image=gap_img)
            
            self.panelA.image = image
            self.panelB.image = dets
            self.panelC.image = clone
            self.panelD.image = line_img
            self.panelE.image = gap_img

    def reset_scales(self):
        self.corner_num = 30
        self.quality = 0.1
        self.distance = 4
        self.detection_threshold = 0.00
        self.line_threshold= .95
        self.corner_total = 0
        self.line_total = 0
        self.c_scale.set(self.corner_num)
        self.q_scale.set(self.quality)
        self.d_scale.set(self.distance)
        self.det_var.set( self.detection_threshold)
        self.line_var.set(self.line_threshold)
        self.multipent.set(0)
        self.flag_var.set(0)
        self.shi_c = []
        self.shi_lines = []     


root = Tk()
my_gui = PentGui(root)
root.mainloop()
