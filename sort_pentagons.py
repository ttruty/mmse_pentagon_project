import os
import shutil
import corner_dets_methods

path = r'C:\Users\KinectProcessing\Desktop\test_mmse_classify\extracted_images\mmse_data'
for _,_, file_list in os.walk(path):
    for file in file_list:
        filename = os.path.join(path, file)
        print(filename)
        detections, found_dets = corner_dets_methods.find_contours(filename, 50, .2, 2, 0)
        print(len(found_dets))
        if found_dets != [] and len(found_dets) == 1:
            shutil.copy(filename, r'C:\Users\KinectProcessing\Desktop\new_potitves')
            print("copied")
