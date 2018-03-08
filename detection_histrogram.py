import re, os, cv2
import corner_dets_methods

def parse_filename(filename):
    
    proj_find = re.compile('\d{8}')
    if proj_find.findall(filename) != []:
        proj_id = proj_find.findall(filename)[0]
    else:
        proj_id = 'FILENAME_ERROR'
        
    date_find = re.compile('\d+-\d+-\d+')
    if date_find.findall(filename) != []:
        date = date_find.findall(filename)[0]
    else:
        date = "DATE_ERROR"
    print(filename)
    fu_reg_list = [re.compile('FU\d+'), re.compile('FU \d+'), re.compile('F-U\d+'), re.compile('F-U \d+'),  re.compile('Baseline'), re.compile('BL')]
    if any(r.findall(filename) for r in fu_reg_list):
        for r in fu_reg_list:
            if r.findall(filename) != []:
               fu_year = r.findall(filename)[0]
               print(fu_year)
               if fu_year == 'Baseline' or fu_year == 'BL':
                   fu_year = '00'
               digit = re.compile(r'\d+')
               fu = digit.findall(fu_year)
               fu_y = (fu[0].zfill(2))
    else:
        fu_y = '66'
               

    #print(proj_id)
    #print(date)
    with open(r'F:\pent_python\mmse_q20.dat') as f:
        for line in f:
            line = line.split("|")
            #print(line)
            dat_id = line[0]
            dat_date = line[2]
            dat_date = dat_date[0:2] + "-" +  dat_date[2:4] + "-"  + dat_date[-2:]
            dat_fuyear = line[1]
            dat_score = line[3]
            if dat_id == proj_id and date == dat_date:
##                    print("Found in DATA q20")
##                    print(dat_id)
##                    print(datf_date)
##                    print(dat_fuyear)
##                    print(dat_score)
                print('Used date')
                return dat_id, dat_date, dat_fuyear, dat_score
            elif dat_id == proj_id and dat_fuyear == fu_y:
                print('Used fu_year')
                return dat_id, dat_date, dat_fuyear, dat_score
            

def detection_funct(path):
        detections, found_dets = corner_dets_methods.find_contours(path, 20, 0.1, 15, -3.0)
        return detections, found_dets

path = r'F:\pent_python\pent_bins\neg_score_dir'

with open("negs_detection_histogram.txt", 'w') as f:
    for file in os.listdir(path):
        print(file)
        try:
            dat_id, dat_date, dat_fuyear, dat_score = parse_filename(file)
        except TypeError:
            dat_id, dat_date, dat_fuyear, dat_score = 66,66,66,66
##            print(dat_id)
##            print(dat_date)
##            print(dat_fuyear)
##            print(dat_score)
        full_file = os.path.join(path,file)
        det, found_dets = detection_funct(full_file)
        #print(found_dets)
        if found_dets != []:
            line = str(dat_id) + "|" + str(dat_date) + "|" + str(dat_fuyear) + "|" + str(dat_score) + "|" + str(found_dets[0][2]) + "\n" 
        else:
            line = str(66) + "|" + str(66) + "|" + str(66) + "|" + str(66) + "|" + str(66) + "\n"
        f.write(file + "|" + line)
    
    
