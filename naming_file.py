import re
import os



def parse_filename(filename):
    cwd = os.getcwd()
    data_file = os.path.join(cwd, 'mmse_q20.dat')

    try:
        proj_find = re.compile('\d{8}')
        proj_id = proj_find.findall(filename)[0]
        date_fine = re.compile('\d+-\d+-\d+')
        date = date_fine.findall(filename)[0]
        
        with open(data_file) as f:
            for line in f:
                line = line.split("|")
                #print(line)
                dat_id = line[0]
                dat_date = line[2]
                dat_date = dat_date[0:2] + "-" +  dat_date[2:4] + "-"  + dat_date[-2:]
                dat_fuyear = line[1]
                dat_score = line[3]
                if dat_id == proj_id and date == dat_date:
                    print("Found in DATA q20")
                    print(dat_id)
                    print(dat_date)
                    print(dat_fuyear)
                    print(dat_score)
                    return dat_id, dat_date, dat_fuyear, dat_score
            

    except:
        return 66, 66, 66, 66

    
