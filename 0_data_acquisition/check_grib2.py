import os
import shutil
import pickle

def main():

    print('check_grib2.py main')

    #declare the forecast hour based on the slurm array ID
    hours = []#int
    for i in range(33):
        hours.append(i*3)
    print(hours)

    f_hours = []#string
    for hr in hours:
        f_hours.append('f'+f"{hr:03}")
    print(f_hours)

    yrs = ['2019','2020','2021','2022','2023','2024']
    print(yrs)

    yrs_dict = {}
    for yr in yrs:#for each year
        if yr=='2019':
            mos = ['07','08','09','10','11','12']
        
        elif yr=='2024':
            mos = ['01','02','03','04','05','06']
        else:
            mos=[]
            for i in range(1,13):
                mos.append(f"{i:02}")
        mos_dict = {}
        for mo in mos: #for each month
            days = []
            mo_jul = []
            if mo=='01' or mo=='03' or mo=='05' or mo=='07' or mo=='08' or mo=='10' or mo=='12':
                for t in range(1,32):
                    days.append(f"{t:02}")
            elif mo=='02':
                if yr=='2020' or yr=='2024':
                    for t in range(1,30):
                        days.append(f"{t:02}")
                else:
                    for t in range(1,29):
                        days.append(f"{t:02}")
            else:
                for t in range(1,31):
                    days.append(f"{t:02}")
            mos_dict.update({mo:days})
        yrs_dict.update({yr:mos_dict})

    init_times = ['00','06','12','18']
    print(init_times)

    bad_urls = []
    for yr in yrs:#for each year
        if yr=='2019':
            mos = ['07','08','09','10','11','12']
        elif yr=='2024':
            mos = ['01','02','03','04','05','06']
        else:
            mos=[]
            for i in range(1,13):
                mos.append(f"{i:02}")
        for mo in mos:
            days = yrs_dict[yr][mo]
            print(yr,mo)
            print(mo,days)
            for day in days:
                for init_time in init_times:
                    for f_hour in f_hours:
                        grib_dir = '/ourdisk/hpc/ai2es/datasets/GFS/%sZ/%s/'%(init_time,f_hour)
                        fbase = 'gfs.0p25.'
                        datestr = yr+mo+day+init_time+'.'
                        fname = fbase+datestr+f_hour+'.grib2'
                        if os.path.isfile(grib_dir+fname)==False:
                            # print('no_file, ',fname)
                            gfs_url = 'https://data.rda.ucar.edu/ds084.1/'+yr+'/'+yr+mo+day+'/'+fname
                            bad_urls.append(gfs_url)
                        #gfs.0p25.2021092306.f000.grib2
    print('number of bad_urls: ',len(bad_urls))
    pickle.dump(bad_urls,open('bad_urls.pkl','wb'))
    
    url_txt = open('bad_urls.txt','w')
    for u, url in enumerate(bad_urls):
        url_txt.write(str(u)+'\n')
        url_txt.write(url+'\n')
    url_txt.close()

if __name__=='__main__':
    main()
