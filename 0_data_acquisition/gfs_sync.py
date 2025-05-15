import os
import glob

hours = []
for i in range(65):
    hours.append(i*3)

f_hours = []
for hr in hours:
    f_hours.append('f'+f"{hr:03}")

init_times = ['00Z','06Z','12Z','18Z']
for init_time in init_times:
    for h,f_hour in enumerate(f_hours):

        grib_dir = '/ourdisk/hpc/ai2es/datasets/GFS/'+init_time+'/'+f_hour+'/'
        print(grib_dir)
        if os.path.exists(grib_dir):
            print('directory exists')
            yr = '2019'
            for mo in ['06']:
                glob_call = grib_dir+'gfs.0p25.'+yr+mo+'*.'+f_hour+'.grib2'
                print(glob_call)
                files = glob.glob(glob_call)
                for file in files:
                    os.remove(file)

