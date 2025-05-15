import os
import shutil
import glob

print("shutil_mv_grib.py")

#declare the forecast hour based on the slurm array ID
hours = []#int
for i in range(33):
    hours.append(i*3)

f_hours = []#string
for hr in hours:
    f_hours.append('f'+f"{hr:03}")
print(f_hours)

init_times = ['00Z','06Z','12Z','18Z']

for init_time in init_times:
    for f_hour in f_hours:

        source_dir = '/scratch/bmac87/GFS/'+init_time+'/'+f_hour+'/'
        target_dir = '/ourdisk/hpc/ai2es/datasets/GFS/'+init_time+'/'+f_hour+'/'
        
        if not os.path.isdir(target_dir):
            os.makedirs(target_dir)

        glob_call = source_dir+'*.grib2'
        files = glob.glob(glob_call)

        for file in files:
            fname = file[-30:]
            print(fname)
            shutil.move(file, target_dir+fname)

