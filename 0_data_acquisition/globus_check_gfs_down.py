import os
import shutil

init_time = '18Z'

year = '2020'
mos = ['01','02','03','04','05','06','07','08','09','10','11','12']

hours = []
for i in range(65):
    hours.append(i*3)

f_hours = []#string
for hr in hours:
    f_hours.append('f'+f"{hr:03}")
print(f_hours)

for f_hour in f_hours:
    gfs_dir = '/scratch/bmac87/BoltCast_scratch/GFS_processing/'+init_time+'/'
    gfs_dir = gfs_dir+f_hour+'/'
    print(gfs_dir)
    for mo in mos:
        fname = 'BC_GFS_'+year+mo+'.nc'
        if os.path.isfile(gfs_dir+fname)==False:
            print(fname, init_time, f_hour)