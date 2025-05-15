import os
import glob
import shutil

#declare the forecast hour based on the slurm array ID
hours = []#int
for i in range(33):
    hours.append(i*3)

f_hours = []#string
for hr in hours:
    f_hours.append('f'+f"{hr:03}")
print(f_hours)

#declare the months
mos = []
for i in range(1,13):
    mos.append(f"{i:02}")

yrs = ['2019','2020','2021','2022','2023','2024']
init_times = ['00','06','12','18']

julian_days = []
yrs_dict = {}
for yr in yrs:#for each year
    julian_day = 1 #start counting a julian day
    mos_dict = {}
    for mo in mos: #for each month
        days = []
        mo_jul = []
        if mo=='01' or mo=='03' or mo=='05' or mo=='07' or mo=='08' or mo=='10' or mo=='12':
            for t in range(1,32):
                days.append(f"{t:02}")
                julian_days.append(f"{julian_day:03}")
                mo_jul.append(f"{julian_day:03}")
                julian_day = julian_day+1
        elif mo=='02':
            if yr=='2020':
                for t in range(1,30):
                    days.append(f"{t:02}")
                    julian_days.append(f"{julian_day:03}")
                    mo_jul.append(f"{julian_day:03}")
                    julian_day = julian_day+1
            else:
                for t in range(1,29):
                    days.append(f"{t:02}")
                    julian_days.append(f"{julian_day:03}")
                    mo_jul.append(f"{julian_day:03}")
                    julian_day = julian_day+1
        else:
            for t in range(1,31):
                days.append(f"{t:02}")
                julian_days.append(f"{julian_day:03}")
                mo_jul.append(f"{julian_day:03}")
                julian_day = julian_day+1
        mos_dict.update({mo:days})
        mos_dict.update({mo+'_jul':mo_jul})
    yrs_dict.update({yr:mos_dict})

for f_hour in f_hours:
    for init_time in init_times:

        gfs_dir = '/ourdisk/hpc/ai2es/datasets/GFS/'+init_time+'Z/'+f_hour+'/'
        if not os.path.isdir(gfs_dir):
            os.makedirs(gfs_dir)

        globus_dir = '/ourdisk/hpc/ai2es/datasets/GFS/20240519/'
        glob_call = globus_dir+'*'+init_time+'.'+f_hour+'.grib2'
        print(glob_call)
        files = glob.glob(glob_call)
        for file in files:
            fname = file[-30:]
            print(fname)
            shutil.move(file, gfs_dir+fname)

# for f_hour in f_hours:
#     for init_time in init_times:
#         globus_dir = '/ourdisk/hpc/ai2es/datasets/GFS/GFS_v2/GFS/'+init_time+'Z/'+f_hour+'/'
#         files = os.listdir(globus_dir)
#         if len(files)>0:
#             print(globus_dir)
# print("end of directory search")
