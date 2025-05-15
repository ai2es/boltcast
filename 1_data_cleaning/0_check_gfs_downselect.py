import os

yr = '2019'

#declare the forecast hour based on the slurm array ID
hours = []#int
for i in range(33):
    hours.append(i*3)

f_hours = []#string
for hr in hours:
    f_hours.append('f'+f"{hr:03}")

init_times = ['00Z','06Z','12Z','18Z']

#declare the months based on the years
if yr=='2019':
    mos = ['07','08','09','10','11','12']
elif yr=='2024':
    mos = ['01','02','03','04','05','06']
else:
    mos=['01','02','03','04','05','06','07','08','09','10','11','12']

job_dict = {}
job_id = 1
bad_jobs = []
for init_time in init_times:
    for f_hour in f_hours:
        for mo in mos:
            job_dict[job_id] = [init_time,f_hour,yr,mo]
            
            gfs_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/data/0_GFS_downselect/'
            gfs_dir = gfs_dir+'%s/%s/'%(init_time,f_hour)
            fname = 'BC_GFS_%s%s.nc'%(yr,mo)

            #check if the file is there 
            if os.path.isfile(gfs_dir+fname)==False:
                bad_jobs.append(job_id)
            job_id = job_id+1

print(bad_jobs)
job_str=''
for job in bad_jobs:
    job_str = job_str+str(job)+','
print(job_str)