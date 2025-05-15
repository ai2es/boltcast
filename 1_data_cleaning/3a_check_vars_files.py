import os


var_list = ['cape','lifted_idx','reflectivity','precip_rate',
                'rain_q','ice_q','snow_q','graupel_q','w']

days = {'day_1':['f000','f003','f006','f009','f012','f015','f018','f021'],
        'day_2':['f024','f027','f030','f033','f036','f039','f042','f045'],
        'day_3':['f048','f051','f054','f057','f060','f063','f066','f069'],
        'day_4':['f072','f075','f078','f081','f084','f087','f090','f093']}

init_times=['00Z','06Z','12Z','18Z']

for d,day in enumerate(days):
    f_hours = days[day]
    for v,var in enumerate(var_list):
        for f_hour in f_hours:
            for init_time in init_times:

                save_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast_ourdisk/data/3_vars_only/'+var+'/'
                fname = '%s_%s_%s_%s.nc'%(day,var,f_hour,init_time)
                if os.path.isfile(save_dir+fname)==False:
                    print(fname)