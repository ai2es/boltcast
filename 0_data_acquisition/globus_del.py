import os
import shutil
import glob

print("hello world")

#declare the months
mos = []
for i in range(1,13):
    mos.append(f"{i:02}")

julian_days = []

yrs = ['2024']
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

globus_base = '/scratch/bmac87/Globus_GFS/'

if yrs[0]=='2019':
    mos = ['06','07','08','09','10','11','12']
elif yrs[0]=='2024':
    mos = ['01','02','03','04','05','06']
else:
    mos = mos

inits_times = ['00','06','12','18']

remove_fhours = ['f099','f102','f105','f108','f111','f114',
                'f117','f120','f123','f126','f129','f132',
                'f135','f138','f141','f144','f147','f150',
                'f153','f156','f159','f162','f165','f168',
                'f171','f174','f177','f180','f183','f186','f189','f192',
                'f195','f198','f201','f204','f207','f210',
                'f213','f216','f219','f222','f225','f228',
                'f231','f234','f237','f240','f246','f252','f258','f264','f270','f276','f282',
                'f288','f294','f300','f306','f312','f318','f324','f330','f336','f342','f348','f354','f360',
                'f366','f372','f378','f384']

yr = yrs[0]
for mo in mos:
    days = yrs_dict[yr][mo]
    for day in days:
        # print(yr+mo+day)
        for f_hour in remove_fhours:
            globus_call = globus_base+yr+'/'+yr+mo+day+'/gfs.0p25.'+yr+mo+day+'*.'+f_hour+'.grib2'
            remove_files = glob.glob(globus_call)
            print(globus_call)
            for file in remove_files:
                os.remove(file)