import wget
from urllib.error import HTTPError
import os


"""
This file downloads the ENTLN data from the webpage provided. 
"""

#declare the years of interest
yrs = ['2024']

#declare the months
mos = []
for i in range(1,13):
    mos.append(f"{i:02}")

#declare the hrs
hrs = []
for i in range(0,24):
    hrs.append(f"{i:02}")

#declare the minutes
mins = []
for i in range(0,60):
    mins.append(f"{i:02}")

#build an array with the daily numbers
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
            if yr=='2020' or yr=='2024':
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

base_url = 'http://lxarchive.ensb.us/media/1724787097/University_of_Oklahoma_-_TLN_pulses_-_010124_05_UTC_to_080224_05_UTC_pulse'
stor_dir = '/ourdisk/hpc/ai2es/bmac87/OG_datasets/ENTLN/'

#'http://lxarchive.ensb.us/media/1724787097/University_of_Oklahoma_-_TLN_pulses_-_010124_05_UTC_to_080224_05_UTC_pulse'#20240101.csv

for yr in yrs:
    yr_dict = yrs_dict[yr]
    for mo in mos:
        days = yr_dict[mo]
        for i,day in enumerate(days):
            if i<=99999999999:
                temp_url = base_url+yr+mo+day+'.csv'
                fname = yr+mo+day+'.csv'                    
                try:
                    if os.path.isfile(stor_dir+fname):
                        print(fname, ' already exists')
                    else:
                        filename = wget.download(temp_url,out=stor_dir+fname)
                except HTTPError:
                    print(temp_url, 'HTTP error')
                except IndexError:
                    print(temp_url,' Index Error')