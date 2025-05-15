import wget
from urllib.error import HTTPError
import os


"""
This file downloads the GLM data from the webpage provided. 
"""

#declare the years of interest
yrs = ['2019',
       '2020',
       '2021',
       '2022']

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

    
glm_base_url = 'https://lightningdev.umd.edu/feng_data_sharing/113_g16_glmgrid_arch/'
glm_stor = '/scratch/bmac87/GLM/G16/'

for yr in yrs:
    yr_dict = yrs_dict[yr]
    temp_url = glm_base_url+yr+'/'

    for mo in mos:
        jul_days = yr_dict[mo+'_jul']
        days = yr_dict[mo]
        for i,jul_day in enumerate(jul_days):
            temp_url2 = temp_url+yr+jul_day+'/OR_GLM-L2-GLMF-M3_G16_e'+yr+mo+days[i]
            fbase = 'OR_GLM-L2-GLMF-M3_G16_e'+yr+mo+days[i]

            for hr in hrs:
                temp_url3 = temp_url2+hr
                fbase1 = fbase+hr
                for minute in mins:
                    temp_url4 = temp_url3+minute+'00.nc'
                    fbase2 = fbase1+minute+'00.nc'                    
                    try:
                        if os.path.isfile(glm_stor+fbase2):
                            pass
                            # print(fbase2, ' already exists')
                        else:
                            # filename = wget.download(temp_url4, out=glm_stor)
                            if minute=='59' and hr=='23':
                                if i<len(jul_days):
                                    temp_url5 = temp_url+yr+jul_days[i+1]+'/OR_GLM-L2-GLMF-M3_G16_e'+yr+mo+days[i+1]+'000000.nc'
                                    print(temp_url5)
                                    # filename2 = wget.download(temp_url5,out=glm_stor)
                    except HTTPError:
                        pass
                        # print(fbase2, 'HTTP error')
                    except IndexError:
                        pass
                        # print(fbase2,' Index Error')
                        
           

# # test_url = 'https://lightningdev.umd.edu/feng_data_sharing/213_g17_glmgrid_arch/2019/2019001/OR_GLM-L2-GLMF-M3_G17_e20190101000100.nc'
#eYYYYMMDDHHMMSS.nc
# # filename = wget.download(test_url, out=glm17_stor)