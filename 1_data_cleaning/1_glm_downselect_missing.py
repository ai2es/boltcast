import os
import glob

"""
This code checks for missing files of the GLM_downselect process
"""

sats = ['G16','G17','G18']
for sat in sats:
    print(sat)
    if sat=='G16':
        years = ['2019','2020','2021','2022','2023','2024']
    elif sat=='G17':
        years = ['2019','2020','2021','2022']
    else:
        years = ['2023','2024']
    
    for year in years:
        print(sat, year)
        j_days = []
        for i in range(1,366):
            j_days.append(f"{i:03}")

        if year=='2020' or year=='2024':
            print('leap year')
            j_days.append('366')
        else:
            j_days = j_days

        for day in j_days:
            nc_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast/GLM/'+sat+'/'   
            fname = sat+'_'+year+'_'+day+'_BC_df.nc'
            if not os.path.isfile('/ourdisk/hpc/ai2es/bmac87/BoltCast/GLM/'+sat+'/'+fname):
                print('missing: '+fname)
    
