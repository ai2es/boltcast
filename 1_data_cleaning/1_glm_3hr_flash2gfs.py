


#################################################################
# Default plotting parameters
FIGURESIZE=(10,6)
FONTSIZE=  18
plt.rcParams['figure.figsize'] = FIGURESIZE
plt.rcParams['font.size'] = FONTSIZE
plt.rcParams['xtick.labelsize'] = FONTSIZE
plt.rcParams['ytick.labelsize'] = FONTSIZE
#################################################################


"""
This file grids the already downselected GLM lightning flashes. It grids the lightning to 3-hour intervals
to be included in the GFS temporal grid, that was already downselected. If there was no lightning for a
3-hour interval, then a grid of zeros is created. 
"""

def grid_lightning(sat, year, day, xmid, ymid, xedge, yedge):

    #declare empty lists for the flash extent density (fed)
    #and binary classifications 
    data_list = []
    time_list = []

    #set the storage directory
    glm_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast/GLM/%s/'%(sat)

    #create the file name
    glm_file = '%s_%s_%s_BC_df.nc'%(sat,year,day)

    month, day = get_day_mo(year,int(day))
    #load the xarray dataset if the file exists, otherwise move to the next day 
    try:

        ds = xr.open_dataset(glm_dir+glm_file,engine='netcdf4')
        #convert the dataset to a dataframe
        df = ds.to_dataframe()

        #get the days for processing out the first flashes from the previous day
        temp_days = df['day'].drop_duplicates().values

        #track if it is the first day
        first_grid=True
        idx=0
        
        for hr in range(24):

            valid_time = datetime.datetime(year=int(year),day=int(day),month=int(month),hour=hr)
            
            if idx==0:
                df_3hr = df.loc[df['hour']==hr]

                #append the last days lightning strikes to the 00 hour due to data writing error
                if hr==0 and len(temp_days)>1:
                    df_3hr = pd.concat([df_3hr, df.loc[df['day']==temp_days[0]]])
            else:
                df_3hr = pd.concat([df_3hr,df.loc[df['hour']==hr]])
            idx = idx+1
        

            if idx==3:
                idx=0

                #declare storage valid time
                file_valid_time = datetime.datetime(year=int(year),day=int(day),month=int(month),hour=hr-2)

                if len(df_3hr)>0:#there is lightning somewhere in CONUS over the last three hours

                    #get the number of flashes per GFS grid cell
                    fed = boxbin(df_3hr['flash_lon']+360,df_3hr['flash_lat'],xedge,yedge,mincnt=0)
                    fed = np.flip(fed.transpose(),axis=0) #need to make sure the lightning moves the right way
                    
                    #set the bindary classification
                    binary = (fed>=1).filled(fill_value=0)
                    time_list.append(file_valid_time)

                    #store the counts and binary classification into an xarray dataset
                    ds = xr.Dataset(
                            data_vars=dict(binary_ltg = (["lat","lon"],binary),
                                            FED = (["lat","lon"],fed)),
                            coords=dict(lon=(["lon"],xmid),lat=(["lat"],ymid)),
                            attrs=dict(description="Binary classification and FED of GLM on GFS grid. 256x128.")
                        )
                    data_list.append(ds)

                else:#there is no lightning

                    print("no lightning, ", sat, year, day, hr)
                    time_list.append(file_valid_time)
                    
                    #there is no lightning 
                    binary = np.zeros([128,256])
                    fed = np.zeros([128,256])

                    #store the counts and binary classification into an xarray dataset
                    ds = xr.Dataset(
                            data_vars=dict(binary_ltg = (["lat","lon"],binary),
                                            FED = (["lat","lon"],fed)),
                            coords=dict(lon=(["lon"],xmid),lat=(["lat"],ymid)),
                            attrs=dict(description="Binary classification and FED of GLM on GFS grid. 256x128.")
                        )
                    
                    data_list.append(ds)

        #generate a new xarray dataset from the dataset list for each day
        ds2 = xr.concat(data_list, data_vars='all', dim='time')
        ds2 = ds2.assign_coords(time=time_list)
        ds2 = ds2.sortby('time')
        return ds2

    except FileNotFoundError:
        print("file not found")
        glm_file = '%s_%s_%s_BC_df.nc'%(sat,year,day)
        print(glm_file)
        return []
        

##################################
def main():

    #create a parser for the initialization time and the forecast hour for easy HPC use
    parser = argparse.ArgumentParser()
    parser.add_argument('--year',type=str,default='2019',help='The year')
    args = vars(parser.parse_args())
    
    #set the satellite and year
    year = args['year']
    sat = 'G16'
    if year=='2019' or year=='2020' or year=='2021' or year=='2022':
        sat1 = 'G17'
    else:
        sat1 = 'G18' 

    #build the array of julian days. add 366 for leap year
    j_days = []
    for i in range(1,366):
        j_days.append(f"{i:03}")
    if year=='2020' or year=='2024':
        j_days.append('366')

    #load the gfs grid
    static_inputs = pickle.load(open('./Archive/static_inputs.pkl','rb'))
    lat = static_inputs['lat']#2D
    lon = static_inputs['lon']#2D

    #add the next grid point so the binning algorithm 
    #size is 256x128
    bin_lat = np.concatenate([lat[:,0],[53.25]])
    bin_lon = np.concatenate([lon[0,:],[298]])

    #store the sorted grid
    xedge = np.sort(bin_lon)
    yedge = np.sort(bin_lat)

    xmid = [] #Blank array
    ymid = [] #Blank array

    #calcuate the midpoints for the sorting algorithm
    i=0
    while(i < len(xedge)-1):
        xmid.append((xedge[i]+xedge[i+1])/2) #Calculate and append midpoints
        i+=1 
    i=0
    while(i < len(yedge)-1):
        ymid.append((yedge[i]+yedge[i+1])/2) #Calculate and append midpoints
        i+=1

    #loop through the files
    for i,day in enumerate(j_days):

        #bookkeeping/progress bar
        print(i,len(j_days),year)

        #grid the lightning using the self define function above
        ds_east = grid_lightning(sat, year, day, xmid, ymid, xedge, yedge) #xarray datasets
        ds_west = grid_lightning(sat1, year, day, xmid, ymid, xedge, yedge) #xarray datasets

        #save the datasets
        e_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast/GLM/'+sat+'/'
        e_file = '%s_%s_%s_BC_ds.nc'%(sat,year,day)
        ds_east.to_netcdf(e_dir+e_file,engine='netcdf4')

        w_dir = '/ourdisk/hpc/ai2es/bmac87/BoltCast/GLM/'+sat1+'/'
        w_file = '%s_%s_%s_BC_ds.nc'%(sat1,year,day)
        if ds_west:
            ds_west.to_netcdf(w_dir+w_file,engine='netcdf4')

        del ds_east, ds_west

if __name__=='__main__':
    main()