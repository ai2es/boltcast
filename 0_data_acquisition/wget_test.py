import xarray as xr
import os
import wget

glm_base_url = 'https://lightningdev.umd.edu/feng_data_sharing/113_g16_glmgrid_arch/'
glm_stor = '/scratch/bmac87/GLM/test/'
yr='2019'
jul_day='001'
temp_url = glm_base_url+yr+'/'

filename = wget.download(temp_url+yr+jul_day+'/*.nc',out=glm_stor)
