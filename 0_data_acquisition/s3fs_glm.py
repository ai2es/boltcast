import s3fs
import os
import glob
import shutil
import argparse

# aws s3 cp s3://noaa-goes16/<Product>/<Year>/<Day of Year>/<Hour>/<Filename> . --no-sign-request
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--year',type=int,default=6,required=True,help='The year to get the GLM data for, via slurm array')
    parser.add_argument('--sat',type=str,default='G16',help='The satellite to get')
    parser.add_argument('--download',action='store_true',default=False,help='Flag to download the data')
    
    args = vars(parser.parse_args())

    if args['year']==1:
        yr='2019'
    elif args['year']==2:
        yr='2020'
    elif args['year']==3:
        yr='2021'
    elif args['year']==4:
        yr='2022'
    elif args['year']==5:
        yr='2023'
    elif args['year']==6:
        yr='2024'
    else:
        yr='2025'

    print(yr)
    sat = args['sat']
    print(sat)

    s3 = s3fs.S3FileSystem(anon=True)

    #declare the hrs
    hrs = []
    for i in range(0,24):
        hrs.append(f"{i:02}")

    days_noleap = []
    for i in range(1,366):
        days_noleap.append(f"{i:03}")

    days_leap = []
    for i in range(1,367):
        days_leap.append(f"{i:03}")

    if yr=='2020' or yr=='2024':
        days = days_leap
    else:
        days = days_noleap

    for t,d in enumerate(days):
        if t>=0:
            for i,h in enumerate(hrs):
                print(d,h)
                if sat=='G17':
                    mybucket = 's3://noaa-goes17/GLM-L2-LCFA/'+yr+'/'+d+'/'+h+'/'
                elif sat=='G18':
                    mybucket = 's3://noaa-goes18/GLM-L2-LCFA/'+yr+'/'+d+'/'+h+'/'
                else:
                    mybucket = 's3://noaa-goes16/GLM-L2-LCFA/'+yr+'/'+d+'/'+h+'/'
                
                try:
                    files = s3.ls(mybucket)
                    save_path = '/scratch/bmac87/GLM/'+sat+'/'+yr+'/'+d+'/'
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    if args['download']:
                        print(mybucket)
                        s3.get(files,lpath=save_path)
                    else:
                        pass
                except FileNotFoundError:
                    print(mybucket,' notfound')

if __name__ == "__main__":
    main()