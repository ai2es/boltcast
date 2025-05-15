import wget
from urllib.error import HTTPError
import os
import argparse
import pickle

def main():

    print("wget_gfs2.py")
    urls = pickle.load(open('bad_urls.pkl','rb'))

    for u,url in enumerate(urls):
        if u<=10:
            print(url)
            fname = url[-30:]
            print(fname)
            f_hour = fname[-10:-6]
            print(f_hour)
            init_time = fname[-13:-11]+'Z'
            print(init_time)

            target_dir = '/scratch/bmac87/'+init_time+'/'+f_hour+'/'
            if not os.path.isdir(target_dir):
                os.makedirs(target_dir)
                
            try:
                if os.path.isfile(target_dir+fname):
                    pass
                else:
                    filename=wget.download(url,out=target_dir)
            except HTTPError:
                print('bad_url: ',url)

    # gfs_base_url = 'https://data.rda.ucar.edu/ds084.1/2019/20190101/gfs.0p25.2019010100.f180.grib2'

if __name__ == "__main__":
    main()





