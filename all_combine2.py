# -*- coding: utf-8 -*-
"""
Created on Mon Nov 15 23:02:22 2021

@author: WavingFlag
"""


import os
from pyhdf import HDF, VS, SD
from netCDF4 import Dataset
import numpy as np
import numpy.ma as ma
import pandas as pd
import multiprocessing

pathdl = '/data3/wangfx/cloudsat/NEW/chun'
pathdl2 = '/data3/wangfx/modis/NEW/chun'
pathdl3 = '/data3/wangfx/cloudsat/NEW3label/chun'
class reader2:
    '''用于读取CloudSat R05 2B产品的类.'''

    def __init__(self, fname):
        '''打开HDF,Vdata table,和scientific dataset.'''

        self.hdf = HDF.HDF(fname, HDF.HC.READ)
        self.vs = self.hdf.vstart()
        self.sd = SD.SD(fname, SD.SDC.READ)

    def attach_vdata(self, varname):
        '''从Vdata table中读取一个变量的所有记录.'''

        vdata = self.vs.attach(varname)
        data = vdata[:]
        vdata.detach()

        return data

    def scale_and_mask(self, data, varname):
        '''依据变量的factor进行放缩,并根据valid_range来mask掉填充值.'''

        factor = self.attach_vdata(f'{varname}.factor')[0][0]
        valid_min, valid_max = \
            self.attach_vdata(f'{varname}.valid_range')[0][0]

        invalid = np.logical_or(data <= valid_min, data >= valid_max)
        data = ma.array(data, mask=invalid)
        data = data / factor

        return data
    
    
    def read_geo(self, process=True):
        '''读取经纬度和地形高度.参数process指定是否放缩并mask地形高度.'''

        lon = np.array(self.attach_vdata('Longitude'))[:,0]
        lat = np.array(self.attach_vdata('Latitude'))[:,0]
        elv = np.array(self.attach_vdata('DEM_elevation'))[:,0]

        if process:
            elv = self.scale_and_mask(elv, 'DEM_elevation')
        
        return lon, lat, elv
    
    def read_time(self, datetime=True):
        '''读取每个数据点对应的时间.
        
        datetime=True: 返回所有数据点的日期时间组成的DatetimeIndex.
        datetime=False: 返回所有数据点相对于第一个点所经过的秒数组成的numpy数组.
        '''

        seconds = np.array(self.attach_vdata('Profile_time'))[:,0]

        if datetime:
            TAI = self.attach_vdata('TAI_start')[0][0]
            start = pd.to_datetime('1993-01-01') + pd.Timedelta(seconds=TAI)
            offsets = pd.to_timedelta(seconds, unit='s')
            time = pd.date_range(start=start, end=start, periods=offsets.size)
            time = time + offsets
            return time
        else:
            return seconds
    
    def read_sds(self, varname, process=True):
        '''读取scientific dataset.参数process指定是否放缩并mask.'''

        data = self.sd.select(varname)[:]
        if process:
            data = self.scale_and_mask(data, varname)
        
        return data
    
    def read_single(self, process=False):
        '''读取经纬度和地形高度.参数process指定是否放缩并mask地形高度.'''

        lon = np.array(self.attach_vdata('Cloud_Water_Path'))[:,0]
        lat = np.array(self.attach_vdata('Latitude'))[:,0]
        # elv = np.array(self.attach_vdata('DEM_elevation'))[:,0]

        # if process:
        #     elv = self.scale_and_mask(elv, 'DEM_elevation')
        
        return lon,lat

    def close(self):
        '''关闭HDF文件.'''

        self.vs.end()
        self.sd.end()
        self.hdf.close()
filelist=[]
scenes=[]
tau_c=[]
p_top=[]
r_e=[]
twp=[]
modis_mask=[]
lat=[]
lon=[]
labels=[]
def Sum_matrix(matrix):
    sum=0
    for i in matrix:
        for j in i:
            sum+=j
    return sum


for file in os.listdir(pathdl):
    orname=os.path.join(file)
    file2=file.split('_')
    file2[3]='MOD06-1KM-AUX'
    file3='_'.join(file2)
    file2[3]='2B-CLDCLASS'
    file4='_'.join(file2)
    fnamecloudsat=pathdl+'//'+orname,"utf-8"
    fnamemodis=pathdl2+'//'+file3,"utf-8"
    fnamelabel=pathdl3+'//'+file4,"utf-8"
    
    try:
        f = reader2(fnamemodis[0])
        f3 = reader2(fnamelabel[0])
    except:
        print("未找到匹配")
        continue
    f2 = reader2(fnamecloudsat[0])
    CWP = np.array(f.sd.select('Cloud_Water_Path')[:,7],dtype=float)
    Tc = np.array(f.sd.select('Cloud_Optical_Thickness')[:,7],dtype=float)
    re = np.array(f.sd.select('Cloud_Effective_Radius')[:,7],dtype=float)
    Ptop = np.array(f.sd.select('Cloud_top_pressure_1km')[:,7],dtype=float)
    mylat = f.sd.select('MODIS_latitude')[:,7]
    mylon = f.sd.select('MODIS_longitude')[:,7]
    # mask127 = np.array(f.sd.select('Cloud_Mask_1km')[0,:,7],dtype=int)
    mask127 = f.sd.select('Cloud_Mask_1km')[0,:,7]
    RR = f2.read_sds('Radar_Reflectivity')
    Label = np.array(f3.sd.select('cloud_scenario'),dtype=int)&0b0001100000011110
    for i in range(len(Label)):
        for j in range(len(Label[0])):
            if(Label[i][j]<1000):
                Label[i][j]=9
                print(i,j)
            else:
                Label[i][j]=(Label[i][j]>>1)-1024
    # lon, lat = f2.read_geo()
    # print(CWP)
    lengthall=len(mask127)
    length64=lengthall//64
    flag= [1 for i in range (lengthall)]
    mycloudflag= [0 for i in range (lengthall)]
    my64= [1 for i in range (64)]
    my64_0= [0 for i in range (64)]

    for i in range(lengthall):
        if (mask127[i]&0b00000110)==0:
            mycloudflag[i]=1
        if CWP[i]==-9999:
            mycloudflag[i]=0
            CWP[i]=120.2
        if re[i]==-9999:
            #缺测判断
            mycloudflag[i]=0
            re[i]=2132.8
        if Tc[i]==-9999:
            #缺测判断
            mycloudflag[i]=0
            Tc[i]=902.5
        if Ptop[i]==-999:
            #缺测判断
            mycloudflag[i]=0
            Ptop[i]=5320   
        if (mask127[i]&0b00001000)==0:
            # print()
            #白天判断
            flag[i]=0
            # day+=1
            continue
        if (mask127[i]&0b11000000)!=0:
            #地表判断
            flag[i]=0
            continue
        if (mask127[i]&0b00000001)==0:
            #可信度判断
            flag[i]=0
    # print("云比例：",sum(mycloudflag)/len(mycloudflag))
    # print("白天比例：",day/len(mycloudflag))
    for i in range(length64):
        if flag[i*64:i*64+64]==my64:
            if sum(mycloudflag[i*64:i*64+64])>32:
                # print(sum(mycloudflag[i*64:i*64+64]))
                tempRR1=RR[i*64:i*64+64,38:102]
                tempRR2 = [[0 for p in range (64)]for q in range (64)]   
                for m in range(64):
                    for n in range(64):
                        if tempRR1[m][n]:
                            # tempRR2[n][m]=((tempRR1[m,n]+35)*2/55-1)
                            tempRR2[n][m]=((tempRR1[m,n]+27)/47)
                            if tempRR2[n][m]<0:
                                tempRR2[n][m]=0
                            elif tempRR2[n][m]>1:
                                tempRR2[n][m]=1
                if(Sum_matrix(tempRR2)<200):
                    continue
                scenes.append(tempRR2)
                labels.append(Label[i*64:i*64+64,38:102].T)
                # print(Label[i*64+32,38:102])
                Tc_=(np.log(Tc[i*64:i*64+64]*0.01)-2.2)/1.13
                tau_c.append(Tc_.tolist())
                Ptop_=Ptop[i*64:i*64+64]*0.1
                p_top.append(((Ptop_-532)/265).tolist())
                re_=(np.log(re[i*64:i*64+64]*0.01)-3.06)/0.542
                r_e.append(re_.tolist())
                CWP_=(np.log(CWP[i*64:i*64+64]*0.01)-0.184)/1.11
                twp.append(CWP_.tolist())
                modis_mask.append(mycloudflag[i*64:i*64+64])
                lat.append(mylat[i*64:i*64+64])
                lon.append(mylon[i*64:i*64+64])
                # print(i*64)
    
    # data =f.read_single()
    print(len(modis_mask))
    # print(modis_mask[0])
    f.close()
    f2.close()
    f3.close()
pathdl = '/data3/wangfx/cloudsat/NEW/xia'
pathdl2 = '/data3/wangfx/modis/NEW/xia'
pathdl3 = '/data3/wangfx/cloudsat/NEW3label/xia'
for file in os.listdir(pathdl):
    orname=os.path.join(file)
    file2=file.split('_')
    file2[3]='MOD06-1KM-AUX'
    file3='_'.join(file2)
    file2[3]='2B-CLDCLASS'
    file4='_'.join(file2)
    fnamecloudsat=pathdl+'//'+orname,"utf-8"
    fnamemodis=pathdl2+'//'+file3,"utf-8"
    fnamelabel=pathdl3+'//'+file4,"utf-8"
    
    try:
        f = reader2(fnamemodis[0])
        f3 = reader2(fnamelabel[0])
    except:
        print("未找到匹配")
        continue
    f2 = reader2(fnamecloudsat[0])
    CWP = np.array(f.sd.select('Cloud_Water_Path')[:,7],dtype=float)
    Tc = np.array(f.sd.select('Cloud_Optical_Thickness')[:,7],dtype=float)
    re = np.array(f.sd.select('Cloud_Effective_Radius')[:,7],dtype=float)
    Ptop = np.array(f.sd.select('Cloud_top_pressure_1km')[:,7],dtype=float)
    mylat = f.sd.select('MODIS_latitude')[:,7]
    mylon = f.sd.select('MODIS_longitude')[:,7]
    # mask127 = np.array(f.sd.select('Cloud_Mask_1km')[0,:,7],dtype=int)
    mask127 = f.sd.select('Cloud_Mask_1km')[0,:,7]
    RR = f2.read_sds('Radar_Reflectivity')
    Label = np.array(f3.sd.select('cloud_scenario'),dtype=int)&0b0001100000011110
    for i in range(len(Label)):
        for j in range(len(Label[0])):
            if(Label[i][j]<1000):
                Label[i][j]=9
                print(i,j)
            else:
                Label[i][j]=(Label[i][j]>>1)-1024
    # lon, lat = f2.read_geo()
    # print(CWP)
    lengthall=len(mask127)
    length64=lengthall//64
    flag= [1 for i in range (lengthall)]
    mycloudflag= [0 for i in range (lengthall)]
    my64= [1 for i in range (64)]
    my64_0= [0 for i in range (64)]

    for i in range(lengthall):
        if (mask127[i]&0b00000110)==0:
            mycloudflag[i]=1
        if CWP[i]==-9999:
            mycloudflag[i]=0
            CWP[i]=120.2
        if re[i]==-9999:
            #缺测判断
            mycloudflag[i]=0
            re[i]=2132.8
        if Tc[i]==-9999:
            #缺测判断
            mycloudflag[i]=0
            Tc[i]=902.5
        if Ptop[i]==-999:
            #缺测判断
            mycloudflag[i]=0
            Ptop[i]=5320   
        if (mask127[i]&0b00001000)==0:
            # print()
            #白天判断
            flag[i]=0
            # day+=1
            continue
        if (mask127[i]&0b11000000)!=0:
            #地表判断
            flag[i]=0
            continue
        if (mask127[i]&0b00000001)==0:
            #可信度判断
            flag[i]=0
    # print("云比例：",sum(mycloudflag)/len(mycloudflag))
    # print("白天比例：",day/len(mycloudflag))
    for i in range(length64):
        if flag[i*64:i*64+64]==my64:
            if sum(mycloudflag[i*64:i*64+64])>32:
                # print(sum(mycloudflag[i*64:i*64+64]))
                tempRR1=RR[i*64:i*64+64,38:102]
                tempRR2 = [[0 for p in range (64)]for q in range (64)]   
                for m in range(64):
                    for n in range(64):
                        if tempRR1[m][n]:
                            # tempRR2[n][m]=((tempRR1[m,n]+35)*2/55-1)
                            tempRR2[n][m]=((tempRR1[m,n]+27)/47)
                            if tempRR2[n][m]<0:
                                tempRR2[n][m]=0
                            elif tempRR2[n][m]>1:
                                tempRR2[n][m]=1
                if(Sum_matrix(tempRR2)<200):
                    continue
                scenes.append(tempRR2)
                labels.append(Label[i*64:i*64+64,38:102].T)
                # print(Label[i*64+32,38:102])
                Tc_=(np.log(Tc[i*64:i*64+64]*0.01)-2.2)/1.13
                tau_c.append(Tc_.tolist())
                Ptop_=Ptop[i*64:i*64+64]*0.1
                p_top.append(((Ptop_-532)/265).tolist())
                re_=(np.log(re[i*64:i*64+64]*0.01)-3.06)/0.542
                r_e.append(re_.tolist())
                CWP_=(np.log(CWP[i*64:i*64+64]*0.01)-0.184)/1.11
                twp.append(CWP_.tolist())
                modis_mask.append(mycloudflag[i*64:i*64+64])
                lat.append(mylat[i*64:i*64+64])
                lon.append(mylon[i*64:i*64+64])
                # print(i*64)
    
    # data =f.read_single()
    print(len(modis_mask))
    # print(modis_mask[0])
    f.close()
    f2.close()
    f3.close()
pathdl = '/data3/wangfx/cloudsat/NEW/qiu'
pathdl2 = '/data3/wangfx/modis/NEW/qiu'
pathdl3 = '/data3/wangfx/cloudsat/NEW3label/qiu'
for file in os.listdir(pathdl):
    orname=os.path.join(file)
    file2=file.split('_')
    file2[3]='MOD06-1KM-AUX'
    file3='_'.join(file2)
    file2[3]='2B-CLDCLASS'
    file4='_'.join(file2)
    fnamecloudsat=pathdl+'//'+orname,"utf-8"
    fnamemodis=pathdl2+'//'+file3,"utf-8"
    fnamelabel=pathdl3+'//'+file4,"utf-8"
    
    try:
        f = reader2(fnamemodis[0])
        f3 = reader2(fnamelabel[0])
    except:
        print("未找到匹配")
        continue
    f2 = reader2(fnamecloudsat[0])
    CWP = np.array(f.sd.select('Cloud_Water_Path')[:,7],dtype=float)
    Tc = np.array(f.sd.select('Cloud_Optical_Thickness')[:,7],dtype=float)
    re = np.array(f.sd.select('Cloud_Effective_Radius')[:,7],dtype=float)
    Ptop = np.array(f.sd.select('Cloud_top_pressure_1km')[:,7],dtype=float)
    mylat = f.sd.select('MODIS_latitude')[:,7]
    mylon = f.sd.select('MODIS_longitude')[:,7]
    # mask127 = np.array(f.sd.select('Cloud_Mask_1km')[0,:,7],dtype=int)
    mask127 = f.sd.select('Cloud_Mask_1km')[0,:,7]
    RR = f2.read_sds('Radar_Reflectivity')
    Label = np.array(f3.sd.select('cloud_scenario'),dtype=int)&0b0001100000011110
    for i in range(len(Label)):
        for j in range(len(Label[0])):
            if(Label[i][j]<1000):
                Label[i][j]=9
                print(i,j)
            else:
                Label[i][j]=(Label[i][j]>>1)-1024
    # lon, lat = f2.read_geo()
    # print(CWP)
    lengthall=len(mask127)
    length64=lengthall//64
    flag= [1 for i in range (lengthall)]
    mycloudflag= [0 for i in range (lengthall)]
    my64= [1 for i in range (64)]
    my64_0= [0 for i in range (64)]

    for i in range(lengthall):
        if (mask127[i]&0b00000110)==0:
            mycloudflag[i]=1
        if CWP[i]==-9999:
            mycloudflag[i]=0
            CWP[i]=120.2
        if re[i]==-9999:
            #缺测判断
            mycloudflag[i]=0
            re[i]=2132.8
        if Tc[i]==-9999:
            #缺测判断
            mycloudflag[i]=0
            Tc[i]=902.5
        if Ptop[i]==-999:
            #缺测判断
            mycloudflag[i]=0
            Ptop[i]=5320   
        if (mask127[i]&0b00001000)==0:
            # print()
            #白天判断
            flag[i]=0
            # day+=1
            continue
        if (mask127[i]&0b11000000)!=0:
            #地表判断
            flag[i]=0
            continue
        if (mask127[i]&0b00000001)==0:
            #可信度判断
            flag[i]=0
    # print("云比例：",sum(mycloudflag)/len(mycloudflag))
    # print("白天比例：",day/len(mycloudflag))
    for i in range(length64):
        if flag[i*64:i*64+64]==my64:
            if sum(mycloudflag[i*64:i*64+64])>32:
                # print(sum(mycloudflag[i*64:i*64+64]))
                tempRR1=RR[i*64:i*64+64,38:102]
                tempRR2 = [[0 for p in range (64)]for q in range (64)]   
                for m in range(64):
                    for n in range(64):
                        if tempRR1[m][n]:
                            # tempRR2[n][m]=((tempRR1[m,n]+35)*2/55-1)
                            tempRR2[n][m]=((tempRR1[m,n]+27)/47)
                            if tempRR2[n][m]<0:
                                tempRR2[n][m]=0
                            elif tempRR2[n][m]>1:
                                tempRR2[n][m]=1
                if(Sum_matrix(tempRR2)<200):
                    continue
                scenes.append(tempRR2)
                labels.append(Label[i*64:i*64+64,38:102].T)
                # print(Label[i*64+32,38:102])
                Tc_=(np.log(Tc[i*64:i*64+64]*0.01)-2.2)/1.13
                tau_c.append(Tc_.tolist())
                Ptop_=Ptop[i*64:i*64+64]*0.1
                p_top.append(((Ptop_-532)/265).tolist())
                re_=(np.log(re[i*64:i*64+64]*0.01)-3.06)/0.542
                r_e.append(re_.tolist())
                CWP_=(np.log(CWP[i*64:i*64+64]*0.01)-0.184)/1.11
                twp.append(CWP_.tolist())
                modis_mask.append(mycloudflag[i*64:i*64+64])
                lat.append(mylat[i*64:i*64+64])
                lon.append(mylon[i*64:i*64+64])
                # print(i*64)
    
    # data =f.read_single()
    print(len(modis_mask))
    # print(modis_mask[0])
    f.close()
    f2.close()
    f3.close()
pathdl = '/data3/wangfx/cloudsat/NEW/dong'
pathdl2 = '/data3/wangfx/modis/NEW/dong'
pathdl3 = '/data3/wangfx/cloudsat/NEW3label/dong'
for file in os.listdir(pathdl):
    orname=os.path.join(file)
    file2=file.split('_')
    file2[3]='MOD06-1KM-AUX'
    file3='_'.join(file2)
    file2[3]='2B-CLDCLASS'
    file4='_'.join(file2)
    fnamecloudsat=pathdl+'//'+orname,"utf-8"
    fnamemodis=pathdl2+'//'+file3,"utf-8"
    fnamelabel=pathdl3+'//'+file4,"utf-8"
    
    try:
        f = reader2(fnamemodis[0])
        f3 = reader2(fnamelabel[0])
    except:
        print("未找到匹配")
        continue
    f2 = reader2(fnamecloudsat[0])
    CWP = np.array(f.sd.select('Cloud_Water_Path')[:,7],dtype=float)
    Tc = np.array(f.sd.select('Cloud_Optical_Thickness')[:,7],dtype=float)
    re = np.array(f.sd.select('Cloud_Effective_Radius')[:,7],dtype=float)
    Ptop = np.array(f.sd.select('Cloud_top_pressure_1km')[:,7],dtype=float)
    mylat = f.sd.select('MODIS_latitude')[:,7]
    mylon = f.sd.select('MODIS_longitude')[:,7]
    # mask127 = np.array(f.sd.select('Cloud_Mask_1km')[0,:,7],dtype=int)
    mask127 = f.sd.select('Cloud_Mask_1km')[0,:,7]
    RR = f2.read_sds('Radar_Reflectivity')
    Label = np.array(f3.sd.select('cloud_scenario'),dtype=int)&0b0001100000011110
    for i in range(len(Label)):
        for j in range(len(Label[0])):
            if(Label[i][j]<1000):
                Label[i][j]=9
                print(i,j)
            else:
                Label[i][j]=(Label[i][j]>>1)-1024
    # lon, lat = f2.read_geo()
    # print(CWP)
    lengthall=len(mask127)
    length64=lengthall//64
    flag= [1 for i in range (lengthall)]
    mycloudflag= [0 for i in range (lengthall)]
    my64= [1 for i in range (64)]
    my64_0= [0 for i in range (64)]

    for i in range(lengthall):
        if (mask127[i]&0b00000110)==0:
            mycloudflag[i]=1
        if CWP[i]==-9999:
            mycloudflag[i]=0
            CWP[i]=120.2
        if re[i]==-9999:
            #缺测判断
            mycloudflag[i]=0
            re[i]=2132.8
        if Tc[i]==-9999:
            #缺测判断
            mycloudflag[i]=0
            Tc[i]=902.5
        if Ptop[i]==-999:
            #缺测判断
            mycloudflag[i]=0
            Ptop[i]=5320   
        if (mask127[i]&0b00001000)==0:
            # print()
            #白天判断
            flag[i]=0
            # day+=1
            continue
        if (mask127[i]&0b11000000)!=0:
            #地表判断
            flag[i]=0
            continue
        if (mask127[i]&0b00000001)==0:
            #可信度判断
            flag[i]=0
    # print("云比例：",sum(mycloudflag)/len(mycloudflag))
    # print("白天比例：",day/len(mycloudflag))
    for i in range(length64):
        if flag[i*64:i*64+64]==my64:
            if sum(mycloudflag[i*64:i*64+64])>32:
                # print(sum(mycloudflag[i*64:i*64+64]))
                tempRR1=RR[i*64:i*64+64,38:102]
                tempRR2 = [[0 for p in range (64)]for q in range (64)]   
                for m in range(64):
                    for n in range(64):
                        if tempRR1[m][n]:
                            # tempRR2[n][m]=((tempRR1[m,n]+35)*2/55-1)
                            tempRR2[n][m]=((tempRR1[m,n]+27)/47)
                            if tempRR2[n][m]<0:
                                tempRR2[n][m]=0
                            elif tempRR2[n][m]>1:
                                tempRR2[n][m]=1
                if(Sum_matrix(tempRR2)<200):
                    continue
                scenes.append(tempRR2)
                labels.append(Label[i*64:i*64+64,38:102].T)
                # print(Label[i*64+32,38:102])
                Tc_=(np.log(Tc[i*64:i*64+64]*0.01)-2.2)/1.13
                tau_c.append(Tc_.tolist())
                Ptop_=Ptop[i*64:i*64+64]*0.1
                p_top.append(((Ptop_-532)/265).tolist())
                re_=(np.log(re[i*64:i*64+64]*0.01)-3.06)/0.542
                r_e.append(re_.tolist())
                CWP_=(np.log(CWP[i*64:i*64+64]*0.01)-0.184)/1.11
                twp.append(CWP_.tolist())
                modis_mask.append(mycloudflag[i*64:i*64+64])
                lat.append(mylat[i*64:i*64+64])
                lon.append(mylon[i*64:i*64+64])
                # print(i*64)
    
    # data =f.read_single()
    print(len(modis_mask))
    # print(modis_mask[0])
    f.close()
    f2.close()
    f3.close()
    
      
try:
    # Open the NetCDF file
    ncfile1 = Dataset(".//day5_cs_modis_scenes.nc", "w", format="NETCDF4")
    
    ncfile1.createDimension('nsamples', 0)
    ncfile1.createDimension('npix_horiz', 64)
    ncfile1.createDimension('npix_vert', 64)
    
    nsamples = ncfile1.createVariable('nsamples', 'i4', ('nsamples',))
    npix_horiz64 = ncfile1.createVariable('npix_horiz', 'i4', ('npix_horiz',))
    npix_vert64 = ncfile1.createVariable('npix_vert', 'i4', ('npix_vert',))
    
    # npix_horiz64 = np.arange(0,64)
    # npix_vert64 = np.arange(0,64)
    label1 = ncfile1.createVariable('label', np.float32, ('nsamples', 'npix_horiz', 'npix_vert'))
    scene1 = ncfile1.createVariable('scenes', np.float32, ('nsamples', 'npix_horiz', 'npix_vert'))
    tau_c1 = ncfile1.createVariable('tau_c', np.float32, ('nsamples', 'npix_horiz'))
    p_top1 = ncfile1.createVariable('p_top', np.float32, ('nsamples', 'npix_horiz'))
    r_e1 = ncfile1.createVariable('r_e', np.float32, ('nsamples', 'npix_horiz'))
    twp1 = ncfile1.createVariable('twp', np.float32, ('nsamples', 'npix_horiz'))
    modis_mask1 = ncfile1.createVariable('modis_mask', np.float32, ('nsamples', 'npix_horiz'))
    lat1 = ncfile1.createVariable('lat', np.float32, ('nsamples', 'npix_horiz'))
    lon1 = ncfile1.createVariable('lon', np.float32, ('nsamples', 'npix_horiz'))
    
    label1[:,:,:] = labels
    scene1[:,:,:] = scenes
    tau_c1[:,:] = tau_c 
    p_top1[:,:] = p_top
    r_e1[:,:] = r_e
    twp1[:,:] = twp
    modis_mask1[:,:] = modis_mask
    lat1[:,:] = lat
    lon1[:,:] = lon
finally:
    ncfile1.close()