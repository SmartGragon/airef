# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 16:04:05 2021

@author: WavingFlag
"""


from netCDF4 import Dataset

# from wrf import (getvar, to_np, vertcross, smooth2d, CoordPair, GeoBounds,
#                  get_cartopy, latlon_coords, cartopy_xlim, cartopy_ylim)

from tqdm import tqdm
from tqdm._tqdm import trange
import xlwt
import numpy as np

try:
    # Open the NetCDF file
    ncfile = Dataset("/data3/wangfx/ref_nc/dong5_cs_modis_scenes.nc")
    # ncfile1 = Dataset("D://1data_set//high_cs_modis_scenes.nc", "w", format="NETCDF4")
    ncfile2 = Dataset("/data3/wangfx/ref_nc/Nlat4/Nmid_dong5_modis_scenes.nc", "w", format="NETCDF4")
    # ncfile3 = Dataset("D://1data_set//low_cs_modis_scenes.nc", "a", format="NETCDF4")
    
    # ncfile1.createDimension('nsamples', 0)
    # ncfile1.createDimension('npix_horiz', 64)
    # ncfile1.createDimension('npix_vert', 64)
    
    # nsamples = ncfile1.createVariable('nsamples', 'i4', ('nsamples',))
    # npix_horiz64 = ncfile1.createVariable('npix_horiz', 'i4', ('npix_horiz',))
    # npix_vert64 = ncfile1.createVariable('npix_vert', 'i4', ('npix_vert',))
    
    # # npix_horiz64 = np.arange(0,64)
    # # npix_vert64 = np.arange(0,64)
    
    # scene1 = ncfile1.createVariable('scenes', np.float32, ('nsamples', 'npix_horiz', 'npix_vert'))
    # tau_c1 = ncfile1.createVariable('tau_c', np.float32, ('nsamples', 'npix_horiz'))
    # p_top1 = ncfile1.createVariable('p_top', np.float32, ('nsamples', 'npix_horiz'))
    # r_e1 = ncfile1.createVariable('r_e', np.float32, ('nsamples', 'npix_horiz'))
    # twp1 = ncfile1.createVariable('twp', np.float32, ('nsamples', 'npix_horiz'))
    # modis_mask1 = ncfile1.createVariable('modis_mask', np.float32, ('nsamples', 'npix_horiz'))
    
    ncfile2.createDimension('nsamples', 0)
    ncfile2.createDimension('npix_horiz', 64)
    ncfile2.createDimension('npix_vert', 64)
    
    nsamples = ncfile2.createVariable('nsamples', 'i4', ('nsamples',))
    npix_horiz64 = ncfile2.createVariable('npix_horiz', 'i4', ('npix_horiz',))
    npix_vert64 = ncfile2.createVariable('npix_vert', 'i4', ('npix_vert',))
    
    # npix_horiz64 = np.arange(0,64)
    # npix_vert64 = np.arange(0,64)
    
    scene2 = ncfile2.createVariable('scenes', np.float32, ('nsamples', 'npix_horiz', 'npix_vert'))
    label2 = ncfile2.createVariable('label', np.float32, ('nsamples', 'npix_horiz', 'npix_vert'))
    tau_c2 = ncfile2.createVariable('tau_c', np.float32, ('nsamples', 'npix_horiz'))
    p_top2 = ncfile2.createVariable('p_top', np.float32, ('nsamples', 'npix_horiz'))
    r_e2 = ncfile2.createVariable('r_e', np.float32, ('nsamples', 'npix_horiz'))
    twp2 = ncfile2.createVariable('twp', np.float32, ('nsamples', 'npix_horiz'))
    modis_mask2 = ncfile2.createVariable('modis_mask', np.float32, ('nsamples', 'npix_horiz'))
    
    # ncfile3.createDimension('nsamples', 0)
    # ncfile3.createDimension('npix_horiz', 64)
    # ncfile3.createDimension('npix_vert', 64)
    
    # nsamples = ncfile3.createVariable('nsamples', 'i4', ('nsamples',))
    # npix_horiz64 = ncfile3.createVariable('npix_horiz', 'i4', ('npix_horiz',))
    # npix_vert64 = ncfile3.createVariable('npix_vert', 'i4', ('npix_vert',))
    
    # # npix_horiz64 = np.arange(0,64)
    # # npix_vert64 = np.arange(0,64)
    
    # scene3 = ncfile3.createVariable('scenes', np.float32, ('nsamples', 'npix_horiz', 'npix_vert'))
    # tau_c3 = ncfile3.createVariable('tau_c', np.float32, ('nsamples', 'npix_horiz'))
    # p_top3 = ncfile3.createVariable('p_top', np.float32, ('nsamples', 'npix_horiz'))
    # r_e3 = ncfile3.createVariable('r_e', np.float32, ('nsamples', 'npix_horiz'))
    # twp3 = ncfile3.createVariable('twp', np.float32, ('nsamples', 'npix_horiz'))
    # modis_mask3 = ncfile3.createVariable('modis_mask', np.float32, ('nsamples', 'npix_horiz'))
    
    labelMid=[]
    
    
    sceneLow=[]
    sceneMid=[]
    sceneHigh=[]
    
    tau_cLow=[]
    tau_cMid=[]
    tau_cHigh=[]
    
    p_topLow=[]
    p_topMid=[]
    p_topHigh=[]
    
    r_eLow=[]
    r_eMid=[]
    r_eHigh=[]
    
    twpLow=[]
    twpMid=[]
    twpHigh=[]
    
    modis_maskLow=[]
    modis_maskMid=[]
    modis_maskHigh=[]
    
    lat=(ncfile.variables['lon'][:,0]).tolist()
    lon=(ncfile.variables['lat'][:,0]).tolist()
    scene_all=(ncfile.variables['scenes'][...]).tolist()
    label_all=(ncfile.variables['label'][...]).tolist()
    tau_c_all=(ncfile.variables['tau_c'][...]).tolist()
    p_top_all=(ncfile.variables['p_top'][...]).tolist()
    r_e_all=(ncfile.variables['r_e'][...]).tolist()
    twp_all=(ncfile.variables['twp'][...]).tolist()
    modis_mask_all=(ncfile.variables['modis_mask'][...]).tolist()
    
    for i in tqdm(range (len(lat))):
        # if (abs(lon[i]>=65)):
        #     sceneHigh.append(scene_all[i])
        #     tau_cHigh.append(tau_c_all[i])
        #     p_topHigh.append(p_top_all[i])
        #     r_eHigh.append(r_e_all[i])
        #     twpHigh.append(twp_all[i])
        #     modis_maskHigh.append(modis_mask_all[i])
            
        # elif (abs(lon[i]<20)):
        #     sceneLow.append(scene_all[i])
        #     tau_cLow.append(tau_c_all[i])
        #     p_topLow.append(p_top_all[i])
        #     r_eLow.append(r_e_all[i])
        #     twpLow.append(twp_all[i])
        #     modis_maskLow.append(modis_mask_all[i])
            
        # else:
        #     sceneMid.append(scene_all[i]) 
        #     tau_cMid.append(tau_c_all[i]) 
        #     p_topMid.append(p_top_all[i]) 
        #     r_eMid.append(r_e_all[i]) 
        #     twpMid.append(twp_all[i]) 
        #     modis_maskMid.append(modis_mask_all[i]) 
        if (lon[i]<=65 and lon[i]>20):
            sceneMid.append(scene_all[i]) 
            labelMid.append(label_all[i]) 
            tau_cMid.append(tau_c_all[i]) 
            p_topMid.append(p_top_all[i]) 
            r_eMid.append(r_e_all[i]) 
            twpMid.append(twp_all[i]) 
            modis_maskMid.append(modis_mask_all[i])             
    # # lon=(ncfile.variables['scenes'][0:100,:,:]).tolist()
    # # print(np.shape(lon))
    # # lat=(ncfile.variables['lon'][:,0])
    
    # # lats = np.arange(89.75,-90,-0.5) 
    
    # tau_cs[:] = lats
    
    
    # scene1[:,:,:] = sceneHigh
    # tau_c1[:,:] = tau_cHigh 
    # p_top1[:,:] = p_topHigh
    # r_e1[:,:] = r_eHigh
    # twp1[:,:] = twpHigh
    # modis_mask1[:,:] = modis_maskHigh
    label2[:,:,:] = labelMid
    scene2[:,:,:] = sceneMid
    tau_c2[:,:] = tau_cMid 
    p_top2[:,:] = p_topMid
    r_e2[:,:] = r_eMid
    twp2[:,:] = twpMid
    modis_mask2[:,:] = modis_maskMid
    
    # scene3[:,:,:] = sceneLow
    # tau_c3[:,:] = tau_cLow 
    # p_top3[:,:] = p_topLow
    # r_e3[:,:] = r_eLow
    # twp3[:,:] = twpLow
    # modis_mask3[:,:] = modis_maskLow


finally:
    ncfile.close()
    # ncfile1.close()
    ncfile2.close()
    # ncfile3.close()
