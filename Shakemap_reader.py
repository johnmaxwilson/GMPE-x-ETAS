# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 15:51:18 2017

@author: jmwilson
"""

import numpy as np
numpy = np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from rtree import index
import os
import sys


def readToRtree(file_path):
    lonlats = []
    pga = []
    i = -1
    curlat = None
    with open(file_path, "r") as gridfile:
        gridfile.readline()
        for line in gridfile:
            line = line.split()
            #if curlat != line[1]:
            #    i += 1
            #    pga.append([])
            #curlat = line[1]
            lonlats.append([float(line[0]), float(line[1])])
            #pga[i].append(float(line[2]))
            pga.append(float(line[2]))

    londict = {}
    latdict = {}
    for coord in lonlats:
        londict[coord[0]] = 1
        latdict[coord[1]] = 1
    
    lons = np.array(sorted(londict.keys()))
    lats = np.array(sorted(latdict.keys()))
    
    idx = index.Index()
    
    dlon = lons[1]-lons[0]
    dlat = lats[1]-lats[0]
    
    for i, coord in enumerate(lonlats):
        idx.insert(i, (coord[0]-dlon/2, coord[1]-dlat/2, coord[0]+dlon/2, coord[1]+dlat/2), obj=pga[i])
    
    return idx, lons, lats, pga


def basemapPGAPlot(lons, lats, pga):
        
    lonmesh, latmesh = np.meshgrid(lons, lats)
    pga = np.array(pga)
    pga = pga.reshape((len(lons), len(lats)))
    
    
    plt.close()
    m = Basemap(projection='cyl', llcrnrlat=lats.min(), urcrnrlat=lats.max(), llcrnrlon=lons.min(), urcrnrlon=lons.max(), resolution='i')
    m.drawcoastlines()
    
    #m.drawmapboundary(fill_color='PaleTurquoise')
    #m.fillcontinents(color='lemonchiffon',lake_color='PaleTurquoise', zorder=0)
    m.drawmapboundary()#fill_color='lightgray')
    #m.fillcontinents(color='darkgray',lake_color='lightgray', zorder=0)
    m.drawcountries()
    
    m.drawparallels(np.arange(round(lats.min()),round(lats.max()),2), labels=[1,0,0,0])
    m.drawmeridians(np.arange(round(lons.min()),round(lons.max()),2), labels=[0,0,0,1])
    
    #m.contourf(lonmesh, latmesh, pga)
    #m.plot(simx[::], simy[::], 'm.')
    m.pcolormesh(lonmesh, latmesh, pga, vmin = 0, vmax = 20, cmap="jet")
    cb = m.colorbar()
    cb.set_label("Peak Ground Acceleration (% g)")
    m.show()

def open_xyz_file(fname=None):
	with open(fname, 'r') as xyz:
		return np.core.records.fromarrays(zip(*[[float(x) for x in rw.split()] for rw in xyz if not rw[0] in ('#', chr(32), chr(10), chr(13), chr(9))]), dtype=[('x', '>f8'), ('y', '>f8'), ('z', '>f8')])


def exceedanceMap(shake_dir, gmpe_file = 'pickles/GMPE_rec.p', shake_threshold = 0.2):
    # This function reads in all ShakeMap files and compiles a grid of number of exceedances of the threshold acceleration
    #  for each cell in the GMPE array
    
    # Get file paths for the ShakeMap files
    filepaths = []
    for filename in os.listdir(shake_dir):
        if filename.endswith(".xyz") and "_01_" not in filename:
            filepaths.append(os.path.join(shake_dir, filename))
            continue
        else:
            continue
    
    # Load in the GMPE Pickle to get starting map bounds and grid points
    gmpe_rec = np.load(gmpe_file)
    gmpe_grid_lons = gmpe_rec['x']
    gmpe_grid_lats = gmpe_rec['y']
    minlon  = min(gmpe_grid_lons)
    maxlon  = max(gmpe_grid_lons)
    minlat  = min(gmpe_grid_lats)
    maxlat  = max(gmpe_grid_lats)
    num_gmpe_lats = 0
    first_lon = gmpe_grid_lons[0]
    for lon in gmpe_grid_lons:
        if lon == first_lon:
            num_gmpe_lats += 1
        else:
            break
    
    dlon    = gmpe_grid_lons[num_gmpe_lats]-gmpe_grid_lons[0]
    dlat    = gmpe_grid_lats[1]-gmpe_grid_lats[0]
    
    # Now load in all ShakeMap data, and get final bounds on max and min coords
    rtrIndeces = []
    pgas   = []
    for filepath in filepaths:
        rtrIndex, thislons, thislats, thispga = readToRtree(filepath)
        minlon = min(min(thislons), minlon)
        maxlon = max(max(thislons), maxlon)
        minlat = min(min(thislats), minlat)
        maxlat = max(max(thislats), maxlat)
        rtrIndeces.append(rtrIndex)
        pgas.append(thispga)
        print("Read "+filepath.split('/')[-1])
    print("boundaries = ("+str(minlon)+', '+str(maxlon)+', '+str(dlon)+'), ('+str(minlat)+', '+str(maxlat)+', '+str(dlat)+')')
    
    grid_lons = np.arange(minlon, maxlon, dlon)
    grid_lats = np.arange(minlat, maxlat, dlat)
    numcoords = len(grid_lons)*len(grid_lats)
    mglats, mglons = np.meshgrid(grid_lats, grid_lons)
    coords1d = np.dstack((mglons, mglats))
    coords1d = coords1d.reshape(numcoords, 2)
    exceedance_array = np.zeros((numcoords,1))
    for i, rtrIndex in enumerate(rtrIndeces):
        for cellcount, coord in enumerate(coords1d):
            pga = list(rtrIndex.nearest((coord[0], coord[1]), objects=True))[0].object
            
            if pga >= shake_threshold*100:
                exceedance_array[cellcount][0] += 1
            #pgaindex = list(rtrIndex.nearest((coord[0], coord[1])))[0]
            #if pgas[i][pgaindex] >= shake_threshold:
            #    exceedance_array[cellcount] += 1
        print("Checked for exceedance "+ str(i+1) +'/'+ str(len(pgas)))
    
    exceedance_rec = np.hstack((coords1d, exceedance_array))
    exceedance_rec = np.core.records.fromarrays(exceedance_rec.T, dtype=[('x', '>f8'), ('y', '>f8'), ('z', '>f8')])
    return exceedance_rec


def plot_xyz_image(xyz, fignum=0, logz=True, needTranspose=False, interp_type='nearest', cmap='jet', do_map=True):
	#
	if not hasattr(xyz, 'dtype'):
		xyz = numpy.core.records.fromarrays(zip(*xyz), dtype=[('x','>f8'), ('y','>f8'), ('z','>f8')])
	#
	xyz.sort(order='x')
	xyz.sort(order='y')
	X = sorted(list(set(xyz['x'])))
	Y = sorted(list(set(xyz['y'])))
	mgx, mgy = np.meshgrid(X, Y)
	print("len(X) = "+str(len(X)))
	print("len(Y) = "+str(len(Y)))
	print("Total size = "+str(len(xyz)))
	#

	if logz: zz=numpy.log(xyz['z'].copy())
	else: zz=xyz['z'].copy()
	#zz.shape=(len(Y), len(X))
	if needTranspose==True:
		zz.shape=(len(X), len(Y))
		zz = zz.T
	else:
		zz.shape=(len(Y), len(X))
	#
	
	plt.figure(fignum)
	plt.clf()
	#plt.imshow(numpy.flipud(zz.transpose()), interpolation=interp_type, cmap=cmap)
	#plt.colorbar()
	#
	if do_map:
		m = Basemap(projection='cyl', llcrnrlat=min(Y), urcrnrlat=max(Y), llcrnrlon=min(X), urcrnrlon=max(X), resolution='i')
		m.drawcoastlines()
		m.drawmapboundary()#fill_color='PaleTurquoise')
		#m.fillcontinents(color='lemonchiffon',lake_color='PaleTurquoise', zorder=0)
		m.drawstates()
		m.drawcountries()
		m.pcolor(mgx, mgy, zz, cmap=cmap)
	plt.colorbar()
	
	#plt.figure(fignum)
	#plt.clf()
	#plt.imshow(zz, interpolation=interp_type, cmap=cmap)
	#plt.colorbar()



if __name__ == '__main__':
    
    kwds={}
    pargs=[]        # positional arguments.
    for arg in sys.argv[1:]:
        # note: module name is the first argument.
        # assume some mistakes might be made. fix them here.
        arg.replace(',', '')
        #
        if '=' in arg:
            kwds.update(dict([arg.split('=')]))
        else:
            pargs+=[arg]
    
    kwds['shake_dir'] = "/home/jmwilson/Desktop/ShakeMaps/nepal_shakemaps/"
    kwds['shake_threshold'] = 0.2
    kwds['gmpe_file'] = 'pickles/GMPE_nepal.pkl'
    
    
    exceed_rec = exceedanceMap(**kwds)
    
    plot_xyz_image(exceed_rec, logz=False, fignum=2)

#    predict_exceed_rec = np.load("/home/jmwilson/Dropbox/GMPE/GMPE-x-ETAS/nepal_GMPE_array.p")
#    plot_xyz_image(predict_exceed_rec, logz=False, fignum=2)

#    aftershock_number = np.load("/home/jmwilson/Dropbox/GMPE/globalETAS/etas_outputs/nepal_Aftershock_num_rec.p")
#    plot_xyz_image(aftershock_number, logz=False)
    
    
    
    
    
    
    
    
    
    




