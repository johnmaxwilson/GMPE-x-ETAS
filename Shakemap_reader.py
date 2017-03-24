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
import itertools


class ShakingExceedanceVerifier:
    """
    Read in GMPE array and ShakeMap data, make shakemaps exceedance array of
    same shape as GMPE array, then verify forecast
    """    
    def __init__(self, shake_dir, shake_threshold=0.2, shake_rtrees_pickle=None):
        """
        Read in GMPE array and ShakeMap data, make shakemaps exceedance array of
        same shape as GMPE array, then verify forecast
        """
        self.shake_dir                   = shake_dir
        self.shake_threshold             = shake_threshold
        self.have_made_obs_rec             = False
        
        # Load all ShakeMap data, create rTrees of observed exceedance values
        if shake_rtrees_pickle==None:
            self.calc_exceedance_from_shakemaps()
        else:
            try: self.exceed_rtrees = np.load(shake_rtrees_pickle)
            except: self.calc_exceedance_from_shakemaps()
        
        
    def calc_exceedance_from_shakemaps(self):
        """
        Create rTree indeces with all ShakeMap data, and select subset closest
        to the grid points of the GMPE array, get final bounds on region
        """
        # Get file paths for the ShakeMap files
        shakemap_filepaths = []
        for filename in os.listdir(self.shake_dir):
            if filename.endswith(".xyz") and "_01_" not in filename:
                shakemap_filepaths.append(os.path.join(self.shake_dir, filename))
                continue
            else:
                continue 
        
        self.exceed_rtrees = []
        for j, filepath in enumerate(shakemap_filepaths):
            thislonlats, thispga = self.read_shakemap(filepath)
            
            thislons = np.unique(thislonlats[:,0])
            thislats = np.unique(thislonlats[:,1])
            dlon = thislons[1]-thislons[0]
            dlat = thislats[1]-thislats[0]
            
            #This might be unnecessary # Keeping track of the outer bounds of our PGA data
            #self.min_obs_lon = min(min(thislons), self.min_obs_lon)
            #self.max_obs_lon = max(max(thislons), self.max_obs_lon)
            #self.min_obs_lat = min(min(thislats), self.min_obs_lat)
            #self.min_obs_lat = max(max(thislats), self.min_obs_lat)
            #print("boundaries = ("+str(self.minlon)+', '+str(self.maxlon)+', '+str(self.dlon)+'), ('+str(self.minlat)+', '+str(self.maxlat)+', '+str(self.dlat)+')')
            
            this_rtrindex = index.Index()
            
            for i, coord in enumerate(thislonlats):
                if thispga[i] >= self.shake_threshold:
                    this_rtrindex.insert(i, (coord[0]-dlon/2, coord[1]-dlat/2, coord[0]+dlon/2, coord[1]+dlat/2), obj=1)
                else:
                    this_rtrindex.insert(i, (coord[0]-dlon/2, coord[1]-dlat/2, coord[0]+dlon/2, coord[1]+dlat/2), obj=0)
            
            self.exceed_rtrees.append(this_rtrindex)
            print("Made rTree for {}, {}/{}".format(filepath.split('/')[-1], j+1, len(shakemap_filepaths)))

        return


    def read_shakemap(self, file_path):
        """
        Read USGS PGA ShakeMap file
        """
        lonlats = []
        pga = []
        with open(file_path, "r") as gridfile:
            gridfile.readline() #skip the header
            for line in gridfile:
                line = line.split()
                lonlats.append([float(line[0]), float(line[1])])
                pga.append(float(line[2])/100.0) #pga in file given in %g, we want g
                
        return np.array(lonlats), np.array(pga)

    
    def verify_GMPE(self, gmpe_file_path, forecast_scaling_multiplier=1):
        """
        Perform all the verification tasks.
        """
        #self.gmpe_file_path = gmpe_file_path
        self.forecast_scaling_multiplier = forecast_scaling_multiplier
        
        # Load the GMPE rec array
        self.gmpe_rec = np.load(gmpe_file_path)
        print("Loaded GMPE rec")
        
        # Use an the existing obs_exceedance_rec if the gmpe grids are the same
        if self.have_made_obs_rec:
            if (self.lons_gmpe==np.unique(self.gmpe_rec['x']) and self.lats_gmpe==np.unique(self.gmpe_rec['y'])):
                print("GMPE on same grid, using existing observed array")
        else:
            # Make an observed exceedence array that matches the grid of GMPE rec
            self.obs_exceedance_rec, self.valid_data_mask = self.sample_matching_obs_data(self.gmpe_rec)
            print("Created observed exceedance array")
        
        # Do the actual verification
        self.gmss_verification()
        
        return
    
    
    def sample_matching_obs_data(self, gmpe_rec):
        """
        Create an observed exceedence array that matches the grid of GMPE rec,
        as well as a mask for cells with no matching observation
        """
        self.lons_gmpe = np.unique(gmpe_rec['x'])
        self.lats_gmpe = np.unique(gmpe_rec['y'])
        dlon_gmpe = self.lons_gmpe[1]-self.lons_gmpe[0]
        dlat_gmpe = self.lats_gmpe[1]-self.lats_gmpe[0]
        
        obs_exceedance_rec = np.copy(gmpe_rec)
        obs_exceedance_rec['z'] = np.zeros_like(gmpe_rec['z'], dtype=int)
        valid_data_mask = np.ones_like(gmpe_rec['z'], dtype=bool)
        
        for j, idx in enumerate(self.exceed_rtrees):
            for i, cell in enumerate(obs_exceedance_rec):
                contained_exceeds = [n.object for n in idx.intersection((cell['x']-dlon_gmpe/2.0, cell['y']-dlat_gmpe/2.0, cell['x']+dlon_gmpe/2.0, cell['y']+dlat_gmpe/2.0), objects=True)]
                if len(contained_exceeds) > 0:
                    cell['z'] += max(contained_exceeds)
                    valid_data_mask[i] = (False)
                
            print('Searched '+str(j+1)+'/'+str(len(self.exceed_rtrees))+' rTrees')
        
        self.created_obs_rec = True
        return obs_exceedance_rec, np.array(valid_data_mask)
    
    
    def gmss_verification(self):
        '''
        Gandin Murphy skill score for forecast with multiple ordinal categories
        using Gerrity scoring matrix
        From Jolliffe & Stephenson, section 4.3.3
        '''
        self.p_contingency = self.p_contingency_calc()
        
        self.s_scoring = self.gerrity_score_calc(self.p_contingency)
        
        self.score = np.sum(self.p_contingency*self.s_scoring)
        return
    
    
    def p_contingency_calc(self):
        """
        Calculate contingency table for our GMPE forecast against observed shaking data
        """
        max_obs_exceedance_number = max(self.obs_exceedance_rec['z'])
        p_contingency = np.zeros((int(max_obs_exceedance_number+1), int(max_obs_exceedance_number+1)))
        
        obs_exceedance_array_valid = self.obs_exceedance_rec['z'][~self.valid_data_mask]
        gmpe_array_valid = self.gmpe_rec['z'][~self.valid_data_mask]
        
        for forecast_value, observed_value in zip(gmpe_array_valid, obs_exceedance_array_valid):
            # Need to forecast integer values of exceedance, scaled by external multiplier
            forecast_value = round(forecast_value*self.forecast_scaling_multiplier)
            
            # We're taking advantage of the fact that these exceedance categories
            # correspond to indeces of our matrix
            p_contingency[int(forecast_value), int(observed_value)] += 1
        
        
        p_contingency = p_contingency/np.sum(p_contingency)
        
        return p_contingency
        
    
    def gerrity_score_calc(self, p_contingency):
        """
        Calculate the Gerrity scoring matrix for the Gandin Murphy skill score
        """
        #total observations in each category
        p_obs = np.sum(p_contingency, axis=0)
        
        #Scoring matrix initialization
        s_scoring = np.zeros_like(p_contingency)
        
        K = len(p_obs)
        
        if K==1:
            print("Only one observed category!")
            return s_scoring
        
        # Calculate the a_r values of Gerrity method
        for i in range(K-1):
            a_list = (1-np.cumsum(p_obs[:-1]))/(np.cumsum(p_obs[:-1]))
                
        # Now calculate each element of the scoring matrix
        b = 1./(K-1)
        for i, j in itertools.combinations_with_replacement(range(K), 2): 
            if i==0:
                first_term = 0
            else:
                first_term = np.cumsum(a_list**-1)[i-1]
            
            third_term = np.sum(a_list[j:])
            
            s_scoring[i,j] = b*(first_term - (j-i) + third_term)
            
            s_scoring[j,i] = s_scoring[i,j]
        
        return s_scoring



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
    
    kwargs={}
    pargs=[]        # positional arguments.
    for arg in sys.argv[1:]:
        # note: module name is the first argument.
        # assume some mistakes might be made. fix them here.
        arg.replace(',', '')
        #
        if '=' in arg:
            kwargs.update(dict([arg.split('=')]))
        else:
            pargs+=[arg]
    
    kwargs['shake_dir'] = "/home/jmwilson/Desktop/ShakeMaps/nepal_shakemaps/"
    kwargs['shake_threshold'] = 0.2
    #kwargs['shake_rtrees_pickle'] = None#"/home/jmwilson/Desktop/nepal_rtrees" # None #
    
    gmpe_file_path = '/home/jmwilson/Dropbox/GMPE/GMPE-x-ETAS/pickles/nepal_GMPE_ab1-0_magInt_nfcorrection_percSource1-0.pkl'
    scaling_multiplier = 20
    
    
#    nepal_verifier = ShakingExceedanceVerifier(*pargs, **kwargs)
#    
#    nepal_verifier.verify_GMPE(gmpe_file_path, scaling_multiplier)
#    
#    print("Score: {:0.4f}".format(nepal_verifier.score))
    
    plot_xyz_image(nepal_verifier.obs_exceedance_rec, logz=False, fignum=2)

#    aftershock_number = open_xyz_file("/home/jmwilson/Dropbox/GMPE/globalETAS/etas_outputs/nepal_tInt_etas_2015-04-25 06:13:00+00:00/etas_tInt_nepal_2015_04_2015-04-25_06:13:00+00:00.xyz")
#    plot_xyz_image(aftershock_number, logz=False, fignum=3)
#
#
    predict_exceed_rec = open_xyz_file("/home/jmwilson/Dropbox/GMPE/GMPE-x-ETAS/pickles/nepal_GMPE_magInt_nfcorrection_percSource1-0.xyz")
    predict_exceed_rec['z'] = np.round(predict_exceed_rec['z']*20)
    plot_xyz_image(predict_exceed_rec, logz=False, fignum=4)

    
    
    
    
    
    
    
    
    
    




