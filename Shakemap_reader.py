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
from xml.etree import ElementTree
import scipy.optimize as opt
from scipy.interpolate import interp1d


class ShakingExceedanceVerifier:
    """
    Read in GMPE array and ShakeMap data, make shakemaps exceedance array of
    same shape as GMPE array, then verify forecast
    """    
    def __init__(self, shake_dir, shake_threshold=0.2):
        """
        Read in GMPE array and ShakeMap data, make shakemaps exceedance array of
        same shape as GMPE array, then verify forecast
        """
        self.shake_dir                   = shake_dir
        self.shake_threshold             = shake_threshold
        self.have_made_obs_rec           = False
        self.have_loaded_gmpe            = False
        
        # Load all ShakeMap data, create rTrees of observed exceedance values
        self.calc_exceedance_from_shakemaps()
        
        
    def calc_exceedance_from_shakemaps(self):
        """
        Create rTree indeces with all ShakeMap data, and select subset closest
        to the grid points of the GMPE array, get final bounds on region
        """
        # Get file paths for the ShakeMap files
        shakemap_filepaths = []
        for filename in os.listdir(self.shake_dir):
            if (filename.endswith(".xyz") or filename.endswith(".xml")) and "_01_" not in filename:
                shakemap_filepaths.append(os.path.join(self.shake_dir, filename))
                continue
            else:
                continue 
        
        self.exceed_rtrees = []
        for j, filepath in enumerate(shakemap_filepaths):
            if filepath.endswith(".xyz"):
                thislonlats, thispga = self.read_shakemap_xyz(filepath)
            elif filepath.endswith(".xml"):
                thislonlats, thispga = self.read_shakemap_xml(filepath)
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


    def read_shakemap_xyz(self, file_path):
        """
        Read USGS PGA ShakeMap xyz text file
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
    
    
    def read_shakemap_xml(self, file_path):
        """
        Read USGS PGA ShakeMap xml file
        """
        lonlats = []
        pga = []
        
        etree = ElementTree.parse(file_path).getroot()
        
        for line in etree.getchildren()[-1].text.splitlines()[1:]:
            line = line.split()
            lonlats.append([float(line[0]), float(line[1])])
            pga.append(float(line[2])/100.0) #pga in file given in %g, we want g
                
        return np.array(lonlats), np.array(pga)
    
    
    def verify_GMPE(self, gmpe_file_path, forecast_scaling_multiplier=1):
        """
        Perform all the verification tasks.
        """
        self.forecast_scaling_multiplier = forecast_scaling_multiplier
        
        # Use existing gmpe_rec if it's already loaded
        if self.have_loaded_gmpe and self.gmpe_file_path == gmpe_file_path:
            pass #Use the already-loaded file
        else:
            self.gmpe_file_path = gmpe_file_path
            # Load the GMPE rec array
            self.gmpe_rec = np.load(gmpe_file_path)
            self.have_loaded_gmpe = True
            print("Loaded GMPE array")
        
        # Use an the existing obs_exceedance_rec if the gmpe grids are the same
        if self.have_made_obs_rec and (self.lons_gmpe==np.unique(self.gmpe_rec['x'])).all() and (self.lats_gmpe==np.unique(self.gmpe_rec['y'])).all():
            pass #Use existing observed array
        else:
            # Make an observed exceedence array that matches the grid of GMPE rec
            self.obs_exceedance_rec, self.valid_data_mask = self.sample_matching_obs_data(self.gmpe_rec)
            print("Created observed exceedance array")
        
        # Do the actual verification
        self.gmss_verification()
        return self.score
    
    
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
        
        self.have_made_obs_rec = True
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
    
    
    def p_contingency_calc(self):
        """
        Calculate contingency table for our GMPE forecast against observed shaking data
        """
        obs_exceedance_array_valid = self.obs_exceedance_rec['z'][~self.valid_data_mask]
        gmpe_array_valid = self.gmpe_rec['z'][~self.valid_data_mask]
        # Need to forecast integer values of exceedance, scaled by external multiplier
        gmpe_array_valid = np.round(gmpe_array_valid*self.forecast_scaling_multiplier)    
        
        max_obs_exceedance_number = max(obs_exceedance_array_valid)
        p_contingency = np.zeros((int(max_obs_exceedance_number+1), int(max_obs_exceedance_number+1)))
        
        for forecast_value, observed_value in zip(gmpe_array_valid, obs_exceedance_array_valid):            
            # We're taking advantage of the fact that these exceedance categories
            # correspond to indeces of our matrix.  An IndexError will occur when
            # the forecast value is greater than the largest observed exceedance number,
            # so the point should go to the largest category.
            try:
                p_contingency[int(forecast_value), int(observed_value)] += 1
            except IndexError:
                p_contingency[int(max_obs_exceedance_number), int(observed_value)] += 1
        
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
#            print("Only one category!")
            return s_scoring
        
        # Calculate the a_r values of Gerrity method
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
    
def plot_xyz_image(xyz, fignum=0, logz=True, needTranspose=False, interp_type='nearest', cmap='jet', do_map=True, colorbounds=None):
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
    if colorbounds==None:
        colorbounds = [zz.min(), zz.max()]
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
        m.pcolor(mgx, mgy, zz, cmap=cmap, vmin=colorbounds[0], vmax=colorbounds[1])
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
    
    
    regions = ['nepal', 'chile', 'sichuan', 'tohoku', 'newzealand', 'sumatra', 'iquique', 'swnz', 'hokkaido']
    regind = 1
    region = regions[regind]
    
    abrat_lists = {'nepal':[1.5, 1.75, 2.0, 2.25, 2.5],
                   'chile':[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
                   'sichuan':[1.0, 1.125, 1.25, 1.5, 2.0],
                   'tohoku':[1.0, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 3.5, 4.0],
                   'newzealand':[1.0, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 3.5, 4.0],
                   'sumatra':[1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
                   }
    
    kwargs['shake_dir'] = "/home/jmwilson/Desktop/ShakeMaps/{}_shakemaps/".format(region)
    kwargs['shake_threshold'] = 0.2
    
    verifier = ShakingExceedanceVerifier(*pargs, **kwargs)
    
    abrat_list = abrat_lists[region]
    scaling_mult_list = np.arange(0, 10, 0.01)
    opt_bnds = [4.05, 5.0]
    scores = np.zeros((len(abrat_list), len(scaling_mult_list)))
    opt_scores = []
    opt_multipliers = []
    optimized_abrats = []
    
    for i, abrat in enumerate(abrat_list):
        abrat_str = str(abrat).replace('.','-')
        gmpe_file_path = '/home/jmwilson/Dropbox/GMPE/GMPE-x-ETAS/gmpe_outputs/{}/{}_GMPE_ab{}_magInt_nfcorrection_percSource1-0.pkl'.format(region, region, abrat_str)
        
        for j, scaling_multiplier in enumerate(scaling_mult_list):
            score = verifier.verify_GMPE(gmpe_file_path, scaling_multiplier)
            scores[i, j] = score

#        opt_result = opt.minimize_scalar(lambda scaling_multiplier: -1*verifier.verify_GMPE(gmpe_file_path, scaling_multiplier), bounds=opt_bnds, method='Bounded')
#        if opt_result.success:
#            optimal_multiplier = opt_result.x
#            opt_scores.append(verifier.verify_GMPE(gmpe_file_path, optimal_multiplier))
#            opt_multipliers.append(optimal_multiplier)
#            optimized_abrats.append(abrat)
#        
#
#    optimized_abrats_interp_func = interp1d(optimized_abrats, opt_scores, kind='cubic')
#    ab_opt_result = opt.minimize_scalar(lambda x: -1*optimized_abrats_interp_func(x), bounds=[1.5,2.0], method='Bounded')
    
#    
    plt.close(1)
    plt.figure(1)
    plt.title("Scores vs scaling multiplier")
    for i, scorelist in enumerate(scores):
        plt.plot(scaling_mult_list, scorelist, label="ab ratio = {}".format(abrat_list[i]))
    plt.ylabel("GMSS")
    plt.xlabel("Scaling factor")
    plt.legend()
    

#    plt.close(2)
#    plt.figure(2)
#    abrats_complete = np.arange(min(optimized_abrats), max(optimized_abrats), 0.01)
#    plt.title("Optimal Scores")
#    plt.plot(optimized_abrats, opt_scores, 'bo')
#    plt.plot(abrats_complete, optimized_abrats_interp_func(abrats_complete), 'b-')
#    plt.plot(ab_opt_result.x, optimized_abrats_interp_func(ab_opt_result.x), 'r*', label="Optimal $\eta$: {:.3}".format(ab_opt_result.x))
#    plt.ylabel("GMSS")
#    plt.xlabel("$\eta$")
#    plt.legend()
    
    plt.show()
#    
#    
#    plot_xyz_image(verifier.obs_exceedance_rec, logz=False, fignum=3, colorbounds=[0,3])
#
##    aftershock_number = open_xyz_file("/home/jmwilson/Dropbox/GMPE/globalETAS/etas_outputs/nepal_tInt_etas_2015-04-25 06:13:00+00:00/etas_tInt_nepal_2015_04_2015-04-25_06:13:00+00:00.xyz")
##    plot_xyz_image(aftershock_number, logz=False, fignum=4)
##
##
#    predict_exceed_rec = open_xyz_file("/home/jmwilson/Dropbox/GMPE/GMPE-x-ETAS/gmpe_outputs/{0}/{0}_GMPE_ab2-0_magInt_nfcorrection_percSource1-0.xyz".format(region))
#    predict_exceed_rec['z'] = np.round(predict_exceed_rec['z']*1.93)
#    plot_xyz_image(predict_exceed_rec, logz=False, fignum=5, colorbounds=[0,3])

    
    
    
    
    
    
    
    
    
    




