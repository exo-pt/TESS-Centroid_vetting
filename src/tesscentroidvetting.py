#
#numpy, matplotlib, lightkurve, astroquery, astropy, scipy
import warnings
import numpy as np
from numpy import ones, array
import matplotlib.pyplot as plt
from matplotlib import patches
import lightkurve as lk
from astroquery.mast import Catalogs
from astropy.time import Time
from astropy import constants
from astropy.coordinates import SkyCoord, Angle
from astropy import units as u
from astropy.table import Table
from scipy.optimize import minimize
import matplotlib.ticker as ticker
import tessprfmodel as prfmodel

#global dictionary for returned results
res = {
    'inTransit_cadences' : None,       # no. of cadences in In Transir Image
    'ooTransit_cadences' : None,       # no. of cadences in Out Of Transir Image
    'inTransit_margin' : None,         # In Transit windows (epochs-inTransit_margin <-> epochs+inTransit_margin)
    'ooTransit_inner_margin' : None,   # Out of Transit wibdows:
    'ooTransit_outer_margin' : None,   #  (epochs +/- inner -~<->to epochs +/- outer)
    'tic_pos' : None,
    'flux_centroid_pos' : None,        #relative base 0 position
    'prf_centroid_pos' : None,         #relative base 0 position
    'prf_fit_quality' : None,          # 0..1
    'tic_offset' : None,
    'nearest_tics' : None,             # nearest TICs to prf centroid Table.    
    'img_diff' : None,                 #difference image
    'img_oot' : None,                  #out of transit image
    'img_intr' : None                  #in transit image
}

def centroid_vetting(tpf, epochs, tot_transit_dur, plot=True, **kwargs):
    #, oot_outer_margin, oot_inner_margin, pixel_mask=None, plot=True):
    #epochs: float or list of floats
    if isinstance(epochs, float): epochs = [epochs]
    #    
    img_diff, img_intr, img_oot = _get_in_out_diff_img(tpf, epochs, tot_transit_dur, **kwargs)
    #
    res['img_diff'] = img_diff
    res['img_oot'] = img_oot
    res['img_intr'] = img_intr
    ntransits = len(epochs)
    TIC_ID = tpf.get_header()['TICID']
    TIC2str = tpf.get_header()['OBJECT']
    sector = tpf.get_header()['SECTOR']
    tpf_mag = tpf.meta.get("TESSMAG") 
    maglim = max(round(tpf_mag) + 8, 18)
    shapeX = img_diff.shape[1]
    shapeY = img_diff.shape[0]
    yy, xx = np.unravel_index(img_diff.argmax(), img_diff.shape)
    circular_mask = np.full(img_diff.shape, False, dtype=bool)
    circular_mask[max(yy-2,0):min(yy+3,shapeY+1), max(xx-1,0):min(xx+2,shapeX+1)] = True
    circular_mask[max(yy-1,0):min(yy+2,shapeY+1), max(xx-2,0):min(xx+3,shapeX+1)] = True        
    fluxCentroid_X, fluxCentroid_Y = lk.utils.centroid_quadratic(img_diff, mask=circular_mask)
    # 
    res['flux_centroid_pos'] = (fluxCentroid_X, fluxCentroid_Y)
    #print
    radSearch = 1 / 10 # radius in degrees
    medx = tpf.shape[1:][1]/2
    medy = tpf.shape[1:][0]/2
    coord_center = tpf.wcs.pixel_to_world(medx, medy)
    pix_scale = 21.0 #Tess
    radius=Angle(np.min(tpf.shape[1:]) * pix_scale, "arcsec")
    attempts = 0
    while attempts < 3:
        try:
            catDatatpf = Catalogs.query_region(coordinates=coord_center, radius = radius, catalog = "TIC")
            break
        except:
            attempts += 1
    #prf 
    catDatatpf = catDatatpf[catDatatpf['Tmag']<maglim]
    coord_corr_tpf = _correct_with_proper_motion(
        np.nan_to_num(np.asarray(catDatatpf['ra'])) * u.deg, 
        np.nan_to_num(np.asarray(catDatatpf['dec'])) * u.deg,
        np.nan_to_num(np.asarray(catDatatpf['pmRA'])) * u.milliarcsecond / u.year,
        np.nan_to_num(np.asarray(catDatatpf['pmDEC'])) * u.milliarcsecond / u.year,    
        Time('2000', format='byear'), # TIC is in J2000 epoch
        tpf.time[0] ,  # the observed time for the tpf
    )
    catDatatpf['ra_corr'] = coord_corr_tpf.ra.to_value(unit="deg")
    catDatatpf['dec_corr'] = coord_corr_tpf.dec.to_value(unit="deg")
    s2_ra = catDatatpf['ra_corr'].value.tolist()
    s2_dec = catDatatpf['dec_corr'].value.tolist()
    
    #at_tic= Table(catDatatpf[catDatatpf['ID'] == str(TIC_ID)][0])
    #
    prfCentroid_X, prfCentroid_Y, prfFitQuality, img_prf= _get_PRF_centroid(tpf, img_diff, fluxCentroid_X, fluxCentroid_Y)
    #
    res['prf_centroid_pos'] = (prfCentroid_X, prfCentroid_Y)
    res['prf_fit_quality'] = prfFitQuality
    res['img_prf'] = img_prf
    coord_d = tpf.wcs.pixel_to_world(prfCentroid_X, prfCentroid_Y)
    #end prf
    attempts = 0
    while attempts < 3:
        try:
            catalogData = Catalogs.query_region(coordinates=coord_d, radius = radSearch, catalog = "TIC")
            break
        except:
            attempts += 1
    catalogData = catalogData[catalogData['Tmag'] < maglim]
    # Proper motion correction implementation by Sam Lee (orionlee)
    if 'ra_no_pm' not in catalogData.colnames:
        coord_corrected = _correct_with_proper_motion(  
            np.nan_to_num(np.asarray(catalogData['ra'])) * u.deg, 
            np.nan_to_num(np.asarray(catalogData['dec'])) * u.deg,
            np.nan_to_num(np.asarray(catalogData['pmRA'])) * u.milliarcsecond / u.year,
            np.nan_to_num(np.asarray(catalogData['pmDEC'])) * u.milliarcsecond / u.year,    
            Time('2000', format='byear'), # TIC is in J2000 epoch
            tpf.time[0]   # the observed time for the tpf
        )
        dst_corrected = coord_d.separation(coord_corrected)
        # preserve catalog's RA/DEC to new _no_pm columns
        # and use the PM-corrreced ones in the original columns
        catalogData['ra_no_pm'] = catalogData['ra']
        catalogData['dec_no_pm'] = catalogData['dec']
        catalogData['dstArcSec_no_pm'] = catalogData['dstArcSec']
        catalogData['ra'] = coord_corrected.ra.to_value(unit="deg")
        catalogData['dec'] = coord_corrected.dec.to_value(unit="deg")
        catalogData['dstArcSec'] = dst_corrected.to_value(unit="arcsec")
        catalogData['has_pm'] = ~np.isnan(catalogData['pmRA']) & ~np.isnan(catalogData['pmDEC'])
        # resort the data using PM-corrected distance
        catalogData.sort(['dstArcSec', 'ID'])
    else:
        warnings.warn("Proper motion correction appears to have been applied to the data. No-Op.")
    tic_row = catalogData[catalogData['ID'] == str(TIC_ID)][0]
    tic_coords = SkyCoord(tic_row['ra']*u.deg, tic_row['dec']*u.deg, frame='icrs')
    coord_d = tpf.wcs.pixel_to_world(prfCentroid_X, prfCentroid_Y)
    ra0, dec0 = _get_offset(tic_coords,coord_d)  

    coord_d2 = tpf.wcs.pixel_to_world(fluxCentroid_X, fluxCentroid_Y)
    ra1, dec1 = _get_offset(tic_coords,coord_d2)  

    
    tx, ty = tpf.wcs.world_to_pixel(tic_coords)
    tic_pos = (tx.item(0), ty.item(0))
    res['tic_pos'] = tic_pos
    tic_offset = catalogData[catalogData['ID'] == str(TIC_ID)]['dstArcSec'][0]
    res['tic_offset'] = tic_offset
    res['nearest_tics'] = TcatData = catalogData[:20]['ID','ra','dec','Tmag','dstArcSec','has_pm','ra_no_pm','dec_no_pm','dstArcSec_no_pm']
    
    nstars = 9  # maximum number of TICs to show in table
    
    if len(catalogData) < nstars:
        nstars = len(catalogData)
    near = [];ra = []; dec = []; has_pm = []; cat = []; tmag = []; sep=[]; s_ra=[]; s_dec=[]; nstar=[]
    i_tic = -1
    for i in range(nstars):
        near.append(SkyCoord(catalogData[i]['ra'], catalogData[i]['dec'], frame='icrs', unit='deg'))
        cat.append(catalogData[i]['ID'])
        tmag.append(catalogData[i]['Tmag'])
        sep.append(round(catalogData[i]['dstArcSec'],2))
        has_pm.append(catalogData[i]['has_pm'])
        if cat[i] == str(TIC_ID):
            i_tic = i
            tic_ra = catalogData[i]['ra']
            tic_dec = catalogData[i]['dec']
        else:
            s_ra.append(catalogData[i]['ra'])
            s_dec.append(catalogData[i]['dec'])
        nstar.append(i+1)
    min_r=0;max_r=0;min_d=0;max_d=0
    nstars_radec = nstars
    maxsep = max(tic_offset+1, 40)
    for i in range(nstars):
        r, d = _get_offset(tic_coords, near[i])
        min_r=min(min_r,r);max_r=max(max_r,r)
        min_d=min(min_d,d);max_d=max(max_d,d)
        if (max_r-min_r>maxsep) or (max_d-min_d>maxsep):
            nstars_radec = i
            break
        ra.append(r)
        dec.append(d)
    if not plot:
        return res
    #######################################
    ###########  plot  ####################
    #######################################
    cols=4
    fig, axes = plt.subplots(1, cols, figsize=(12.5, 4.5),subplot_kw=dict(box_aspect=1)) #,constrained_layout = True)
    axes[0].imshow(img_oot, cmap=plt.cm.viridis, origin = 'lower',aspect='auto')
    axes[0].set_title("Mean Out of Transit Flux".format(tot_transit_dur), fontsize = 11 )
    #display pipeline mask
    points = [[None for j in range(tpf.pipeline_mask.shape[0])] for i in range(tpf.pipeline_mask.shape[1])]
    for x in range(tpf.pipeline_mask.shape[0]):
        for y in range(tpf.pipeline_mask.shape[1]):
            if tpf.pipeline_mask[x,y] == True:
                points[x][y] = patches.Rectangle(
                                    xy=(y - 0.5, x - 0.45),
                                    width=1,
                                    height=1,
                                    color='white',
                                    lw=0.6,
                                    fill=False)
                axes[0].add_patch(points[x][y])
    axes[1].imshow(img_diff, cmap=plt.cm.viridis, origin = 'lower',aspect='auto')
    # prf centroid cross
    axes[1].scatter(prfCentroid_X,prfCentroid_Y, marker = '+', s=400,lw=0.6,c='k')

    coords = SkyCoord(s_ra*u.deg, s_dec*u.deg, frame='icrs')
    g1,g2=tpf.wcs.world_to_pixel(coords)

    j = 0
    for i in range(nstars):
        if i != i_tic:
            if sep[i] < 25:
                axes[1].text(g1[j]+0.15, g2[j], nstar[i], fontsize=10, c='black')
            else:
                axes[1].text(g1[j]+0.15, g2[j], nstar[i], fontsize=10, c='white')
            j = j + 1
    
    # Print TIC stars (code based in TPFPlotter-Gaia stars)
    coords2 = SkyCoord(s2_ra*u.deg, s2_dec*u.deg, frame='icrs')
    g21, g22 = tpf.wcs.world_to_pixel(coords2)    
    sizes = np.array(64.0 / 2**(catDatatpf['Tmag']/5.0))
    j = 0
    for i in range(len(s2_ra)): 
        if g21[j] > -0.4 and g21[j] < 10.4:
            if g22[j] > -0.4 and g22[j] < 10.4:
                size=sizes[i]/100
                star_color = "white"
                zorder = 1
                if catDatatpf[i]['ID'] == str(TIC_ID):
                    star_color = "red"
                    zorder = 3
                circle1 = plt.Circle((g21[j], g22[j]),size,color=star_color, zorder = zorder)
                circle2 = plt.Circle((g21[j], g22[j]),size,color=star_color, zorder = zorder)
                axes[0].add_patch(circle1)  
                axes[1].add_patch(circle2)
        j = j + 1   
    # end Print TICs
    x_list = []; y_list = []
    for i in range(tpf.shape[1:][0]):
        y_list.append(str(tpf.row+i))
    for i in range(tpf.shape[1:][1]):
        x_list.append(str(tpf.column+i))

    axes[0].set_xticks(np.arange(len(x_list)))
    axes[0].set_xticklabels(x_list)
    axes[0].set_yticks(np.arange(len(y_list)))
    axes[0].set_yticklabels(y_list)
    axes[0].tick_params(axis='x', labelrotation=90)
    
    axes[1].set_xticks(np.arange(len(x_list)))
    axes[1].set_xticklabels(x_list)
    axes[1].set_yticks(np.arange(len(y_list)))
    axes[1].set_yticklabels(y_list)
    axes[1].tick_params(axis='x', labelrotation=90)
    axes[1].set_title("Difference Image\nMean Out-In Transit Flux".format(tot_transit_dur), fontsize = 11 )

    axes[2].grid(lw=0.2)
    axes[2].tick_params(axis='both', which='major', labelsize=9)
    axes[2].minorticks_on()
    # centroid cross 
    x01 = np.array([ra0-10.5, ra0+10.5])
    y01 = np.array([dec0, dec0])
    x02 = np.array([ra0, ra0])
    y02 = np.array([dec0-10.5, dec0+10.5])
    axes[2].plot(x01, y01, x02, y02, c='green', lw=0.8)
    #axes[2].scatter(fluxCentroid_X,fluxCentroid_Y, marker = '+', s=400,lw=0.6,c='k')
    axes[2].scatter(ra1,dec1, marker = '+', s=200,lw=0.6,c='k')
    coord_fc = tpf.wcs.pixel_to_world(fluxCentroid_X, fluxCentroid_Y)
    ra_fc, dec_fc = _get_offset(tic_coords,coord_fc)        
    #circle around centroid cross, radius in pixels = 10.5 (TESS_pixel/2)
    #circle2 = plt.Circle((ra_fc, dec_fc), 10.5, color='grey', clip_on=False,fill=False, lw=0.3)
    #axes[2].add_patch(circle2)
    
    axes[2].scatter(ra,dec, marker = '*', c='blue')
    axes[2].scatter(0,0, marker = '*', c='red')
    axes[2].set_title("Difference Image PRF Centroid\nPRF fit Quality = "+str(round(prfFitQuality,3)), fontsize = 11 )
    
    xmin, xmax = axes[2].get_xlim()
    ymin, ymax = axes[2].get_ylim()
    tam1 = abs(xmax-xmin)
    tam2 = abs(ymax-ymin)
    if tam1 > tam2:  
        ymin=ymin-(tam1-tam2)/2
        ymax=ymax+(tam1-tam2)/2
    if tam2 > tam1:
        xmin=xmin-(tam2-tam1)/2
        xmax=xmax+(tam2-tam1)/2
    tam1 = abs(xmax-xmin)
    z = 0
    for i in range(nstars_radec):
        scolor = 'black'
        if cat[i] == str(TIC_ID):
            scolor = 'red'
        z =z + 1
        axes[2].text(ra[i]+tam1/35,dec[i], z, fontsize=9, c=scolor, zorder=2)
        
    axes[2].set_xlabel("RA Offset (arcsec)", fontsize=10,labelpad=2)
    axes[2].set_ylabel("Dec Offset (arcsec)", fontsize=10,labelpad=0)
    axes[2].set_xlim([xmin, xmax])
    axes[2].set_ylim([ymin, ymax])
    z = 0
    axes[3].axis('off')
    axes[3].set_xlim([0, 1])
    axes[3].set_ylim([0, 1])
    yl1 = 1.05 #1.036
    axes[3].set_title('Nearest TICs to centroid   \n(Tmag<'+str(maglim)+')  ', fontsize = 11 )
    axes[3].text(0.5, yl1-0.1,'Tmag', fontsize=9, c='black') 
    axes[3].text(0.67, yl1-0.1,'Sep.arcsec', fontsize=9, c='black') 
    star_ok = False
    pm_any = False
    for i in range(nstars):
        colort = 'black'
        if cat[i] == str(TIC_ID):
            star_ok = True
            colort = 'red'
        z =z + 1
        axes[3].text(-0.06, yl1-0.09-z/10, str(z)+' - TIC '+cat[i], fontsize=8, c=colort)    
        axes[3].text(0.48, yl1-0.09-z/10, f'{tmag[i]:6.3f}', fontsize=9, c=colort)   
        sep_label = f'{sep[i]:6.2f}'
        if not has_pm[i]:
            sep_label += ' (#)'
            pm_any = True
        axes[3].text(0.78, yl1-0.09-z/10, sep_label, fontsize=9, c=colort)

    if not star_ok:
        z=z+1
        tic_row = catalogData[catalogData['ID'] == str(TIC_ID)][0]
        sep_label =  f"{tic_row['dstArcSec']:6.2f}"
        if not tic_row['has_pm']:
            sep_label += ' (#)'
        axes[3].text(-0.06, yl1-0.09-z/10, '* - TIC '+str(TIC_ID), fontsize=8, c='red')    
        axes[3].text(0.48, yl1-0.09-z/10, f"{tic_row['Tmag']:6.3f}", fontsize=9, c='black')    
        axes[3].text(0.78, yl1-0.09-z/10, sep_label, fontsize=9, c='black')                

    if pm_any:
        pm_text = '(#) No proper motion correction.'
    else:
        pm_text = ' Stars proper motion corrected.'
    z = z+1    
    axes[3].text(0.01, yl1-0.12-z/10, pm_text, fontsize=10, c='black')    

    plt.gcf().subplots_adjust(bottom=0.25, top=0.95, left=0.05,right=0.95,wspace=0.26)

    if len(epochs) < 6:
        epochs3 = epochs.copy()
        for i,t in enumerate(epochs3):
            epochs3[i] = round(t,3)
        fig.suptitle(TIC2str+' Sector ' +str(sector)+'            Transit epochs (BTJD)= '+str(epochs3)+'            Transit duration (hours)= '+ f'{tot_transit_dur*24:2.3f}',
                     fontsize=12, x=0.49, y=1.04)
    else:
        fig.suptitle(TIC2str+' Sector ' +str(sector)+'            Total transits = '+str(ntransits)+'            Transit duration (hours)= '+ f'{tot_transit_dur*24:2.3f}',
                     fontsize=12, x=0.49, y=1.04)
        
    com_xlabel = "In Transit cadences: " + str(res['inTransit_cadences'])+"  (Epoch ± "+f'{res["inTransit_margin"]:2.3f}d)   '
    com_xlabel += "         Out Of Transit cadences: "+str(res['ooTransit_cadences'])+"  (Epoch ± "+f'{res["ooTransit_inner_margin"]:2.3f}d' 
    com_xlabel += "  to  Epoch ± "+ f'{res["ooTransit_outer_margin"]:2.3f}d)'
    fig.text(0.5, 0.97, com_xlabel, ha='center', fontsize = 10)    
    plt.show()
    return res

# Difference Image calculation - Adapted from :
#     @noraeisner - Planet Hunters Coffee Chat - False Positives - In and Out of Flux Comparison 
# https://github.com/noraeisner/PH_Coffee_Chat/blob/main/False%20Positive/False%20positives%20-%20(2)%20in%20out%20transit%20flux.ipynb
#============================================================================================================
def _get_in_out_diff_img(tpf, epochs, tot_transit_dur, **kwargs): 
    #full_transit_dur, oot_outer_margin, oot_inner_margin, pixel_mask=None):
    #epochs: float or list of floats - If more than one, a mean image of all transits is calculated
    mask = False
    pixel_mask = kwargs.get('pixel_mask', None)
    if pixel_mask is not None and pixel_mask.any(): mask = True
    tpf_list = [tpf.flux.value]
    t_list = [tpf.time.value]
    if isinstance(epochs, float): epochs = [epochs]
    T0_list = epochs
    if tot_transit_dur == 0:
        tot_transit_dur = 0.3
    full_transit_dur = kwargs.get('full_transit_dur', tot_transit_dur * 0.8)    
    transit_tot_half = round(tot_transit_dur/2, 3)
    transit_full_half = round(full_transit_dur/2, 3)
    oot_inner_margin = kwargs.get('oot_inner_margin', round(transit_tot_half * 1.5, 3))
    oot_outer_margin = kwargs.get('oot_outer_margin', oot_inner_margin + tot_transit_dur)
    # loop through all of the list of PCA corrected flux vs time arrays for each marked transit-event
    imgs_intr = []; imgs_oot = []
    for idx, tpf_filt in enumerate(tpf_list): # idx is for each marked transit-event
        t = t_list[idx] # the time array
        intransit = 0; ootransit = 0
        for T0 in epochs:
            intr = abs(T0 - t) < transit_full_half  # mask of in transit times
            oot = (abs(T0 - t) < oot_outer_margin) * (abs(T0 - t) > oot_inner_margin)  # mask of out transit times
            intransit = intransit + len(intr[intr==True])
            ootransit = ootransit + len(oot[oot==True])
            img_intr = np.nanmean(tpf_filt[intr,:,:], axis=0)
            if mask: img_intr[pixel_mask] = np.nan
            img_oot = np.nanmean(tpf_filt[oot, :, :], axis=0)
            if mask: img_oot[pixel_mask] = np.nan
            imgs_intr.append(img_intr) 
            imgs_oot.append(img_oot)
        img_intr = np.nanmean(imgs_intr, axis=0)
        img_oot = np.nanmean(imgs_oot, axis=0)
        #img_diff = np.abs(img_oot - img_intr)  # calculate the difference image (out of transit minus in-transit)
        img_diff = img_oot - img_intr
        res['inTransit_cadences'] = intransit
        res['ooTransit_cadences'] = ootransit
        res['inTransit_margin'] = transit_full_half
        res['ooTransit_inner_margin'] = oot_inner_margin
        res['ooTransit_outer_margin'] = oot_outer_margin
    return img_diff, img_intr, img_oot

def _get_offset(coord1,coord2):
    ra_offset = (coord2.ra - coord1.ra) * np.cos(coord1.dec.to('radian'))
    dec_offset = (coord2.dec - coord1.dec)
    return float(str(ra_offset.to('arcsec'))[:-6]), float(str(dec_offset.to('arcsec'))[:-6])

#====================================================================
# proper motion correction implemented by Sam Lee (@orionlee)  https://github.com/orionlee
#====================================================================
def _correct_with_proper_motion(ra, dec, pm_ra, pm_dec, equinox, new_time):
    """Return proper-motion corrected RA / Dec.
       It also return whether proper motion correction is applied or not."""
    # all parameters have units

    # To be more accurate, we should have supplied distance to SkyCoord
    # in theory, for Gaia DR2 data, we can infer the distance from the parallax provided.
    # It is not done for 2 reasons:
    # 1. Gaia DR2 data has negative parallax values occasionally. Correctly handling them could be tricky. See:
    #    https://www.cosmos.esa.int/documents/29201/1773953/Gaia+DR2+primer+version+1.3.pdf/a4459741-6732-7a98-1406-a1bea243df79
    # 2. For our purpose (ploting in various interact usage) here, the added distance does not making
    #    noticeable significant difference. E.g., applying it to Proxima Cen, a target with large parallax
    #    and huge proper motion, does not change the result in any noticeable way.
    #
    c = SkyCoord(ra, dec, pm_ra_cosdec=pm_ra, pm_dec=pm_dec, frame='icrs', obstime=equinox)

    # Suppress ErfaWarning temporarily as a workaround for:
    #   https://github.com/astropy/astropy/issues/11747
    with warnings.catch_warnings():
        # the same warning appears both as an ErfaWarning and a astropy warning
        # so we filter by the message instead
        warnings.filterwarnings("ignore", message="ERFA function")
        new_c = c.apply_space_motion(new_obstime=new_time)
    return new_c

#  PRF centroid  calculation 
# ================================================================================
# simplified from:           TESS-plots (@mkunimoto)   - https://github.com/mkunimoto/TESS-plots    
# ================================================================================
def _get_PRF_centroid(tpf, img_diff, flux_centroid_x, flux_centroid_y):
    cols = tpf.flux.shape[2]
    rows = tpf.flux.shape[1]
    extent = (tpf.column - 0.5, tpf.column+cols-0.5,tpf.row-0.5,tpf.row+rows-0.5)
    prf = prfmodel.SimpleTessPRF(shape=img_diff.shape,
                             sector = tpf.sector,
                             camera = tpf.camera,
                             ccd = tpf.ccd,
                             column = extent[0],
                             row = extent[2])
    sector = tpf.sector
    fluxcentroid = (tpf.column + flux_centroid_x, tpf.row + flux_centroid_y)
    #
    fitVector, prfFitQuality, simData = _tess_PRF_centroid(prf, img_diff, fluxcentroid) 
    #
    prf_x = fitVector[0]-tpf.column
    prf_y = fitVector[1]-tpf.row
    return prf_x, prf_y, prfFitQuality, simData
def _tess_PRF_centroid(prf, diffImage, qfc):
    data = diffImage.copy()
    seed = np.array([qfc[0], qfc[1], 1, 0])
    r = minimize(_sim_image_data_diff, seed, method="L-BFGS-B",args = (prf, data))
    simData = _render_prf(prf, r.x)
    prfFitQuality = np.corrcoef(simData.ravel(), data.ravel())[0,1]
    return r.x, prfFitQuality, simData
def _sim_image_data_diff(c, prf, data):
    pix = _render_prf(prf, c)
    return np.sum((pix.ravel() - data.ravel())**2)
def _render_prf(prf, coef):
    # coef = [x, y, a, o]
    return coef[3] + coef[2]*prf.evaluate(coef[0] + 0.5, coef[1] + 0.5)
# end =============================================================================