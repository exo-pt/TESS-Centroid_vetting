#
#numpy, matplotlib, lightkurve, astroquery, astropy, scipy
import sys, warnings
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


def centroid_vetting(tpf, epochs, transit_dur, intransit_margin=0, ootransit_inner=0, ootransit_outer=0,
                                plot=True, plot_flux_centroid=False, mask_edges= False, discard_epochs_with_gaps=True):
    '''
    Minimum Parameters:,
            tpf                     [Tess TargetPixelFile]
            epochs               [list] or [float] of transit time(s) to analyse
            transit_dur          [float] Transit duration in days

    Optional parameters: 
            intransit_margin    Margin in days around transit time(s) 
                                     [float]  default = transit_dur * 0.4
            ootransit_inner     Margin in days of out of transit (inner limit)
                                     [float]  default = transit_dur * 0.75
            ootransit_outer     Margin in days of out of transit (outer limit)
                                     [float]  default = oot_inner_margin + transit_dur
            plot                    Display images 
                                     [boolean] default = True 
            plot_flux_centroid  Also plots a small cross for the Flux Centroid)                                     
                                     [boolean] default = False  
            mask_edges         Mask pixels at edge if brighter than brightest pixel in central region
                                     [boolean] default = False (Only can be set if difference image brightest pixel is at edge)
            
        In transit times      :  epoch(s) ± intransit_margin
        Out of transit times:  Between <epoch(s) ± ootransit_outer>  and  <epoch(s) ± ootransit_inner> 
    '''
    if isinstance(epochs, float): epochs = [epochs]
    #
    inTMargin, ooTInnerM, ooTOuterM = _get_margins(transit_dur, intransit_margin, ootransit_inner, ootransit_outer)
    #
    if discard_epochs_with_gaps:
        gap_factor = 8  # discard epochs with data gap > 8 * cadence
    else:
        gap_factor = 999999999
    
    validEpochs, inTransitCad, ooTransitCad = _check_epochs(tpf, epochs,  inTMargin, ooTInnerM, ooTOuterM, gap_factor)   
    if len(validEpochs) < 1:
        raise Exception(f"All supplied epochs are deemed invalid due to gaps. epochs = {epochs}")
    if len(epochs) > len(validEpochs):
        warnings.warn(f"Some epochs are deemed invalid due to gaps. Excluded epochs: {set(epochs) - set(validEpochs)}")    
    #
    img_diff, img_intr, img_oot = _get_in_out_diff_img(tpf, validEpochs, inTMargin, ooTInnerM, ooTOuterM)
    img_diff_to_return = img_diff.copy()  # the instance that will be returned to call\er
    # img_diff is used to determining centroid. 
    # the implementation (lk.utils.centroid_quadratic()) might not be able to handle negative flux in soem cases, 
    # so we apply an offset to re-zero the flux to be non-negative in the copy used internally.
    if np.min(img_diff) < 0:
        img_diff = img_diff - np.min(img_diff)
    #
    ntransits = len(validEpochs)
    warnings.formatwarning = formatwarning
    
    ticid = tpf.get_header()['TICID']
    TIC2str = tpf.get_header()['OBJECT']
    sector = tpf.get_header()['SECTOR']
    tpf_mag = tpf.meta.get("TESSMAG") 
    maglim = max(round(tpf_mag) + 8, 19)
    shapeX = img_diff.shape[1]
    shapeY = img_diff.shape[0]
    yy, xx = np.unravel_index(img_diff.argmax(), img_diff.shape)
    
    if yy in [0, shapeY] or xx in [0, shapeX]:
        #brightest pixel in edge - centroid cannot be calculated 
        if mask_edges:
            fluxCentroid_X, fluxCentroid_Y, img_diff, points_mask = _mask_edges(img_diff)       
        else:
            warnings.warn("Brightest pixel on edge. Flux centroid couldn't be calculated. Use mask_edges=True if suitable.")
            plot_flux_centroid = False
            fluxCentroid_X, fluxCentroid_Y = xx, yy  #return brightest pixel in edge  
    else:
        #calculate flux centroid with circular mask
        circular_mask = np.full(img_diff.shape, False, dtype=bool)
        circular_mask[max(yy-2,0):min(yy+3,shapeY+1), max(xx-1,0):min(xx+2,shapeX+1)] = True
        circular_mask[max(yy-1,0):min(yy+2,shapeY+1), max(xx-2,0):min(xx+3,shapeX+1)] = True        
        fluxCentroid_X, fluxCentroid_Y = lk.utils.centroid_quadratic(img_diff, mask=circular_mask)
    # 
    if np.isnan(fluxCentroid_X) or np.isnan(fluxCentroid_Y):
        raise Exception(f"Failed to derive flux centroid: {fluxCentroid_X, fluxCentroid_Y} . Consider to adjust the in-transit / out-of-transit margins.")
    #
    flux_centroid_pos = (fluxCentroid_X, fluxCentroid_Y)
    #
    prfCentroid_X, prfCentroid_Y, prfFitQuality, img_prf= _get_PRF_centroid(tpf, img_diff, fluxCentroid_X, fluxCentroid_Y)
    #
    prfError = False
    if not((round(prfCentroid_X) in range(shapeX)) and (round(prfCentroid_X) in range(shapeX))): 
        prfError = True
        warnings.warn("Error calculating PRF centroid. Returned Flux Centroid instead.")
        prfCentroid_X, prfCentroid_Y = fluxCentroid_X, fluxCentroid_Y
        plot_flux_centroid = False
        
    prf_centroid_pos = (prfCentroid_X, prfCentroid_Y)
    
    coord_d = tpf.wcs.pixel_to_world(prfCentroid_X, prfCentroid_Y)
    radSearch = 1 / 10 # radius in degrees
    attempts = 0
    while attempts < 3:
        try:
            catalogData = Catalogs.query_region(coordinates=coord_d, radius = radSearch, catalog = "TIC")
            break
        except Exception as error:
            attempts += 1
            if attempts > 2:
                raise
                
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
        
    tic_row = catalogData[catalogData['ID'] == str(ticid)][0]
    tic_coords = SkyCoord(tic_row['ra']*u.deg, tic_row['dec']*u.deg, frame='icrs')
    coord_d = tpf.wcs.pixel_to_world(prfCentroid_X, prfCentroid_Y)
    ra0, dec0 = _get_offset(tic_coords,coord_d)  

    coord_d2 = tpf.wcs.pixel_to_world(fluxCentroid_X, fluxCentroid_Y)
    ra1, dec1 = _get_offset(tic_coords,coord_d2)  
    
    tx, ty = tpf.wcs.world_to_pixel(tic_coords)
    tic_pos = (tx.item(0), ty.item(0))
    tic_offset = catalogData[catalogData['ID'] == str(ticid)]['dstArcSec'][0]
    nearest_tics = catalogData[:20]['ID','ra','dec','Tmag','dstArcSec','has_pm','ra_no_pm','dec_no_pm','dstArcSec_no_pm']
    
    if not plot:
        return _get_results(ticid, sector, validEpochs, transit_dur, inTransitCad, ooTransitCad, inTMargin, ooTInnerM, ooTOuterM, 
                                    tic_pos, flux_centroid_pos, prf_centroid_pos, prfFitQuality, tic_offset, nearest_tics,
                                    img_diff, img_oot, img_intr, img_prf)
    #######################################
    ###########  plot  ####################
    #######################################
    #
    nstars = 9  # maximum number of TICs to show in table
    # catalogData is sorted by distance to centroid
    if len(catalogData) < nstars:
        nstars = len(catalogData)
        
    near = [];ra = []; dec = []; has_pm = []; cat = []; tmag = []; sep=[]; s_ra=[]; s_dec=[]
    i_tic = -1
    for i in range(nstars):
        near.append(SkyCoord(catalogData[i]['ra'], catalogData[i]['dec'], frame='icrs', unit='deg'))
        r, d = _get_offset(tic_coords, near[i])
        ra.append(r)
        dec.append(d)   
        cat.append(catalogData[i]['ID'])
        tmag.append(catalogData[i]['Tmag'])
        sep.append(round(catalogData[i]['dstArcSec'],2))
        has_pm.append(catalogData[i]['has_pm'])
        if cat[i] == str(ticid):
            i_tic = i
            tic_ra = catalogData[i]['ra']
            tic_dec = catalogData[i]['dec']
        else:
            s_ra.append(catalogData[i]['ra'])
            s_dec.append(catalogData[i]['dec'])
        
    cols=4
    fig, axes = plt.subplots(1, cols, figsize=(12.5, 4.5),subplot_kw=dict(box_aspect=1)) #,constrained_layout = True)
    axes[0].imshow(img_oot, cmap=plt.cm.viridis, origin = 'lower',aspect='auto')
    axes[0].set_title("Mean Out of Transit Flux".format(transit_dur), fontsize = 11 )
    #display pipeline mask in Out of Transit image
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
    # prf centroid cross in Diff Image
    axes[1].scatter(prfCentroid_X,prfCentroid_Y, marker = '+', s=400,lw=0.6,c='k')

    # Stars 1 to 9 in Diff Image
    coords = SkyCoord(s_ra*u.deg, s_dec*u.deg, frame='icrs')
    xs , ys =tpf.wcs.world_to_pixel(coords)
    j = 0
    for i in range(nstars):
       if i != i_tic:
            if sep[i] < 25:
               axes[1].text(xs[j]+0.15, ys[j], i+1, fontsize=10, c='black')
            else:
                axes[1].text(xs[j]+0.15, ys[j], i+1, fontsize=10, c='white')
            j = j + 1
    
    # Plot all TIC stars in Out of Transit and Diff Images (code based in TPFPlotter-Gaia stars)
    catDatatpf = _get_tpf_stars_catalog(tpf, maglim)
    stars_ra = catDatatpf['ra_corr'].value.tolist()
    stars_dec = catDatatpf['dec_corr'].value.tolist()    

    coords2 = SkyCoord(stars_ra*u.deg, stars_dec*u.deg, frame='icrs')
    g21, g22 = tpf.wcs.world_to_pixel(coords2)    
    sizes = np.array(64.0 / 2**(catDatatpf['Tmag']/5.0))
    j = 0
    for i in range(len(stars_ra)): 
        if g21[j] > -0.4 and g21[j] < shapeX - 0.6: #10.4:
            if g22[j] > -0.4 and g22[j] < shapeY - 0.6: #10.4:
                size=sizes[i]/100
                star_color = "white"
                zorder = 1
                if catDatatpf[i]['ID'] == str(ticid):
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
    if mask_edges:
        for i in range(len(points_mask[0])):
            for j in range(len(points_mask[1])):
                if points_mask[i][j] is not None:
                    axes[1].add_patch(points_mask[i][j]) 
    if mask_edges:
        axes[1].set_title("Difference Image (edge mask)\nMean Out - In Transit Flux".format(transit_dur), fontsize = 11 )
    else:
        axes[1].set_title("Difference Image\nMean Out - In Transit Flux".format(transit_dur), fontsize = 11 )

    axes[2].grid(lw=0.2)
    axes[2].tick_params(axis='both', which='major', labelsize=9)
    axes[2].minorticks_on()

    # prf centroid cross (21px)
    x01 = np.array([ra0-10.5, ra0+10.5])
    y01 = np.array([dec0, dec0])
    x02 = np.array([ra0, ra0])
    y02 = np.array([dec0-10.5, dec0+10.5])
    axes[2].plot(x01, y01, x02, y02, c='green', lw=0.8)

    if plot_flux_centroid:
        axes[2].scatter(ra1,dec1, marker = '+', s=200,lw=0.6,c='blue')
    
    axes[2].scatter(0,0, marker = '*', c='red')
    if prfError:
        axes[2].set_title("Difference Image Flux Centroid\n(Error in PRF centroid)", fontsize = 11 )
    else:
        axes[2].set_title("Difference Image PRF Centroid\nPRF fit Quality = "+str(round(prfFitQuality,3)), fontsize = 11 )

    xmin, xmax = axes[2].get_xlim()
    ymin, ymax = axes[2].get_ylim() 
    for i in range(nstars):
        bkxmin = min(ra[i]-2, xmin)
        bkxmax = max(ra[i]+2, xmax)
        bkymin = min(dec[i]-2, ymin)
        bkymax = max(dec[i]+2, ymax)
        size = max(abs(bkxmax-bkxmin), abs(bkymax-bkymin))
        if size > 54:  
           break
        xmin, xmax, ymin, ymax = bkxmin, bkxmax, bkymin, bkymax
    tam1 = abs(xmax-xmin)
    tam2 = abs(ymax-ymin)
    if tam1 > tam2:  
        ymin=ymin-(tam1-tam2)/2
        ymax=ymax+(tam1-tam2)/2
    if tam2 > tam1:
        xmin=xmin-(tam2-tam1)/2
        xmax=xmax+(tam2-tam1)/2
    tam1 = abs(xmax-xmin)

    axes[2].set_xlim([xmin, xmax])
    axes[2].set_ylim([ymin, ymax])
    #xmin, xmax = axes[2].get_xlim()
    #ymin, ymax = axes[2].get_ylim()
    
    for i in range(nstars):
        scolor = 'blue'
        if cat[i] == str(ticid):
            scolor = 'red'
        if ra[i] < xmax and ra[i] > xmin and dec[i] < ymax and dec[i] > ymin: 
            axes[2].scatter(ra[i],dec[i], marker = '*', c=scolor)
            axes[2].text(ra[i]-tam1/34,dec[i], i+1, fontsize=9, c=scolor, zorder=2)      
            
    axes[2].invert_xaxis()
    axes[2].set_xlabel("RA Offset (arcsec)", fontsize=10,labelpad=2)
    axes[2].set_ylabel("Dec Offset (arcsec)", fontsize=10,labelpad=0)
    axes[3].axis('off')
    axes[3].set_xlim([0, 1])
    axes[3].set_ylim([0, 1])
    yl1 = 1.05 #1.036
    axes[3].set_title('Nearest TICs to centroid   \n(Tmag<'+str(maglim)+')  ', fontsize = 11 )
    axes[3].text(0.5, yl1-0.1,'Tmag', fontsize=9, c='black') 
    axes[3].text(0.67, yl1-0.1,'Sep.arcsec', fontsize=9, c='black') 
    star_ok = False
    pm_any = False
    z = 0
    for i in range(nstars):
        colort = 'black'
        if cat[i] == str(ticid):
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
        tic_row = catalogData[catalogData['ID'] == str(ticid)][0]
        sep_label =  f"{tic_row['dstArcSec']:6.2f}"
        if not tic_row['has_pm']:
            sep_label += ' (#)'
        axes[3].text(-0.06, yl1-0.09-z/10, '* - TIC '+str(ticid), fontsize=8, c='red')    
        axes[3].text(0.48, yl1-0.09-z/10, f"{tic_row['Tmag']:6.3f}", fontsize=9, c='black')    
        axes[3].text(0.78, yl1-0.09-z/10, sep_label, fontsize=9, c='black')                

    if pm_any:
        pm_text = '(#) No proper motion correction.'
    else:
        pm_text = ' Stars are proper motion corrected.'
    z = z+1    
    axes[3].text(0.01, yl1-0.12-z/10, pm_text, fontsize=10, c='black')    

    plt.gcf().subplots_adjust(bottom=0.25, top=0.95, left=0.05,right=0.95,wspace=0.26)

    if len(validEpochs) < 6:
        epochs3 = validEpochs.copy()
        for i,t in enumerate(epochs3):
            epochs3[i] = round(t,3)
        fig.suptitle(TIC2str+' Sector ' +str(sector)+'            Transit epochs (BTJD)= '+str(epochs3)+'            Transit duration (hours)= '+ f'{transit_dur*24:2.3f}',
                     fontsize=12, x=0.49, y=1.04)
    else:
        fig.suptitle(TIC2str+' Sector ' +str(sector)+'            Total transits = '+str(ntransits)+'            Transit duration (hours)= '+ f'{transit_dur*24:2.3f}',
                     fontsize=12, x=0.49, y=1.04)
    com_xlabel = "In Transit cadences: " + str(inTransitCad)+"  (Epoch ± "+f'{inTMargin:2.3f}d)   '
    com_xlabel += "         Out Of Transit cadences: "+str(ooTransitCad)+"  (Epoch ± "+f'{ooTInnerM:2.3f}d' 
    com_xlabel += "  to  Epoch ± "+ f'{ooTOuterM:2.3f}d)'
    fig.text(0.5, 0.97, com_xlabel, ha='center', fontsize = 10)    
    plt.show()
    
    if prfError:
        flux_centroid_pos = prf_centroid_pos
        prf_centroid_pos = None
        prfFitQuality = None
        img_prf = None

    return _get_results(ticid, sector, validEpochs, transit_dur, inTransitCad, ooTransitCad, inTMargin, ooTInnerM, ooTOuterM, 
                                tic_pos, flux_centroid_pos, prf_centroid_pos, prfFitQuality, tic_offset, nearest_tics,
                                img_diff_to_return, img_oot, img_intr, img_prf)       

# Difference Image calculation - Adapted from :
# Nora Eisner - Planet Hunters Coffee Chat - False Positives - In Out Transit Flux 
# https://github.com/noraeisner/PH_Coffee_Chat/blob/main/False%20Positive/False%20positives%20-%20(2)%20in%20out%20transit%20flux.ipynb
#============================================================================================================
def _get_in_out_diff_img(tpf, epochs, inTMargin, ooTInnerM, ooTOuterM): 
    #epochs: float or list of floats - If more than one, a mean image of all transits is calculated
    imgs_intr = []; imgs_oot = []
    times = tpf.time.value
    flux = tpf.flux.value
    for T0 in epochs:
        intr = abs(T0 - times) < inTMargin  # mask of in transit times
        oot = (abs(T0 - times) < ooTOuterM) * (abs(T0 - times) > ooTInnerM)  # mask of out transit times
        img_intr = np.nanmean(flux[intr,:,:], axis=0)
        img_oot = np.nanmean(flux[oot, :, :], axis=0)
        imgs_intr.append(img_intr) 
        imgs_oot.append(img_oot)
    img_intr = np.nanmean(imgs_intr, axis=0)
    img_oot = np.nanmean(imgs_oot, axis=0)
    img_diff = img_oot - img_intr
    return img_diff, img_intr, img_oot

def _get_margins(transit_dur, intransit_margin=0, ootransit_inner=0, ootransit_outer=0): 
    if intransit_margin == 0:
        intransit_margin = round(transit_dur * 0.4,3)
    if ootransit_inner == 0:
        ootransit_inner = round(transit_dur * 0.75, 3)
    if ootransit_outer == 0:
        ootransit_outer = round(ootransit_inner + transit_dur, 3)
    return intransit_margin, ootransit_inner, ootransit_outer
    
def _check_epochs(tpf, epochs, intransit_margin, ootransit_inner, ootransit_outer, gap_factor):
    validEpochs = []
    inTransitCad = 0
    ooTransitCad = 0
    tv = tpf.time.value
    for T0 in epochs:
        t = tv[abs(T0-tv)<ootransit_outer]
        diff = np.diff(t)
        if np.max(diff) < np.min(diff) * gap_factor:   #discard transits wuth gaps
            validEpochs.append(T0)
            inTransitCad += len(t[abs(T0 - t) < intransit_margin])
            ooTransitCad += len(t[(abs(T0-t) < ootransit_outer) * (abs(T0-t) > ootransit_inner)])
        # error if validEpochs == []        
    return validEpochs, inTransitCad, ooTransitCad

def _get_tpf_stars_catalog(tpf, maglim):
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
        except Exception as error:
            attempts += 1
            if attempts > 2:
                raise
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
    return catDatatpf
 
def _get_results(*args):
    return dict(zip(
                (
                    'ticid' ,
                    'sector',
                    'valid_transit_epochs',     # (list)   #     
                    'transit_duration' ,          # in days
                    'inTransit_cadences' ,      # no. of cadences in In Transir Image
                    'ooTransit_cadences' ,     # no. of cadences in Out Of Transir Image
                    'inTransit_margin' ,          # In Transit windows (epochs-inTransit_margin <-> epochs+inTransit_margin)
                    'ooTransit_inner_margin' , # Out of Transit wibdows:
                    'ooTransit_outer_margin' , #  (epochs +/- inner -~<->to epochs +/- outer)
                    'tic_pos' ,                       # relative tic position to (0,0)
                    'flux_centroid_pos' ,         #relative flux centroid position to (0,0)
                    'prf_centroid_pos' ,          #relative prf centroid position to (0,0)
                    'prf_fit_quality' ,              # 0..1
                    'tic_offset' ,                   # tic offset to centroid (arcsec)
                    'nearest_tics' ,               # nearest TICs to centroid Table.    
                    'img_diff' ,                     #difference image
                    'img_oot' ,                     #out of transit image
                    'img_intr' ,                     #in transit image
                    'img_prf'                       # prf image
                ), args))

def _mask_edges(img):
    emask = np.full(img.shape, True, dtype=bool)
    emask[2:-2, 2:-2] = False
    cimg = img.copy()
    cimg[emask]=0
    yy, xx = np.unravel_index(cimg.argmax(), cimg.shape)
    maxflux = img[yy,xx]        
    points = [[None for j in range(emask.shape[0])] for i in range(emask.shape[1])]
    for i in range(emask.shape[0]):
        for j in range(emask.shape[1]):
            if emask[i, j]:
                if img[i,j]>=maxflux:
                    img[i,j] = 0            
                    points[i][j] = patches.Rectangle(
                        xy=(j - 0.5, i - 0.5),
                        width=1,
                        height=1,
                        color='coral',
                        fill=False,
                        hatch="///",
                    )
    return xx, yy, img, points
    #ax.add_patch(points[i][j])    
                    
def _get_offset(coord1,coord2):
    ra_offset = (coord2.ra - coord1.ra) * np.cos(coord1.dec.to('radian'))
    dec_offset = (coord2.dec - coord1.dec)
    return float(str(ra_offset.to('arcsec'))[:-6]), float(str(dec_offset.to('arcsec'))[:-6])

def formatwarning(msg, *args, **kwargs):
    return 'Warning:\n' + str(msg) + '\n'
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
    
# =================================================================
# ---- additional lightcurve plots to help visualization of the time span measured --------
#--------------- Sam Lee - https://github.com/orionlee    -----------------------------
#------------------------------------------------------------------------------------

def show_transit_margins(tpf, epoch, transit_dur, intransit_margin=0, ootransit_inner=0, ootransit_outer=0, interact=False):
    #def iPlot(inT, ooTI, ooTO, T0):
    #   plot(inT.value, ooTI.value, ooTO.value, T0)
    #   return inT, ooTO, ooTI
    def plot(inT, ooTI, ooTO, T0):
        ax = plt.figure(figsize=(8, 4.5)).gca()
        ax = lc.scatter(ax=ax, s=20, c='k')
        #xmin, xmax = ax.get_xlim()
        ax.axvline(T0, color="red", ymax=0.15, linewidth=1, linestyle="--", label="epoch")
        ax.axvspan(T0 - inT, T0 + inT, facecolor="red", alpha=0.3, label="In Transit")
        ax.axvspan(T0 - ooTO, T0 - ooTI, facecolor="green", alpha=0.3, label="Out of Transit")
        ax.axvspan(T0 + ooTI, T0 + ooTO, facecolor="green", alpha=0.3)
        ax.set_xticks([T0 - ooTO, T0 - ooTI,T0 - inT, T0, T0 + inT, T0 + ooTI, T0 + ooTO])
        ax.set_xticklabels(['-%.3f' % ooTO,'-%.3f' % ooTI,'\n-%.3f' % inT,
                      '%.3f' % T0, '\n+%.3f' % inT, '+%.3f' % ooTI, '+%.3f' % ooTO])    
        ax.legend(loc="lower right", fontsize=8)
        for item in (ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(10)
        return
        
    if isinstance(epoch, float): epoch = [epoch]
    #
    inTMargin, ooTInnerM, ooTOuterM = _get_margins(transit_dur, intransit_margin, ootransit_inner, ootransit_outer)
    #
    lc_target = tpf.to_lightcurve().remove_nans()
    T0 = epoch[0]
    lc = lc_target.truncate(T0 - ooTOuterM * 1.25, T0 + ooTOuterM * 1.25)
    if interact == False:
        plot(inTMargin, ooTInnerM, ooTOuterM, T0)
        return
    #------------ Interact = True -------------------
    from ipywidgets import interact,fixed, Label, HBox
    import ipywidgets as widgets
    #
    xmax = T0 + ooTOuterM * 1.25
    inT = widgets.FloatSlider(value=inTMargin, min=0.01, max=ooTInnerM, step=0.002,disabled=False,readout_format='.3f')
    ooTI = widgets.FloatSlider(value=ooTInnerM, min=inTMargin, max=ooTOuterM, step=0.002, disabled=False,readout_format='.3f')
    ooTO = widgets.FloatSlider(value=ooTOuterM, min=ooTInnerM, max=xmax-T0-0.01, step=0.002, disabled=False,readout_format='.3f')
    BinT = HBox([Label('InTransit Margin'), inT])
    BooTI = HBox([Label('ooT Inner Margin'), ooTI])
    BooTO = HBox([Label('ooT Outer Margin'), ooTO])
    box_layout = widgets.Layout(display='flex', flex_flow='column', align_items='center', width='570px')
    ui = widgets.VBox([BinT, BooTI, BooTO], layout=box_layout)
    out = widgets.interactive_output(plot, {'inT': inT, 'ooTI': ooTI, 'ooTO': ooTO, 'T0':fixed(T0)})
    display(out,ui)
    return inT, ooTI, ooTO
