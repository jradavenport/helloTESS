import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import os
from glob import glob
from matplotlib.colors import LogNorm

from scipy.optimize import curve_fit

from astropy.table import Table
import astropy.io.fits as fits
from astropy.stats import LombScargle, BoxLeastSquares
import exoplanet as xo
from stuff import FINDflare, EasyE

matplotlib.rcParams.update({'font.size':18})
matplotlib.rcParams.update({'font.family':'serif'})

ftype = '.pdf'


def RunSectors(tess_dir = '/Users/james/Desktop/tess/', run_dir = '/Users/james/Desktop/helloTESS/'):
    '''
    Do some simplier things on stars that are observed in mulitple sectors

    should probably be combined with run_sector.py.... but oh well for now!
    '''

    sectors = ['sector001', 'sector002', 'sector003', 'sector004', 'sector005', 'sector006', 'sector007']

    # just in case glob wants to re-order things, be sure grab them in Sector order
    files = []
    for k in range(len(sectors)):
        files = files + glob(tess_dir + sectors[k] + '/*.fits', recursive=True)

    # get the unique object IDs (NOT the simplest way, but matches the next step)
    obj = pd.Series(files).str.split('-', expand=True).groupby(by=2).count().index

    # get the count of unique object IDs
    Nobj = pd.Series(files).str.split('-', expand=True).groupby(by=2).count()[0]

    # for k in range(max(Nobj)):
    #     print(k+1, sum(Nobj > k))
    # obj[0] # example Object ID (TIC #)

    o5 = np.where((Nobj > 3))[0] # was named "o5" because originally wanted Over 5 observations. Now pick other N

    for k in range(0, len(o5)):
        print(k, obj[o5][k])
        files_k = pd.Series(files)[np.where((pd.Series(files).str.split('-', expand=True)[2] == obj[o5][k]))[0]].values

        rot_out_k = MultiSector(files_k)
        if k==0:
            rot_out = rot_out_k
        else:
            rot_out = pd.concat([rot_out, rot_out_k], ignore_index=True, sort=False)
    rot_out.o_csv(run_dir + 'longerP_flare_out.csv')
    return


def MultiSector(TICs, tess_dir = '/Users/james/Desktop/tess/',
                run_dir = '/Users/james/Desktop/helloTESS/',
                clobber=False):
    '''
    Run the basic set of tools on every light curve -> NOW FOR MULTI-SECTOR DATA

    Produce a diagnostic plot for each light curve

    '''

    if not os.path.isdir(run_dir + 'figures/longerP'):
        os.makedirs(run_dir + 'figures/longerP')

    tbit = False
    for k in range(len(TICs)):
        tbl = -1
        try:
            tbl = Table.read(TICs[k], format='fits')
            tbl['PDCSAP_FLUX'] = tbl['PDCSAP_FLUX'] - np.nanmedian(tbl['PDCSAP_FLUX'])

            if tbit == False:
                df_tbl = tbl.to_pandas()
                tbit = True
            else:
                df_tbl = pd.concat([df_tbl, tbl.to_pandas()], ignore_index=True, sort=False)

        except:
            tbl = -1
            print('bad file: ' + TICs[k])



    df_tbl['PDCSAP_FLUX'] = df_tbl['PDCSAP_FLUX'] + np.nanmedian(df_tbl['SAP_FLUX'])

    # make harsh quality cuts, and chop out a known bad window of time (might add more later)
    AOK = (df_tbl['QUALITY'] == 0) & ((df_tbl['TIME'] < 1347) | (df_tbl['TIME'] > 1350))

    # do a running median for a basic smooth
    smo = df_tbl['PDCSAP_FLUX'][AOK].rolling(128, center=True).median().values
    med = np.nanmedian(smo)


    # make an output plot for every file
    figname = run_dir + 'figures/longerP/' + TICs[0].split('-')[2] + '.jpeg'
    makefig = ((not os.path.exists(figname)) | clobber)

    if makefig:
        plt.figure(figsize=(14,6))
        plt.errorbar(df_tbl['TIME'][AOK], df_tbl['PDCSAP_FLUX'][AOK]/med, yerr=df_tbl['PDCSAP_FLUX_ERR'][AOK]/med,
                     linestyle=None, alpha=0.25, label='PDC_FLUX')
        plt.plot(df_tbl['TIME'][AOK], smo/med, label='128pt MED', c='orange')

#     Smed = np.nanmedian(df_tbl['SAP_FLUX'][AOK])
#     plt.errorbar(df_tbl['TIME'][AOK], df_tbl['SAP_FLUX'][AOK]/Smed, yerr=df_tbl['SAP_FLUX_ERR'][AOK]/Smed,
#                  linestyle=None, alpha=0.25, label='SAP_FLUX')


    # require at least 1000 good datapoints for analysis
    if sum(AOK) > 1000:
        # find OK points in the smoothed LC
        SOK = np.isfinite(smo)


        # Lomb Scargle
        LS = LombScargle(df_tbl['TIME'][AOK][SOK], smo[SOK]/med, dy=df_tbl['PDCSAP_FLUX_ERR'][AOK][SOK]/med)
        frequency, power = LS.autopower(minimum_frequency=1./40.,
                                        maximum_frequency=1./0.1,
                                        samples_per_peak=7)
        best_frequency = frequency[np.argmax(power)]

        per_out = 1./best_frequency
        per_amp = np.nanmax(power)
        per_med = np.nanmedian(power)
        per_std = np.nanstd(smo[SOK]/med)

        if np.nanmax(power) > 0.2:
            LSmodel = LS.model(df_tbl['TIME'][AOK][SOK], best_frequency)
            if makefig:
                plt.plot(df_tbl['TIME'][AOK][SOK], LSmodel,
                     label='L-S P='+format(1./best_frequency, '6.3f')+'d, pk='+format(np.nanmax(power), '6.3f'),
                     c='green')


        # ACF w/ Exoplanet package
        acf = xo.autocorr_estimator(df_tbl['TIME'][AOK][SOK].values, smo[SOK]/med,
                                    yerr=df_tbl['PDCSAP_FLUX_ERR'][AOK][SOK].values/med,
                                    min_period=0.1, max_period=40, max_peaks=2)
        ACF_1pk = -1
        ACF_1dt = -1
        if len(acf['peaks']) > 0:
            ACF_1dt = acf['peaks'][0]['period']
            ACF_1pk = acf['autocorr'][1][np.where((acf['autocorr'][0] == acf['peaks'][0]['period']))[0]][0]

            if makefig:
                plt.plot(df_tbl['TIME'][AOK][SOK],
                         np.nanstd(smo[SOK]/med) * ACF_1pk * np.sin(df_tbl['TIME'][AOK][SOK] / ACF_1dt * 2 * np.pi) + 1,
                         label = 'ACF=' + format(ACF_1dt, '6.3f') + 'd, pk=' + format(ACF_1pk, '6.3f'), lw=2,
                         alpha=0.7, c='FireBrick')


        # here is where a simple Eclipse (EB) finder goes
        EE = EasyE(smo[SOK]/med, df_tbl['PDCSAP_FLUX_ERR'][AOK][SOK]/med, N1=5, N2=3, N3=2)
        EclFlg = 0
        if np.size(EE) > 0:
            EclFlg = 1
            if makefig:
                for j in range(len(EE[0])):
                    plt.scatter(df_tbl['TIME'][AOK][SOK][(EE[0][j]):(EE[1][j]+1)],
                                smo[SOK] [(EE[0][j]):(EE[1][j]+1)] / med,
                                color='k', marker='s', s=5, alpha=0.75, label='_nolegend_')
                plt.scatter([],[], color='k', marker='s', s=5, alpha=0.75, label='Ecl?')



        # add BLS
#         bls = BoxLeastSquares(df_tbl['TIME'][AOK][SOK], smo[SOK]/med, dy=df_tbl['PDCSAP_FLUX_ERR'][AOK][SOK]/med)
#         blsP = bls.autopower(0.1, method='fast', objective='snr')
#         blsPer = blsP['period'][np.argmax(blsP['power'])]
#         if ((4*np.nanstd(blsP['power']) + np.nanmedian(blsP['power']) < np.nanmax(blsP['power'])) &
#             (np.nanmax(blsP['power']) > 50.) &
#             (blsPer < 0.95 * np.nanmax(blsP['period']))
#            ):
#             blsPeriod = blsPer
#             blsAmpl = np.nanmax(blsP['power'])
#             plt.plot([],[], ' ', label='BLS='+format(blsPer, '6.3f')+'d')

    if makefig:
        plt.title(TICs[0].split('-')[2], fontsize=12)
        plt.ylabel('Flux')
        plt.xlabel('BJD - 2457000 (days)')
        plt.legend(fontsize=10)

        plt.savefig(figname, bbox_inches='tight', pad_inches=0.25, dpi=100)
        plt.close()



#     # write per-sector output files
#     ALL_TIC = pd.Series(files_i).str.split('-', expand=True).iloc[:,-3].astype('int')

#     flare_out = pd.DataFrame(data={'TIC':ALL_TIC[FL_id], 'i0':FL_t0, 'i1':FL_t1, 'med':FL_f0, 'peak':FL_f1})
#     flare_out.to_csv(run_dir + sector + '_flare_out.csv')

    rot_out = pd.DataFrame(data={'TIC':TICs[0].split('-')[2],
                                 'per':per_out, 'Pamp':per_amp, 'Pmed':per_med, 'StdLC':per_std,
                                 'acf_pk':ACF_1pk, 'acf_per':ACF_1dt, 'ecl_flg':EclFlg}, index=[0])
                                 # 'bls_period':blsPeriod, 'bls_ampl':blsAmpl, )
#     rot_out.to_csv(run_dir + sector + '_rot_out.csv')

    return rot_out


if __name__ == "__main__":
    '''
      let this file be called from the terminal directly. e.g.:
      $ python analysis.py
    '''
    RunSectors()




### junk code i probably dont need
# sect1 = glob(tess_dir + sectors[0] + '/*.fits', recursive=True)
# sect2 = glob(tess_dir + sectors[1] + '/*.fits', recursive=True)
# sect3 = glob(tess_dir + sectors[2] + '/*.fits', recursive=True)
# sect4 = glob(tess_dir + sectors[3] + '/*.fits', recursive=True)
# sect5 = glob(tess_dir + sectors[4] + '/*.fits', recursive=True)
# sect6 = glob(tess_dir + sectors[5] + '/*.fits', recursive=True)
#
# files = sect1 + sect2 + sect3 + sect4 + sect5 + sect6
# # make into an array for looping later!
# s_lens = [len(sect1), len(sect2), len(sect3), len(sect4), len(sect5), len(sect6)]
# print(s_lens, len(files))
