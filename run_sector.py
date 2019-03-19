import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import os
from glob import glob
import sys

from scipy.optimize import curve_fit

from astropy.table import Table
import astropy.io.fits as fits
from astropy.stats import LombScargle, BoxLeastSquares
import exoplanet as xo
from stuff import FINDflare, EasyE

matplotlib.rcParams.update({'font.size':18})
matplotlib.rcParams.update({'font.family':'serif'})

ftype = '.pdf'


# tess_dir = '/data/epyc/data/tess/'
# tess_dir = '/Users/james/Desktop/tess/'
#
# sectors = ['sector001', 'sector002', 'sector003', 'sector004', 'sector005', 'sector006']
#
# # just in case glob wants to re-order things, be sure grab them in Sector order
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


def BasicActivity(sector, tess_dir = '/Users/james/Desktop/tess/', run_dir = '/Users/james/Desktop/helloTESS/', clobber=False):
    '''
    Run the basic set of tools on every light curve

    Produce a diagnostic plot for each light curve

    Save a file on Rotation stats and a file on Flare stats
    '''

    print('running ' + tess_dir + sector)
    files_i = glob(tess_dir + sector + '/*.fits', recursive=True)
    print(str(len(files_i)) + ' .fits files found')

    # arrays to hold outputs
    per_out = np.zeros(len(files_i)) -1
    per_amp = np.zeros(len(files_i)) -1
    per_med = np.zeros(len(files_i)) -1
    per_std = np.zeros(len(files_i)) -1

    ACF_1pk = np.zeros(len(files_i)) -1
    ACF_1dt = np.zeros(len(files_i)) -1

    blsPeriod = np.zeros(len(files_i)) -1
    blsAmpl = np.zeros(len(files_i)) -1

    EclFlg = np.zeros(len(files_i)) -1

    FL_id = np.array([])
    FL_t0 = np.array([])
    FL_t1 = np.array([])
    FL_f0 = np.array([])
    FL_f1 = np.array([])


    if not os.path.isdir(run_dir + 'figures/' + sector):
        os.makedirs(run_dir + 'figures/' + sector)

    for k in range(len(files_i)):
        tbl = -1
        df_tbl = -1
        try:
            tbl = Table.read(files_i[k], format='fits')
            df_tbl = tbl.to_pandas()
        except (OSError, KeyError, TypeError, ValueError):
            print('k=' + str(k) + ' bad file: ' + files_i[k])

        # this is a bit clumsy, but it made sense at the time when trying to chase down some bugs...
        if tbl != -1:
            # make harsh quality cuts, and chop out a known bad window of time (might add more later)
            AOK = (tbl['QUALITY'] == 0) & ((tbl['TIME'] < 1347) | (tbl['TIME'] > 1350))

            # do a running median for a basic smooth
            smo = df_tbl['PDCSAP_FLUX'][AOK].rolling(128, center=True).median()
            med = np.nanmedian(smo)
            Smed = np.nanmedian(tbl['SAP_FLUX'][AOK])

            # make an output plot for every file
            figname = run_dir + 'figures/' + sector + '/' + files_i[k].split('/')[-1] + '.jpeg' #run_dir + 'figures/longerP/' + TICs[0].split('-')[2] + '.jpeg'
            makefig = ((not os.path.exists(figname)) | clobber)

            if makefig:
                plt.figure(figsize=(12,9))
                plt.errorbar(tbl['TIME'][AOK], tbl['PDCSAP_FLUX'][AOK]/med, yerr=tbl['PDCSAP_FLUX_ERR'][AOK]/med,
                             linestyle=None, alpha=0.25, label='PDC_FLUX')
                plt.plot(tbl['TIME'][AOK], smo/med, label='128pt MED')

                plt.errorbar(tbl['TIME'][AOK], tbl['SAP_FLUX'][AOK]/Smed, yerr=tbl['SAP_FLUX_ERR'][AOK]/Smed,
                             linestyle=None, alpha=0.25, label='SAP_FLUX')

            # require at least 1000 good datapoints for analysis
            if sum(AOK) > 1000:
                # find OK points in the smoothed LC
                SOK = np.isfinite(smo)

                # flares
                FL = FINDflare((df_tbl['PDCSAP_FLUX'][AOK][SOK] - smo[SOK])/med,
                               df_tbl['PDCSAP_FLUX_ERR'][AOK][SOK]/med,
                               N1=4, N2=2, N3=5, avg_std=False)

                if np.size(FL) > 0:
                    for j in range(len(FL[0])):
                        FL_id = np.append(FL_id, k)
                        FL_t0 = np.append(FL_t0, FL[0][j])
                        FL_t1 = np.append(FL_t1, FL[1][j])
                        FL_f0 = np.append(FL_f0, med)
                        FL_f1 = np.append(FL_f1, np.nanmax(tbl['PDCSAP_FLUX'][AOK][SOK][(FL[0][j]):(FL[1][j]+1)]))

                if makefig:
                    if np.size(FL) > 0:
                        for j in range(len(FL[0])):
                            plt.scatter(tbl['TIME'][AOK][SOK][(FL[0][j]):(FL[1][j]+1)],
                                        tbl['PDCSAP_FLUX'][AOK][SOK][(FL[0][j]):(FL[1][j]+1)] / med, color='r',
                                        label='_nolegend_')
                        plt.scatter([],[], color='r', label='Flare?')



                # Lomb Scargle
                LS = LombScargle(df_tbl['TIME'][AOK][SOK], smo[SOK]/med, dy=df_tbl['PDCSAP_FLUX_ERR'][AOK][SOK]/med)
                frequency, power = LS.autopower(minimum_frequency=1./40.,
                                                maximum_frequency=1./0.1,
                                                samples_per_peak=7)
                best_frequency = frequency[np.argmax(power)]

                per_out[k] = 1./best_frequency
                per_amp[k] = np.nanmax(power)
                per_med[k] = np.nanmedian(power)
                per_std[k] = np.nanstd(smo[SOK]/med)

                if np.nanmax(power) > 0.2:
                    LSmodel = LS.model(df_tbl['TIME'][AOK][SOK], best_frequency)
                    if makefig:
                        plt.plot(df_tbl['TIME'][AOK][SOK], LSmodel,
                                 label='L-S P='+format(1./best_frequency, '6.3f')+'d, pk='+format(np.nanmax(power), '6.3f'))


                # ACF w/ Exoplanet package
                acf = xo.autocorr_estimator(tbl['TIME'][AOK][SOK], smo[SOK]/med,
                                            yerr=tbl['PDCSAP_FLUX_ERR'][AOK][SOK]/med,
                                            min_period=0.1, max_period=40, max_peaks=2)
                if len(acf['peaks']) > 0:
                    ACF_1dt[k] = acf['peaks'][0]['period']
                    ACF_1pk[k] = acf['autocorr'][1][np.where((acf['autocorr'][0] == acf['peaks'][0]['period']))[0]][0]

                if (ACF_1dt[k] > 0) & makefig:
                    plt.plot(tbl['TIME'][AOK][SOK],
                             np.nanstd(smo[SOK]/med) * ACF_1pk[k] * np.sin(tbl['TIME'][AOK][SOK] / ACF_1dt[k] * 2 * np.pi) + 1,
                             label = 'ACF=' + format(ACF_1dt[k], '6.3f') + 'd, pk=' + format(ACF_1pk[k], '6.3f'), lw=2, alpha=0.7)


                # here is where a simple Eclipse (EB) finder goes
                EE = EasyE(smo[SOK]/med, df_tbl['PDCSAP_FLUX_ERR'][AOK][SOK]/med, N1=5, N2=3, N3=5)
                if (np.size(EE) > 0):
                    if makefig:
                        for j in range(len(EE[0])):
                            plt.scatter(tbl['TIME'][AOK][SOK][(EE[0][j]):(EE[1][j]+1)],
                                        smo[SOK] [(EE[0][j]):(EE[1][j]+1)] / med,
                                        color='k', marker='s', s=5, alpha=0.75, label='_nolegend_')
                        plt.scatter([],[], color='k', marker='s', s=5, alpha=0.75, label='Ecl?')
                    EclFlg[k] = 1


                # add BLS
                bls = BoxLeastSquares(df_tbl['TIME'][AOK][SOK], smo[SOK]/med, dy=df_tbl['PDCSAP_FLUX_ERR'][AOK][SOK]/med)
                blsP = bls.autopower(0.1, method='fast', objective='snr')
                blsPer = blsP['period'][np.argmax(blsP['power'])]
                if ((4*np.nanstd(blsP['power']) + np.nanmedian(blsP['power']) < np.nanmax(blsP['power'])) &
                    (np.nanmax(blsP['power']) > 50.) &
                    (blsPer < 0.95 * np.nanmax(blsP['period']))
                   ):
                    blsPeriod[k] = blsPer
                    blsAmpl[k] = np.nanmax(blsP['power'])
                    if makefig:
                        plt.plot([],[], ' ', label='BLS='+format(blsPer, '6.3f')+'d')


            if makefig:
                plt.title(files_i[k].split('/')[-1] + ' k='+str(k), fontsize=12)
                plt.ylabel('Flux')
                plt.xlabel('BJD - 2457000 (days)')
                plt.legend(fontsize=10)
                plt.savefig(figname, bbox_inches='tight', pad_inches=0.25, dpi=100)
                plt.close()



    # write per-sector output files
    ALL_TIC = pd.Series(files_i).str.split('-', expand=True).iloc[:,-3].astype('int')

    flare_out = pd.DataFrame(data={'TIC':ALL_TIC[FL_id], 'i0':FL_t0, 'i1':FL_t1, 'med':FL_f0, 'peak':FL_f1})
    flare_out.to_csv(run_dir + 'outputs/' + sector + '_flare_out.csv')

    rot_out = pd.DataFrame(data={'TIC':ALL_TIC,
                                 'per':per_out, 'Pamp':per_amp, 'Pmed':per_med, 'StdLC':per_std,
                                 'acf_pk':ACF_1pk, 'acf_per':ACF_1dt,
                                 'bls_period':blsPeriod, 'bls_ampl':blsAmpl, 'ecl_flg':EclFlg})
    rot_out.to_csv(run_dir + 'outputs/' + sector + '_rot_out.csv')



if __name__ == "__main__":
    '''
      let this file be called from the terminal directly. e.g.:
      $ python analysis.py
    '''
    BasicActivity(sys.argv[1])
