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
import pymc3 as pm
import theano.tensor as tt
from stuff import FINDflare, EasyE, IRLSSpline

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


def BasicActivity(sector, tess_dir = '/Users/james/Desktop/tess/',
                  run_dir = '/Users/james/Desktop/helloTESS/',
                  clobber=False):
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

    EclNum = np.zeros(len(files_i)) -1
    EclDep = np.zeros(len(files_i)) -1

    FL_id = np.array([])
    FL_t0 = np.array([])
    FL_t1 = np.array([])
    FL_f0 = np.array([])
    FL_f1 = np.array([])


    if not os.path.isdir(run_dir + 'figures/' + sector):
        os.makedirs(run_dir + 'figures/' + sector)

    for k in range(len(files_i)):
        # print(files_i[k])

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
            med = np.nanmedian(df_tbl['PDCSAP_FLUX'][AOK])

            # ACF w/ Exoplanet package
            acf = xo.autocorr_estimator(tbl['TIME'][AOK], tbl['PDCSAP_FLUX'][AOK] / med,
                                        yerr=tbl['PDCSAP_FLUX_ERR'][AOK] / med,
                                        min_period=0.1, max_period=27, max_peaks=2)
            if len(acf['peaks']) > 0:
                ACF_1dt[k] = acf['peaks'][0]['period']
                ACF_1pk[k] = acf['autocorr'][1][np.where((acf['autocorr'][0] == acf['peaks'][0]['period']))[0]][0]

                s_window = int(ACF_1dt[k] / np.abs(np.nanmedian(np.diff(tbl['TIME']))) / 6.)
            else:
                s_window = 128

            # do a running median for a basic smooth
            # smo = (df_tbl['PDCSAP_FLUX'][AOK].rolling(128, center=True).median() + df_tbl['PDCSAP_FLUX'][AOK].rolling(256, center=True).median()) / 2.
            smo = df_tbl['PDCSAP_FLUX'][AOK].rolling(s_window, center=True).median()

            # make an output plot for every file
            figname = run_dir + 'figures/' + sector + '/' + files_i[k].split('/')[-1] + '.jpeg' #run_dir + 'figures/longerP/' + TICs[0].split('-')[2] + '.jpeg'
            makefig = ((not os.path.exists(figname)) | clobber)

            if makefig:
                plt.figure(figsize=(12,9))
                plt.errorbar(tbl['TIME'][AOK], tbl['PDCSAP_FLUX'][AOK]/med, yerr=tbl['PDCSAP_FLUX_ERR'][AOK]/med,
                             linestyle=None, alpha=0.15, label='PDC_FLUX')
                plt.plot(tbl['TIME'][AOK], smo/med, label=str(s_window)+'pt MED')

                if (ACF_1dt[k] > 0):
                    plt.plot(tbl['TIME'][AOK],
                             np.nanstd(smo / med) * ACF_1pk[k] * np.sin(tbl['TIME'][AOK] / ACF_1dt[k] * 2 * np.pi) + 1,
                             label='ACF=' + format(ACF_1dt[k], '6.3f') + 'd, pk=' + format(ACF_1pk[k], '6.3f'), lw=2,
                             alpha=0.7)
                # plt.errorbar(tbl['TIME'][AOK], tbl['SAP_FLUX'][AOK]/Smed, yerr=tbl['SAP_FLUX_ERR'][AOK]/Smed,
                #              linestyle=None, alpha=0.25, label='SAP_FLUX')

            # require at least 1000 good datapoints for analysis
            if sum(AOK) > 1000:
                # find OK points in the smoothed LC
                SOK = np.isfinite(smo)

                # do some SPLINE'ing
                # spl = IRLSSpline(df_tbl['TIME'].values[AOK][SOK], df_tbl['PDCSAP_FLUX'].values[AOK][SOK] / med,
                #                  df_tbl['PDCSAP_FLUX_ERR'].values[AOK][SOK] / med)

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
                LS = LombScargle(df_tbl['TIME'][AOK], df_tbl['PDCSAP_FLUX'][AOK]/med, dy=df_tbl['PDCSAP_FLUX_ERR'][AOK]/med)
                frequency, power = LS.autopower(minimum_frequency=1./40.,
                                                maximum_frequency=1./0.1,
                                                samples_per_peak=7)
                best_frequency = frequency[np.argmax(power)]

                per_out[k] = 1./best_frequency
                per_amp[k] = np.nanmax(power)
                per_med[k] = np.nanmedian(power)
                per_std[k] = np.nanstd(smo[SOK]/med)

                if np.nanmax(power) > 0.2:
                    LSmodel = LS.model(df_tbl['TIME'][AOK], best_frequency)
                    if makefig:
                        plt.plot(df_tbl['TIME'][AOK], LSmodel,
                                 label='L-S P='+format(1./best_frequency, '6.3f')+'d, pk='+format(np.nanmax(power), '6.3f'))


                # here is where a simple Eclipse (EB) finder goes
                # EE = EasyE(smo[SOK]/med, df_tbl['PDCSAP_FLUX_ERR'][AOK][SOK]/med,
                #            N1=5, N2=3, N3=2)
                EE = EasyE(df_tbl['PDCSAP_FLUX'][AOK][SOK]/med - smo[SOK]/med,
                           df_tbl['PDCSAP_FLUX_ERR'][AOK][SOK] / med, N1=5, N2=3, N3=2)

                if (np.size(EE) > 0):
                    # need to test if EE outputs look periodic-ish, or just junk
                    noE = np.arange(len(SOK))

                    for j in range(len(EE[0])):
                        if makefig:
                            plt.scatter(tbl['TIME'][AOK][SOK][(EE[0][j]):(EE[1][j]+1)],
                                        df_tbl['PDCSAP_FLUX'][AOK][SOK][(EE[0][j]):(EE[1][j]+1)] / med,
                                        color='k', marker='s', s=5, alpha=0.75, label='_nolegend_')

                        noE[(EE[0][j]):(EE[1][j]+1)] = -1

                        EclDep[k] = EclDep[k] + np.nanmin(df_tbl['PDCSAP_FLUX'][AOK][SOK][(EE[0][j]):(EE[1][j] + 1)] / med  - smo[SOK][(EE[0][j]):(EE[1][j] + 1)]/med)

                    if makefig:
                        plt.scatter([],[], color='k', marker='s', s=5, alpha=0.75, label='Ecl: '+str(len(EE[0])))

                    EclNum[k] = len(EE[0])
                    EclDep[k] = EclDep[k] / np.float(len(EE[0]))

                    okE = np.where((noE > -1))[0]
                else:
                    okE = np.arange(len(SOK))

                # do some GP'ing, from:
                # https://exoplanet.dfm.io/en/stable/tutorials/stellar-variability/
                '''
                with pm.Model() as model:

                    # The mean flux of the time series
                    mean = pm.Normal("mean", mu=1.0, sd=10.0)

                    # A jitter term describing excess white noise
                    # print(AOK.shape, SOK.shape, okE.shape)
                    yerr = df_tbl['PDCSAP_FLUX_ERR'].values[AOK][SOK] / med
                    y = df_tbl['PDCSAP_FLUX'].values[AOK][SOK] / med
                    x = df_tbl['TIME'].values[AOK][SOK]

                    logs2 = pm.Normal("logs2", mu=2 * np.log(np.min(yerr)), sd=5.0)

                    # The parameters of the RotationTerm kernel
                    logamp = pm.Normal("logamp", mu=np.log(np.var(y)), sd=5.0)
                    logperiod = pm.Normal("logperiod", mu=np.log(acf['peaks'][0]['period']), sd=5.0)
                    logQ0 = pm.Normal("logQ0", mu=1.0, sd=10.0)
                    logdeltaQ = pm.Normal("logdeltaQ", mu=2.0, sd=10.0)
                    mix = pm.Uniform("mix", lower=0, upper=1.0)

                    # Track the period as a deterministic
                    period = pm.Deterministic("period", tt.exp(logperiod))

                    # Set up the Gaussian Process model
                    kernel = xo.gp.terms.RotationTerm(
                        log_amp=logamp,
                        period=period,
                        log_Q0=logQ0,
                        log_deltaQ=logdeltaQ,
                        mix=mix
                    )
                    gp = xo.gp.GP(kernel, x, yerr ** 2 + tt.exp(logs2), J=4)

                    # Compute the Gaussian Process likelihood and add it into the
                    # the PyMC3 model as a "potential"
                    pm.Potential("loglike", gp.log_likelihood(y - mean))

                    # Compute the mean model prediction for plotting purposes
                    pm.Deterministic("pred", gp.predict())

                    # Optimize to find the maximum a posteriori parameters
                    map_soln = xo.optimize(start=model.test_point)

                gpspl = map_soln["pred"]
                plt.plot(df_tbl['TIME'].values[AOK][SOK], gpspl+1, label='GP')
                ''';


                # add BLS
                bls = BoxLeastSquares(df_tbl['TIME'][AOK][SOK], smo[SOK]/med, dy=df_tbl['PDCSAP_FLUX_ERR'][AOK][SOK]/med)
                blsP = bls.autopower([0.025, 0.1], method='fast', objective='snr',
                                     minimum_n_transit=2, minimum_period=0.2)

                blsPer = blsP['period'][np.argmax(blsP['power'])]

                if ((3*np.nanstd(blsP['power']) + np.nanmedian(blsP['power']) < np.nanmax(blsP['power'])) &
                    (np.nanmax(blsP['power']) > 25.) &
                    (blsPer < 0.95 * np.nanmax(blsP['period']))
                   ):
                    blsPeriod[k] = blsPer
                    blsAmpl[k] = np.nanmax(blsP['power'])
                    if makefig:
                        plt.plot([],[], ' ', label='BLS='+format(blsPer, '6.3f')+'d')


            if makefig:
                # plt.plot(df_tbl['TIME'].values[AOK][SOK], spl, label='spl')
                plt.title(files_i[k].split('/')[-1] + ' k='+str(k), fontsize=12)
                plt.ylabel('Flux')
                plt.xlabel('BJD - 2457000 (days)')
                plt.legend(fontsize=10)
                # plt.show()
                plt.savefig(figname, bbox_inches='tight', pad_inches=0.25, dpi=100)
                plt.close()



    # write per-sector output files
    ALL_TIC = pd.Series(files_i).str.split('-', expand=True).iloc[:,-3].astype('int')

    flare_out = pd.DataFrame(data={'TIC':ALL_TIC[FL_id], 'i0':FL_t0, 'i1':FL_t1, 'med':FL_f0, 'peak':FL_f1})
    flare_out.to_csv(run_dir + 'outputs/' + sector + '_flare_out.csv')

    rot_out = pd.DataFrame(data={'TIC':ALL_TIC,
                                 'LSper':per_out, 'LSamp':per_amp, 'LSmed':per_med, 'LSstd':per_std,
                                 'acf_pk':ACF_1pk, 'acf_per':ACF_1dt,
                                 'bls_per':blsPeriod, 'bls_ampl':blsAmpl,
                                 'ecl_num':EclNum, 'ecl_dep':EclDep})
    rot_out.to_csv(run_dir + 'outputs/' + sector + '_rot_out.csv')



if __name__ == "__main__":
    '''
      let this file be called from the terminal directly. e.g.:
      $ python analysis.py
    '''
    BasicActivity(sys.argv[1])
