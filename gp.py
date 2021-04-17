import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import pandas as pd
from argparse import ArgumentParser
import itertools

#SR variable (1.6 is the nominal cutoff)
def Xhh(m1,m2):
    return np.sqrt(((m1-120)/(0.1*m1))**2+((m2-110)/(0.1*m2))**2)

#CR variable (45 is the nominal cutoff)
def CR_hh(m1,m2):
    return np.sqrt((m1 - (120*1.05))**2 + (m2 - (110*1.05))**2)

#VR variable (30 is the nominal cutoff)
def VR_hh(m1,m2):
    return np.sqrt((m1 - (120*1.03))**2 + (m2 - (110*1.03))**2)

#Find midpoints of bins array
def mid(bins):
    return np.array([bins[i]+(bins[i+1]-bins[i])/2. for i in range(len(bins)-1)])

#Find widths of bins array
def widths(bins):
    return np.array([bins[i+1]-bins[i] for i in range(len(bins)-1)])

#s = f'm_h1>{126-45} & m_h1<{126+45} & m_h2>{116-45} & m_h2<{116+45}'
def box_sel(arr, m1_lo, m1_hi, m2_lo, m2_hi):
    mask = ((arr['m_h1']>m1_lo) & (arr['m_h1']<m1_hi) &
            (arr['m_h2']>m2_lo) & (arr['m_h2']<m2_hi))
    
    return arr[mask] 

class GPdensity:
    def __init__(self, data, bins, blind=True):
        self.data = data
        self.bins = bins
        self.blind = blind
        
    def train(self):  
        arr = self.data
        bins = self.bins
        if self.blind:
            mask = (arr['kinematic_region'] != 0)
            bkgd_2d, xedges, yedges = np.histogram2d(arr[mask]['m_h1'], arr[mask]['m_h2'],
                                                       bins=bins)
        else:
            bkgd_2d, xedges, yedges = np.histogram2d(arr['m_h1'], arr['m_h2'],
                                                       bins=bins)

        stat_err = (np.maximum(bkgd_2d, np.ones(bkgd_2d.shape))/bkgd_2d).flatten()

        #Protect against 0 bins
        stat_err[np.isinf(stat_err)]=1

        bkgd = bkgd_2d.flatten()

        mean = np.mean(bkgd)
        std = np.std(bkgd)

        norm_bkgd = (bkgd-mean)/std
        norm_stat_err = stat_err/std

        m1m2=np.array(list(itertools.product(mid(xedges), mid(yedges))))

        #Remove any bins with any signal region overlap
        w_x = widths(xedges)
        w_y = widths(yedges)
        mids_x = mid(xedges)
        mids_y = mid(yedges)

        m1m2_ur=np.array(list(itertools.product(mids_x+w_x/2., mids_y+w_y/2.)))
        m1m2_dr=np.array(list(itertools.product(mids_x+w_x/2., mids_y-w_y/2.)))
        m1m2_ul=np.array(list(itertools.product(mids_x-w_x/2., mids_y+w_y/2.)))
        m1m2_dl=np.array(list(itertools.product(mids_x-w_x/2., mids_y-w_y/2.)))

        blind = ((Xhh(m1m2_ur[:,0], m1m2_ur[:,1]) > 1.6)
             & (Xhh(m1m2_dr[:,0], m1m2_dr[:,1]) > 1.6)
             & (Xhh(m1m2_ul[:,0], m1m2_ul[:,1]) > 1.6)
             & (Xhh(m1m2_dl[:,0], m1m2_dl[:,1]) > 1.6))

        kern = RBF(length_scale=[50,50], length_scale_bounds=(20,150))
        gpr = GaussianProcessRegressor(kernel=kern,
                                       alpha=norm_stat_err[blind])

        gpr.fit(m1m2[blind], norm_bkgd[blind])
        print(gpr.kernel_)

        (bkgdmodel, uncert) = gpr.predict(m1m2, return_std=True)

        self.pred = bkgdmodel*std+mean
        self.uncert = uncert*std
        self.std = std
        self.mean = mean
        self.bkgd = bkgd
        self.gpr = gpr
        self.xedges = xedges
        self.yedges = yedges
        self.stat_err = stat_err
        
    def sample(self, nEvents):
        '''
        Does an inverse transform sampling of the GP prediction. Note that
        this sampling is done over the entire GP training box, not just the SR.
        A smearing is applied so that sampled (m1,m2) values are not just the bin 
        centers, but rather are some uniformly random values in the given bin.

        - nEvents: number of events to sample. In principle, this is arbitrary,
                   SR normalization should be set using getNorm.
        '''

        widths_x = widths(self.xedges)
        widths_y = widths(self.yedges)

        mid_xbins = mid(self.xedges)
        mid_ybins = mid(self.yedges)

        cdf = np.cumsum(self.pred)
        cdf = cdf / cdf[-1]

        values = np.random.rand(nEvents)
        value_bins = np.searchsorted(cdf, values)
        x_idx, y_idx = np.unravel_index(value_bins,
                                        (len(mid_xbins),
                                         len(mid_ybins)))
        random_from_cdf = np.column_stack((mid_xbins[x_idx],
                                           mid_ybins[y_idx]))
        new_x, new_y = random_from_cdf.T

        smear_x = []
        smear_y = []
        for i in range(len(new_x)):
            width_x = widths_x[mid_xbins==new_x[i]][0]
            width_y = widths_y[mid_ybins==new_y[i]][0]
            x = new_x[i] + np.random.uniform(-width_x/2.,width_x/2.)
            y = new_y[i] + np.random.uniform(-width_y/2.,width_y/2.)

            smear_x.append(x)
            smear_y.append(y)

        return np.asarray(smear_x), np.asarray(smear_y)

    def getNorm(self, nsamp=20000, low_bounds=[126-45,116-45], high_bounds=[126+45,116+45]):
        '''
        Does an MC sampling of the GP to approximate the relative fraction of
        events predicted both in (frac_sig) and outside of (frac_out) the SR relative to 
        the full box. Since we know the number of events outside, we can then use this to 
        get a prediction for events inside. 

        Note, box can be a subset of GP training box.
    
        - nsamp: number of draws from the GP to determine fractions
        - low_bounds: low boundaries for box [m1_lo, m2_lo]
        - high_bounds: high boundaries for box [m1_hi, m2_hi]  
        '''

        m1m2 = np.random.uniform(low=low_bounds, 
                                 high=high_bounds, size=(nsamp,2))
        pred_all = self.gpr.predict(m1m2)*self.std+self.mean

        selected = box_sel(self.data, low_bounds[0], high_bounds[0],
                           low_bounds[1], high_bounds[1])
        nout = np.sum(selected['kinematic_region'] != 0)
        frac_out = np.sum(pred_all[Xhh(m1m2[:, 0],m1m2[:, 1]) > 1.6])/np.sum(pred_all)
        frac_sig = np.sum(pred_all[Xhh(m1m2[:, 0],m1m2[:, 1]) < 1.6])/np.sum(pred_all)
        ntot = nout/frac_out
        n_sig = ntot*frac_sig

        return n_sig
 


def main(fname, nbinsx, nbinsy, nEvents, ntag_reg):
    cols = ['m_h1', 'm_h2', 'ntag', 'eta_h1', 'eta_h2'] 

    #Selects ntag box, dEta_hh < 1.5 (replace with whatever pre-processing) 
    full_data = pd.read_hdf(fname)

    if ntag_reg == 2 or ntag_reg == 3:
        ntag_mask = (full_data['ntag']==ntag_reg)
    elif ntag_reg == 4:
        ntag_mask = (full_data['ntag']>=ntag_reg)
    else:
        print("ntag must be 2, 3, or 4!")
        return 0

    full_data = full_data[ntag_mask & 
                          (abs(full_data['eta_h1']-full_data['eta_h2'])<1.5)]

    full_data = box_sel(full_data, 126-45, 126+45, 116-45, 116+45)


    #Train on full box - SR will be blinded in training
    bins = [np.linspace(126-45, 126+45, nbinsx+1), 
            np.linspace(116-45, 116+45, nbinsy+1)]

    density = GPdensity(full_data, bins, blind=True)
    density.train()
  
    m1,m2 =density.sample(nEvents)

    out_df = pd.DataFrame({'m_h1': m1, 'm_h2': m2})
    out_df.to_hdf('test_pred_%db.h5' % ntag_reg, 'df')

    n_sig = density.getNorm()
    print("Predicted number of SR events:", n_sig)


if __name__ == '__main__':
    parser = ArgumentParser()
    #"/eos/user/h/hartman/hh4b/pairAGraph/data16_PFlow-FEB20-5jets/df_SM_2b_p_0.01_2b.h5"
    parser.add_argument("-d", "--data", dest="data_file", default="",
                        help="Input data filenames")
    parser.add_argument("--ntag", dest="ntag_reg", default=2, type=int,
                        help="n-tag region to evaluate on")
    parser.add_argument("--nbinsx", dest="nbinsx", default=50, type=int,
                        help="Number of bins in m_h1")
    parser.add_argument("--nbinsy", dest="nbinsy", default=50, type=int,
                        help="Number of bins in m_h2")
    parser.add_argument("--nEvents", dest="nEvents", default=10000, type=int,
                        help="Number of events to sample from pred (note: sampling done in full box)")
    args = parser.parse_args()

    main(args.data_file, args.nbinsx, args.nbinsy, args.nEvents, args.ntag_reg)

