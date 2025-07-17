import numpy as np
import pandas as pd
import torch


# helper to wrap variables and conditionals into pytorch dataset
class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        super(Dataset, self).__init__()
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.X[idx], self.y[idx]


# helper to mask SR in gaussian process data (for evaluation)
def SR(m1,m2):
    return np.sqrt(((m1-120)/(0.1*m1))**2+((m2-110)/(0.1*m2))**2)


# train data loader
def load(train_loc, eval_loc, ntag, kinematic_region, features=None):
    '''
    train_loc: path to file containing kinematic variables
    eval_loc: path to file containing gaussian process massplane data
    '''

    if features is None:
        features = ['ntag',
                    'kinematic_region',
                    'pass_vbf_sel',
                    'm_h1',
                    'm_h2',
                    'log_pT_h1',
                    'log_pT_h2',
                    'eta_h1',
                    'eta_h2',
                    'log_dphi_hh']

    # load train data
    k = pd.read_parquet(train_loc, engine='pyarrow', columns=features)

    # mask 2b/4b events & kinematic region
    if ntag == 2:
        mask_ntag = k['ntag'] == 2
    elif ntag == 4:
        mask_ntag = k['ntag'] >= 4
    if kinematic_region == 'CRVR':
        mask_reg = (k['kinematic_region'] == 1) | (k['kinematic_region'] == 2)
    elif kinematic_region == 'SR':
        mask_reg = k['kinematic_region'] == 0
    mask_vbf = k['pass_vbf_sel'] == False

    # apply cuts
    m_train = k.loc[(mask_ntag & mask_reg & mask_vbf), ('m_h1', 'm_h2')]
    k_train = k.loc[(mask_ntag & mask_reg & mask_vbf), ('log_pT_h1', 'log_pT_h2', 'eta_h1', 'eta_h2', 'log_dphi_hh')]
    del k

    # load evaluation data (GP)
    m = pd.read_parquet(eval_loc, engine='pyarrow')

    # mask SR
    mask_SR_gp = SR(m['m_h1'], m['m_h2']) < 1.6

    # apply cuts
    m_eval = m.loc[(mask_SR_gp), ('m_h1', 'm_h2')]
    del m

    return k_train, m_train, m_eval

