import argparse
import pyro.distributions as dist
import pyro.distributions.transforms as T
from pyro.nn import ConditionalAutoRegressiveNN
from sklearn.preprocessing import StandardScaler
from uproot_methods import TLorentzVectorArray as lv
from dataloader import *


def train(args):

    # initialise CUDA
    if args.cuda:
        device = torch.device("cuda")
        torch.set_default_tensor_type("torch.cuda.FloatTensor")
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")
        torch.set_default_tensor_type("torch.FloatTensor")

    # load data
    k_train, m_train, m_eval = load(args.train_loc, args.eval_loc, args.ntag, args.kinematic_region)

    # normalize
    scaler_m_train = StandardScaler()
    m_train = torch.tensor(scaler_m_train.fit_transform(m_train)).float()
    m_eval = torch.tensor(scaler_m_train.transform(m_eval)).float()
    scaler_k_train = StandardScaler()
    k_train = torch.tensor(scaler_k_train.fit_transform(k_train)).float()

    # prepare dataloaders
    trainset = Dataset(k_train, m_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size)

    # base distributions chosen to be gaussian (can change to distribution of choice)
    dist_base_k = dist.Normal(torch.zeros(args.dim_k), torch.ones(args.dim_k), validate_args=False)

    # bootstrap loop
    bootstrap = 0
    while bootstrap < args.bootstraps:
        bootstrap = bootstrap + 1

        # conditional autoregressive spline flow
        hypernet = ConditionalAutoRegressiveNN(
            input_dim=args.dim_k,
            context_dim=args.dim_m,
            hidden_dims=[args.dim_m * 10] * args.layers,
            param_dims=[args.count_bins, args.count_bins, args.count_bins - 1, args.count_bins],
            skip_connections=False)
        k_transform = [T.ConditionalSplineAutoregressive(
            args.dim_k,
            hypernet,
            count_bins=args.count_bins)
            for _ in range(args.num_flows)]
        dist_k_given_m = dist.ConditionalTransformedDistribution(dist_base_k, k_transform)

        modules = torch.nn.ModuleList(k_transform).to(device)
        optimizer = torch.optim.Adam(modules.parameters(), lr=args.lr, weight_decay=args.beta)

        # train loop
        for epoch in range(args.epochs):

            running_train_loss = 0
            for batch, (k_batch, m_batch) in enumerate(trainloader):
                optimizer.zero_grad()

                # train log liklihood
                ln_p_k_given_m = dist_k_given_m.condition(m_batch).log_prob(k_batch)
                loss_train = - ln_p_k_given_m.mean()

                # take optimization step
                loss_train.backward()
                optimizer.step()

                dist_k_given_m.clear_cache()
                running_train_loss += loss_train.item()

                # progress
                if batch % args.print_interval == args.print_interval - 1:
                    print('bootstrap: {}, epoch: {}, [{}/{} ({:.0f}%)], train loss: {:.3f}'.format(
                        bootstrap,
                        epoch,
                        batch * args.batch_size,
                        len(trainloader.dataset),
                        100. * batch / len(
                        trainloader),
                        running_train_loss / args.print_interval))
                    running_train_loss = 0

            # re-train if unstable
            if np.isnan(running_train_loss):
                bootstrap = bootstrap - 1
                print('Unstable training. Restarting bootstrap...')
                break

        # proceed to next bootstrap if stable
        if not np.isnan(running_train_loss):

            # sample from transformed distribution
            k_flow_ = dist_k_given_m.condition(m_train).sample(torch.Size([len(m_train), ]))

            # evaulate in SR
            k_eval_ = dist_k_given_m.condition(m_eval).sample(torch.Size([len(m_eval), ]))

            # denormalize
            k_train_ = scaler_k_train.inverse_transform(k_train.cpu().detach().numpy())
            k_flow_ = scaler_k_train.inverse_transform(k_flow_.cpu().detach().numpy())
            k_eval_ = scaler_k_train.inverse_transform(k_eval_.cpu().detach().numpy())
            m_train_ = scaler_m_train.inverse_transform(m_train.cpu().detach().numpy())
            m_eval_ = scaler_m_train.inverse_transform(m_eval.cpu().detach().numpy())

            # convert to pandas
            k_train_ = pd.DataFrame(k_train_, columns=['log_pT_h1', 'log_pT_h2', 'eta_h1', 'eta_h2', 'log_dphi_hh'])
            k_flow_ = pd.DataFrame(k_flow_, columns=['log_pT_h1', 'log_pT_h2', 'eta_h1', 'eta_h2', 'log_dphi_hh'])
            k_eval_ = pd.DataFrame(k_eval_, columns=['log_pT_h1', 'log_pT_h2', 'eta_h1', 'eta_h2', 'log_dphi_hh'])
            m_train_ = pd.DataFrame(m_train_, columns=['m_h1', 'm_h2'])
            m_eval_ = pd.DataFrame(m_eval_, columns=['m_h1', 'm_h2'])

            # undo preprocessing
            for i in ('log_pT_h1', 'log_pT_h2', 'log_dphi_hh'):
                k_flow_[i[4:]] = np.exp(k_flow_[i])
                k_train_[i[4:]] = np.exp(k_train_[i])
                k_eval_[i[4:]] = np.exp(k_eval_[i])
            k_flow_['dphi_hh'] = np.pi - k_flow_['dphi_hh']
            k_train_['dphi_hh'] = np.pi - k_train_['dphi_hh']
            k_eval_['dphi_hh'] = np.pi - k_eval_['dphi_hh']

            # find HC 4-vectors
            hc1_pred = lv.from_ptetaphim(k_flow_['pT_h1'], k_flow_['eta_h1'], np.zeros_like(k_flow_.index).astype(float),
                                         m_train_['m_h1'])
            hc2_pred = lv.from_ptetaphim(k_flow_['pT_h2'], k_flow_['eta_h2'], k_flow_['dphi_hh'], m_train_['m_h2'])
            hc1_pred_SR = lv.from_ptetaphim(k_eval_['pT_h1'], k_eval_['eta_h1'], np.zeros_like(k_eval_.index).astype(float),
                                            m_eval_['m_h1'])
            hc2_pred_SR = lv.from_ptetaphim(k_eval_['pT_h2'], k_eval_['eta_h2'], k_eval_['dphi_hh'], m_eval_['m_h2'])

            # boost into the di-Higgs rest frame to get cos(theta*)
            hh_pred = hc1_pred + hc2_pred
            hh_pred_SR = hc1_pred_SR + hc2_pred_SR
            boost = - hh_pred.boostp3
            boost_SR = - hh_pred_SR.boostp3
            rest_hc1 = hc1_pred._to_cartesian().boost(boost)
            rest_hc1_SR = hc1_pred_SR._to_cartesian().boost(boost_SR)

            # calculate high level variables
            k_flow_['m_hh'] = hh_pred.mass
            k_flow_['m_hh_cor2'] = hh_pred.mass - hc1_pred.mass - hc2_pred.mass + 250
            k_flow_['absCosThetaStar'] = np.abs(np.cos(rest_hc1.theta))
            k_flow_['pt_hh'] = hh_pred.pt
            k_eval_['m_hh'] = hh_pred_SR.mass
            k_eval_['m_hh_cor2'] = hh_pred_SR.mass - hc1_pred_SR.mass - hc2_pred_SR.mass + 250
            k_eval_['absCosThetaStar'] = np.abs(np.cos(rest_hc1_SR.theta))
            k_eval_['pt_hh'] = hh_pred_SR.pt

            # append conditional variables
            k_flow_['m_h1'] = m_train_['m_h1']
            k_flow_['m_h2'] = m_train_['m_h2']
            k_eval_['m_h1'] = m_eval_['m_h1']
            k_eval_['m_h2'] = m_eval_['m_h2']

            # save results
            k_flow_.to_hdf('train_bootstrap_{}.h5'.format(bootstrap), key='df', mode='w')
            k_eval_.to_hdf('eval_bootstrap_{}.h5'.format(bootstrap), key='df', mode='w')


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--train_loc", help="path to kinematics file", type=str, default='/home/taymaz/Documents/Flows/Data/data.kinematics.MDR_VEC.2018.parquet')
    parser.add_argument("--eval_loc", help="path to gaussian process massplane file", type=str, default='/home/taymaz/Documents/Flows/Data/test_pred_2b_MDR_VEC.parquet')
    parser.add_argument("--ntag", help="2b or 4b events (2 or 4)", type=int, default=2)
    parser.add_argument("--kinematic_region", help="kinematic train region", type=str, default='CRVR')
    parser.add_argument("--dim_m", help="conditional variable dimension (Higgs candidate mass)", type=int, default=2)
    parser.add_argument("--dim_k", help="train variable dimension (kinematics)", type=int, default=5)
    parser.add_argument("--batch_size", help="train batch size", type=int, default=2**15)
    parser.add_argument("--num_flows", help="number of spline autoregressive flows for density estimation", type=int, default=2)
    parser.add_argument("--count_bins", help="number of spline bins for density estimation", type=int, default=10)
    parser.add_argument("--bound", help="spline tail bound", type=int, default=1.0)
    parser.add_argument("--layers", help="number of hidden layers in flow hypernet", type=int, default=5)
    parser.add_argument("--beta", help="L2 regularization", type=int, default=1e-5)
    parser.add_argument("--lr", help="adam learning rate", type=int, default=5e-3)
    parser.add_argument("--print_interval", help="interval to print batch loss", type=int, default=10)
    parser.add_argument("--epochs", help="train epochs", type=int, default=25)
    parser.add_argument("--bootstraps", help="number of re-trainings", type=int, default=25)
    parser.add_argument("--cuda", help="use CUDA acceleration", default=True, action='store_false')
    args = parser.parse_args()

    train(args)
