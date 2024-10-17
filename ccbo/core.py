import torch
import numpy as np
from botorch.models import MixedSingleTaskGP, ModelListGP
from botorch.fit import fit_gpytorch_model
from botorch.acquisition.monte_carlo import qExpectedImprovement
from botorch.sampling.normal import SobolQMCNormalSampler
from botorch.optim import optimize_acqf_mixed
from botorch.models.transforms.outcome import Standardize
from botorch.utils.transforms import unnormalize, normalize, normalize_indices
from botorch.acquisition.objective import GenericMCObjective
from botorch.models.gpytorch import GPyTorchModel
from botorch.models.kernels.categorical import CategoricalKernel

import gpytorch
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.constraints import GreaterThan
from gpytorch.kernels.matern_kernel import MaternKernel
from gpytorch.kernels.scale_kernel import ScaleKernel

from functools import partial

class GP_vi_mixed(ApproximateGP, GPyTorchModel):
    def __init__(self, train_x: torch.Tensor, train_y: torch.Tensor, cat_dims):
        self.train_inputs = (train_x,)
        self.train_targets = train_y

        variational_distribution = CholeskyVariationalDistribution(train_x.size(0))
        variational_strategy = VariationalStrategy(
            self, train_x, variational_distribution
        )

        # for mixed model kernels
        def get_batch_dimensions(train_X, train_Y):
            input_batch_shape = train_X.shape[:-2]
            aug_batch_shape = input_batch_shape
            num_outputs = train_Y.shape[-1]
            if num_outputs > 1:
                aug_batch_shape += torch.Size([num_outputs])
            return input_batch_shape, aug_batch_shape

        _, aug_batch_shape = get_batch_dimensions(
            train_X=train_x, train_Y=train_y.unsqueeze(-1)
        )

        def cont_kernel_factory(batch_shape, ard_num_dims, active_dims):
            return MaternKernel(
                nu=2.5,
                batch_shape=batch_shape,
                ard_num_dims=ard_num_dims,
                active_dims=active_dims,
                lengthscale_constraint=GreaterThan(1e-04),
            )

        d = train_x.shape[-1]
        cat_dims = normalize_indices(indices=cat_dims, d=d)
        ord_dims = sorted(set(range(d)) - set(cat_dims))

        sum_kernel = ScaleKernel(
            cont_kernel_factory(
                batch_shape=aug_batch_shape,
                ard_num_dims=len(ord_dims),
                active_dims=ord_dims,
            )
            + ScaleKernel(
                CategoricalKernel(
                    batch_shape=aug_batch_shape,
                    ard_num_dims=len(cat_dims),
                    active_dims=cat_dims,
                    lengthscale_constraint=GreaterThan(1e-06),
                )
            )
        )
        prod_kernel = ScaleKernel(
            cont_kernel_factory(
                batch_shape=aug_batch_shape,
                ard_num_dims=len(ord_dims),
                active_dims=ord_dims,
            )
            * CategoricalKernel(
                batch_shape=aug_batch_shape,
                ard_num_dims=len(cat_dims),
                active_dims=cat_dims,
                lengthscale_constraint=GreaterThan(1e-06),
            )
        )
        covar_module = sum_kernel + prod_kernel

        super(GP_vi_mixed, self).__init__(variational_strategy)

        self.mean_module = gpytorch.means.ConstantMean(batch_shape=aug_batch_shape)
        self.covar_module = covar_module
        self.likelihood = gpytorch.likelihoods.BernoulliLikelihood()

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def sim_espray_constrained(x, noise_se=None):
    # Define the equations
    conc = x[:, 0]
    flow_rate = x[:, 1]
    voltage = x[:, 2]
    solvent = x[:, 3]
    diameter = (
        (conc.pow(1 / 2)) * (flow_rate.pow(1 / 2)) / torch.log2(voltage) * 10
        + 0.4
        + solvent
    )  # Diameter in micrometers
    if noise_se != None:
        diameter = diameter + noise_se * torch.randn_like(diameter)
    exp_con = (torch.log(flow_rate) * (solvent - 0.5) + 1.40 >= 0).float()
    return torch.cat((diameter.reshape(-1, 1), exp_con.reshape(-1, 1)), dim=1)


def _neg_sq_dist(y, y_target, offset=0, X=None):
    return -torch.square(y - y_target) + offset

def optimize_acqf_and_get_recommendation(
    X_raw: torch.tensor,
    y_raw: torch.tensor,
    bounds: torch.tensor,
    y_target: torch.tensor = None,
    y_var: float = 1e-8,
    y_log_transform: bool = True,
    fr_log_transform: bool = True,
    strategy: str = "qEI",
    batch_size: int = 2,
):
    """
    Use the defined strategy to optimize the acquisition function for the constrained optimization.

    args:
    X: torch.tensor, the input features
    y: torch.tensor, the output
    bounds: torch.tensor, the bounds for the input features
    y_target: torch.tensor, the target value for the objective
    y_log_transform: bool, whether to log transform the output
    fr_log_transform: bool, whether to log transform the flow rate
    strategy: str, the strategy to use, available strategies are: 'rnd', 'qEI', 'qEI_vi_mixed_con', 'qEICF_vi_mixed_con'
    batch_size: int, the number of experiments

    return:
    next_experiment: torch.tensor, the next experiment to run
    """

    standard_bounds = torch.tensor([[0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]]).to(
        X_raw.device, X_raw.dtype
    )

    # check if y_target is provided
    if y_target is None:
        raise ValueError("y_target must be provided")

    # transform y if needed and separate the objective and constraint
    if y_log_transform:
        y_obj = torch.log(y_raw[:, [0]]).detach()
        y_con = y_raw[:, 1].detach()
        y_target = torch.log(y_target)
    else:
        y_obj = y_raw[:, [0]].detach()
        y_con = y_raw[:, 1].detach()

    # Standardize the y_con
    y_con = 2 * y_con - 1  # convert {0,1} to {-1,1}

    # check if all items in X are within bound
    if not ((X_raw >= bounds[0, :]).all() and (X_raw <= bounds[1, :]).all()):
        raise ValueError(
            "X must be set within bounds, current X: ", X_raw, "bounds: ", bounds
        )

    # Normalize X
    if fr_log_transform:
        X_raw_fr_log = X_raw.clone()
        X_raw_fr_log[:, 1] = torch.log(X_raw_fr_log[:, 1])

        bounds_fr_log = bounds.clone()
        bounds_fr_log[:, 1] = torch.log(bounds_fr_log[:, 1])

        X_normalized = normalize(X_raw_fr_log, bounds_fr_log)

    else:
        X_normalized = normalize(X_raw, bounds)

    if strategy == "rnd":
        # randomly select the next experiment
        candidates = torch.rand((batch_size, 4)).to(X_raw.device, X_raw.dtype)
        # round the cat_dim (-1)
        candidates[:, -1] = torch.round(candidates[:, -1])
        next_experiment = unnormalize(candidates, bounds)
        return next_experiment

    acqf = _build_acqf_constrained(X_normalized, y_obj, y_con, y_target, strategy)

    # run the optimization function
    candidates, _ = optimize_acqf_mixed(
        acq_function=acqf,
        bounds=standard_bounds,
        q=batch_size,
        num_restarts=10,
        raw_samples=512,
        fixed_features_list=[{3: 0.0}, {3: 1.0}],
        options={"batch_limit": 5, "maxiter": 400},
    )

    # unnormalize the candidates
    if fr_log_transform:
        next_experiment = unnormalize(candidates.detach(), bounds_fr_log)
        next_experiment[:, 1] = torch.exp(next_experiment[:, 1])
    else:
        next_experiment = unnormalize(candidates.detach(), bounds)

    return next_experiment


def _build_acqf_constrained(X_normalized, y_obj, y_con, y_target, strategy):
    """
    Build the acquisition function for the constrained optimization.
    Available strategies are:

    - qEI (this is baseline with NOT constraint modelling)

    - qEI_vi_mixed_con (qEI with variational inference model for constraint,
        specifically for mixed input features - categorical and continuous,
        relies on botorch default constraint implementation - SampleReducingMCAcquisitionFunction)

    - qEICF_vi_mixed_con (qEICF version of qEI_vi_mixed_con)

    args:
    X_normalized: torch.tensor, the normalized input features
    y_obj: torch.tensor, the objective
    y_con: torch.tensor, the constraint
    y_target: torch.tensor, the target value for the objective
    strategy: str, the strategy to use

    return:
    acqf: botorch.acquisition.AcquisitionFunction, the acquisition function
    """

    def obj_direct_pass(Z, X=None):
        """
        directly pass the objective
        """
        return Z[..., 0]

    def obj_composite(Z, y_target, offset, X=None):
        """
        integrate the composite function into the objective
        """
        return _neg_sq_dist(Z[..., 0], y_target, offset=offset)

    # to comply with botorch constraints, we need to convert negative values for feasibility
    def con_vi_unsigmoid(Z, model_con, X=None):
        """
        Botorch does sigmoid transformation for the constraint by default,
        therefore we need to unsigmoid our probability (0-1) to (-inf,inf)
        also we need to invert the probability, where -inf means the constraint is satisfied

        we add 1e-8 to avoid log(0).

        model_con: GPyTorchModel, the constraint model

        """
        y_con = Z[..., 1]
        # calculate the probability of satisfying the constraint (0,1) from variational inference
        prob = model_con.likelihood(y_con).probs
        prob_unsigmoid_neg = torch.log(1 - prob + 1e-8) - torch.log(prob + 1e-8)
        return prob_unsigmoid_neg

    sampler = SobolQMCNormalSampler(sample_shape=torch.Size([512]))

    # for non-composite acquisition functions, use y_distance as the target
    y_distance = _neg_sq_dist(y_obj, y_target, offset=10.0).reshape(-1, 1)
    best_f = np.ma.masked_array(y_distance, mask=~y_con.bool()).max().item()

    if strategy == "qEI":
        # define the model
        model_obj = MixedSingleTaskGP(
            train_X=X_normalized,
            train_Y=y_distance,
            cat_dims=[-1],
            outcome_transform=Standardize(m=1),
        )
        mll_obj = ExactMarginalLogLikelihood(model_obj.likelihood, model_obj)
        fit_gpytorch_model(mll_obj)

        acqf = qExpectedImprovement(
            model=model_obj,
            best_f=best_f,
            sampler=sampler,
        )

    elif strategy == "qEI_vi_mixed_con":
        # define the model
        model_obj = MixedSingleTaskGP(
            train_X=X_normalized,
            train_Y=y_distance,
            cat_dims=[-1],
            outcome_transform=Standardize(m=1),
        )
        mll_obj = ExactMarginalLogLikelihood(model_obj.likelihood, model_obj)
        fit_gpytorch_model(mll_obj)

        model_con = GP_vi_mixed(X_normalized, y_con, cat_dims=[-1])
        mll_con = gpytorch.mlls.VariationalELBO(
            model_con.likelihood, model_con, num_data=y_con.size(0)
        )
        model_con.double()
        mll_con.double()
        fit_gpytorch_model(mll_con)

        model = ModelListGP(model_obj, model_con)

        acqf = qExpectedImprovement(
            model=model,
            best_f=best_f,
            sampler=sampler,
            objective=GenericMCObjective(obj_direct_pass),
            constraints=[partial(con_vi_unsigmoid, model_con=model_con)],
        )

    elif strategy == "qEICF_vi_mixed_con":
        # define the model
        model_obj = MixedSingleTaskGP(
            train_X=X_normalized,
            train_Y=y_obj,
            cat_dims=[-1],
            outcome_transform=Standardize(m=1),
        )
        mll_obj = ExactMarginalLogLikelihood(model_obj.likelihood, model_obj)
        fit_gpytorch_model(mll_obj)

        model_con = GP_vi_mixed(X_normalized, y_con, cat_dims=[-1])
        mll_con = gpytorch.mlls.VariationalELBO(
            model_con.likelihood, model_con, num_data=y_con.size(0)
        )
        model_con.double()
        mll_con.double()
        fit_gpytorch_model(mll_con)

        model = ModelListGP(model_obj, model_con)

        acqf = qExpectedImprovement(
            model=model,
            best_f=best_f,
            sampler=sampler,
            objective=GenericMCObjective(
                partial(obj_composite, y_target=y_target, offset=10.0)
            ),
            constraints=[partial(con_vi_unsigmoid, model_con=model_con)],
        )

    else:
        raise ValueError("strategy not available")

    return acqf
