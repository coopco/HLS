import numpy as np
import scipy.stats as st
from scipy.special import logsumexp
from scipy.sparse import csc_matrix, csr_array
from scipy.sparse import diags_array as sparse_diag
from sksparse.cholmod import analyze as sks_analyze
from tqdm import tqdm
from pgdraw import pgdraw


class Model:
    def __init__(self, X, y, prior, a=0.5, b=0.5, fit_intercept=False,
                 fixed_tau=None, scale_prior="cauchy",
                 global_scale=True, group_scale=False):
        self.X = X
        self.y = y
        self.n = y.sum(axis=1)
        self.num_predictors = X.shape[1]
        self.num_classes = y.shape[1]

        self.prior = prior
        self.a = a
        self.b = b
        self.fit_intercept = fit_intercept
        self.fixed_tau = fixed_tau
        assert scale_prior in ["cauchy", "inverse-gamma"]
        self.scale_prior = scale_prior
        # global_scale = True => A tau for across all classes
        # group_scale = True => A tau for for each class
        # Note that both can be true
        self.global_scale = global_scale
        self.group_scale = group_scale

        # TODO: check what of these is needed
        self.X = X.astype(float)
        self.y = y.astype(float)
        self.n = self.n.astype(float)

        # For sparse cholesky
        self.X = csr_array(X)
        self.XtX = csc_matrix(self.X.T @ self.X)
        # NOTE: ordering_method="natural" means no permutation
        self.XtX_factor = sks_analyze(self.XtX, mode="auto", ordering_method="natural")

        # handle zero counts
        self.n[self.n == 0] += 1e-3

        if prior == "hs":
            self.prior = "horseshoe"

    def sample_beta(self, state):
        try:
            beta, omega = sparse_sample_coefficients(
                self.X, self.y, self.n,
                state['beta'], state['beta_0'],
                state['tau'], state['vu'], state['lambda'],
                self.XtX_factor)
        except Exception as e:
            print(e)
            return state['beta'], state['omega']
        return beta, omega

    def sample_beta_0(self, state):
        raise Exception("Not implemented")

    def sample_lambda(self, state):
        if self.prior == "ridge" or self.prior == "none":
            return np.ones((self.num_predictors, self.num_classes))
        elif self.prior == "lasso":
            raise Exception("Not implemented")
        elif self.prior == "horseshoe":
            if self.scale_prior == "cauchy":
                lambdas = np.zeros(state['lambda'].shape)
                lambdas[:, :-1] = sample_horseshoe_local(
                    state['beta'][:, :-1],
                    state['lambda'][:, :-1],
                    state['vu'][:-1],
                    state['tau'], 1, self.a, self.b)
                lambdas[:, -1] = 1
            elif self.scale_prior == "inverse-gamma":
                lambdas = np.zeros(state['lambda'].shape)
                lambdas[:, :-1] = sample_lambda_inverse_gamma(
                    state['beta'][:, :-1],
                    state['lambda'][:, :-1],
                    state['vu'][:-1],
                    state['tau'], 1, self.a, self.b)
                lambdas[:, -1] = 1
            return lambdas
        else:
            raise Exception

    def sample_vu(self, state):
        """Per-class scales"""
        if self.scale_prior == "cauchy":
            vu = np.zeros(state['vu'].shape)
            vu[:-1] = sample_vu_ghs(
                state['beta'][:, :-1],
                state['lambda'][:, :-1],
                state['vu'][:-1],
                state['tau'], 1, self.a, self.b)
            vu[-1] = 1
        elif self.scale_prior == "inverse-gamma":
            vu = np.zeros(state['vu'].shape)
            vu[:-1] = sample_vu_inverse_gamma(
                state['beta'][:, :-1],
                state['lambda'][:, :-1],
                state['vu'][:-1],
                state['tau'], 1, self.a, self.b)
            vu[-1] = 1
        else:
            raise NotImplementedError
        return vu

    def sample_tau(self, state):
        if self.prior == "none":
            return np.ones(state['tau'].shape)
        elif self.fixed_tau is not None:
            return self.fixed_tau
        elif self.scale_prior == "cauchy":
            tau = sample_horseshoe_global(
                state['beta'][:, :-1], state['lambda'][:, :-1],
                state['vu'][:-1], state['tau'], 1, self.a, self.b)
        elif self.scale_prior == "inverse-gamma":
            tau = sample_tau_inverse_gamma(
                state['beta'][:, :-1], state['lambda'][:, :-1],
                state['vu'][:-1], state['tau'], 1, self.a, self.b)
        # Constrain beta_K to 0
        return tau

    def new_samples(self, n_samples):
        samples = {
            'beta': np.zeros((n_samples, self.num_predictors, self.num_classes)),
            'beta_0': np.zeros((n_samples, self.num_classes)),
            'lambda': np.zeros((n_samples, self.num_predictors, self.num_classes)),
            'vu': np.zeros((n_samples, self.num_classes)),
            'tau': np.zeros((n_samples)),
        }
        return samples

    def new_state(self):
        state = {
            'beta': np.zeros((self.num_predictors, self.num_classes)),
            'beta_0': np.zeros(self.num_classes),
            'omega': np.ones((self.X.shape[0], self.num_classes)),
            'lambda': np.ones((self.num_predictors, self.num_classes)),
            'vu': np.ones(self.num_classes),
            'tau': 1
        }
        return state

    def gibbs_step(self, state):
        state['beta'], state['omega'] = self.sample_beta(state)
        if self.fit_intercept:
            state['beta_0'] = self.sample_beta_0(state)
        if self.global_scale:
            state['tau'] = self.sample_tau(state)
        if self.group_scale:
            state['vu'] = self.sample_vu(state)
        state['lambda'] = self.sample_lambda(state)

        return state


###
### Scale mixture samplers
###

def sparse_sample_coefficients(X, y, n, beta, beta_0, tau, vu, lambdas, chol_factor,
                                    sparse_matrix=False):
    K = y.shape[1]

    # TODO: sparse
    Xb = X @ beta + beta_0

    omega_new = np.ones((X.shape[0], y.shape[1]))
    beta_new = np.zeros(beta.shape)

    # Constrain beta_K to zero
    for k in range(K-1):
        # TODO: precompute somehow
        c_k = logsumexp(Xb[:, np.arange(K)!=k], axis=1)
        eta_k = Xb[:, k] - c_k

        # Sample beta_k | Omega_k
        kappa_k = y[:, k] - n/2  # TODO: precompute

        # Use Rue's algorithm to avoid computing the matrix inverse
        omega_k_new = 1 / np.sqrt(np.maximum(1e-10, pgdraw(n.astype(float), eta_k)))

        z = (kappa_k) * omega_k_new**2
        alpha = z - beta_0[k] + c_k

        # TODO: faster option?
        # Multiply each row of X by 1/omega_k_new
        X0 = X * np.repeat(1/omega_k_new, X.shape[1]).reshape((-1, X.shape[1]))

        Lambda = tau * vu[k] * lambdas[:, k]
        beta_k_new, _ = sparse_rue_nongaussian(chol_factor, X0, alpha, Lambda, omega_k_new)

        omega_new[:, k] = omega_k_new
        beta_new[:, k] = beta_k_new

    return beta_new, omega_new

def sparse_rue_nongaussian(XtX_factor, Phi, alpha, d, omega,
                           inplace=False):  # TODO: inplace should be faster
    p = Phi.shape[1]
    # TODO: sparse

    Dinv = sparse_diag(1/d, format="csc")

    # TODO: use precomuted XtX to compute X0.T @ X0

    # TODO: avoid converting between csc and csr
    PtP = Phi.T @ Phi # Is this equal to D in block matrix inverse?

    XtX_chol = XtX_factor.cholesky(csc_matrix(PtP + Dinv))

    # use_LDLt_decomposition is required
    v = XtX_chol.solve_L(Phi.T @ (alpha/omega), use_LDLt_decomposition=False)
    m = XtX_chol.solve_Lt(v, use_LDLt_decomposition=False)
    y = st.norm.rvs(loc = 0, scale = 1, size=p)
    w = XtX_chol.solve_Lt(y, use_LDLt_decomposition=False)

    # NOTE: x, m is not sparse
    x = m + w

    return x, m


###
### Horseshoe samplers
###

def density(z, m, p, a, b):
    return z**(2*a-p-1) * (1+z**2)**(-a-b) * np.exp(-m/z**2)

# Experiment
def sample_horseshoe(z_old, m, p, a, b):
    # shape, scale
    v = st.invgamma.rvs(a + b, scale = 1 + 1/z_old**2)
    z = np.sqrt(st.invgamma.rvs(p/2 + b, scale = m + 1/v))
    return z

def sample_horseshoe_local(beta, lambdas, vu, tau, sigma2, a, b):
    m = 1/(2*sigma2 * tau**2) * (beta**2 / vu**2)
    p = 1
    return sample_horseshoe(lambdas, m, p, a, b)

def sample_horseshoe_global(beta, lambdas, vu, tau, sigma2, a, b):
    m = 1/(2*sigma2) * (beta**2 / (lambdas**2 * vu**2)).sum()
    p = beta.shape[0] if len(beta.shape) == 1 else beta.shape[0] * beta.shape[1]
    return sample_horseshoe(tau, m, p, a, b)

def sample_vu_ghs(beta, lambdas, vu, tau, sigma2, a, b):
    assert len(beta.shape) == 2
    # Sum across class
    m = 1/(2*sigma2 * tau**2) * (beta**2 / lambdas**2).sum(0)
    p = beta.shape[0]
    new_vu = sample_horseshoe(vu, m, p, a, b)
    return new_vu

def sample_tau_inverse_gamma(beta, lambdas, vu, tau, sigma2, a, b):
    k = beta.shape[0] if len(beta.shape) == 1 else beta.shape[0] * beta.shape[1]
    shape = k/2 + 2*a + 1
    scale = b + (beta**2 / (lambdas**2 * vu**2)).sum() / (2*sigma2)
    tau2 = st.invgamma.rvs(shape, scale = scale)
    return np.sqrt(tau2)

def sample_lambda_inverse_gamma(beta, lambdas, vu, tau, sigma2, a, b):
    k = beta.shape[0] if len(beta.shape) == 1 else beta.shape[0] * beta.shape[1]
    shape = k/2 + 2*a + 1
    scale = b + (beta**2 / vu**2) / (2*sigma2 * tau**2)
    lambdas = st.invgamma.rvs(shape, scale = scale)
    return np.sqrt(lambdas)

def sample_vu_inverse_gamma(beta, lambdas, vu, tau, sigma2, a, b):
    k = beta.shape[0] if len(beta.shape) == 1 else beta.shape[0] * beta.shape[1]
    shape = k/2 + 2*a + 1
    scale = b + (beta**2 / lambdas**2).sum(0) / (2*sigma2 * tau**2)
    vu = st.invgamma.rvs(shape, scale = scale)
    return np.sqrt(vu)

###
### Sampler
###

def sampler(X, y, prior,
            fit_intercept=False,
            n_burn=10000,
            n_samples=10000,
            state=None,
            thinning=1, # TODO:
            fixed_tau=None,
            seed=123,
            a=0.5,
            b=0.5,
            global_scale = True,
            group_scale = False,
            scale_prior="cauchy"):
    np.random.seed(seed)
    model = Model(X, y, prior, fixed_tau=fixed_tau,
                  fit_intercept=fit_intercept, a=a, b=b,
                  scale_prior=scale_prior, global_scale=global_scale, group_scale=group_scale)
    if state is None: state = model.new_state()
    samples = model.new_samples(n_samples)

    for i in tqdm(range(n_burn + n_samples)):
        state = model.gibbs_step(state)

        if i >= n_burn:
            sample_i = i - n_burn
            samples['tau'][sample_i] = state['tau']
            samples['vu'][sample_i] = state['vu']
            samples['lambda'][sample_i, :] = state['lambda']
            samples['beta_0'][sample_i] = state['beta_0']
            samples['beta'][sample_i, :] = state['beta']

    return samples


