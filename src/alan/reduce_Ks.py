import opt_einsum
from .utils import *
from .unravel_index import unravel_index
from functorch.dim import Dim

def einsum_args(lps, sum_dims):
    """
    opt_einsum requires pretty weird arguments to get an optimal path.
    This function constructs the required arguments.
    """
    #There shouldn't be any non-torchdim dimensions.
    #Should eventually be able to implement this as a straight product-sum
    for lp in lps:
        assert lp.shape == ()

    set_sum_dims = set(sum_dims)

    all_dims = unify_dims(lps)
    dim_to_idx = {dim: i for (i, dim) in enumerate(all_dims)}
    out_dims = [dim for dim in all_dims if dim not in set_sum_dims]
    out_idxs = [dim_to_idx[dim] for dim in out_dims]

    undim_lps = []
    arg_idxs = []
    for lp in lps:
        dims = generic_dims(lp)
        arg_idxs.append([dim_to_idx[dim] for dim in dims])
        undim_lps.append(generic_order(lp, dims))

    assert all(not is_dimtensor(lp) for lp in undim_lps)

    return [val for pair in zip(undim_lps, arg_idxs) for val in pair] + [out_idxs], out_dims


def sample_Ks(lps, Ks_to_sum, N_dim, num_samples):

    """
    Fundamental method that returns K samples from the posterior
    opt_einsum gives an "optimization path", i.e. the indicies of lps to reduce.
    We use this path to do our reductions, handing everything off to a simple t.einsum
    call (which ensures a reasonably efficient implementation for each reduction).
    """
    assert_unique_dim_iter(Ks_to_sum)
    assert set(unify_dims(lps)).issuperset(Ks_to_sum)
    
    _, lps_for_sampling, Ks_to_sample = collect_lps(lps, Ks_to_sum)

    #Now that we have the list of reduced factors and which Kdims to sample from each factor we can sample from each factor in turn
    indices = {}

    for lps, kdims_to_sample in zip(lps_for_sampling[::-1], Ks_to_sample[::-1]): 
        
        lp = sum(lps)

        for dim in list(set(generic_dims(lp)).intersection(set(indices.keys()))):
            lp = lp.order(dim)[indices[dim]]

        #If there is more than one Kdim to sample from this factor we need to sample from the joint distribution
        #To do this we sample from a multinomial over the indices of the lp tensor
        #We then unravel the indices and assign them to the appropriate Kdim

        # shift lps up by the max value in each kdim_to_sample to avoid numerical issues
        lp_max = lp.amax(kdims_to_sample)

        logits = t.exp(lp.order(*kdims_to_sample) - lp_max).ravel()
        
        assert generic_all(logits.isnan().sum() == 0)
        assert generic_all(logits.isinf().sum() == 0)
        assert generic_all(logits >= 0)
        
        sampled_flat_idx = t.multinomial(logits, num_samples, replacement=True)
        unravelled_indices = unravel_index(sampled_flat_idx, shape=[dim.size for dim in kdims_to_sample])
        
        for idx, kdim in zip(unravelled_indices, kdims_to_sample):
            indices[kdim] = idx[N_dim]

    return indices

def sample_Ks_timeseries(lps, Ks_to_sum, ts_init_Ks, N_dim, num_samples, T_dim, indices):

    """
    Fundamental method that returns K samples from the posterior *where Ks_to_sum are timeseries K_dims*
    opt_einsum gives an "optimization path", i.e. the indicies of lps to reduce.
    We use this path to do our reductions, handing everything off to a simple t.einsum
    call (which ensures a reasonably efficient implementation for each reduction).
    """
    assert_unique_dim_iter(Ks_to_sum)
    assert set(unify_dims(lps)).issuperset(Ks_to_sum)
    _, lps_for_sampling, Ks_to_sample = collect_lps(lps, Ks_to_sum)

    #Now that we have the list of reduced factors and which Kdims to sample from each factor we can sample from each factor in turn
    indices = {**indices}

    for lps, kdims_to_sample, init_K_dim in zip(lps_for_sampling[::-1], Ks_to_sample[::-1], ts_init_Ks[::-1]):
        assert len(kdims_to_sample) == 1
        K_dim = kdims_to_sample[0]

        lp = sum(lps)


        assert K_dim in set(generic_dims(lp))
        assert T_dim in set(generic_dims(lp))
        assert init_K_dim in set(generic_dims(lp))

        assert init_K_dim in indices.keys()

        # index into lp EXCEPT for ts_init_Ks
        for dim in list(set(generic_dims(lp)).intersection(set(indices.keys())).difference(set(ts_init_Ks))):
            lp = lp.order(dim)[indices[dim]]

        #get plate_dims
        plate_dims = list(set(generic_dims(lp)).difference(set(indices.keys()) | set(ts_init_Ks)).difference(set([N_dim, K_dim, T_dim])))
        plate_dim_sizes = [dim.size for dim in plate_dims]

        if len(plate_dims) == 0:
            ts_indices = t.zeros((num_samples, T_dim.size, ), dtype=t.int64)[N_dim, ...]
        else:
            ts_indices = t.zeros((num_samples, *plate_dim_sizes, T_dim.size, ), dtype=t.int64)[N_dim, plate_dims, ...]


        # Precompute the forward filtering prefix products incrementally: cumulative[t_idx]
        # is chain_logmmexp(lp_ordered[:t_idx+1]), i.e. exactly what the loop below used to
        # recompute from scratch at every t_idx (T separate chain_logmmexp calls, over
        # prefixes of average length T/2 -> O(T^2) total matrix multiplies). Building the
        # T prefix products once, incrementally, is O(T) total instead.
        lp_ordered = lp.order(T_dim, init_K_dim, K_dim)
        cumulative = [lp_ordered[0]]
        for t_idx in range(1, T_dim.size):
            cumulative.append(logmmexp(cumulative[-1], lp_ordered[t_idx]))

        filtered_t_plus_one = None
        smoothed_t_plus_one = None

        for t_idx in range(T_dim.size-1,-1,-1):
            # this filtering/forward-run gives us log p(x_t | y_{1:t}, x_{1:t-1}) as a K x K tensor
            filtered_t = cumulative[t_idx][init_K_dim, K_dim]

            # index into filtered_t with the ts_init_Ks indices to get the filtering distribution as a K-long vector
            filtered_t = filtered_t.order(init_K_dim)[indices[init_K_dim]]
            filtered_t = filtered_t.order(N_dim)
            filtered_t = t.logsumexp(filtered_t, 0) #- t.tensor(num_samples).log().to(filtered_t.device)

            # normalise
            filtered_t = filtered_t - t.logsumexp(filtered_t.order(K_dim), 0)

            # now do smoothing/backward-run to get log p(x_t | y_{1:T}, x_{1:T})
            # this is calculated as 
            #       p(x_t | y_{1:t}, x_{1:t-1}) * INTEGRATE{dx_{t+1} * p(x_{t+1} | y_{1:T}) * p(x_{t+1} | x_t) / p(x_{t+1} | y_{1:t}) }
            # i.e.  filtered_t * INTEGRATE{dx_{t+1} * smoothed_{t+1} * p(x_{t+1} | x_t) / filtered_{t+1} }
            # 
            # see e.g. http://www.gatsby.ucl.ac.uk/~byron/nlds/briers04.pdf
            if t_idx < T_dim.size-1:
                transition = lp.order(T_dim)[t_idx+1]

                integrand = ((smoothed_t_plus_one - filtered_t_plus_one) + transition) # [2, init_K_dim, K_dim] with torchdims

                smoothed_t = filtered_t + integrand.logsumexp(K_dim)

                # index into smoothed_t with the ts_init_Ks indices to get the smoothed distribution as a K-long vector
                smoothed_t = smoothed_t.order(init_K_dim)[indices[init_K_dim]]
                smoothed_t = smoothed_t.order(N_dim)
                smoothed_t = t.logsumexp(smoothed_t, 0)

                # normalise
                smoothed_t = smoothed_t - t.logsumexp(smoothed_t.order(K_dim), 0)

            else:
                smoothed_t = filtered_t

            # save for next iteration
            filtered_t_plus_one = filtered_t
            smoothed_t_plus_one = smoothed_t

            # shift lps up by the max value in each kdim_to_sample to avoid numerical issues
            lp_max = smoothed_t.amax(kdims_to_sample)

            sampled_flat_idx = t.multinomial(t.exp(smoothed_t.order(K_dim) - lp_max).ravel(), num_samples, replacement=True)
            ts_indices[t_idx] = sampled_flat_idx[N_dim]

        indices[K_dim] = ts_indices[T_dim] # TODO: try just the final timestep (as we were doing before)

    return indices


def timeseries_pairwise_marginal(lps, Ks_to_sum, ts_init_Ks, N_dim, T_dim, indices, target_K_dim, i, j):
    """
    Computes the pairwise smoothed marginal xi[k_i, k_j] = p(x_i=k_i, x_j=k_j | data)
    for a Timeseries variable whose K-dimension is target_K_dim, at two (0-indexed)
    timesteps i < j -- the standard forward-backward pairwise-smoothing formula
    (Rabiner 1989's xi_t, generalised from adjacent timesteps to arbitrary lags):

        xi(k_i, k_j)  ~  alpha_i(k_i) * Bridge_{i->j}(k_i, k_j) * beta_j(k_j)

    alpha_i is exactly filtered_t at t=i (see sample_Ks_timeseries). beta_j is the
    backward-only message: since smoothed_t = filtered_t + logsumexp[(smoothed_{t+1}
    - filtered_{t+1}) + transition], the quantity (smoothed_t - filtered_t) satisfies
    its own clean recursion, which is what's computed here directly rather than via
    the full smoothed_t. Bridge_{i->j} is the one piece not otherwise computed: the
    UNcontracted chain_logmmexp over just the (i, j] sub-range, instead of summed
    all the way through like the main forward filtering pass.

    Mirrors sample_Ks_timeseries's own preamble (collect_lps, indexing into lps with
    already-resolved indices, building lp.order(T_dim, init_K_dim, K_dim)) so this
    uses exactly the same lp construction as alan's own posterior sampling -- any
    discrepancy against ground truth would reflect the formula, not a mismatch
    against how alan actually represents the model.

    Unlike a single-timestep marginal (which sample.moments() already computes
    exactly, with no sampling), this conditions on (and averages over N_dim) the
    already-resolved indices of anything upstream of this Timeseries (e.g. an
    `init` variable) -- so it inherits whatever finite-N noise that resolution
    carries, the same way sample_Ks_timeseries's own filtered_t/smoothed_t do.

    Returns xi as a plain [K, K] tensor (rows = index at i, cols = index at j,
    normalised to sum to 1), plus the plain [K] alpha_i and lp_ordered used to
    derive it (callers can reuse lp_ordered for e.g. multiple (i,j) queries against
    the same sample without rebuilding it).
    """
    assert i < j
    assert_unique_dim_iter(Ks_to_sum)
    assert set(unify_dims(lps)).issuperset(Ks_to_sum)
    _, lps_for_sampling, Ks_to_sample = collect_lps(lps, Ks_to_sum)

    for group_lps, kdims_to_sample, init_K_dim in zip(lps_for_sampling[::-1], Ks_to_sample[::-1], ts_init_Ks[::-1]):
        assert len(kdims_to_sample) == 1
        K_dim = kdims_to_sample[0]
        if K_dim is not target_K_dim:
            continue

        lp = sum(group_lps)

        assert K_dim in set(generic_dims(lp))
        assert T_dim in set(generic_dims(lp))
        assert init_K_dim in set(generic_dims(lp))
        assert init_K_dim in indices.keys()

        # index into lp with already-resolved indices, EXCEPT for this Timeseries's
        # own init_K_dim, which we condition on/average over explicitly below --
        # mirrors sample_Ks_timeseries exactly.
        for dim in list(set(generic_dims(lp)).intersection(set(indices.keys())).difference({init_K_dim})):
            lp = lp.order(dim)[indices[dim]]

        # From here on, everything is plain PyTorch (no torchdims): lp_ordered[t]
        # is [K,K] with row=index at t-1 (or init, for t=0) and col=index at t --
        # this convention is what makes chain_logmmexp correctly chain as ordinary
        # matrix multiplication, and is all the rest of this function relies on.
        lp_ordered = lp.order(T_dim, init_K_dim, K_dim)
        T = lp_ordered.shape[0]
        assert 0 <= i < j <= T - 1

        # alpha_i: forward-filtered log-mass at i. This is the one place we go back
        # to torchdim, to condition on/average over the already-resolved init index
        # exactly as sample_Ks_timeseries's own filtered_t does.
        cumulative_i = lp_ordered[0]
        for t_idx in range(1, i + 1):
            cumulative_i = logmmexp(cumulative_i, lp_ordered[t_idx])
        alpha_i = cumulative_i[init_K_dim, K_dim]
        alpha_i = alpha_i.order(init_K_dim)[indices[init_K_dim]]
        alpha_i = alpha_i.order(N_dim)
        alpha_i = t.logsumexp(alpha_i, 0).order(K_dim)
        alpha_i = alpha_i - t.logsumexp(alpha_i, 0)

        # beta_j: backward message, beta_{T-1} = 0 (log-space), plain [K] throughout.
        K = alpha_i.shape[0]
        beta = t.zeros(K)
        for t_idx in range(T - 1, j, -1):
            beta = t.logsumexp(lp_ordered[t_idx] + beta[None, :], dim=1)
        beta_j = beta

        # Bridge_{i->j}: uncontracted chain over (i, j], plain [K,K],
        # row = index at i, col = index at j.
        bridge = lp_ordered[i + 1]
        for t_idx in range(i + 2, j + 1):
            bridge = logmmexp(bridge, lp_ordered[t_idx])

        log_xi = alpha_i[:, None] + bridge + beta_j[None, :]
        log_xi = log_xi - t.logsumexp(log_xi.reshape(-1), 0)
        return log_xi.exp(), alpha_i, lp_ordered

    raise Exception(f"No timeseries group found with K-dimension {target_K_dim}")

def reduce_Ks(lps, Ks_to_sum):
    """
    Sum over Ks_to_sum, returning a single tensor.
    """
    assert_unique_dim_iter(Ks_to_sum)

    result, _, _ = collect_lps(lps, Ks_to_sum)

    return result

def checkpoint_reduce_Ks(lps, Ks_to_sum):
    return t.utils.checkpoint.checkpoint(reduce_Ks, lps, Ks_to_sum, use_reentrant=False)

def logsumexp_sum(_Ks_to_sum, *lps_to_reduce):
    #Needs a strange argument order, because checkpoint doesn't work with lists of lps.
    return logsumexp_dims(sum(lps_to_reduce), _Ks_to_sum, ignore_extra_dims=True)



def collect_lps(lps, Ks_to_sum):
    """
    Helper method that sums over Ks and returns a list of the reduced tensors along with a list of which Ks were reduced over for each reduced tensor.
    opt_einsum gives an "optimization path", i.e. the indicies of lps to reduce.
    We use this path to do our reductions, handing everything off to a simple t.einsum
    call (which ensures a reasonably efficient implementation for each reduction).
    """
    assert_unique_dim_iter(Ks_to_sum)
    
    args, out_dims = einsum_args(lps, Ks_to_sum)
    path = opt_einsum.contract_path(*args)[0]
    
    all_reduced_lps = [[*lps]]
    Ks_to_sample = []
    
    for lp_idxs in path:
        #Split lps into two groups: those we're going to reduce, and the rest.
        lps_to_reduce = tuple(lps[i] for i in lp_idxs)
        lps = [lps[i] for i in range(len(lps)) if i not in lp_idxs]

        #In this step, sum over all Ks in Ks_to_sample, and not in lps (i.e. the other tensors)
        _Ks_to_sum = tuple(set(Ks_to_sum).difference(unify_dims(lps)).intersection(unify_dims(lps_to_reduce)))
        Ks_to_sample.append(_Ks_to_sum)

        #Instantiates but doesn't save lp with _Ks_to_sample dims
        lps.append(logsumexp_sum(_Ks_to_sum, *lps_to_reduce))
        all_reduced_lps.append([*lps])

    all_reduced_lps = all_reduced_lps[:-1]

    assert 1==len(lps)
    result = lps[0]

    # Find indices of any empty K sets
    empty_K_idxs = []
    for i in range(len(Ks_to_sample)):
        if Ks_to_sample[i] == ():
            empty_K_idxs.append(i)

    # Remove empty K sets and corresponding reduced lps
    all_reduced_lps = [lps for i, lps in enumerate(all_reduced_lps) if i not in empty_K_idxs]
    Ks_to_sample = [Ks for i, Ks in enumerate(Ks_to_sample) if i not in empty_K_idxs]
    
    return result, all_reduced_lps, Ks_to_sample
