import math
from typing import Optional, Union

from .Plate import Plate, tree_values, update_scope
from .Group import Group
from .Data import Data
from .Enumerate import Enumerate
from .Timeseries import Timeseries
from .dist import Dist, datagroup, enumerategroup

from .utils import *
from .reduce_Ks import reduce_Ks
from .Split import Split, checkpoint, no_checkpoint
from .Sampler import Sampler

def logPQ_plate(
        name:Optional[str],
        P:Plate,
        Q:Plate,
        sample: dict,
        inputs_params: dict,
        data: dict,
        extra_log_factors: dict,
        scope: dict[str, Tensor],
        active_platedims:list[Dim],
        all_platedims:dict[str: Dim],
        groupvarname2Kdim:dict[str, Tensor],
        varname2groupvarname:dict[str, str],
        sampler:Sampler,
        computation_strategy:Optional[Split],
        pairwise_query=None,
        pairwise_result=None):
    """
    pairwise_query, pairwise_result: optional hook used by
    Sample.timeseries_pairwise_covariance_exact to insert a source term at
    two timestep positions of a single Timeseries variable and get an EXACT
    (autodiff, no sampling) cross-timestep joint moment out, mirroring how
    ordinary cross-variable joint moments already work (Sample._marginal_idxs).
    pairwise_query is (target_K_dim, i, j, J_tensor); when set, and recursion
    reaches the Timeseries group with that K-dimension, a source term J is
    inserted at the (i, j) split point instead of the ordinary uniform
    chain_logmmexp reduction, and pairwise_result['done'] is set. No effect
    on the ordinary path -- both default to None. Scoped to exactly one
    Timeseries group per plate (asserts otherwise).
    """

    #Returns a tuple of dicts, with computation_strategy samples, inputs_params, extra_log_factors, data and all_platedims.
    siedas = computation_strategy.split_args(
        name=name,
        sample=sample,
        inputs_params=inputs_params,
        extra_log_factors=extra_log_factors,
        data=data,
        all_platedims=all_platedims,
    )

    lpq_func = _logPQ_plate if computation_strategy is no_checkpoint else _logPQ_plate_checkpointed

    lpq = None
    for sieda in siedas:
        lpq = lpq_func(
            name=name,
            P=P,
            Q=Q,
            scope=scope,
            active_platedims=active_platedims,
            groupvarname2Kdim=groupvarname2Kdim,
            varname2groupvarname=varname2groupvarname,
            sampler=sampler,
            computation_strategy=computation_strategy,
            **sieda,
            prev_lpq = lpq,
            pairwise_query=pairwise_query,
            pairwise_result=pairwise_result,
        )
    assert isinstance(lpq, Tensor)

    return lpq, (), (), ()

def _logPQ_plate_checkpointed(*args, **kwargs):
    return t.utils.checkpoint.checkpoint(_logPQ_plate_args_kwargs, args, kwargs, use_reentrant=False)

def _logPQ_plate_args_kwargs(args, kwargs):
    return _logPQ_plate(*args, **kwargs)

def _logPQ_plate(
        name:Optional[str],
        P:Plate,
        Q:Plate,
        sample: dict,
        inputs_params: dict,
        data: dict,
        extra_log_factors: dict,
        scope: dict[str, Tensor],
        active_platedims:list[Dim],
        all_platedims:dict[str: Dim],
        groupvarname2Kdim:dict[str, Tensor],
        varname2groupvarname:dict[str, str],
        sampler:Sampler,
        computation_strategy:Optional[Split],
        prev_lpq,
        pairwise_query=None,
        pairwise_result=None):

    assert isinstance(P, Plate)
    assert isinstance(Q, Plate)

    assert isinstance(sample, dict)
    assert isinstance(inputs_params, dict)
    assert isinstance(data, dict)
    assert isinstance(extra_log_factors, dict)


    #Push an extra plate, if not the top-layer plate (top-layer plate is signalled
    #by name=None.
    if name is not None:
        new_platedim = all_platedims[name]
        active_platedims = [*active_platedims, new_platedim]

    scope = update_scope(scope, inputs_params)
    scope = update_scope(scope, sample)

    lps, all_Ks, K_currs, K_inits = lp_getter(
        name=name,
        P=P,
        Q=Q,
        sample=sample,
        inputs_params=inputs_params,
        data=data,
        extra_log_factors=extra_log_factors,
        scope=scope,
        active_platedims=active_platedims,
        all_platedims=all_platedims,
        groupvarname2Kdim=groupvarname2Kdim,
        varname2groupvarname=varname2groupvarname,
        sampler=sampler,
        computation_strategy=computation_strategy,
        pairwise_query=pairwise_query,
        pairwise_result=pairwise_result,
    )

    assert len(K_currs) == len(K_inits)

    #Sum out Ks for non-timeseries variables.
    #Returns a single tensor, with dimensions:
    # Higher plates
    # K_prevs from timeseries
    # K_currs from timeseries
    # Ks from higher plates.
    lp = reduce_Ks(lps, all_Ks)

    #Sum over new_platedim
    if name is not None:
        if 0 < len(K_inits):
            ##Timeseries
            #NOTE: deliberately NOT gated on pairwise_result['done'] here -- that flag is
            #written below purely so callers (e.g. Sample.timeseries_pairwise_covariance_exact)
            #can tell the hook fired. Reading it back here to gate control flow would make
            #the branch taken depend on mutable state set by a previous call, which breaks
            #under checkpointed computation_strategy: t.utils.checkpoint.checkpoint calls this
            #function twice (once to compute, once to recompute activations for the backward
            #pass), and the second call must take the SAME branch as the first, or checkpoint's
            #save-tensor consistency check fails. K_currs[0] is pairwise_query[0] already
            #uniquely identifies the one target plate (Dims are unique per Timeseries
            #K-dimension), so no separate guard is needed.
            use_pairwise = (
                pairwise_query is not None
                and 1 == len(K_currs)
                and K_currs[0] is pairwise_query[0]
            )
            if use_pairwise:
                lp = _timeseries_pairwise_source_term(
                    lp.order(new_platedim, K_inits[0], K_currs[0]), pairwise_query)
                pairwise_result['done'] = True
            else:
                lp = lp.order(new_platedim, K_inits, K_currs)    # Removes torchdims from T, and Ks
                lp = chain_logmmexp(lp)# Kprevs x Kcurrs
                assert 2 == lp.ndim

                #lp = lp.logsumexp(-1)
                lp = t.logsumexp(lp, -1)
            assert 1 == lp.ndim
            #Put torchdim back.
            #Stupid trailing None is necessary, because otherwise the list of K_inits is just splatted in, rather than being treated as a group.
            lp = lp[K_inits, None].squeeze(-1)

            assert prev_lpq is None

        else:
            #No timeseries
            lp = lp.sum(new_platedim)

            if prev_lpq is not None:
                assert set(generic_dims(lp)) == set(generic_dims(prev_lpq))
                lp = prev_lpq + lp

    return lp

def _timeseries_pairwise_source_term(lp_ordered, pairwise_query):
    """
    lp_ordered: plain tensor [T, Kinit, Kcurr] (T_dim, K_init_dim, K_curr_dim already
    ordered out by the caller). Follows the same convention as
    reduce_Ks.timeseries_pairwise_marginal / sample_Ks_timeseries: lp_ordered[t] is the
    log-transition matrix into position t (row = index at t-1, or the init variable's
    index when t==0; col = index at t).

    pairwise_query = (target_K_dim, i, j, J), with 0 <= i < j <= T-1: builds the
    [Kinit, K_i, K_j] tensor

        combined[k_init, k_i, k_j] = alpha_i[k_init, k_i] + (Bridge_{i->j}[k_i, k_j] + J[k_i, k_j]) + beta_j[k_j]

    (see reduce_Ks.timeseries_pairwise_marginal's docstring for the forward-backward
    derivation of alpha/Bridge/beta), inserts the source term J additively into the
    Bridge, then immediately contracts k_i and k_j back out via ordinary logsumexp.
    The returned tensor is [Kinit]-shaped -- EXACTLY what the unmodified
    chain_logmmexp(lp); logsumexp(lp, -1) path this replaces would have returned --
    so nothing downstream (autograd's scalar-L requirement, outer plate nesting,
    Kinit_dim propagation) needs to change. Differentiating the resulting scalar ELBO
    w.r.t. J recovers the exact joint marginal xi(k_i, k_j), via the same
    softmax-gradient identity Sample._marginal_idxs already uses for ordinary
    (non-timeseries) joint moments.
    """
    _, i, j, J = pairwise_query
    T = lp_ordered.shape[0]
    assert 0 <= i < j <= T - 1

    #alpha_i: [Kinit, K_i], folding lp_ordered[0..i].
    alpha_i = lp_ordered[0]
    for t_idx in range(1, i + 1):
        alpha_i = logmmexp(alpha_i, lp_ordered[t_idx])

    #Bridge_{i->j}: [K_i, K_j], uncontracted chain over (i, j].
    bridge = lp_ordered[i + 1]
    for t_idx in range(i + 2, j + 1):
        bridge = logmmexp(bridge, lp_ordered[t_idx])
    bridge = bridge + J

    #beta_j: [K_j], folding lp_ordered[(j, T-1]] backwards; beta_{T-1} = 0.
    beta = t.zeros(lp_ordered.shape[-1], dtype=lp_ordered.dtype, device=lp_ordered.device)
    for t_idx in range(T - 1, j, -1):
        beta = t.logsumexp(lp_ordered[t_idx] + beta[None, :], dim=1)

    combined = alpha_i[:, :, None] + bridge[None, :, :] + beta[None, None, :]
    return t.logsumexp(combined, dim=(1, 2))

def logPQ_gdt(
        name:str,
        P:dict,
        Q:dict,
        sample: dict, 
        inputs_params: dict,
        data: None,
        extra_log_factors: None, 
        scope: dict[str, Tensor], 
        active_platedims:list[Dim],
        all_platedims:dict[str: Dim],
        groupvarname2Kdim:dict[str, Tensor],
        varname2groupvarname:dict[str, str],
        sampler:Sampler,
        computation_strategy:Optional[Split]):

    assert isinstance(sample, dict)
    assert inputs_params is None
    assert extra_log_factors is None
    assert isinstance(P, dict)
    assert isinstance(Q, dict)

    prog_P = P
    prog_Q = Q

    assert 1<=len(prog_P) 
    assert set(prog_P.keys()) == set(prog_Q.keys())

    #Immediately return if data.
    if datagroup(prog_Q):
        assert len(prog_Q) == 1
        k = next(iter(prog_Q))

        assert isinstance(prog_Q[k], Data)
        assert sample[k] is None
        assert isinstance(data[k], Tensor)

        lp, _ = prog_P[k].log_prob(data[k], scope, None, None)

        return lp, (), (), ()

    Kdim = groupvarname2Kdim[name]

    #Exact marginalisation via enumeration (see Enumerate's docs): Q is
    #never consulted -- sample[k] already holds the deterministic
    #particle-i-is-category-i assignment sample_gdt built. P's log-prob at
    #those K=cardinality particles IS the group's contribution; ordinary
    #logsumexp reduction over Kdim (done by the caller, same as any other
    #group) then gives exactly log(sum_x P(x)), with no -log(K)/Q correction.
    if enumerategroup(prog_Q):
        assert len(prog_Q) == 1
        k = next(iter(prog_Q))

        assert isinstance(prog_Q[k], Enumerate)
        assert isinstance(sample[k], Tensor)

        T_dim = active_platedims[-1] if 1<=len(active_platedims) else None
        lp, Kinit_p = prog_P[k].log_prob(sample[k], scope=scope, T_dim=T_dim, K_dim=Kdim)
        if Kinit_p is not None:
            raise Exception("Enumerate doesn't yet support Timeseries variables")

        return lp, (Kdim,), (), ()

    total_logP = 0.
    total_logQ = 0.

    #Gather extra dimensions for timeseries
    Kinit_dims = []
    for v in prog_P.values():
        if isinstance(v, Timeseries):
            Kinit_dims.append(groupvarname2Kdim[varname2groupvarname[v.init]])

    T_dim = active_platedims[-1] if 1<=len(active_platedims) else None

    Kinits = []
    for k in prog_P:
        dist_P   = prog_P[k]
        dist_Q   = prog_Q[k]
        sample_k = sample[k]

        assert isinstance(dist_P, (Dist, Timeseries))
        assert isinstance(dist_Q, (Dist, Timeseries))
        assert isinstance(sample_k, Tensor)
        assert data[k] is None

        lp, Kinit_p = dist_P.log_prob(sample_k, scope=scope, T_dim=T_dim, K_dim=Kdim)
        lq, Kinit_q = dist_Q.log_prob(sample_k, scope=scope, T_dim=T_dim, K_dim=Kdim)

        if Kinit_q is not None:
            assert Kinit_p is Kinit_q

        if Kinit_p is not None:
            assert isinstance(Kinit_p, Dim)
            Kinits.append(Kinit_p)

        total_logP = total_logP + lp
        total_logQ = total_logQ + lq

    total_logQ = sampler.reduce_logQ(total_logQ, active_platedims, Kdim)
    lp = total_logP - total_logQ - math.log(Kdim.size)

    if 1 <= len(Kinits):
        #There's at least one timeseries in the group.
        Kinit0 = Kinits[0]
        for Kid in Kinit_dims:
            assert Kid is Kinit0
        Knon_timeseries = ()
        Ktimeseries = (Kdim,)
        Kinits = (Kinit0,)
    else:
        #No timeseries in the group.
        Knon_timeseries = (Kdim,)
        Ktimeseries = ()
        Kinits = ()

    return lp, Knon_timeseries, Ktimeseries, Kinits


def lp_getter(
        name:Optional[str],
        P:Plate, 
        Q:Plate, 
        sample: dict, 
        inputs_params: dict,
        data: dict,
        extra_log_factors: dict, 
        scope: dict[str, Tensor], 
        active_platedims:list[Dim],
        all_platedims:dict[str: Dim],
        groupvarname2Kdim:dict[str, Tensor],
        varname2groupvarname:dict[str, str],
        sampler:Sampler,
        computation_strategy:Optional[Split],
        pairwise_query=None,
        pairwise_result=None):
    """Traverses Q according to the structure of P collecting log probabilities

    pairwise_query, pairwise_result: forwarded, unchanged, to any nested Plate
    (see logPQ_plate's docstring). Not forwarded to logPQ_gdt, since the
    Timeseries reduction the hook targets happens one level up, in the
    enclosing _logPQ_plate, after this function returns.
    """

    assert isinstance(P, Plate)
    assert isinstance(Q, Plate)

    #We want to pass back just the incoming scope, as nothing outside the plate can see
    #variables inside the plate.  So `scope` is the internal scope, and `parent_scope`
    #is the external scope we will pass back.

    assert set(P.flat_prog.keys()) == set(Q.flat_prog.keys())

    lps = list(tree_values(extra_log_factors).values())
    Knon_timeseries = []
    Ktimeseries = []
    Kinits = []

    for childname, childQ in Q.grouped_prog.items():
        if isinstance(childQ, dict):
            childP = {varname: P.flat_prog[varname] for varname in childQ}
            method = logPQ_gdt
            extra_kwargs = {}
        else:
            assert isinstance(childQ, Plate)
            childP = P.flat_prog[childname]
            assert isinstance(childP, Plate)
            method = logPQ_plate
            extra_kwargs = dict(pairwise_query=pairwise_query, pairwise_result=pairwise_result)

        lp, _Knon_timeseries, _Ktimeseries, _Kinits = method(
            name=childname,
            P=childP,
            Q=childQ,
            sample=Q.grouped_get(sample, childname),
            data=Q.grouped_get(data, childname),
            inputs_params=inputs_params.get(childname),
            extra_log_factors=extra_log_factors.get(childname),
            scope=scope,
            active_platedims=active_platedims,
            all_platedims=all_platedims,
            groupvarname2Kdim=groupvarname2Kdim,
            varname2groupvarname=varname2groupvarname,
            sampler=sampler,
            computation_strategy=computation_strategy,
            **extra_kwargs,
        )

        lps.append(lp)
        Knon_timeseries.extend(_Knon_timeseries)
        Ktimeseries.extend(_Ktimeseries)
        Kinits.extend(_Kinits)
        

    #Collect all non-timeseries Ks in the plate (note: iterating through Q!!)
    #all_Ks = []
    #for varname, dist in Q.grouped_prog.items():
    #    if isinstance(dist, dict):
    #        if not datagroup(dist):
    #            all_Ks.append(groupvarname2Kdim[varname])
    #    else:
    #        assert isinstance(dist, Plate)
            
    return lps, Knon_timeseries, Ktimeseries, Kinits
