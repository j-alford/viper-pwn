"""Corner plots and TeX labels for MCMC chains."""

from __future__ import annotations

import numpy as np




_PARAM_TEX = {
    'esn': r'$E_{\rm SN}$ [$10^{51}$ erg]',
    'mej': r'$M_{\rm ej}$ [$M_\odot$]',
    'n_ism': r'$n_{\rm ISM}$ [cm$^{-3}$]',
    'braking_index': r'$n$',
    'tau_sd': r'$\tau_{\rm sd}$ [yr]',
    'eta_b': r'$\eta_B$',
    'e_min': r'$E_{\rm min}$ [GeV]',
    'e_max': r'$E_{\rm max}$ [TeV]',
    'e_break': r'$E_{\rm break}$ [GeV]',
    'p1': r'$p_1$',
    'p2': r'$p_2$',
    'distance': r'$d$ [kpc]',
    'ic_temp_1': r'$T_{\rm IC,1}$ [K]',
    'ic_norm_1': r'$u_{\rm IC,1}$',
    'ic_temp_2': r'$T_{\rm IC,2}$ [K]',
    'ic_norm_2': r'$u_{\rm IC,2}$',
}

_PARAM_TEX_SHORT = {
    'esn': r'$E_{\rm SN}$',
    'mej': r'$M_{\rm ej}$',
    'n_ism': r'$n_{\rm ISM}$',
    'braking_index': r'$n$',
    'tau_sd': r'$\tau_{\rm sd}$',
    'eta_b': r'$\eta_B$',
    'e_min': r'$E_{\rm min}$',
    'e_max': r'$E_{\rm max}$',
    'e_break': r'$E_{\rm break}$',
    'p1': r'$p_1$',
    'p2': r'$p_2$',
    'distance': r'$d$',
    'ic_temp_1': r'$T_{\rm IC,1}$',
    'ic_norm_1': r'$u_{\rm IC,1}$',
    'ic_temp_2': r'$T_{\rm IC,2}$',
    'ic_norm_2': r'$u_{\rm IC,2}$',
}

_PARAM_UNIT_LABEL = {
    'esn': r'$10^{51}$ erg',
    'mej': r'$M_\odot$',
    'n_ism': r'cm$^{-3}$',
    'tau_sd': 'yr',
    'e_min': 'GeV',
    'e_max': 'TeV',
    'e_break': 'GeV',
    'distance': 'kpc',
    'ic_temp_1': 'K',
    'ic_temp_2': 'K',
}

_PARAM_SCALE = {
    'e_max': 1e-3,
    'distance': 1e-3,
}

_PARAM_TITLE_FMT = {
    'esn': '.3f',
    'mej': '.1f',
    'n_ism': '.2f',
    'braking_index': '.2f',
    'tau_sd': '.0f',
    'eta_b': '.1e',
    'e_min': '.0f',
    'e_max': '.0f',
    'e_break': '.0f',
    'p1': '.2f',
    'p2': '.2f',
    'distance': '.2f',
    'ic_temp_1': '.0f',
    'ic_norm_1': '.1e',
    'ic_temp_2': '.0f',
    'ic_norm_2': '.1e',
}


def plot_corner(h5file, config, burn_frac=0.2, params=None,
                figsize=None, save=None, chi2_cut=None, **corner_kwargs):
    """Make a corner plot from an MCMC chain.

    Parameters
    ----------
    h5file : str
        Path to the emcee HDF5 backend file.
    config : _FitSpec
        The fit specification (from build_fit_spec).
    burn_frac : float
        Fraction of chain to discard as burn-in (default 0.2).
    params : list of str, optional
        Subset of parameter names to plot. None for all.
    figsize : tuple, optional
        Figure size. Defaults to (20, 20) for all or (12, 12) for subset.
    save : str, optional
        Path to save the figure.
    chi2_cut : float, optional
        Drop samples with chi2 > chi2_cut (default best_chi2 + 15).
    **corner_kwargs
        Extra kwargs passed to corner.corner().

    Returns
    -------
    fig : Figure
    flat_phys : ndarray
        The physical-units samples used for plotting.
    """
    import emcee
    try:
        import corner
    except ImportError:
        raise ImportError("pip install corner")

    backend = emcee.backends.HDFBackend(h5file, read_only=True)
    chain = backend.get_chain()
    lps = backend.get_log_prob()
    nburn = int(chain.shape[0] * burn_frac)
    flat = chain[nburn:].reshape(-1, chain.shape[2])
    flat_lps = lps[nburn:].reshape(-1)

    best_lp = np.max(flat_lps)
    if chi2_cut is None:
        chi2_cut = -2 * best_lp + 15
    mask = -2 * flat_lps < chi2_cut
    flat = flat[mask]
    n_dropped = (~mask).sum()
    if n_dropped > 0:
        print(f"Dropped {n_dropped}/{len(mask)} samples "
              f"(chi2 > {chi2_cut:.0f})")

    flat_phys = flat.copy()
    for i, fp in enumerate(config.free_params):
        if fp.log:
            flat_phys[:, i] = 10**flat[:, i]
        scale = _PARAM_SCALE.get(fp.name, 1.0)
        if scale != 1.0:
            flat_phys[:, i] *= scale

    all_names = config.param_names
    all_labels = [_PARAM_TEX.get(n, n) for n in all_names]

    if params is not None:
        idx = [all_names.index(n) for n in params]
        flat_phys = flat_phys[:, idx]
        labels = [all_labels[i] for i in idx]
        names_used = [all_names[i] for i in idx]
        default_size = (12, 12)
    else:
        labels = all_labels
        names_used = list(all_names)
        default_size = (20, 20)

    ranges = []
    for i in range(flat_phys.shape[1]):
        lo, hi = np.percentile(flat_phys[:, i], [5, 95])
        pad = (hi - lo) * 0.2
        ranges.append((lo - pad, hi + pad))

    defaults = dict(quantiles=[0.16, 0.5, 0.84], show_titles=False,
                    max_n_ticks=3, labelpad=0.08,
                    range=ranges, label_kwargs={'fontsize': 12})
    defaults.update(corner_kwargs)

    fig = corner.corner(flat_phys, labels=labels, **defaults)
    fig.set_size_inches(figsize or default_size)

    ndim = flat_phys.shape[1]
    axes = np.array(fig.axes).reshape(ndim, ndim)
    for i, name in enumerate(names_used):
        ax = axes[i, i]
        q16, q50, q84 = np.percentile(flat_phys[:, i], [16, 50, 84])
        fmt = _PARAM_TITLE_FMT.get(name, '.2f')
        unit = _PARAM_UNIT_LABEL.get(name, '')
        sym = _PARAM_TEX_SHORT.get(name, name)
        up = q84 - q50
        lo = q50 - q16
        med_s = f"{q50:{fmt}}"
        lo_s = f"{lo:{fmt}}"
        up_s = f"{up:{fmt}}"
        title_str = f"{sym} = ${med_s}_{{-{lo_s}}}^{{+{up_s}}}$"
        if unit:
            title_str += f" {unit}"
        ax.set_title(title_str, fontsize=11, loc='right', pad=6)

    if save:
        fig.savefig(save, dpi=100, bbox_inches='tight')

    return fig, flat_phys
