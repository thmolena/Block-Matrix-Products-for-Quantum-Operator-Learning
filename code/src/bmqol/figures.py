"""Publication figures generated only from the locked numerical result."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _setup():
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9,
            "legend.fontsize": 7.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    return plt


def _save(figure, output: Path, name: str) -> None:
    for suffix in ("pdf", "png"):
        figure.savefig(output / f"{name}.{suffix}", bbox_inches="tight")


def write_figures(payload: dict[str, Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    plt = _setup()
    from matplotlib.ticker import LogLocator, NullFormatter, FuncFormatter, FixedLocator
    colors = ("#1f4e79", "#c44e52", "#2a9d8f", "#7b2cbf", "#d97706")
    records = payload["records"]
    figure, axes = plt.subplots(2, 2, figsize=(6.8, 4.5), sharey=True)
    for axis, record in zip(axes.ravel(), records):
        frontier = record["frontier"]
        axis.semilogy([x["depth"] for x in frontier], [x["certificate"] for x in frontier],
                      "o-", label="preflight bound", color=colors[0])
        axis.semilogy([x["depth"] for x in frontier], [max(x["gradient_error"],1e-18) for x in frontier],
                      "s--", label="reference discrepancy", color=colors[1])
        axis.axhline(payload["tolerance"],ls=":",color="gray")
        axis.axvline(record["selected_depth"],color="gray",alpha=.3)
        axis.set(title=record["name"],xlabel="Krylov depth",ylabel="gradient error / bound")
        axis.grid(alpha=.2)
    axes[0,0].legend(frameon=False)
    figure.tight_layout(); _save(figure,output,"certificate_frontier"); plt.close(figure)

    figure,axes=plt.subplots(2,2,figsize=(6.8,4.8),sharey=True)
    for axis,record in zip(axes.ravel(),records):
        for method,label,color,marker in zip(("compressed","full","full_adjoint","scalar_polarization"),
                ("compressed","full block","full adjoint","scalar polarization"),colors,("o","s","D","^")):
            rows=[x for x in record['comparisons']['frontiers'] if x['method']==method]
            axis.loglog([x['timing']['median_seconds'] for x in rows],
                        [max(x['error'],1e-17) for x in rows],marker+'-',label=label,color=color)
        axis.scatter(record['proposed_timing']['median_seconds'],max(record['gradient_error'],1e-17),
                     marker='*',s=90,color=colors[4],label='joint certified',zorder=5)
        axis.axhline(1e-3,color='gray',ls=':')
        axis.set(title=record['name'],xlabel='gradient runtime (s)',ylabel='reference discrepancy')
        axis.xaxis.set_major_locator(LogLocator(base=10, numticks=4))
        axis.xaxis.set_minor_formatter(NullFormatter())
        axis.grid(alpha=.2)
    axes[0,0].legend(frameon=False,fontsize=6.7)
    figure.tight_layout(); _save(figure,output,'work_accuracy'); plt.close(figure)

    figure,axis=plt.subplots(figsize=(6.8,2.8))
    offsets=np.arange(len(records))
    for j,(method,label,color) in enumerate(zip(('joint','full','incremental','oracle','reference'),
            ('joint certified','full certified','incremental adjoint','adjoint observed oracle','augmented reference'),colors)):
        timings=[]
        for record in records:
            if method in ('joint','full'):
                timing=next(x['timing'] for x in record['comparisons']['certified_policies'] if x['policy']==method)
            elif method=='incremental':
                timing=next(x['timing'] for x in record['comparisons']['incremental_adaptive']
                            if x['method']=='full' and x['tolerance']==1e-3)
            elif method=='oracle':
                timing=next(x['timing'] for x in record['comparisons']['observed_accuracy_oracles']
                            if x['method']=='full_adjoint' and x['tolerance']==1e-3)
            else: timing=record['independent_reference_timing']
            timings.append(timing)
        med=np.array([x['median_seconds'] for x in timings])
        low=np.array([min(x['seconds']) for x in timings]); high=np.array([max(x['seconds']) for x in timings])
        axis.bar(offsets+(j-2)*.16,med,.15,color=color,label=label,
                 yerr=np.array([med-low,high-med]),capsize=2,error_kw={'linewidth':.7})
    axis.set_yscale('log'); axis.set_ylabel('gradient runtime (s)')
    axis.set_xticks(offsets,[x['name'] for x in records]); axis.legend(frameon=False,ncol=2,fontsize=7)
    axis.grid(axis='y',alpha=.2); figure.tight_layout()
    _save(figure,output,'runtime_scaling');plt.close(figure)

    figure,axes=plt.subplots(2,2,figsize=(6.8,4.7))
    for row,n in enumerate(payload['scalable_learning']['config']['sizes']):
        for method,label,color,marker in zip(('joint','full','fixed_depth_4','reference'),
                ('joint certified','full certified','fixed depth 4','reference gradient'),colors,('o','s','^','D')):
            runs=[r for r in payload['scalable_learning']['runs']
                  if r['n']==n and r['method']==method and r['stationarity_tolerance']==1e-7]
            selected=next(r for r in runs if r['noise']==0 and r['truth_index']==0 and r['start_index']==0)
            trajectory=selected['history']
            axes[row,0].loglog([max(h['seconds'],1e-5) for h in trajectory],
                [max(h['parameter_error'],1e-13) for h in trajectory],marker+'-',
                color=color,lw=1,ms=3,label=label)
            axes[row,1].scatter([r['optimization_seconds'] for r in runs],
                [max(r['parameter_error'],1e-13) for r in runs],
                marker=marker,color=color,s=13,alpha=.65,label=label)
        for column in (0,1):
            axis=axes[row,column]
            axis.set_xscale('log');axis.set_yscale('log');axis.grid(alpha=.2)
            axis.xaxis.set_major_locator(FixedLocator(np.geomspace(*axis.get_xlim(),3)))
            axis.xaxis.set_major_formatter(FuncFormatter(lambda value,pos:f'{value:.2g}'))
            axis.xaxis.set_minor_formatter(NullFormatter())
            axis.set(xlabel='optimization time (s)',ylabel='parameter error',
                     title=f'n = {n}: '+('accepted iterates' if column==0 else 'all final results'))
        axes[row,1].axhline(1e-3,color='gray',ls=':',lw=.8)
    axes[0,0].legend(frameon=False,fontsize=6.5)
    figure.tight_layout();_save(figure,output,'probe_spectrum');plt.close(figure)

    sweep=payload['rank_depth_sweep']
    figure,axes=plt.subplots(1,2,figsize=(6.8,2.4))
    for axis,key,title,cmap in zip(axes,('selected_ranks','selected_depths'),
                                   ('selected rank','selected depth'),('Blues','Oranges')):
        values=np.array(sweep[key]); im=axis.imshow(values,origin='lower',aspect='auto',cmap=cmap)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]): axis.text(j,i,str(values[i,j]),ha='center',va='center',fontsize=8)
        axis.set_xticks(range(4),sweep['horizons']);axis.set_yticks(range(3),['1e-2','1e-3','1e-4'])
        axis.set(title=title,xlabel='maximum evolution time',ylabel='tolerance')
        figure.colorbar(im,ax=axis,fraction=.046,pad=.04)
    figure.tight_layout();_save(figure,output,'rank_depth_map');plt.close(figure)
