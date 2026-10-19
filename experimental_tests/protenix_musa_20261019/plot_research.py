import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

r=Path(__file__).resolve().parent;out=r/'图表';out.mkdir(exist_ok=True)
d=json.loads((r/'研究分析.json').read_text())
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
colors=['#78869b','#3478bf','#25a18e']
core=d['main_pairs']
if core:
    fig,axes=plt.subplots(1,2,figsize=(14,max(5,.38*len(core)+2)),sharey=True)
    y=np.arange(len(core));h={x['pdb']:x for x in d['h100']}
    axes[0].barh(y-.23,[h[x['pdb']]['h100_s'] for x in core],height=.22,label='H100 reference',color=colors[0])
    axes[0].barh(y,[x['native_s'] for x in core],height=.22,label='MUSA native',color=colors[1])
    axes[0].barh(y+.23,[x['fused_s'] for x in core],height=.22,label='MUSA fused',color=colors[2])
    axes[0].set_xscale('log');axes[0].set_xlabel('Synchronized prediction time (s, log scale)')
    axes[1].barh(y-.13,[x['native_gib'] for x in core],height=.25,color=colors[1])
    axes[1].barh(y+.13,[x['fused_gib'] for x in core],height=.25,color=colors[2])
    axes[1].set_xlabel('Peak active tensor memory (GiB)')
    axes[0].set_yticks(y,[f"{x['pdb']} ({x['tokens']})" for x in core]);axes[0].invert_yaxis()
    for ax in axes:ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    axes[0].legend();fig.suptitle(f'Protenix full-budget inference: {len(core)} completed matched tasks')
    fig.text(.5,.015,'Same frozen tensors within MUSA. H100 is a same-task reference with platform-native kernels; not identical serialized tensors.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.04,1,.95]);fig.savefig(out/'整模型性能与显存.png',dpi=180);plt.close(fig)

ms=d['multiseed']
if ms:
    fig,ax=plt.subplots(1,2,figsize=(12,4.6))
    for i,pdb in enumerate(('7bca','6zsv','7b8b')):
        values=[x for x in ms if x['pdb']==pdb];positions=np.arange(len(values))+i*4
        ax[0].bar(positions,[x['time_reduction_pct'] for x in values],color=colors[i],label=pdb)
        qs=[x for x in values if x['lddt_delta'] is not None]
        ax[1].scatter([i*4+j for j in range(len(qs))],[x['lddt_delta']*1e4 for x in qs],color=colors[i],s=45,label=pdb)
    labels=[f"{x['pdb']}\n{x['seed']}" for pdb in ('7bca','6zsv','7b8b') for x in ms if x['pdb']==pdb]
    pos=[i*4+j for i,pdb in enumerate(('7bca','6zsv','7b8b')) for j,x in enumerate([x for x in ms if x['pdb']==pdb])]
    for a in ax:a.set_xticks(pos,labels,rotation=45,ha='right');a.axhline(0,color='#333',lw=.7);a.grid(axis='y',alpha=.2);a.set_axisbelow(True)
    ax[0].set_ylabel('Prediction time reduction (%)');ax[1].set_ylabel('Mean raw-index-0 lDDT change (x 1e-4)')
    ax[0].set_title('Three seeds per real task');ax[1].set_title('Observed quality change (zoomed scale)')
    fig.text(.5,.01,'Three timing repetitions per variant/seed; repeated calls are not independent proteins. Engineering lDDT screen limit: -0.005.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.06,1,.97]);fig.savefig(out/'多种子性能与质量.png',dpi=180);plt.close(fig)

ab=r/'ablation.json'
if ab.exists():
    rows=json.loads(ab.read_text())['records'];variants=['bias_only','softmax_only','out_of_place','full_in_place']
    fig,ax=plt.subplots(1,2,figsize=(13,4.6))
    for j,dtype in enumerate(('torch.bfloat16','torch.float32')):
        filtered=[x for x in rows if x['dtype']==dtype and x['layout']=='model_transposed'];xs=np.arange(3)
        for k,v in enumerate(variants):
            values=[]
            for n in (234,840,2053):
                a=next((x for x in filtered if x['tokens']==n and x['variant']=='native'),None)
                b=next((x for x in filtered if x['tokens']==n and x['variant']==v),None)
                values.append(100*(1-b['median_ms']/a['median_ms']) if a and b else np.nan)
            ax[j].bar(xs+(k-1.5)*.2,values,width=.19,label=v)
        ax[j].set_xticks(xs,['234','840','2053']);ax[j].set_xlabel('Token-shaped synthetic workload');ax[j].set_ylabel('Attention time reduction (%)');ax[j].set_title(dtype);ax[j].axhline(0,color='#333',lw=.7);ax[j].grid(axis='y',alpha=.2);ax[j].set_axisbelow(True)
    ax[0].legend(fontsize=8);fig.suptitle('Ablation of bias addition, softmax, and output buffer reuse')
    fig.text(.5,.01,'Synthetic inputs, batch <=64, transposed Q/K/V, FP32 mask; eight timed repetitions. Local attention, not whole-model acceleration.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.06,1,.93]);fig.savefig(out/'融合机制消融.png',dpi=180);plt.close(fig)

cross=d['crossover']
if cross:
    fig,ax=plt.subplots(figsize=(8,4.5));xs=np.arange(len(cross));v=[x['time_reduction_pct'] for x in cross]
    low=[max(0,x['time_reduction_pct']-x['bootstrap_low_pct']) for x in cross];high=[max(0,x['bootstrap_high_pct']-x['time_reduction_pct']) for x in cross]
    ax.bar(xs,v,color=colors[2]);ax.errorbar(xs,v,yerr=[low,high],fmt='none',ecolor='#333',capsize=5)
    ax.set_xticks(xs,[x['pdb'] for x in cross]);ax.set_ylabel('Prediction time reduction (%)');ax.axhline(0,color='#333',lw=.7);ax.set_title('Interleaved ABBA / BAAB validation, 8 calls per variant')
    ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.text(.5,.01,'Bars: median-based reduction. Error bars: bootstrap 95% timing intervals, not confidence about new proteins.',ha='center',fontsize=8)
    fig.tight_layout(rect=[0,.06,1,1]);fig.savefig(out/'交替复测与波动.png',dpi=180);plt.close(fig)
print('FIGURES',len(list(out.glob('*.png'))))
