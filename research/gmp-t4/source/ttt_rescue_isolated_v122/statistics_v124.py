from review import A
from audit_results import np,bootstrap

def average(values):
    valid=[v for v in values if v is not None]
    return float(np.mean(valid)) if valid else None

def paired(values,rows):
    x=np.asarray(values,dtype=float);keep=np.isfinite(x)
    if not keep.any():return dict(mean=None,ci95=[None,None],tasks=0)
    result=bootstrap(x[keep],[r for r,k in zip(rows,keep) if k]);result['tasks']=int(keep.sum());return result

def summarize(rows):
    out={};arms=list(rows[0]['arms'])
    for group in ['all','low','middle','high','near','broad']:
        rs=[r for r in rows if group=='all' or r['spec']['region']==group or r['spec']['cohort']==group];out[group]={}
        for arm in arms:
            aa=[r['arms'][arm] for r in rs];gs=np.array([a['mean_return'] for a in aa],dtype=float);base=np.array([r['arms']['base']['mean_return'] for r in rs],dtype=float);new=np.array([r['arms']['new']['mean_return'] for r in rs],dtype=float);old=np.array([r['arms']['old']['mean_return'] for r in rs],dtype=float)
            delta=gs-base;valid_delta=delta[np.isfinite(delta)]
            z=dict(mean_return=float(np.nanmean(gs)),successes=sum(a['successes'] for a in aa),judged=sum(a['judged'] for a in aa),slots=6*len(rs),censored=6*len(rs)-sum(a['judged'] for a in aa),tasks_with_judgment=int(np.isfinite(gs).sum()),last3_all=sum(a['last_three_all'] for a in aa),lost_success=sum(a['failure_after_success'] for a in aa),first_trial=average([a['returns'][0] for a in aa]),first_trial_judged=sum(a['returns'][0] is not None for a in aa),last3_return=average([g for a in aa for g in a['returns'][-3:]]),gain_new=paired(gs-new,rs),gain_old=paired(gs-old,rs),gain_base=paired(gs-base,rs),damage_tasks=int(sum(valid_delta<0)),worst_task_gain_base=float(valid_delta.min()) if len(valid_delta) else None,p10_gain_base=float(np.quantile(valid_delta,.1)) if len(valid_delta) else None)
            complete=np.array([a['judged']==6 and r['arms']['new']['judged']==6 for a,r in zip(aa,rs)])
            z['complete_task_gain_new']=paired(np.where(complete,gs-new,np.nan),rs)
            z['censor_penalty_sensitivity']={}
            for penalty in [-5.,-10.]:
                impute=lambda key:np.array([np.mean([penalty if g is None else g for g in r['arms'][key]['returns']]) for r in rs])
                v=impute(arm);z['censor_penalty_sensitivity'][str(penalty)]=dict(mean_return=float(v.mean()),gain_new=paired(v-impute('new'),rs),gain_old=paired(v-impute('old'),rs))
                if arm.startswith(('hindsight','grounded')):z['censor_penalty_sensitivity'][str(penalty)]['gain_matched_mc']=paired(v-impute('mc_'+arm.split('_')[-1]),rs)
            if arm.startswith(('hindsight','grounded')):
                mc=np.array([r['arms']['mc_'+arm.split('_')[-1]]['mean_return'] for r in rs],dtype=float);z['gain_matched_mc']=paired(gs-mc,rs)
            out[group][arm]=z
    return out
