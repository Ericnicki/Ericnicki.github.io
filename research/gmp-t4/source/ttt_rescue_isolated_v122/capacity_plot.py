from review import A
from audit_results import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
setup();P=R/'ttt_rescue_capacity_v125_output';d=json.loads((P/'results.json').read_text());fig,ax=plt.subplots(1,2,figsize=(10,4.5),constrained_layout=True)
for key,label,col in [('1.0','Original action scale','#3677af'),('10.0','Action input ×10','#d97935')]:
 t=d[key]['trace'];ax[0].plot([r['step'] for r in t],[r['contrast_mae'] for r in t],label=label,color=col)
ax[0].set(xlabel='Head-only full-batch updates',ylabel='TRAIN relative-return MAE',title='Stronger fitting can learn the TRAIN contrasts');ax[0].legend()
xx=np.arange(2);w=.3
ax[1].bar(xx-w/2,[d[k]['trace'][-1]['contrast_mae'] for k in ['1.0','10.0']],w,label='TRAIN');ax[1].bar(xx+w/2,[d[k]['development']['contrast_mae'] for k in ['1.0','10.0']],w,label='Diagnostic queries');ax[1].set_xticks(xx,['Original scale','Action ×10']);ax[1].set(yscale='log',ylabel='Relative-return MAE (log scale)',title='The fitted action effects extrapolate poorly');ax[1].legend()
fig.suptitle('Critic-fit diagnostic · frozen writer · no actor updates or control evaluation')
for ext in ['png','pdf','svg']:fig.savefig(P/f'capacity_diagnostic.{ext}',dpi=170)
print('CAPACITY_PLOT_COMPLETE')
