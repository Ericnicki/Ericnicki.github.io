from review import A
from audit_results import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
setup();assert json.loads((OUT/'audit.json').read_text())['status']=='passed';data=json.loads((OUT/'metrics.json').read_text());rows=json.loads((CONF/'summary.json').read_text())
fig,axes=plt.subplots(2,2,figsize=(13,10),constrained_layout=True)
keys=['base','old','new','old_actor','half','mc_2033000','mc_2033001','hindsight_2033000','hindsight_2033001','grounded_2033000','grounded_2033001']
labels=['BASE','v100','v115','old actor / new writer','half acceleration','MC seed 0','MC seed 1','hindsight seed 0','hindsight seed 1','grounded seed 0','grounded seed 1']
colors=['#888888','#3677af','#ce4f45','#329a68','#9c793b','#84a3c0','#84a3c0','#9871b5','#9871b5','#db9948','#db9948']
ax=axes[0,0];vals=[data['all'][k]['mean_return'] for k in keys];ax.barh(labels,vals,color=colors);ax.invert_yaxis();ax.axvline(0,color='black',lw=.8);ax.set_xlabel('Mean native return');ax.set_title('60 fresh tasks · six trials each')
ax=axes[0,1];v=[data['all'][k]['gain_new']['mean'] for k in keys];bounds=[data['all'][k]['gain_new']['ci95'] for k in keys];ax.errorbar(v,np.arange(len(keys)),xerr=np.array([[m-b[0],b[1]-m] for m,b in zip(v,bounds)]).T,fmt='o',color='#34495e');ax.set_yticks(np.arange(len(keys)),labels);ax.invert_yaxis();ax.axvline(0,color='black',lw=.8);ax.set_xlabel('Task-paired return gain vs v115 (95% bootstrap CI)');ax.set_title('Tasks are the resampling unit')
ax=axes[1,0];xx=np.arange(3);width=.11
for j,(key,label,col) in enumerate([('base','BASE','#888888'),('old','v100','#3677af'),('new','v115','#ce4f45'),('old_actor','old actor / new writer','#329a68'),('mc_2033000','MC seed 0','#84a3c0'),('hindsight_2033000','hindsight seed 0','#9871b5'),('grounded_2033000','grounded seed 0','#db9948')]):
 rates=[data[region][key]['successes']/data[region][key]['slots'] for region in ['low','middle','high']];ax.bar(xx+(j-3)*width,rates,width,label=label,color=col)
ax.set_xticks(xx,['Low friction','Middle friction','High friction']);ax.set_ylim(0,1);ax.set_ylabel('Successful trial fraction (censored = not success)');ax.set_title('Seed 0 by friction; both seeds reported above');ax.legend(fontsize=8,ncol=3,loc='upper center',bbox_to_anchor=(.5,-.12))
ax=axes[1,1]
for key,label,col in [('old','v100','#3677af'),('new','v115','#ce4f45'),('old_actor','old actor / new writer','#329a68'),('hindsight_2033000','hindsight seed 0','#9871b5'),('grounded_2033000','grounded seed 0','#db9948')]:
 means=[np.mean([r['arms'][key]['returns'][ti] for r in rows if r['arms'][key]['returns'][ti] is not None]) for ti in range(6)];ax.plot(range(1,7),means,'o-',label=label,color=col)
ax.set_xlabel('Physical trial (memory persists)');ax.set_ylabel('Mean native return');ax.set_title('Adaptation trajectory');ax.legend(fontsize=8)
fig.suptitle('TTT degradation rescue: fixed-endpoint fresh confirmation',fontsize=16)
for ext in ['png','pdf','svg']:fig.savefig(OUT/f'rescue_summary_v2.{ext}',dpi=170)
plt.close(fig)
print('PLOTS_COMPLETE')
