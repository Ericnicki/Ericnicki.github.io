import ast,logging
from types import SimpleNamespace
import numpy as np
import torch
from common import O,write
from official import CONFIG,run_one

def official_sequence(cfg,friction):
    path=CONFIG.parent/'env/modules/tasks/push_cube.py'
    tree=ast.parse(path.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='PushCube')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='get_vel_sequence')
    ns=dict(np=np,logger=logging.getLogger('official_sequence'))
    exec(compile(ast.Module(body=[method],type_ignores=[]),str(path),'exec'),ns)
    table=SimpleNamespace(sampled_vels_m_per_s=np.array(cfg.task.sampled_vels_m_per_s),
        sampled_sliding_frictions=np.array(cfg.task.sampled_sliding_frictions),
        sliding_friction_min=float(cfg.task.sliding_friction_min),
        sliding_friction_max=float(cfg.task.sliding_friction_max),trial_max_step=4)
    return ns['get_vel_sequence'](table,friction).tolist()

def calibrate(env,acfg,cfg,spec,start,budget,tp):
    folder=tp/'calibration';folder.mkdir();rows=[]
    def attempt(speed):
        assert len(rows)<24 and budget['near_starts']<24000
        budget['near_starts']+=1
        budget['entries'].append(dict(kind='calibration',task=spec['task'],attempt=len(rows),speed=float(speed)))
        write(O/'budget.json',budget)
        result=run_one(env,acfg,start,[float(speed)]*6)
        result["speed"]=float(speed)
        torch.save(result,folder/f'attempt{len(rows):02d}.pt')
        row={k:v for k,v in result.items() if k not in ['events','records','metadata']}
        row['attempt']=len(rows)
        if row['status']=='completed':
            xy=np.array(row['stop_xy']);row['XY_error_m']=float(np.max(abs(xy-[.1,.15])))
            row['signed_Y_error_m']=float(xy[1]-.15)
            assert row['official_success']==(row['terminal_return']>0)
        rows.append(row);write(folder/'attempts.json',rows);return row
    def qualified(row):
        return row.get('status')=='completed' and not row.get('official_success') and .05<=row['XY_error_m']<=.06 and abs(row['signed_Y_error_m'])>=abs(row['stop_xy'][0]-.1)
    for speed in official_sequence(cfg,spec['friction']):attempt(speed)
    good=[row for row in rows if row.get('official_success')]
    if good and not any(qualified(row) for row in rows):
        succ=min(good,key=lambda row:row['XY_error_m'])
        side=1 if succ['speed']<=.341 else -1;end=.435 if side==1 else .247
        bads=[row for row in rows if row.get('status')=='completed' and not row['official_success'] and (row['speed']-succ['speed'])*side>0 and row['signed_Y_error_m']*side>0]
        if not bads and not any(abs(row['speed']-end)<1e-12 for row in rows):
            row=attempt(end)
            if row.get('status')=='completed' and not row['official_success'] and row['signed_Y_error_m']*side>0:bads=[row]
        if bads:
            bad=min(bads,key=lambda row:abs(row['speed']-succ['speed']))
            while len(rows)<24 and not any(qualified(row) for row in rows):
                row=attempt((succ['speed']+bad['speed'])/2)
                if row.get('status')!='completed':break
                if row['official_success']:succ=row
                elif row['signed_Y_error_m']*side>0:bad=row
                else:break
    candidates=[row for row in rows if qualified(row)]
    selected=min(candidates,key=lambda row:row['XY_error_m']) if candidates else None
    write(folder/'completion.json',dict(qualified=selected is not None,selected=selected,attempts=len(rows)))
    print('CALIBRATION',spec['task'],len(rows),selected,flush=True)
    if selected is None:raise RuntimeError(f"Task {spec['task']} did not qualify within calibration cap; preserve and stop")
    return selected
