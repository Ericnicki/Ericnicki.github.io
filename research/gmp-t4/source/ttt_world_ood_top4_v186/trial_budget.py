import copy
CAP=480
def claim_start(budget,identity):
 assert budget['trial_starts']<CAP
 budget['trial_starts']+=1;budget['entries'].append(copy.deepcopy(identity))
