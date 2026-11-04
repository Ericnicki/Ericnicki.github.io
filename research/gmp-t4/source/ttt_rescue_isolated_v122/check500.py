from compare import *
key='hindsight_2033000';a=torch.load(new/key/'step500.pt',map_location='cpu',weights_only=False);b=torch.load(old/key/'step500.pt',map_location='cpu',weights_only=False)
print({g:dict(exact=all(torch.equal(v,b[g][k]) for k,v in a[g].items()),max_abs_difference=max(float((v-b[g][k]).abs().max()) for k,v in a[g].items())) for g in ['model','actor']})
