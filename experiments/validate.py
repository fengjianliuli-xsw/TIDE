import math
import torch
from run_experiment import Net, losses, proposal, vec

torch.manual_seed(12)
for kind in ['adam','sgdm']:
    p=torch.nn.Parameter(torch.randn(11))
    opt=torch.optim.Adam([p],lr=.001) if kind=='adam' else torch.optim.SGD([p],lr=.001,momentum=.9)
    state={'t':0,'m':torch.zeros_like(p),'v':torch.zeros_like(p)}
    for _ in range(5):
        g=torch.randn_like(p);old=p.detach().clone()
        d,state=proposal(g,state,kind,.001)
        p.grad=g.clone();opt.step()
        torch.testing.assert_close(p.detach(),old+d,rtol=1e-12,atol=1e-14)
    print(kind,'proposal matches torch over five sequential updates')

class Exact(torch.nn.Module):
    def forward(self,x): return torch.sin(3*math.pi*x)+x
p,b=losses(Exact(),torch.linspace(0,1,64)[:,None],3,False)
assert p<1e-24 and b<1e-24
print('Manufactured solution satisfies implemented PDE and boundary')

net=Net();x=torch.linspace(0,1,10)[1:-1,None]
p,b=losses(net,x,1)
g=torch.autograd.grad(p,tuple(net.parameters()),allow_unused=True)
flat=torch.cat([(torch.zeros_like(q) if z is None else z).flatten() for z,q in zip(g,net.parameters())])
from run_experiment import put
v=vec(net);d=torch.randn_like(v);d/=d.norm();eps=1e-5
put(net,v+eps*d);a=losses(net,x,1,False)[0].item()
put(net,v-eps*d);c=losses(net,x,1,False)[0].item()
assert abs((a-c)/(2*eps)-float(flat@d))<1e-6
print('PDE parameter gradient matches centered finite difference')
