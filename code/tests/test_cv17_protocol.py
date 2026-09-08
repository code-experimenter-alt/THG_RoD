import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score
from tools.run_cv17_release import metrics, from_confusion, cluster_health
from src.losses import kd_loss, hakd_y_loss


def test_metrics_match_independent_sklearn():
    y=np.array([0,0,0,1,1,2,2,2,2])
    pred=np.array([0,1,2,1,2,0,2,2,2])
    logits=np.eye(3)[pred]
    m=metrics(y,logits)
    assert abs(m['bal_acc']-balanced_accuracy_score(y,pred))<1e-12
    assert abs(m['macro_f1']-f1_score(y,pred,average='macro'))<1e-12
    assert sum(map(sum,m['confusion']))==len(y)


def test_vectorized_health_matches_scalar():
    c=np.array([[[5,1,0],[1,3,2],[0,1,8]],[[0,2,1],[1,0,2],[3,2,0]]])
    b,f,d=from_confusion(c)
    for i in range(len(c)):
        ys=[];ps=[]
        for a in range(3):
            for z in range(3):
                ys.extend([a]*c[i,a,z]);ps.extend([z]*c[i,a,z])
        m=metrics(np.asarray(ys),np.eye(3)[ps])
        assert np.allclose([b[i],f[i],d[i]],[m['bal_acc'],m['macro_f1'],1-m['maj_pred']])


def test_cluster_bootstrap_is_reproducible_and_counts_speakers():
    y=np.repeat(np.arange(3),12)
    pred=y.copy();pred[::4]=(pred[::4]+1)%3
    groups=np.repeat(np.arange(12),3)
    a=cluster_health(y,np.eye(3)[pred],groups,seed=123,draws=2000)
    b=cluster_health(y,np.eye(3)[pred],groups,seed=123,draws=2000)
    assert a==b and a['clusters']==12 and a['se']>0
    assert 0<=a['interval95'][0]<a['interval95'][1]<=1


def test_uniform_hakd_is_exact_kd_loss_and_gradient():
    torch.manual_seed(70)
    x=torch.randn(19,3,dtype=torch.float64,requires_grad=True)
    t=torch.randn(19,3,dtype=torch.float64)
    y=torch.arange(19)%3
    kd=kd_loss(x,t,y,.7,2.)
    uniform,_=hakd_y_loss(x,t,y,.7,2.,torch.tensor([.1,.5,.9],dtype=x.dtype),'uniform')
    assert torch.allclose(kd,uniform,atol=1e-14,rtol=0)
    a=torch.autograd.grad(kd,x,retain_graph=True)[0]
    b=torch.autograd.grad(uniform,x)[0]
    assert torch.allclose(a,b,atol=1e-14,rtol=0)


def test_unit_feature_bounds_full_linear_ce_gradient():
    torch.manual_seed(0)
    x=torch.randn(30,768,dtype=torch.float64)
    x=x/x.norm(dim=1,keepdim=True)
    w=torch.randn(3,768,dtype=torch.float64,requires_grad=True)
    bias=torch.zeros(3,dtype=torch.float64,requires_grad=True)
    for i in range(len(x)):
        logits=(x[i]@w.T+bias+torch.tensor([1.,2.,3.],dtype=x.dtype)).unsqueeze(0)
        loss=torch.nn.functional.cross_entropy(logits,torch.tensor([i%3]))
        gw,gb=torch.autograd.grad(loss,[w,bias])
        assert torch.sqrt(gw.square().sum()+gb.square().sum())<=2+1e-12


def test_constant_class_health_equals_attenuated_kd():
    torch.manual_seed(31)
    s=torch.randn(12,3,dtype=torch.float64,requires_grad=True)
    t=torch.randn(12,3,dtype=torch.float64);y=torch.arange(12)%3
    loss,_=hakd_y_loss(s,t,y,.7,2.,torch.full((3,),.5,dtype=s.dtype),'instance')
    constant=kd_loss(s,t,y,.35,2.)
    assert torch.allclose(loss,constant,atol=1e-14,rtol=0)
    a=torch.autograd.grad(loss,s,retain_graph=True)[0]
    b=torch.autograd.grad(constant,s)[0]
    assert torch.allclose(a,b,atol=1e-14,rtol=0)


def test_independent_uncertainty_audit_matches_runner():
    from tools.audit_cv17_release import verify_uncertainty
    from src.metrics import compute_teacher_health, estimate_health_se
    from src.routing import certified_rc_tcrd_mode
    y=np.tile(np.arange(3),50)
    pred=y.copy();pred[::4]=(pred[::4]+1)%3
    logits=np.eye(3)[pred];groups=np.repeat(np.arange(30),5)
    m=metrics(y,logits);h=compute_teacher_health(m)['global_health']
    cluster=cluster_health(y,logits,groups,17)
    iid=estimate_health_se(m,seed=17)
    r=dict(health=dict(cluster=cluster,iid_se=iid),
           rc_route=certified_rc_tcrd_mode(h,cluster['se']),
           iid_rc_route=certified_rc_tcrd_mode(h,iid))
    verify_uncertainty(y,logits,groups,17,r)


def test_student_retains_exact_checkpoint_selection_trace():
    from src.train_student import train_student
    from src.models import LinearHead
    from torch.utils.data import DataLoader, TensorDataset
    torch.manual_seed(44)
    x=torch.randn(18,8);y=torch.arange(18)%3
    loader=DataLoader(TensorDataset(x,y),batch_size=6)
    model=LinearHead(8,3)
    info=train_student(model,loader,loader,torch.device('cpu'),3,3,.01,1e-4,'hard',.7,2.)
    trace=info['selection_trace']
    assert len(trace)==3
    best=max(trace,key=lambda v:v['macro_f1'])
    assert best['epoch']==info['best_epoch']
    assert best['macro_f1']==info['best_dev_macro_f1']
