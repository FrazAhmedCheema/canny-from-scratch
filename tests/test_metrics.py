import numpy as np
import pytest
from edgevision.metrics import edge_scores


def test_exact_counts():
    p=np.array([[1,1,0,0]],bool);g=np.array([[0,1,1,0]],bool)
    s=edge_scores(p,g)
    assert (s.true_positive,s.false_positive,s.false_negative)==(1,1,1)
    assert s.precision==s.recall==s.f1==.5


def test_empty_conventions_and_valid_mask():
    z=np.zeros((3,3),bool);o=z.copy();o[1,1]=1
    assert edge_scores(z,z).f1==1
    assert edge_scores(o,z).f1==0 and edge_scores(z,o).f1==0
    v=np.ones((3,3),bool);v[1,1]=0
    assert edge_scores(o,z,valid=v).f1==1


def test_tolerance_does_not_reward_thick_edges():
    p=np.ones((1,3),bool);g=np.array([[0,1,0]],bool)
    s=edge_scores(p,g,tolerance=1)
    assert s.true_positive==1 and s.false_positive==2 and s.recall==1


def test_matching_requires_augmenting_path():
    # Pred (0,1) can match either truth; pred (1,0) can only match (0,0).
    # Nearest-first greedy gets 1; an augmenting path reassigns the first pred.
    p=np.array([[0,1,0],[1,0,0]],bool);g=np.array([[1,0,1],[0,0,0]],bool)
    assert edge_scores(p,g,tolerance=1).true_positive==2


def brute_matching(p,g,r):
    pp=np.argwhere(p);gg=np.argwhere(g)
    adjacency=[[j for j,b in enumerate(gg) if np.sum((a-b)**2)<=r*r] for a in pp]
    def search(i,used):
        if i==len(pp):return 0
        best=search(i+1,used)
        for j in adjacency[i]:
            if j not in used:best=max(best,1+search(i+1,used|{j}))
        return best
    return search(0,set())


def test_sparse_matching_against_exhaustive_oracle():
    rng=np.random.default_rng(19)
    for _ in range(35):
        p=rng.random((2,3))>.55;g=rng.random((2,3))>.55
        for r in (1,2):
            assert edge_scores(p,g,tolerance=r).true_positive==brute_matching(p,g,r)


@pytest.mark.parametrize("tol", [-1,6,1.5,True])
def test_metric_invalid(tol):
    with pytest.raises(ValueError):edge_scores(np.zeros((2,2),bool),np.zeros((2,2),bool),tolerance=tol)


def test_metric_shape_type():
    with pytest.raises(ValueError):edge_scores([[1]],[[1]])
    with pytest.raises(ValueError):edge_scores(np.zeros((1,2),bool),np.zeros((2,1),bool))
