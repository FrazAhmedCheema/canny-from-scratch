import numpy as np
import pytest
from edgevision.edges import non_maximum_suppression as nms, double_threshold, hysteresis


@pytest.mark.parametrize("angle,transpose", [(0,False), (np.pi/2,True)])
def test_nms_thins_plateau(angle,transpose):
    m=np.tile([0.,1,3,3,1,0,0],(7,1))
    if transpose:m=m.T
    out=nms(m,np.full(m.shape,angle))
    expected=np.zeros_like(m);expected[1:-1,2]=3
    if transpose:expected=expected.T
    np.testing.assert_array_equal(out,expected)


def test_nms_bilinear_not_direction_bin():
    m=np.zeros((5,5));m[2,2]=2;m[3,3]=3
    assert nms(m,np.full(m.shape,np.pi/4))[2,2]==2
    m[3,3]=5
    assert nms(m,np.full(m.shape,np.pi/4))[2,2]==0


def test_nms_contrast_reversal_and_boundaries():
    rng=np.random.default_rng(5);m=rng.uniform(size=(9,11));t=rng.uniform(-np.pi,np.pi,m.shape)
    np.testing.assert_allclose(nms(m,t),nms(m,t+np.pi),atol=1e-14)
    out=nms(m,t);assert not out[0].any() and not out[-1].any() and not out[:,0].any() and not out[:,-1].any()
    assert not nms(np.ones((2,8)),np.zeros((2,8))).any()


def test_nms_invalid():
    with pytest.raises(ValueError):nms([[-1]],[[0]])
    with pytest.raises(ValueError):nms([[1]],[[0,1]])
    with pytest.raises(ValueError):nms([[1]],[[np.inf]])


def test_threshold_endpoints_and_blank():
    m=np.array([[0,.1,.2,.3]])
    np.testing.assert_array_equal(double_threshold(m,.1,.2,mode="absolute"),[[0,1,2,2]])
    np.testing.assert_array_equal(double_threshold(m,1/3,2/3),[[0,1,2,2]])
    assert not double_threshold(np.zeros((3,3)),0,0).any()
    np.testing.assert_array_equal(double_threshold([[0,1]],0,0),[[0,2]])


@pytest.mark.parametrize("low,high,mode", [(-1,.2,"absolute"),(.4,.2,"relative"),(0,2,"relative"),(np.nan,1,"absolute"),(0,1,"bad"),(False,1,"relative")])
def test_threshold_invalid(low,high,mode):
    with pytest.raises(ValueError):double_threshold([[1]],low,high,mode=mode)


def test_hysteresis_long_chain_isolated_and_no_mutation():
    labels=np.zeros((5,1200),np.uint8);labels[1,1:1199]=1;labels[1,1]=2;labels[3,50]=1
    before=labels.copy();out=hysteresis(labels)
    assert out[1].sum()==1198 and not out[3,50]
    np.testing.assert_array_equal(labels,before)


def test_hysteresis_diagonal_and_no_wrap():
    a=np.eye(5,dtype=np.uint8);a[0,0]=2
    assert hysteresis(a,8).sum()==5 and hysteresis(a,4).sum()==1
    a=np.zeros((2,4),np.uint8);a[0,0]=2;a[0,-1]=1
    assert hysteresis(a).sum()==1
    assert not hysteresis(np.ones((2,2),np.uint8)).any()
    assert hysteresis([[2]]).item()


@pytest.mark.parametrize("a,conn", [([[3]],8),([[1.]],8),([[1]],6),([[1]],True),([[1]],4.0)])
def test_hysteresis_invalid(a,conn):
    with pytest.raises(ValueError):hysteresis(a,conn)
