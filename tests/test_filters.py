import numpy as np
import pytest
from edgevision.filters import (convolve2d, gaussian_kernel, smooth_image,
    sobel_gradients, magnitude_orientation, to_grayscale)


def scalar_convolution(a, k, mode, cval=0):
    """Independent loop contraction oracle; padding convention tested separately."""
    ph, pw = k.shape[0]//2, k.shape[1]//2
    kw = {"constant_values": cval} if mode == "constant" else {}
    p = np.pad(a, ((ph, ph), (pw, pw)), mode=mode, **kw)
    out = np.zeros_like(a, dtype=float)
    for y in range(a.shape[0]):
        for x in range(a.shape[1]):
            for j in range(k.shape[0]):
                for i in range(k.shape[1]):
                    out[y, x] += p[y+j, x+i]*k[-1-j, -1-i]
    return out


@pytest.mark.parametrize("shape", [(1,1), (1,5), (5,1), (2,2), (7,9)])
@pytest.mark.parametrize("mode", ["reflect", "edge", "constant"])
def test_convolution_matches_scalar_oracle(shape, mode):
    rng = np.random.default_rng(4)
    a, k = rng.normal(size=shape), rng.normal(size=(5,3))
    before = a.copy()
    np.testing.assert_allclose(convolve2d(a,k,boundary=mode,cval=.2), scalar_convolution(a,k,mode,.2), atol=1e-13)
    np.testing.assert_array_equal(a,before)


def test_true_convolution_not_correlation():
    a = np.zeros((5,5)); a[2,2] = 1
    k = np.arange(9).reshape(3,3)
    np.testing.assert_array_equal(convolve2d(a,k,boundary="constant")[1:4,1:4], k)


@pytest.mark.parametrize("mode,expected", [("reflect", [5,6,7]), ("edge", [4,6,8]), ("constant", [3,6,5])])
def test_boundary_semantics_explicit(mode, expected):
    # [1,2,3] padded by reflect: [2,1,2,3,2]; edge: [1,1,2,3,3].
    np.testing.assert_array_equal(convolve2d([[1,2,3]], [[1,1,1]], boundary=mode), [expected])


@pytest.mark.parametrize("image,kernel", [(np.empty((0,2)),[[1]]), ([[np.nan]],[[1]]), ([[1]],[[np.inf]]), ([[1]],np.ones((2,3))), ([[1j]],[[1]]), ([1,2],[[1]])])
def test_bad_convolution(image,kernel):
    with pytest.raises(ValueError):convolve2d(image,kernel)


def test_bad_boundary_and_overflow():
    with pytest.raises(ValueError):convolve2d([[1]],[[1]],boundary="wrap")
    with pytest.raises(ValueError):convolve2d([[1]],[[1]],cval=np.nan)
    with pytest.raises(ValueError):convolve2d([[1e308]],[[1e308]])


@pytest.mark.parametrize("size,sigma", [(1,1), (3,.5), (7,1.2), (11,3), (3,1e-300)])
def test_gaussian_properties(size,sigma):
    k = gaussian_kernel(size,sigma)
    assert k.shape == (size,size) and np.isfinite(k).all() and (k>=0).all()
    np.testing.assert_allclose(k.sum(),1,atol=1e-15)
    np.testing.assert_allclose(k,k[::-1,::-1],atol=0)
    np.testing.assert_allclose(smooth_image(np.full((5,8),.7),size,sigma),.7,atol=1e-15)


@pytest.mark.parametrize("size,sigma", [(2,1), (0,1), (3,0), (3,-1), (3,np.nan), (3,np.inf), (3.5,1), (True,1)])
def test_bad_gaussian(size,sigma):
    with pytest.raises(ValueError):gaussian_kernel(size,sigma)


def test_sobel_sign_scale_and_orientation():
    y,x = np.mgrid[:8,:9]
    gx,gy = sobel_gradients(2*x+3*y)
    np.testing.assert_allclose(gx[1:-1,1:-1],2,atol=1e-14)
    np.testing.assert_allclose(gy[1:-1,1:-1],3,atol=1e-14)
    m,t = magnitude_orientation(gx,gy)
    np.testing.assert_allclose(m[1:-1,1:-1],np.sqrt(13))
    np.testing.assert_allclose(t[1:-1,1:-1],np.arctan2(3,2))


def test_zero_and_large_gradients():
    m,t = magnitude_orientation([[0]],[[0]])
    assert m.item()==t.item()==0
    m,_ = magnitude_orientation([[1e200]],[[1e200]])
    assert np.isfinite(m).all()
    with pytest.raises(ValueError):magnitude_orientation([[1e308]],[[1e308*1.7]])
    with pytest.raises(ValueError):magnitude_orientation([[1]],[[1,2]])


def test_grayscale_contract():
    rgb = np.array([[[255,0,0],[0,255,0],[0,0,255]]],dtype=np.uint8)
    np.testing.assert_allclose(to_grayscale(rgb),[[.2126,.7152,.0722]])
    np.testing.assert_array_equal(to_grayscale(np.array([[0,65535]],np.uint16)),[[0,1]])
    a=np.array([[.3]]); b=to_grayscale(a); b[0,0]=0; assert a.item()==.3


@pytest.mark.parametrize("a", [np.ones((2,2,4)),np.zeros((0,2)),[[2.]],[[np.nan]],[[1+2j]],np.array([[1]],dtype=int)])
def test_bad_grayscale(a):
    with pytest.raises(ValueError):to_grayscale(a)
