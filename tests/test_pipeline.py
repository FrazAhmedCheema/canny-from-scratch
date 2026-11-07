import json
import subprocess
import sys
import numpy as np
import pytest
from edgevision import EdgeConfig, detect_edges
from edgevision.synthetic import make_scenes, save_scenes
from edgevision.metrics import edge_scores


@pytest.mark.parametrize("shape", [(1,1),(1,8),(2,2),(3,3),(16,20)])
def test_blank_and_small_images(shape):
    a=np.full(shape,.5);before=a.copy();r=detect_edges(a)
    assert r.edges.shape==shape and r.edges.dtype==bool and not r.edges.any()
    np.testing.assert_array_equal(a,before)
    assert all(t>=0 for t in r.timings.values())
    assert all(np.isfinite(getattr(r,k)).all() for k in ['gray','smoothed','gx','gy','magnitude','orientation','suppressed'])


def test_step_integration_and_repeatability():
    scene=make_scenes(64)[0];a=scene.image.copy();r=detect_edges(a);r2=detect_edges(a)
    valid=np.zeros((64,64),bool);valid[5:-5,5:-5]=1
    score=edge_scores(r.edges,scene.truth,tolerance=1,valid=valid)
    assert score.f1>.95
    assert (r.edges & valid).sum()==54  # one surviving edge per interior row
    np.testing.assert_array_equal(r.edges,r2.edges)
    np.testing.assert_array_equal(a,scene.image)


def test_threshold_monotonicity_for_fixed_image():
    a=make_scenes(64)[-1].image
    loose=detect_edges(a,EdgeConfig(low=.02,high=.05,threshold_mode='absolute'))
    strict=detect_edges(a,EdgeConfig(low=.08,high=.2,threshold_mode='absolute'))
    assert np.all(~strict.edges | loose.edges)
    assert strict.edges.sum()<loose.edges.sum()


def test_seeded_scenes_and_rgb():
    a,b=make_scenes(64,7),make_scenes(64,7)
    for x,y in zip(a,b):np.testing.assert_array_equal(x.image,y.image)
    assert a[1].image.shape==(64,64,3)
    assert detect_edges(a[1].image).edges.any()


@pytest.mark.parametrize("kwargs", [{'sigma':0},{'kernel_size':4},{'low':.8,'high':.2},{'boundary':'wrap'},{'connectivity':4.0},{'threshold_mode':'x'}])
def test_config_invalid(kwargs):
    with pytest.raises(ValueError):EdgeConfig(**kwargs)


def test_pipeline_bad_config_and_image():
    with pytest.raises(ValueError):detect_edges(np.zeros((3,3)),{})
    with pytest.raises(ValueError):detect_edges(np.full((3,3),np.nan))
    with pytest.raises(ValueError):make_scenes(16)


def test_cli_end_to_end(tmp_path):
    root=tmp_path/'samples'
    proc=subprocess.run([sys.executable,'-m','edgevision','samples','--size','32','--output',str(root)],capture_output=True,text=True)
    assert proc.returncode==0,proc.stderr
    assert len(list(root.glob('*.png')))==12
    out=tmp_path/'demo'
    proc=subprocess.run([sys.executable,'-m','edgevision','detect',str(root/'disk.png'),'--output',str(out)],capture_output=True,text=True)
    assert proc.returncode==0,proc.stderr
    assert (out/'stages.png').stat().st_size>1000 and (out/'edges.png').is_file()
    assert 'hysteresis' in json.loads((out/'timings.json').read_text())
    bad=subprocess.run([sys.executable,'-m','edgevision','detect',str(root/'disk.png'),'--sigma','0'],capture_output=True,text=True)
    assert bad.returncode==2 and 'sigma' in bad.stderr


def test_evaluation_small(tmp_path):
    from edgevision.evaluation import evaluate
    report=evaluate(tmp_path,size=32,repeats=1)
    assert len(report['benchmark'])==6
    assert all(b['total_median_ms']>0 for b in report['benchmark'])
    assert len((tmp_path/'metrics.csv').read_text().splitlines())==37
    assert len(list(tmp_path.glob('*_stages.png')))==6
    assert (tmp_path/'threshold_comparison.png').is_file()
    import matplotlib.pyplot as plt
    assert not plt.get_fignums()


def test_optional_scipy_reference():
    pytest.importorskip('scipy',reason='optional trusted-reference extra')
    from edgevision.reference import compare_reference
    assert compare_reference()['status']=='passed'
