import numpy as np


def test_per_label_thresholds_can_differ_and_use_validation_arrays():
    from training.calibrate_thresholds import calibrate_label_thresholds
    probs=np.array([[.9,.4],[.8,.45],[.2,.8],[.1,.7]])
    truth=np.array([[1,0],[1,0],[0,1],[0,1]])
    assert calibrate_label_thresholds(probs,truth,candidates=[.3,.5,.75])==[0.75,0.5]
