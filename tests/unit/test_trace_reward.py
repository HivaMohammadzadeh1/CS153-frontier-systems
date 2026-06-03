from learning_memory_os.memory.trace import reward_weight


def test_reward_weight_upsamples_good_drops_bad():
    assert reward_weight(None) == 1     # unlabeled -> neutral keep
    assert reward_weight(-1.0) == 0     # negative outcome -> dropped
    assert reward_weight(0.0) == 0      # no gain -> dropped
    assert reward_weight(0.3) == 1
    assert reward_weight(0.7) == 2
    assert reward_weight(0.95) == 3
