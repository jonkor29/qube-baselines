#this testfile was written by an LLM

# test_discrepancy_calculator.py
import numpy as np
import pytest
from SimOpt import D # Assuming D is in this file

# Fixed constants from the D function for manual calculation in tests
WL1_CONST = 0.5
WL2_CONST = 1.0
W_CONST = np.array([1, 1, 0.5, 0.5])

def test_zero_discrepancy():
    """Test that D is 0 when trajectories are identical."""
    T = 5
    traj_a = np.random.rand(T, 1, 4)
    traj_b = np.copy(traj_a)
    assert D(traj_a, traj_b) == pytest.approx(0.0)

def test_single_timestep_known_values():
    """Test D with a single timestep and manually calculated expected value."""
    traj_xi = np.array([[[1.0, 2.0, 3.0, 4.0]]]) # Shape (1, 1, 4)
    traj_real = np.array([[[0.0, 0.0, 0.0, 0.0]]]) # Shape (1, 1, 4)

    diff = traj_xi - traj_real  # [[[1.0, 2.0, 3.0, 4.0]]]
    weighted_diff_vector = W_CONST * diff[0, 0, :] # [1*1, 1*2, 0.5*3, 0.5*4] = [1.0, 2.0, 1.5, 2.0]

    expected_l1_norm = np.sum(np.abs(weighted_diff_vector)) # 1+2+1.5+2 = 6.5
    expected_l2_norm_sq = np.sum(weighted_diff_vector**2)  # 1^2+2^2+1.5^2+2^2 = 1+4+2.25+4 = 11.25

    expected_D = WL1_CONST * expected_l1_norm + WL2_CONST * expected_l2_norm_sq
    # expected_D = 0.5 * 6.5 + 1.0 * 11.25 = 3.25 + 11.25 = 14.5
    assert expected_D == pytest.approx(14.5)
    assert D(traj_xi, traj_real) == pytest.approx(expected_D)

def test_multiple_timesteps_known_values():
    """Test D with multiple timesteps and manually calculated expected value."""
    traj_xi = np.array([
        [[1.0, 2.0, 3.0, 4.0]],  # Timestep 0
        [[2.0, 1.0, 4.0, 2.0]]   # Timestep 1
    ]) # Shape (2, 1, 4)
    traj_real = np.array([
        [[0.0, 0.0, 0.0, 0.0]],  # Timestep 0
        [[1.0, 0.0, 2.0, 0.0]]   # Timestep 1
    ]) # Shape (2, 1, 4)

    # Timestep 0
    diff_0 = traj_xi[0,0,:] - traj_real[0,0,:] # [1, 2, 3, 4]
    weighted_diff_0 = W_CONST * diff_0        # [1, 2, 1.5, 2]
    l1_norm_0 = np.sum(np.abs(weighted_diff_0))      # 6.5
    l2_norm_sq_0 = np.sum(weighted_diff_0**2)        # 11.25

    # Timestep 1
    diff_1 = traj_xi[1,0,:] - traj_real[1,0,:] # [1, 1, 2, 2]
    weighted_diff_1 = W_CONST * diff_1        # [1, 1, 1, 1]
    l1_norm_1 = np.sum(np.abs(weighted_diff_1))      # 4.0
    l2_norm_sq_1 = np.sum(weighted_diff_1**2)        # 4.0

    total_l1_sum = l1_norm_0 + l1_norm_1             # 6.5 + 4.0 = 10.5
    total_l2_sum_sq = l2_norm_sq_0 + l2_norm_sq_1    # 11.25 + 4.0 = 15.25

    expected_D = WL1_CONST * total_l1_sum + WL2_CONST * total_l2_sum_sq
    # expected_D = 0.5 * 10.5 + 1.0 * 15.25 = 5.25 + 15.25 = 20.5
    assert expected_D == pytest.approx(20.5)

    assert D(traj_xi, traj_real) == pytest.approx(expected_D)

def test_trajectory_with_negative_differences():
    """Test D handles negative differences correctly due to abs and squaring."""
    traj_xi = np.array([[[-1.0, -2.0, 1.0, 1.0]]])
    traj_real = np.array([[[0.0, 0.0, 3.0, 3.0]]])

    # diff = [[[-1, -2, -2, -2]]]
    # weighted_diff_vector = W_CONST * diff[0,0,:] = [1*(-1), 1*(-2), 0.5*(-2), 0.5*(-2)]
    #                                            = [-1, -2, -1, -1]
    weighted_diff_vector = np.array([-1.0, -2.0, -1.0, -1.0])

    expected_l1_norm = np.sum(np.abs(weighted_diff_vector)) # 1+2+1+1 = 5.0
    expected_l2_norm_sq = np.sum(weighted_diff_vector**2)  # (-1)^2+(-2)^2+(-1)^2+(-1)^2 = 1+4+1+1 = 7.0

    expected_D = WL1_CONST * expected_l1_norm + WL2_CONST * expected_l2_norm_sq
    # expected_D = 0.5 * 5.0 + 1.0 * 7.0 = 2.5 + 7.0 = 9.5

    assert D(traj_xi, traj_real) == pytest.approx(expected_D)

@pytest.mark.parametrize("traj_xi_data, traj_real_data, expected_d_val", [
    # Case 1: Single timestep, already tested in test_single_timestep_known_values
    (np.array([[[1.0, 2.0, 3.0, 4.0]]]),
     np.array([[[0.0, 0.0, 0.0, 0.0]]]),
     14.5),
    # Case 2: Multiple timesteps, already tested in test_multiple_timesteps_known_values
    (np.array([[[1.0, 2.0, 3.0, 4.0]], [[2.0, 1.0, 4.0, 2.0]]]),
     np.array([[[0.0, 0.0, 0.0, 0.0]], [[1.0, 0.0, 2.0, 0.0]]]),
     20.5),
    # Case 3: Another multiple timestep example
    (np.array([[[0.5, 0.5, 0.5, 0.5]], [[1.5, 1.5, 1.5, 1.5]]]),
     np.array([[[0.0, 0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0, 0.0]]]),
     # T0: diff=[.5,.5,.5,.5], wd=[.5,.5,.25,.25], l1=1.5, l2sq=0.25+0.25+0.0625+0.0625 = 0.625
     # T1: diff=[1.5,1.5,1.5,1.5], wd=[1.5,1.5,.75,.75], l1=4.5, l2sq=2.25+2.25+0.5625+0.5625 = 5.625
     # Total l1_sum = 1.5 + 4.5 = 6.0
     # Total l2_sum_sq = 0.625 + 5.625 = 6.25
     # D = 0.5 * 6.0 + 1.0 * 6.25 = 3.0 + 6.25 = 9.25
     9.25),
])
def test_D_function_parameterized(traj_xi_data, traj_real_data, expected_d_val):
    """Parameterized test for various known input/output pairs."""
    assert D(np.array(traj_xi_data), np.array(traj_real_data)) == pytest.approx(expected_d_val)

def test_mismatched_trajectory_lengths_clips_correctly():
    """Test that D clips to the shorter trajectory and calculates correctly."""
    # traj_xi is longer (3 timesteps)
    traj_xi = np.array([
        [[1.0, 2.0, 3.0, 4.0]],  # Timestep 0
        [[2.0, 1.0, 4.0, 2.0]],  # Timestep 1
        [[9.0, 9.0, 9.0, 9.0]]   # Timestep 2 (should be ignored)
    ])
    # traj_real is shorter (2 timesteps)
    traj_real = np.array([
        [[0.0, 0.0, 0.0, 0.0]],  # Timestep 0
        [[1.0, 0.0, 2.0, 0.0]]   # Timestep 1
    ])

    # Expected D should be based only on the first 2 timesteps (like test_multiple_timesteps_known_values)
    # Manually calculate based on the first 2 timesteps
    total_l1_sum = 0
    total_l2_sum_sq = 0
    # T0
    wd_0 = W_CONST * (traj_xi[0,0,:] - traj_real[0,0,:]) # [1,2,1.5,2]
    total_l1_sum += np.sum(np.abs(wd_0))      # 6.5
    total_l2_sum_sq += np.sum(wd_0**2)        # 11.25
    # T1
    wd_1 = W_CONST * (traj_xi[1,0,:] - traj_real[1,0,:]) # [1,1,1,1]
    total_l1_sum += np.sum(np.abs(wd_1))      # 4.0
    total_l2_sum_sq += np.sum(wd_1**2)        # 4.0
    
    expected_D_clipped = WL1_CONST * total_l1_sum + WL2_CONST * total_l2_sum_sq # 0.5 * 10.5 + 1.0 * 15.25 = 20.5

    assert D(traj_xi, traj_real) == pytest.approx(expected_D_clipped)

    # Test with traj_real being longer
    traj_xi_short = traj_xi[:2,:,:] # Use the first two from previous traj_xi
    traj_real_long = np.array([
        [[0.0, 0.0, 0.0, 0.0]],
        [[1.0, 0.0, 2.0, 0.0]],
        [[8.0, 8.0, 8.0, 8.0]]
    ])
    assert D(traj_xi_short, traj_real_long) == pytest.approx(expected_D_clipped)

def test_empty_trajectories():
    """Test D returns 0.0 if one or both trajectories are empty."""
    T = 5
    traj_full = np.random.rand(T, 1, 4)
    traj_empty_xi = np.empty((0, 1, 4))
    traj_empty_real = np.empty((0, 1, 4))

    assert D(traj_empty_xi, traj_full) == pytest.approx(0.0)
    assert D(traj_full, traj_empty_real) == pytest.approx(0.0)
    assert D(traj_empty_xi, traj_empty_real) == pytest.approx(0.0)