import casadi as ca
import numpy as np
import timeit
import random

# ==============================================================================
# PROVIDED FUNCTIONS (Original and Helper) - quat_dcm, quat_dot, dcm_quat 유지
# ==============================================================================
def quat_dcm(q: ca.MX) -> ca.MX:
    # Ensure q is treated as [q1, q2, q3, q0] matching the original code's usage
    q1, q2, q3, q0 = q[0], q[1], q[2], q[3]

    # Pre-calculate squares and products
    q1q1 = q1**2
    q2q2 = q2**2
    q3q3 = q3**2
    q1q2 = q1 * q2
    q1q3 = q1 * q3
    q2q3 = q2 * q3
    q0q1 = q0 * q1
    q0q2 = q0 * q2
    q0q3 = q0 * q3

    # Construct DCM rows using CasADi's concatenation
    row1 = ca.hcat([1 - 2*(q2q2 + q3q3), 2*(q1q2 - q0q3),     2*(q1q3 + q0q2)])
    row2 = ca.hcat([2*(q1q2 + q0q3),     1 - 2*(q1q1 + q3q3), 2*(q2q3 - q0q1)])
    row3 = ca.hcat([2*(q1q3 - q0q2),     2*(q2q3 + q0q1),     1 - 2*(q1q1 + q2q2)])

    # Combine rows into the final DCM matrix
    return ca.vertcat(row1, row2, row3)


def quat_dot(q: ca.MX, w: ca.MX) -> ca.MX:
    q1, q2, q3, q0 = q[0], q[1], q[2], q[3]
    w1, w2, w3 = w[0], w[1], w[2]
    half = ca.MX(0.5)
    return ca.vertcat(
        half * ( q0 * w1 + q2 * w3 - q3 * w2),
        half * ( q0 * w2 + q3 * w1 - q1 * w3),
        half * ( q0 * w3 + q1 * w2 - q2 * w1),
        half * (-q1 * w1 - q2 * w2 - q3 * w3)
    )

# --- Original dcm_quat ---
def dcm_quat(DCM: ca.MX) -> ca.MX:
    r11 = DCM[0, 0]; r12 = DCM[0, 1]; r13 = DCM[0, 2]
    r21 = DCM[1, 0]; r22 = DCM[1, 1]; r23 = DCM[1, 2]
    r31 = DCM[2, 0]; r32 = DCM[2, 1]; r33 = DCM[2, 2]
    trace = r11 + r22 + r33
    eps = 1e-9

    # Case 1: Trace is largest
    w1 = 0.5 * ca.sqrt(ca.fmax(1 + trace, eps))
    x1 = (r32 - r23) / ca.fmax(4 * w1, eps) # Avoid division by zero
    y1 = (r13 - r31) / ca.fmax(4 * w1, eps)
    z1 = (r21 - r12) / ca.fmax(4 * w1, eps)

    # Case 2: R11 is largest
    x2 = 0.5 * ca.sqrt(ca.fmax(1 + r11 - r22 - r33, eps))
    w2 = (r32 - r23) / ca.fmax(4 * x2, eps)
    y2 = (r12 + r21) / ca.fmax(4 * x2, eps)
    z2 = (r13 + r31) / ca.fmax(4 * x2, eps)

    # Case 3: R22 is largest
    y3 = 0.5 * ca.sqrt(ca.fmax(1 - r11 + r22 - r33, eps))
    w3 = (r13 - r31) / ca.fmax(4 * y3, eps)
    x3 = (r12 + r21) / ca.fmax(4 * y3, eps)
    z3 = (r23 + r32) / ca.fmax(4 * y3, eps)

    # Case 4: R33 is largest
    z4 = 0.5 * ca.sqrt(ca.fmax(1 - r11 - r22 + r33, eps))
    w4 = (r21 - r12) / ca.fmax(4 * z4, eps)
    x4 = (r13 + r31) / ca.fmax(4 * z4, eps)
    y4 = (r23 + r32) / ca.fmax(4 * z4, eps)

    quat = ca.MX.zeros(4)
    cond1 = trace > 0
    cond2 = ca.logic_and(ca.logic_not(cond1), ca.logic_and(r11 > r22, r11 > r33))
    cond3 = ca.logic_and(ca.logic_not(ca.logic_or(cond1, cond2)), r22 > r33)

    quat[0] = ca.if_else(cond1, x1, ca.if_else(cond2, x2, ca.if_else(cond3, x3, x4)))
    quat[1] = ca.if_else(cond1, y1, ca.if_else(cond2, y2, ca.if_else(cond3, y3, y4)))
    quat[2] = ca.if_else(cond1, z1, ca.if_else(cond2, z2, ca.if_else(cond3, z3, z4)))
    quat[3] = ca.if_else(cond1, w1, ca.if_else(cond2, w2, ca.if_else(cond3, w3, w4)))
    return quat

# --- Optimized dcm_quat_opt (Corrected) ---
def dcm_quat_opt(DCM: ca.MX) -> ca.MX:
    """
    Numerically stable and potentially faster DCM to quaternion conversion.
    Input: DCM (3x3 ca.MX)
    Output: Quaternion [q1, q2, q3, q0] (4x1 ca.MX)
    """
    r11 = DCM[0, 0]; r12 = DCM[0, 1]; r13 = DCM[0, 2]
    r21 = DCM[1, 0]; r22 = DCM[1, 1]; r23 = DCM[1, 2]
    r31 = DCM[2, 0]; r32 = DCM[2, 1]; r33 = DCM[2, 2]

    tr = r11 + r22 + r33
    eps = 1e-9 # Small epsilon for sqrt/division robustness

    # --- Define helper functions that return MX vectors ---
    def case1():
        S = ca.sqrt(ca.fmax(tr + 1.0, eps)) * 2
        safe_S = ca.fmax(S, eps) # Avoid division by zero/small numbers
        q0 = 0.25 * S
        q1 = (r32 - r23) / safe_S
        q2 = (r13 - r31) / safe_S
        q3 = (r21 - r12) / safe_S
        return ca.vertcat(q1, q2, q3, q0) # Return MX vector

    def case2():
        S = ca.sqrt(ca.fmax(1.0 + r11 - r22 - r33, eps)) * 2
        safe_S = ca.fmax(S, eps)
        q0 = (r32 - r23) / safe_S
        q1 = 0.25 * S
        q2 = (r12 + r21) / safe_S
        q3 = (r13 + r31) / safe_S
        return ca.vertcat(q1, q2, q3, q0) # Return MX vector

    def case3():
        S = ca.sqrt(ca.fmax(1.0 + r22 - r11 - r33, eps)) * 2
        safe_S = ca.fmax(S, eps)
        q0 = (r13 - r31) / safe_S
        q1 = (r12 + r21) / safe_S
        q2 = 0.25 * S
        q3 = (r23 + r32) / safe_S
        return ca.vertcat(q1, q2, q3, q0) # Return MX vector

    def case4():
        S = ca.sqrt(ca.fmax(1.0 + r33 - r11 - r22, eps)) * 2
        safe_S = ca.fmax(S, eps)
        q0 = (r21 - r12) / safe_S
        q1 = (r13 + r31) / safe_S
        q2 = (r23 + r32) / safe_S
        q3 = 0.25 * S
        return ca.vertcat(q1, q2, q3, q0) # Return MX vector

    # Use nested if_else, now operating on MX vectors returned by case functions
    quat_vector = ca.if_else(tr > 0,
                             case1(),
                             ca.if_else(ca.logic_and(r11 > r22, r11 > r33),
                                        case2(),
                                        ca.if_else(r22 > r33,
                                                   case3(),
                                                   case4()
                                                   )
                                        )
                             )

    # Ensure the quaternion is normalized (optional but recommended for stability)
    # quat_vector = quat_vector / ca.norm_2(quat_vector)

    return quat_vector # Return the resulting vector directly

# ==============================================================================
# UTILITY FUNCTION TO GENERATE RANDOM DCM (No changes needed)
# ==============================================================================
def generate_random_dcm() -> ca.MX:
    """Generates a random valid Direction Cosine Matrix as ca.MX"""
    axis = np.random.rand(3) - 0.5
    axis /= np.linalg.norm(axis)
    angle = random.uniform(0, 2 * np.pi)
    half_angle = angle / 2.0
    sin_half = np.sin(half_angle)
    cos_half = np.cos(half_angle)
    q_np = np.array([
        axis[0] * sin_half,
        axis[1] * sin_half,
        axis[2] * sin_half,
        cos_half
    ])
    q_np /= np.linalg.norm(q_np)
    q_ca = ca.MX(q_np)
    DCM_ca = quat_dcm(q_ca)
    return DCM_ca

# ==============================================================================
# VERIFICATION (No changes needed, should work with corrected function)
# ==============================================================================
print("--- Verification Step ---")
DCM_verify = generate_random_dcm()
q_orig = dcm_quat(DCM_verify)
q_opt = dcm_quat_opt(DCM_verify) # Now call the corrected function

# Evaluate the CasADi expressions to get numerical results
q_orig_np = q_orig.full().flatten()
q_opt_np = q_opt.full().flatten()

# Quaternions q and -q represent the same rotation, so check both
# Add a small tolerance for floating point comparisons
tolerance = 1e-7
match = np.allclose(q_orig_np, q_opt_np, atol=tolerance) or \
        np.allclose(q_orig_np, -q_opt_np, atol=tolerance)

if match:
    print("Verification PASSED: Outputs match numerically.")
else:
    print("Verification FAILED: Outputs do NOT match.")
    print("Original:", q_orig_np)
    print("Optimized:", q_opt_np)
    # Optional: print difference
    print("Difference:", q_orig_np - q_opt_np)
    print("Difference (-):", q_orig_np + q_opt_np)
print("-" * 25)


# ==============================================================================
# TIMING TEST (No changes needed)
# ==============================================================================
print("\n--- Timing Test ---")
num_trials = 20
num_executions = 1000

total_time_orig = 0
total_time_opt = 0
setup_code = "import casadi as ca"
global_dict = {
    'ca': ca,
    'dcm_quat': dcm_quat,
    'dcm_quat_opt': dcm_quat_opt, # Ensure this points to the corrected function
    'DCM_input': None # Placeholder, will be updated in loop
}

for i in range(num_trials):
    print(f"Running trial {i+1}/{num_trials}...")
    DCM_input = generate_random_dcm()
    global_dict['DCM_input'] = DCM_input

    t_orig = timeit.timeit(stmt='dcm_quat(DCM_input)',
                           setup=setup_code,
                           number=num_executions,
                           globals=global_dict)
    total_time_orig += t_orig

    t_opt = timeit.timeit(stmt='dcm_quat_opt(DCM_input)',
                          setup=setup_code,
                          number=num_executions,
                          globals=global_dict)
    total_time_opt += t_opt

avg_time_orig = total_time_orig / (num_trials * num_executions)
avg_time_opt = total_time_opt / (num_trials * num_executions)

print("-" * 25)
print(f"Average execution time over {num_trials} trials ({num_executions} calls each):")
print(f"  Original dcm_quat:    {avg_time_orig * 1e6:.4f} microseconds")
print(f"  Optimized dcm_quat_opt: {avg_time_opt * 1e6:.4f} microseconds")

if avg_time_orig > 0 and avg_time_opt > 0:
    if avg_time_opt < avg_time_orig:
        speedup = avg_time_orig / avg_time_opt
        print(f"\nOptimized version is approximately {speedup:.2f} times faster.")
    else:
        speedup = avg_time_opt / avg_time_orig
        print(f"\nOriginal version is approximately {speedup:.2f} times faster (Optimization ineffective).")
else:
     print("\nCould not calculate speedup (one or both average times were zero).")

print("-" * 25)
print("Note: Timing reflects CasADi MX expression evaluation for numerical inputs.")
print("Performance within a larger optimization context (e.g., with compilation or AD) might differ.")