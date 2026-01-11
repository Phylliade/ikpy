"""
Profiling script to measure JAX overhead vs NumPy for single-point IK.
"""

import numpy as np
import time
import os
import json

# Enable JAX float64 mode for better precision
os.environ['JAX_ENABLE_X64'] = 'True'

from ikpy import chain
from ikpy import JAX_AVAILABLE

# Debug logging helper
LOG_PATH = "/Users/pim/000_COURS/ikpy/.cursor/debug.log"

# Ensure log directory exists
os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)

def log_timing(location, message, data):
    """Append timing data to debug log in NDJSON format"""
    import json
    entry = {
        "timestamp": time.time() * 1000,
        "location": location,
        "message": message,
        "data": data,
        "sessionId": "profile-jax-overhead",
        "hypothesisId": data.get("hypothesis", "general")
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(entry) + "\n")


def profile_single_ik():
    """Profile a single IK call for both NumPy and JAX backends"""
    
    # #region agent log
    log_timing("profile_single_ik:start", "Starting profiling", {"hypothesis": "general"})
    # #endregion
    
    # Load Baxter chain (more complex, realistic use case)
    json_path = os.path.join(os.path.dirname(__file__), "../resources/baxter/baxter_left_arm.json")
    urdf_path = os.path.join(os.path.dirname(__file__), "../resources/baxter/baxter.urdf")
    
    with open(json_path) as f:
        config = json.load(f)
    
    baxter_chain = chain.Chain.from_urdf_file(
        urdf_path,
        base_elements=config["elements"],
        active_links_mask=config["active_links_mask"],
        last_link_vector=config["last_link_vector"],
        name=config["name"]
    )
    
    print(f"\n{'='*70}")
    print(f"Profiling Single-Point IK: Baxter ({len(baxter_chain.links)} links, 7 active)")
    print(f"{'='*70}")
    
    # Target position
    target = [0.6, 0.3, 0.2]
    initial_position = [0.0] * len(baxter_chain.links)
    
    # ============================================
    # Warm-up: Force JAX compilation (AOT)
    # ============================================
    # #region agent log
    t_warmup_start = time.perf_counter()
    # #endregion
    
    _ = baxter_chain.jax_cache  # Force cache creation and AOT compilation
    _ = baxter_chain.inverse_kinematics(target_position=target, backend="jax", optimizer="scipy")
    
    # #region agent log
    t_warmup_end = time.perf_counter()
    log_timing("warmup", "JAX warmup complete", {
        "duration_ms": (t_warmup_end - t_warmup_start) * 1000,
        "hypothesis": "general"
    })
    # #endregion
    
    print(f"\nJAX warmup (compilation): {(t_warmup_end - t_warmup_start)*1000:.2f} ms")
    
    # ============================================
    # SINGLE CALL PROFILING
    # ============================================
    n_calls = 1
    
    print(f"\n--- Single Call Timing (n={n_calls}) ---")
    
    # NumPy single call
    # #region agent log
    t_numpy_start = time.perf_counter()
    # #endregion
    
    result_numpy = baxter_chain.inverse_kinematics(
        target_position=target,
        initial_position=initial_position,
        backend="numpy"
    )
    
    # #region agent log
    t_numpy_end = time.perf_counter()
    numpy_single_ms = (t_numpy_end - t_numpy_start) * 1000
    log_timing("single_ik:numpy", "NumPy single IK call", {
        "duration_ms": numpy_single_ms,
        "hypothesis": "H4"
    })
    # #endregion
    
    # JAX single call
    # #region agent log
    t_jax_start = time.perf_counter()
    # #endregion
    
    result_jax = baxter_chain.inverse_kinematics(
        target_position=target,
        initial_position=initial_position,
        backend="jax",
        optimizer="scipy"
    )
    
    # #region agent log
    t_jax_end = time.perf_counter()
    jax_single_ms = (t_jax_end - t_jax_start) * 1000
    log_timing("single_ik:jax", "JAX single IK call", {
        "duration_ms": jax_single_ms,
        "hypothesis": "H4"
    })
    # #endregion
    
    print(f"NumPy: {numpy_single_ms:.3f} ms")
    print(f"JAX:   {jax_single_ms:.3f} ms")
    print(f"Ratio: JAX is {jax_single_ms/numpy_single_ms:.2f}x {'slower' if jax_single_ms > numpy_single_ms else 'faster'}")
    
    # ============================================
    # DETAILED BREAKDOWN: JAX internals
    # ============================================
    print(f"\n--- Detailed JAX Breakdown ---")
    
    import jax.numpy as jnp
    cache = baxter_chain.jax_cache
    
    # H1: Test conversion overhead
    # #region agent log
    t_conv_start = time.perf_counter()
    # #endregion
    
    for _ in range(100):
        target_frame_jax = jnp.array(np.eye(4), dtype=cache._dtype)
        initial_position_jax = jnp.array(initial_position, dtype=cache._dtype)
    
    # #region agent log
    t_conv_end = time.perf_counter()
    conv_time_per_call = (t_conv_end - t_conv_start) / 100 * 1000
    log_timing("conversion:numpy_to_jax", "NumPy to JAX conversion (100x avg)", {
        "duration_ms_per_call": conv_time_per_call,
        "hypothesis": "H1"
    })
    # #endregion
    
    print(f"H1 - NumPy→JAX conversion: {conv_time_per_call:.4f} ms/call")
    
    # H1b: Test reverse conversion overhead
    # #region agent log
    t_rconv_start = time.perf_counter()
    # #endregion
    
    jax_result = jnp.zeros(len(baxter_chain.links), dtype=cache._dtype)
    for _ in range(100):
        _ = np.array(jax_result)
    
    # #region agent log
    t_rconv_end = time.perf_counter()
    rconv_time_per_call = (t_rconv_end - t_rconv_start) / 100 * 1000
    log_timing("conversion:jax_to_numpy", "JAX to NumPy conversion (100x avg)", {
        "duration_ms_per_call": rconv_time_per_call,
        "hypothesis": "H1"
    })
    # #endregion
    
    print(f"H1 - JAX→NumPy conversion: {rconv_time_per_call:.4f} ms/call")
    
    # H2: Test compiled function dispatch overhead
    target_frame_jax = jnp.array(np.eye(4), dtype=cache._dtype)
    target_frame_jax = target_frame_jax.at[:3, 3].set(jnp.array(target, dtype=cache._dtype))
    initial_position_jax = jnp.array(initial_position, dtype=cache._dtype)
    
    # #region agent log
    t_dispatch_start = time.perf_counter()
    # #endregion
    
    residual_fn = cache._ik_residuals[(None, False)]
    x0 = cache.active_from_full(initial_position_jax)
    for _ in range(100):
        _ = residual_fn(x0, target_frame_jax, initial_position_jax)
    
    # #region agent log
    t_dispatch_end = time.perf_counter()
    dispatch_time_per_call = (t_dispatch_end - t_dispatch_start) / 100 * 1000
    log_timing("dispatch:residual_fn", "Compiled residual fn dispatch (100x avg)", {
        "duration_ms_per_call": dispatch_time_per_call,
        "hypothesis": "H2"
    })
    # #endregion
    
    print(f"H2 - Compiled residual dispatch: {dispatch_time_per_call:.4f} ms/call")
    
    # H2b: Test jacobian compiled function dispatch
    # #region agent log
    t_jac_start = time.perf_counter()
    # #endregion
    
    jacobian_fn = cache._ik_jacobian[(None, False)]
    for _ in range(100):
        _ = jacobian_fn(x0, target_frame_jax, initial_position_jax)
    
    # #region agent log
    t_jac_end = time.perf_counter()
    jac_time_per_call = (t_jac_end - t_jac_start) / 100 * 1000
    log_timing("dispatch:jacobian_fn", "Compiled jacobian fn dispatch (100x avg)", {
        "duration_ms_per_call": jac_time_per_call,
        "hypothesis": "H2"
    })
    # #endregion
    
    print(f"H2 - Compiled jacobian dispatch: {jac_time_per_call:.4f} ms/call")
    
    # H3: Test array mutation overhead
    # #region agent log
    t_mut_start = time.perf_counter()
    # #endregion
    
    for _ in range(100):
        _ = initial_position_jax.at[cache.active_indices].set(x0)
    
    # #region agent log
    t_mut_end = time.perf_counter()
    mut_time_per_call = (t_mut_end - t_mut_start) / 100 * 1000
    log_timing("mutation:at_set", "Array .at[].set() mutation (100x avg)", {
        "duration_ms_per_call": mut_time_per_call,
        "hypothesis": "H3"
    })
    # #endregion
    
    print(f"H3 - Array mutation (.at[].set()): {mut_time_per_call:.4f} ms/call")
    
    # H5: Compare Jacobian computation
    print(f"\n--- Jacobian Computation Comparison ---")
    
    # NumPy: Uses finite differences in scipy
    # JAX: Uses analytical autodiff
    
    # Measure scipy least_squares without JAX
    import scipy.optimize
    
    def numpy_residuals(x):
        full_joints = baxter_chain.active_to_full(x, initial_position)
        fk = baxter_chain.forward_kinematics(full_joints, backend="numpy")
        return fk[:3, 3] - target
    
    x0_np = baxter_chain.active_from_full(initial_position)
    bounds_np = np.array([link.bounds for link in baxter_chain.links])
    bounds_np = baxter_chain.active_from_full(bounds_np)
    bounds_np = np.moveaxis(bounds_np, -1, 0)
    
    # #region agent log
    t_scipy_np_start = time.perf_counter()
    # #endregion
    
    result_scipy_np = scipy.optimize.least_squares(
        numpy_residuals,
        x0_np,
        bounds=bounds_np
    )
    
    # #region agent log
    t_scipy_np_end = time.perf_counter()
    scipy_np_ms = (t_scipy_np_end - t_scipy_np_start) * 1000
    log_timing("scipy:numpy_finite_diff", "SciPy with finite diff Jacobian", {
        "duration_ms": scipy_np_ms,
        "n_function_evals": result_scipy_np.nfev,
        "n_jacobian_evals": result_scipy_np.njev,
        "hypothesis": "H5"
    })
    # #endregion
    
    print(f"NumPy+SciPy (finite diff): {scipy_np_ms:.3f} ms")
    print(f"  Function evaluations: {result_scipy_np.nfev}, Jacobian evaluations: {result_scipy_np.njev}")
    
    # Scipy with JAX analytical Jacobian
    def jax_residuals(x):
        return np.array(residual_fn(
            jnp.array(x, dtype=cache._dtype),
            target_frame_jax,
            initial_position_jax
        ))
    
    def jax_jacobian(x):
        return np.array(jacobian_fn(
            jnp.array(x, dtype=cache._dtype),
            target_frame_jax,
            initial_position_jax
        ))
    
    # #region agent log
    t_scipy_jax_start = time.perf_counter()
    # #endregion
    
    result_scipy_jax = scipy.optimize.least_squares(
        jax_residuals,
        np.array(x0),
        jac=jax_jacobian,
        bounds=(np.array(cache.lower_bounds), np.array(cache.upper_bounds))
    )
    
    # #region agent log
    t_scipy_jax_end = time.perf_counter()
    scipy_jax_ms = (t_scipy_jax_end - t_scipy_jax_start) * 1000
    log_timing("scipy:jax_analytical_jac", "SciPy with JAX analytical Jacobian", {
        "duration_ms": scipy_jax_ms,
        "n_function_evals": result_scipy_jax.nfev,
        "n_jacobian_evals": result_scipy_jax.njev,
        "hypothesis": "H5"
    })
    # #endregion
    
    print(f"JAX+SciPy (analytical jac): {scipy_jax_ms:.3f} ms")
    print(f"  Function evaluations: {result_scipy_jax.nfev}, Jacobian evaluations: {result_scipy_jax.njev}")
    
    # ============================================
    # MULTIPLE CALLS PROFILING (amortized cost)
    # ============================================
    n_iterations = 100
    
    print(f"\n--- Multiple Calls ({n_iterations}x) Amortized Timing ---")
    
    # NumPy multiple calls
    # #region agent log
    t_numpy_multi_start = time.perf_counter()
    # #endregion
    
    for _ in range(n_iterations):
        _ = baxter_chain.inverse_kinematics(
            target_position=target,
            initial_position=initial_position,
            backend="numpy"
        )
    
    # #region agent log
    t_numpy_multi_end = time.perf_counter()
    numpy_multi_ms = (t_numpy_multi_end - t_numpy_multi_start) * 1000
    log_timing("multi_ik:numpy", f"NumPy {n_iterations}x IK calls", {
        "total_duration_ms": numpy_multi_ms,
        "per_call_ms": numpy_multi_ms / n_iterations,
        "hypothesis": "H4"
    })
    # #endregion
    
    # JAX multiple calls
    # #region agent log
    t_jax_multi_start = time.perf_counter()
    # #endregion
    
    for _ in range(n_iterations):
        _ = baxter_chain.inverse_kinematics(
            target_position=target,
            initial_position=initial_position,
            backend="jax",
            optimizer="scipy"
        )
    
    # #region agent log
    t_jax_multi_end = time.perf_counter()
    jax_multi_ms = (t_jax_multi_end - t_jax_multi_start) * 1000
    log_timing("multi_ik:jax", f"JAX {n_iterations}x IK calls", {
        "total_duration_ms": jax_multi_ms,
        "per_call_ms": jax_multi_ms / n_iterations,
        "hypothesis": "H4"
    })
    # #endregion
    
    print(f"NumPy: {numpy_multi_ms:.2f} ms total, {numpy_multi_ms/n_iterations:.3f} ms/call")
    print(f"JAX:   {jax_multi_ms:.2f} ms total, {jax_multi_ms/n_iterations:.3f} ms/call")
    print(f"Ratio: JAX is {jax_multi_ms/numpy_multi_ms:.2f}x {'slower' if jax_multi_ms > numpy_multi_ms else 'faster'}")
    
    # ============================================
    # SUMMARY
    # ============================================
    print(f"\n{'='*70}")
    print("OVERHEAD SUMMARY")
    print(f"{'='*70}")
    
    total_jax_overhead = conv_time_per_call + rconv_time_per_call + dispatch_time_per_call + mut_time_per_call
    print(f"\nEstimated JAX overhead per IK call:")
    print(f"  - NumPy→JAX conversion: {conv_time_per_call:.4f} ms")
    print(f"  - JAX→NumPy conversion: {rconv_time_per_call:.4f} ms")
    print(f"  - Compiled fn dispatch:  {dispatch_time_per_call:.4f} ms")
    print(f"  - Array mutations:       {mut_time_per_call:.4f} ms")
    print(f"  ---------------------------------")
    print(f"  Total measured overhead: ~{total_jax_overhead:.4f} ms")
    print(f"\n  But single call delta:   {jax_single_ms - numpy_single_ms:.3f} ms")
    
    # #region agent log
    log_timing("summary", "Profiling summary", {
        "numpy_single_ms": numpy_single_ms,
        "jax_single_ms": jax_single_ms,
        "conversion_overhead_ms": conv_time_per_call + rconv_time_per_call,
        "dispatch_overhead_ms": dispatch_time_per_call,
        "mutation_overhead_ms": mut_time_per_call,
        "estimated_total_overhead_ms": total_jax_overhead,
        "actual_delta_ms": jax_single_ms - numpy_single_ms,
        "hypothesis": "summary"
    })
    # #endregion
    
    # Verify results are similar
    fk_numpy = baxter_chain.forward_kinematics(result_numpy)[:3, 3]
    fk_jax = baxter_chain.forward_kinematics(result_jax)[:3, 3]
    
    print(f"\nResult verification:")
    print(f"  NumPy error to target: {np.linalg.norm(fk_numpy - target)*1000:.4f} mm")
    print(f"  JAX error to target:   {np.linalg.norm(fk_jax - target)*1000:.4f} mm")


def profile_simple_chain():
    """Profile with a simpler chain to see if overhead becomes more visible"""
    
    # #region agent log
    log_timing("profile_simple:start", "Starting simple chain profiling", {"hypothesis": "general"})
    # #endregion
    
    # Load simpler Poppy Torso chain (4 active joints)
    urdf_path = os.path.join(os.path.dirname(__file__), "../resources/poppy_torso/poppy_torso.URDF")
    simple_chain = chain.Chain.from_urdf_file(
        urdf_path,
        base_elements=[
            "base", "abs_z", "spine", "bust_y", "bust_motors", "bust_x",
            "chest", "r_shoulder_y"
        ],
        last_link_vector=[0, 0.18, 0],
        active_links_mask=[
            False, False, False, False, True, True, True, True, False
        ]
    )
    
    print(f"\n{'='*70}")
    print(f"Profiling Simple Chain: Poppy Torso ({len(simple_chain.links)} links, 4 active)")
    print(f"{'='*70}")
    
    target = [0.1, -0.2, 0.1]
    initial_position = [0.0] * len(simple_chain.links)
    
    # Warmup JAX
    t_warmup_start = time.perf_counter()
    _ = simple_chain.jax_cache
    _ = simple_chain.inverse_kinematics(target_position=target, backend="jax", optimizer="scipy")
    t_warmup_end = time.perf_counter()
    
    # #region agent log
    log_timing("simple:warmup", "Simple chain JAX warmup", {
        "duration_ms": (t_warmup_end - t_warmup_start) * 1000,
        "hypothesis": "general"
    })
    # #endregion
    
    print(f"\nJAX warmup: {(t_warmup_end - t_warmup_start)*1000:.2f} ms")
    
    # Single call comparison
    n_iterations = 100
    
    # NumPy
    t_np_start = time.perf_counter()
    for _ in range(n_iterations):
        _ = simple_chain.inverse_kinematics(target_position=target, initial_position=initial_position, backend="numpy")
    t_np_end = time.perf_counter()
    numpy_ms = (t_np_end - t_np_start) / n_iterations * 1000
    
    # JAX
    t_jax_start = time.perf_counter()
    for _ in range(n_iterations):
        _ = simple_chain.inverse_kinematics(target_position=target, initial_position=initial_position, backend="jax", optimizer="scipy")
    t_jax_end = time.perf_counter()
    jax_ms = (t_jax_end - t_jax_start) / n_iterations * 1000
    
    # #region agent log
    log_timing("simple:comparison", "Simple chain comparison", {
        "numpy_ms_per_call": numpy_ms,
        "jax_ms_per_call": jax_ms,
        "ratio": jax_ms / numpy_ms,
        "hypothesis": "H4"
    })
    # #endregion
    
    print(f"\n--- Per-call timing ({n_iterations}x average) ---")
    print(f"NumPy: {numpy_ms:.3f} ms/call")
    print(f"JAX:   {jax_ms:.3f} ms/call")
    print(f"Ratio: JAX is {jax_ms/numpy_ms:.2f}x {'slower' if jax_ms > numpy_ms else 'faster'}")
    
    # Detailed analysis with function evaluations
    import scipy.optimize
    import jax.numpy as jnp
    
    cache = simple_chain.jax_cache
    target_frame = np.eye(4)
    target_frame[:3, 3] = target
    target_frame_jax = jnp.array(target_frame, dtype=cache._dtype)
    initial_position_jax = jnp.array(initial_position, dtype=cache._dtype)
    
    # NumPy scipy details
    def numpy_residuals(x):
        full_joints = simple_chain.active_to_full(x, initial_position)
        fk = simple_chain.forward_kinematics(full_joints, backend="numpy")
        return fk[:3, 3] - target
    
    x0_np = simple_chain.active_from_full(initial_position)
    bounds_np = np.array([link.bounds for link in simple_chain.links])
    bounds_np = simple_chain.active_from_full(bounds_np)
    bounds_np = np.moveaxis(bounds_np, -1, 0)
    
    t_np_start = time.perf_counter()
    result_np = scipy.optimize.least_squares(numpy_residuals, x0_np, bounds=bounds_np)
    t_np_end = time.perf_counter()
    
    # JAX scipy details
    residual_fn = cache._ik_residuals[(None, False)]
    jacobian_fn = cache._ik_jacobian[(None, False)]
    
    def jax_residuals(x):
        return np.array(residual_fn(
            jnp.array(x, dtype=cache._dtype),
            target_frame_jax,
            initial_position_jax
        ))
    
    def jax_jacobian(x):
        return np.array(jacobian_fn(
            jnp.array(x, dtype=cache._dtype),
            target_frame_jax,
            initial_position_jax
        ))
    
    x0_jax = np.array(cache.active_from_full(initial_position_jax))
    
    t_jax_start = time.perf_counter()
    result_jax = scipy.optimize.least_squares(
        jax_residuals, x0_jax, jac=jax_jacobian,
        bounds=(np.array(cache.lower_bounds), np.array(cache.upper_bounds))
    )
    t_jax_end = time.perf_counter()
    
    # #region agent log
    log_timing("simple:scipy_detail", "Simple chain scipy details", {
        "numpy_time_ms": (t_np_end - t_np_start) * 1000,
        "numpy_nfev": result_np.nfev,
        "numpy_njev": result_np.njev,
        "jax_time_ms": (t_jax_end - t_jax_start) * 1000,
        "jax_nfev": result_jax.nfev,
        "jax_njev": result_jax.njev,
        "hypothesis": "H5"
    })
    # #endregion
    
    print(f"\n--- Single-call scipy details ---")
    print(f"NumPy: {(t_np_end - t_np_start)*1000:.3f} ms, nfev={result_np.nfev}, njev={result_np.njev}")
    print(f"JAX:   {(t_jax_end - t_jax_start)*1000:.3f} ms, nfev={result_jax.nfev}, njev={result_jax.njev}")
    
    # Check position error
    fk_np = simple_chain.forward_kinematics(simple_chain.active_to_full(result_np.x, initial_position))[:3, 3]
    fk_jax = simple_chain.forward_kinematics(simple_chain.active_to_full(result_jax.x, initial_position))[:3, 3]
    print(f"\nPosition errors:")
    print(f"NumPy: {np.linalg.norm(fk_np - target)*1000:.4f} mm")
    print(f"JAX:   {np.linalg.norm(fk_jax - target)*1000:.4f} mm")
    
    # H6: Check if bounds are the same
    print(f"\n--- H6: Bounds comparison ---")
    print(f"NumPy bounds shape: {bounds_np.shape}")
    print(f"NumPy bounds: {bounds_np}")
    print(f"JAX lower_bounds: {np.array(cache.lower_bounds)}")
    print(f"JAX upper_bounds: {np.array(cache.upper_bounds)}")
    
    # H6: Check dtype
    print(f"\nJAX dtype: {cache._dtype}")
    print(f"JAX float64 enabled: {cache._use_float64}")
    
    # #region agent log
    log_timing("simple:bounds_check", "Bounds comparison", {
        "numpy_lower": bounds_np[0].tolist(),
        "numpy_upper": bounds_np[1].tolist(),
        "jax_lower": np.array(cache.lower_bounds).tolist(),
        "jax_upper": np.array(cache.upper_bounds).tolist(),
        "jax_dtype": str(cache._dtype),
        "jax_float64": cache._use_float64,
        "hypothesis": "H6"
    })
    # #endregion
    
    # H6b: Test with same bounds - use numpy bounds for JAX
    print(f"\n--- H6b: JAX with NumPy-style bounds ---")
    t_jax2_start = time.perf_counter()
    result_jax2 = scipy.optimize.least_squares(
        jax_residuals, x0_jax, jac=jax_jacobian,
        bounds=bounds_np  # Use same bounds as numpy
    )
    t_jax2_end = time.perf_counter()
    print(f"JAX (numpy bounds): {(t_jax2_end - t_jax2_start)*1000:.3f} ms, nfev={result_jax2.nfev}, njev={result_jax2.njev}")
    
    # #region agent log
    log_timing("simple:jax_numpy_bounds", "JAX with NumPy bounds", {
        "time_ms": (t_jax2_end - t_jax2_start) * 1000,
        "nfev": result_jax2.nfev,
        "njev": result_jax2.njev,
        "hypothesis": "H6"
    })
    # #endregion
    
    # H7: Compare Jacobians at initial point
    print(f"\n--- H7: Jacobian comparison at x0 ---")
    
    # NumPy finite diff jacobian (approximate)
    eps = 1e-8
    n_active = len(x0_np)
    jac_fd = np.zeros((3, n_active))
    f0 = numpy_residuals(x0_np)
    for i in range(n_active):
        x_plus = x0_np.copy()
        x_plus[i] += eps
        f_plus = numpy_residuals(x_plus)
        jac_fd[:, i] = (f_plus - f0) / eps
    
    # JAX analytical jacobian
    jac_jax = jax_jacobian(x0_jax)
    
    print(f"Finite diff Jacobian:\n{jac_fd}")
    print(f"\nJAX Jacobian:\n{jac_jax}")
    print(f"\nDifference (JAX - FD):\n{jac_jax - jac_fd}")
    print(f"Max abs difference: {np.max(np.abs(jac_jax - jac_fd))}")
    print(f"Relative difference: {np.linalg.norm(jac_jax - jac_fd) / np.linalg.norm(jac_fd) * 100:.4f}%")
    
    # #region agent log
    log_timing("simple:jacobian_comparison", "Jacobian comparison", {
        "fd_jacobian": jac_fd.tolist(),
        "jax_jacobian": jac_jax.tolist(),
        "max_abs_diff": float(np.max(np.abs(jac_jax - jac_fd))),
        "relative_diff_pct": float(np.linalg.norm(jac_jax - jac_fd) / np.linalg.norm(jac_fd) * 100),
        "hypothesis": "H7"
    })
    # #endregion
    
    # H7b: Try scipy with no jacobian hint (let it estimate)
    print(f"\n--- H7b: JAX residuals + scipy auto-jacobian ---")
    t_auto_start = time.perf_counter()
    result_auto = scipy.optimize.least_squares(
        jax_residuals, x0_jax, bounds=bounds_np
    )
    t_auto_end = time.perf_counter()
    print(f"JAX residuals + auto-jac: {(t_auto_end - t_auto_start)*1000:.3f} ms, nfev={result_auto.nfev}, njev={result_auto.njev}")
    
    # #region agent log
    log_timing("simple:jax_auto_jac", "JAX residuals with auto jacobian", {
        "time_ms": (t_auto_end - t_auto_start) * 1000,
        "nfev": result_auto.nfev,
        "njev": result_auto.njev,
        "hypothesis": "H7"
    })
    # #endregion
    
    # H8: Check Jacobian format and order
    print(f"\n--- H8: Jacobian format analysis ---")
    
    # Check what scipy's finite diff produces
    def fd_jacobian(fn, x, eps=1e-8):
        f0 = fn(x)
        n_res = len(f0)
        n_params = len(x)
        jac = np.zeros((n_res, n_params))
        for i in range(n_params):
            x_plus = x.copy()
            x_plus[i] += eps
            jac[:, i] = (fn(x_plus) - f0) / eps
        return jac
    
    # Scipy uses 2-point or 3-point finite differences
    fd_jac_np = fd_jacobian(numpy_residuals, x0_np)
    fd_jac_jax = fd_jacobian(jax_residuals, x0_jax)
    jax_explicit = jax_jacobian(x0_jax)
    
    print(f"FD Jacobian (numpy residuals):\n{fd_jac_np}")
    print(f"\nFD Jacobian (jax residuals):\n{fd_jac_jax}")
    print(f"\nJAX explicit Jacobian:\n{jax_explicit}")
    
    print(f"\nJacobian shapes:")
    print(f"  FD numpy: {fd_jac_np.shape}, dtype: {fd_jac_np.dtype}")
    print(f"  FD jax: {fd_jac_jax.shape}, dtype: {fd_jac_jax.dtype}")
    print(f"  JAX explicit: {jax_explicit.shape}, dtype: {jax_explicit.dtype}")
    
    # Check C-contiguous (required by scipy)
    print(f"\nC-contiguous:")
    print(f"  FD numpy: {fd_jac_np.flags['C_CONTIGUOUS']}")
    print(f"  FD jax: {fd_jac_jax.flags['C_CONTIGUOUS']}")
    print(f"  JAX explicit: {jax_explicit.flags['C_CONTIGUOUS']}")
    
    # H8b: Test with forced contiguous array
    print(f"\n--- H8b: JAX Jacobian with np.ascontiguousarray ---")
    
    def jax_jacobian_contiguous(x):
        jac = jacobian_fn(
            jnp.array(x, dtype=cache._dtype),
            target_frame_jax,
            initial_position_jax
        )
        return np.ascontiguousarray(jac)
    
    t_cont_start = time.perf_counter()
    result_cont = scipy.optimize.least_squares(
        jax_residuals, x0_jax, jac=jax_jacobian_contiguous,
        bounds=bounds_np
    )
    t_cont_end = time.perf_counter()
    print(f"JAX (contiguous jac): {(t_cont_end - t_cont_start)*1000:.3f} ms, nfev={result_cont.nfev}, njev={result_cont.njev}")
    
    # #region agent log
    log_timing("simple:jax_contiguous_jac", "JAX with contiguous jacobian", {
        "time_ms": (t_cont_end - t_cont_start) * 1000,
        "nfev": result_cont.nfev,
        "njev": result_cont.njev,
        "hypothesis": "H8"
    })
    # #endregion
    
    # H9: Check scipy optimization status and cost
    print(f"\n--- H9: Scipy optimization details ---")
    print(f"NumPy:")
    print(f"  status: {result_np.status}, message: {result_np.message}")
    print(f"  cost: {result_np.cost}, optimality: {result_np.optimality}")
    print(f"  x: {result_np.x}")
    
    print(f"\nJAX (explicit jac):")
    print(f"  status: {result_jax.status}, message: {result_jax.message}")
    print(f"  cost: {result_jax.cost}, optimality: {result_jax.optimality}")
    print(f"  x: {result_jax.x}")
    
    print(f"\nJAX (auto jac):")
    print(f"  status: {result_auto.status}, message: {result_auto.message}")
    print(f"  cost: {result_auto.cost}, optimality: {result_auto.optimality}")
    print(f"  x: {result_auto.x}")
    
    # #region agent log
    log_timing("simple:scipy_status", "Scipy optimization status comparison", {
        "numpy_status": result_np.status,
        "numpy_cost": float(result_np.cost),
        "numpy_x": result_np.x.tolist(),
        "jax_explicit_status": result_jax.status,
        "jax_explicit_cost": float(result_jax.cost),
        "jax_explicit_x": result_jax.x.tolist(),
        "jax_auto_status": result_auto.status,
        "jax_auto_cost": float(result_auto.cost),
        "jax_auto_x": result_auto.x.tolist(),
        "hypothesis": "H9"
    })
    # #endregion
    
    # H9b: Check if target is reachable - what's the actual workspace?
    print(f"\n--- H9b: Workspace check ---")
    # Find FK at zero position
    fk_zero = simple_chain.forward_kinematics([0]*len(simple_chain.links))[:3, 3]
    print(f"FK at zero position: {fk_zero}")
    print(f"Target: {target}")
    print(f"Distance to target from zero: {np.linalg.norm(fk_zero - target)*100:.2f} cm")
    
    # Check chain length (approximate reachability)
    link_lengths = [link.length for link in simple_chain.links]
    print(f"Link lengths: {link_lengths}")
    print(f"Total chain length: {sum(link_lengths):.3f} m")


def investigate_scipy_iterations():
    """
    Deep investigation into why scipy does more iterations with explicit Jacobian.
    """
    import scipy.optimize
    import jax.numpy as jnp
    
    print(f"\n{'='*70}")
    print("INVESTIGATION: Why does scipy do more iterations with explicit Jacobian?")
    print(f"{'='*70}")
    
    # Load Poppy Torso (the problematic chain)
    urdf_path = os.path.join(os.path.dirname(__file__), "../resources/poppy_torso/poppy_torso.URDF")
    simple_chain = chain.Chain.from_urdf_file(
        urdf_path,
        base_elements=[
            "base", "abs_z", "spine", "bust_y", "bust_motors", "bust_x",
            "chest", "r_shoulder_y"
        ],
        last_link_vector=[0, 0.18, 0],
        active_links_mask=[
            False, False, False, False, True, True, True, True, False
        ]
    )
    
    target = [0.1, -0.2, 0.1]
    initial_position = [0.0] * len(simple_chain.links)
    
    # Warmup JAX
    _ = simple_chain.jax_cache
    
    cache = simple_chain.jax_cache
    target_frame = np.eye(4)
    target_frame[:3, 3] = target
    target_frame_jax = jnp.array(target_frame, dtype=cache._dtype)
    initial_position_jax = jnp.array(initial_position, dtype=cache._dtype)
    
    # Setup functions
    def numpy_residuals(x):
        full_joints = simple_chain.active_to_full(x, initial_position)
        fk = simple_chain.forward_kinematics(full_joints, backend="numpy")
        return fk[:3, 3] - target
    
    residual_fn = cache._ik_residuals[(None, False)]
    jacobian_fn = cache._ik_jacobian[(None, False)]
    
    def jax_residuals(x):
        return np.array(residual_fn(
            jnp.array(x, dtype=cache._dtype),
            target_frame_jax,
            initial_position_jax
        ))
    
    def jax_jacobian(x):
        return np.array(jacobian_fn(
            jnp.array(x, dtype=cache._dtype),
            target_frame_jax,
            initial_position_jax
        ))
    
    x0 = simple_chain.active_from_full(initial_position)
    bounds = np.array([link.bounds for link in simple_chain.links])
    bounds = simple_chain.active_from_full(bounds)
    bounds = np.moveaxis(bounds, -1, 0)
    
    # ========================================
    # H1 & H2: Track iteration-by-iteration behavior
    # ========================================
    print("\n--- Tracking iteration-by-iteration behavior ---")
    
    # Custom callback to track iterations
    iteration_data_no_jac = []
    iteration_data_with_jac = []
    
    def make_tracker(data_list, residual_func):
        call_count = [0]
        def track_residuals(x):
            r = residual_func(x)
            cost = 0.5 * np.sum(r**2)
            data_list.append({
                'call': call_count[0],
                'x': x.copy(),
                'cost': cost,
                'residual_norm': np.linalg.norm(r)
            })
            call_count[0] += 1
            return r
        return track_residuals
    
    # Run without explicit jacobian
    tracked_jax_no_jac = make_tracker(iteration_data_no_jac, jax_residuals)
    result_no_jac = scipy.optimize.least_squares(
        tracked_jax_no_jac, x0.copy(), bounds=bounds, verbose=0
    )
    
    # Run with explicit jacobian
    tracked_jax_with_jac = make_tracker(iteration_data_with_jac, jax_residuals)
    result_with_jac = scipy.optimize.least_squares(
        tracked_jax_with_jac, x0.copy(), jac=jax_jacobian, bounds=bounds, verbose=0
    )
    
    print(f"\nWithout explicit Jacobian: {len(iteration_data_no_jac)} function calls")
    print(f"With explicit Jacobian: {len(iteration_data_with_jac)} function calls")
    
    # ========================================
    # Analyze convergence paths
    # ========================================
    print("\n--- Cost evolution (first 20 calls) ---")
    print(f"{'Call':<6} {'No Jac Cost':<15} {'With Jac Cost':<15} {'Diff':<15}")
    print("-" * 55)
    
    max_calls = min(20, len(iteration_data_no_jac), len(iteration_data_with_jac))
    for i in range(max_calls):
        cost_no = iteration_data_no_jac[i]['cost']
        cost_with = iteration_data_with_jac[i]['cost']
        diff = cost_with - cost_no
        print(f"{i:<6} {cost_no:<15.6e} {cost_with:<15.6e} {diff:<15.6e}")
    
    # ========================================
    # H2: Check trust region behavior via verbose output
    # ========================================
    print("\n--- Trust Region Analysis (verbose=2) ---")
    print("\nWithout explicit Jacobian:")
    result_verbose_no = scipy.optimize.least_squares(
        jax_residuals, x0.copy(), bounds=bounds, verbose=2, max_nfev=30
    )
    
    print("\nWith explicit Jacobian:")
    result_verbose_with = scipy.optimize.least_squares(
        jax_residuals, x0.copy(), jac=jax_jacobian, bounds=bounds, verbose=2, max_nfev=30
    )
    
    # ========================================
    # H3: Check if scaling matters
    # ========================================
    print("\n--- H3: Effect of x_scale ---")
    
    # Test with different x_scale options
    for x_scale in ['jac', [1.0]*4]:
        result_scale = scipy.optimize.least_squares(
            jax_residuals, x0.copy(), jac=jax_jacobian, bounds=bounds,
            x_scale=x_scale if isinstance(x_scale, str) else np.array(x_scale)
        )
        scale_name = x_scale if isinstance(x_scale, str) else 'manual [1,1,1,1]'
        print(f"x_scale={scale_name}: nfev={result_scale.nfev}, cost={result_scale.cost:.6e}")
    
    # ========================================
    # H4: Check finite difference method
    # ========================================
    print("\n--- H4: Finite difference method comparison ---")
    
    for diff_step in [None, 1e-8, 1e-6, 1e-4]:
        result_fd = scipy.optimize.least_squares(
            jax_residuals, x0.copy(), bounds=bounds, diff_step=diff_step
        )
        step_name = 'default' if diff_step is None else f'{diff_step}'
        print(f"diff_step={step_name}: nfev={result_fd.nfev}, cost={result_fd.cost:.6e}")
    
    # ========================================
    # Key insight: Compare Jacobian at different points
    # ========================================
    print("\n--- Jacobian comparison at multiple points ---")
    
    # Compare at x0
    jac_explicit_x0 = jax_jacobian(x0)
    
    # Scipy's internal FD jacobian approximation
    eps = np.sqrt(np.finfo(float).eps)  # scipy's default
    jac_fd_x0 = np.zeros_like(jac_explicit_x0)
    f0 = jax_residuals(x0)
    for i in range(len(x0)):
        x_plus = x0.copy()
        h = eps * max(1.0, abs(x0[i]))
        x_plus[i] += h
        jac_fd_x0[:, i] = (jax_residuals(x_plus) - f0) / h
    
    print(f"Scipy default eps: {eps}")
    print(f"\nJacobian at x0 (explicit):\n{jac_explicit_x0}")
    print(f"\nJacobian at x0 (FD with scipy's eps):\n{jac_fd_x0}")
    print(f"\nDifference:\n{jac_explicit_x0 - jac_fd_x0}")
    print(f"Max abs diff: {np.max(np.abs(jac_explicit_x0 - jac_fd_x0)):.2e}")
    
    # Condition number
    print(f"\nCondition number (explicit): {np.linalg.cond(jac_explicit_x0):.2e}")
    print(f"Condition number (FD): {np.linalg.cond(jac_fd_x0):.2e}")
    
    # #region agent log
    log_timing("investigation:summary", "Investigation summary", {
        "nfev_no_jac": result_no_jac.nfev,
        "nfev_with_jac": result_with_jac.nfev,
        "cost_no_jac": float(result_no_jac.cost),
        "cost_with_jac": float(result_with_jac.cost),
        "cond_explicit": float(np.linalg.cond(jac_explicit_x0)),
        "cond_fd": float(np.linalg.cond(jac_fd_x0)),
        "hypothesis": "investigation"
    })
    # #endregion


def test_analytical_jacobian_option():
    """
    Test the new use_analytical_jacobian option.
    """
    print(f"\n{'='*70}")
    print("TEST: use_analytical_jacobian option")
    print(f"{'='*70}")
    
    # Load Poppy Torso (the problematic chain)
    urdf_path = os.path.join(os.path.dirname(__file__), "../resources/poppy_torso/poppy_torso.URDF")
    simple_chain = chain.Chain.from_urdf_file(
        urdf_path,
        base_elements=[
            "base", "abs_z", "spine", "bust_y", "bust_motors", "bust_x",
            "chest", "r_shoulder_y"
        ],
        last_link_vector=[0, 0.18, 0],
        active_links_mask=[
            False, False, False, False, True, True, True, True, False
        ]
    )
    
    target = [0.1, -0.2, 0.1]
    
    # Warmup
    _ = simple_chain.inverse_kinematics(target_position=target, backend="jax", optimizer="scipy")
    
    # Test with analytical Jacobian (default)
    t1_start = time.perf_counter()
    for _ in range(10):
        result1 = simple_chain.inverse_kinematics(
            target_position=target, 
            backend="jax", 
            optimizer="scipy",
            use_analytical_jacobian=True
        )
    t1_end = time.perf_counter()
    
    # Test without analytical Jacobian
    t2_start = time.perf_counter()
    for _ in range(10):
        result2 = simple_chain.inverse_kinematics(
            target_position=target, 
            backend="jax", 
            optimizer="scipy",
            use_analytical_jacobian=False
        )
    t2_end = time.perf_counter()
    
    time_analytical = (t1_end - t1_start) / 10 * 1000
    time_fd = (t2_end - t2_start) / 10 * 1000
    
    print(f"\nPoppy Torso - JAX scipy optimizer:")
    print(f"  With analytical Jacobian:    {time_analytical:.2f} ms/call")
    print(f"  Without (finite diff):       {time_fd:.2f} ms/call")
    print(f"  Speedup with FD:             {time_analytical/time_fd:.2f}x")
    
    # Verify results are similar
    fk1 = simple_chain.forward_kinematics(result1)[:3, 3]
    fk2 = simple_chain.forward_kinematics(result2)[:3, 3]
    print(f"\nPosition errors:")
    print(f"  Analytical Jacobian: {np.linalg.norm(fk1 - target)*1000:.4f} mm")
    print(f"  Finite diff:         {np.linalg.norm(fk2 - target)*1000:.4f} mm")
    
    # #region agent log
    log_timing("test:analytical_jacobian", "Analytical Jacobian option test", {
        "time_analytical_ms": time_analytical,
        "time_fd_ms": time_fd,
        "speedup": time_analytical / time_fd,
        "error_analytical_mm": float(np.linalg.norm(fk1 - target)*1000),
        "error_fd_mm": float(np.linalg.norm(fk2 - target)*1000),
        "hypothesis": "solution"
    })
    # #endregion


def test_scipy_options():
    """
    Test different scipy.optimize.least_squares options to improve convergence
    with analytical Jacobian.
    """
    import scipy.optimize
    import jax.numpy as jnp
    
    print(f"\n{'='*70}")
    print("TEST: Scipy options to fix analytical Jacobian convergence")
    print(f"{'='*70}")
    
    # Load Poppy Torso
    urdf_path = os.path.join(os.path.dirname(__file__), "../resources/poppy_torso/poppy_torso.URDF")
    poppy = chain.Chain.from_urdf_file(
        urdf_path,
        base_elements=["base", "abs_z", "spine", "bust_y", "bust_motors", "bust_x", "chest", "r_shoulder_y"],
        last_link_vector=[0, 0.18, 0],
        active_links_mask=[False, False, False, False, True, True, True, True, False]
    )
    
    target = [0.1, -0.2, 0.1]
    initial_position = [0.0] * len(poppy.links)
    
    # Warmup JAX
    _ = poppy.jax_cache
    cache = poppy.jax_cache
    
    target_frame = np.eye(4)
    target_frame[:3, 3] = target
    target_frame_jax = jnp.array(target_frame, dtype=cache._dtype)
    initial_jax = jnp.array(initial_position, dtype=cache._dtype)
    
    residual_fn = cache._ik_residuals[(None, False)]
    jacobian_fn = cache._ik_jacobian[(None, False)]
    
    def jax_res(x):
        return np.array(residual_fn(jnp.array(x, dtype=cache._dtype), target_frame_jax, initial_jax))
    
    def jax_jac(x):
        return np.array(jacobian_fn(jnp.array(x, dtype=cache._dtype), target_frame_jax, initial_jax))
    
    x0 = poppy.active_from_full(initial_position)
    bounds = np.array([link.bounds for link in poppy.links])
    bounds = poppy.active_from_full(bounds)
    bounds = np.moveaxis(bounds, -1, 0)
    
    print(f"\nBaseline (no Jacobian): ", end="")
    r_base = scipy.optimize.least_squares(jax_res, x0.copy(), bounds=bounds)
    print(f"nfev={r_base.nfev}, cost={r_base.cost:.2e}")
    
    print(f"Baseline (with Jacobian): ", end="")
    r_jac = scipy.optimize.least_squares(jax_res, x0.copy(), jac=jax_jac, bounds=bounds)
    print(f"nfev={r_jac.nfev}, cost={r_jac.cost:.2e}")
    
    print(f"\n{'='*70}")
    print("Testing different options with analytical Jacobian:")
    print(f"{'='*70}\n")
    
    # Test 1: x_scale='jac' (auto-scaling based on Jacobian)
    print("1. x_scale='jac' (auto-scaling): ", end="")
    r1 = scipy.optimize.least_squares(jax_res, x0.copy(), jac=jax_jac, bounds=bounds, x_scale='jac')
    print(f"nfev={r1.nfev}, cost={r1.cost:.2e}")
    
    # Test 2: Different tr_solver
    print("2. tr_solver='lsmr': ", end="")
    r2 = scipy.optimize.least_squares(jax_res, x0.copy(), jac=jax_jac, bounds=bounds, tr_solver='lsmr')
    print(f"nfev={r2.nfev}, cost={r2.cost:.2e}")
    
    # Test 3: Relaxed tolerances
    print("3. ftol=1e-4, xtol=1e-4: ", end="")
    r3 = scipy.optimize.least_squares(jax_res, x0.copy(), jac=jax_jac, bounds=bounds, ftol=1e-4, xtol=1e-4)
    print(f"nfev={r3.nfev}, cost={r3.cost:.2e}")
    
    # Test 4: Robust loss function (soft_l1)
    print("4. loss='soft_l1': ", end="")
    r4 = scipy.optimize.least_squares(jax_res, x0.copy(), jac=jax_jac, bounds=bounds, loss='soft_l1')
    print(f"nfev={r4.nfev}, cost={r4.cost:.2e}")
    
    # Test 5: Robust loss function (huber)
    print("5. loss='huber': ", end="")
    r5 = scipy.optimize.least_squares(jax_res, x0.copy(), jac=jax_jac, bounds=bounds, loss='huber')
    print(f"nfev={r5.nfev}, cost={r5.cost:.2e}")
    
    # Test 6: Combination x_scale + relaxed tol
    print("6. x_scale='jac' + ftol=1e-4: ", end="")
    r6 = scipy.optimize.least_squares(jax_res, x0.copy(), jac=jax_jac, bounds=bounds, x_scale='jac', ftol=1e-4)
    print(f"nfev={r6.nfev}, cost={r6.cost:.2e}")
    
    # Test 7: dogbox method instead of trf
    print("7. method='dogbox': ", end="")
    r7 = scipy.optimize.least_squares(jax_res, x0.copy(), jac=jax_jac, bounds=bounds, method='dogbox')
    print(f"nfev={r7.nfev}, cost={r7.cost:.2e}")
    
    # Test 8: lm method (no bounds support, so skip bounds)
    print("8. method='lm' (no bounds): ", end="")
    r8 = scipy.optimize.least_squares(jax_res, x0.copy(), jac=jax_jac, method='lm')
    print(f"nfev={r8.nfev}, cost={r8.cost:.2e}")
    
    # Find best option
    results = [
        ("baseline (no jac)", r_base),
        ("baseline (with jac)", r_jac),
        ("x_scale='jac'", r1),
        ("tr_solver='lsmr'", r2),
        ("relaxed tol", r3),
        ("loss='soft_l1'", r4),
        ("loss='huber'", r5),
        ("x_scale + tol", r6),
        ("method='dogbox'", r7),
        ("method='lm'", r8),
    ]
    
    print(f"\n{'='*70}")
    print("SUMMARY (sorted by nfev):")
    print(f"{'='*70}")
    for name, r in sorted(results, key=lambda x: x[1].nfev):
        fk = poppy.forward_kinematics(poppy.active_to_full(r.x, initial_position))[:3, 3]
        err = np.linalg.norm(fk - target) * 1000
        print(f"{name:25s}: nfev={r.nfev:3d}, cost={r.cost:.2e}, error={err:.4f}mm")


def test_new_scipy_params():
    """Test the new scipy parameters in the JAX backend."""
    print(f"\n{'='*70}")
    print("TEST: New scipy parameters in JAX backend")
    print(f"{'='*70}")
    
    # Load Poppy Torso
    urdf_path = os.path.join(os.path.dirname(__file__), "../resources/poppy_torso/poppy_torso.URDF")
    poppy = chain.Chain.from_urdf_file(
        urdf_path,
        base_elements=["base", "abs_z", "spine", "bust_y", "bust_motors", "bust_x", "chest", "r_shoulder_y"],
        last_link_vector=[0, 0.18, 0],
        active_links_mask=[False, False, False, False, True, True, True, True, False]
    )
    
    target = [0.1, -0.2, 0.1]
    
    # Warmup
    _ = poppy.inverse_kinematics(target_position=target, backend="jax")
    
    configs = [
        {"name": "NumPy baseline", "backend": "numpy"},
        {"name": "JAX + FD (no analytical)", "backend": "jax", "use_analytical_jacobian": False},
        {"name": "JAX default (analytical)", "backend": "jax"},
        {"name": "JAX + x_scale='jac'", "backend": "jax", "scipy_x_scale": "jac"},
        {"name": "JAX + gtol=1e-6", "backend": "jax", "scipy_gtol": 1e-6},
        {"name": "JAX + tr_solver='lsmr'", "backend": "jax", "scipy_tr_solver": "lsmr"},
        {"name": "JAX + lsmr + regularize", "backend": "jax", "scipy_tr_solver": "lsmr", "scipy_tr_options": {"regularize": True}},
        {"name": "JAX + method='dogbox'", "backend": "jax", "scipy_method": "dogbox"},
        {"name": "JAX + x_scale + gtol", "backend": "jax", "scipy_x_scale": "jac", "scipy_gtol": 1e-6},
        {"name": "JAX optimal combo", "backend": "jax", "scipy_x_scale": "jac", "scipy_gtol": 1e-6, "scipy_tr_solver": "lsmr", "scipy_tr_options": {"regularize": True}},
    ]
    
    print(f"\n{'Config':<30} {'Time (ms)':<12} {'Error (mm)':<12}")
    print("-" * 55)
    
    for cfg in configs:
        name = cfg.pop("name")
        
        # Time 10 iterations
        t_start = time.perf_counter()
        for _ in range(10):
            result = poppy.inverse_kinematics(target_position=target, **cfg)
        t_end = time.perf_counter()
        
        avg_time = (t_end - t_start) / 10 * 1000
        
        # Check error
        fk = poppy.forward_kinematics(result)[:3, 3]
        error = np.linalg.norm(fk - target) * 1000
        
        print(f"{name:<30} {avg_time:<12.2f} {error:<12.4f}")
        
        # Restore name for next iteration
        cfg["name"] = name


if __name__ == "__main__":
    if not JAX_AVAILABLE:
        print("JAX is not available. Please install JAX to run this profiling.")
        exit(1)
    
    test_new_scipy_params()
