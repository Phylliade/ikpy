"""Tests for the JAX backend"""

import numpy as np
import pytest
import os

# Enable JAX float64 mode for better precision
os.environ['JAX_ENABLE_X64'] = 'True'

from ikpy import chain
from ikpy import JAX_AVAILABLE


# Skip all tests in this module if JAX is not available
pytestmark = pytest.mark.skipif(not JAX_AVAILABLE, reason="JAX is not installed")


@pytest.fixture
def simple_chain():
    """Create a simple chain for testing"""
    return chain.Chain.from_urdf_file(
        "../resources/poppy_torso/poppy_torso.URDF",
        base_elements=[
            "base", "abs_z", "spine", "bust_y", "bust_motors", "bust_x",
            "chest", "r_shoulder_y"
        ],
        last_link_vector=[0, 0.18, 0],
        active_links_mask=[
            False, False, False, False, True, True, True, True, False
        ]
    )


class TestForwardKinematics:
    """Tests for forward kinematics with JAX backend"""
    
    def test_fk_matches_numpy(self, simple_chain):
        """Test that JAX FK results match NumPy results"""
        joints = [0.0] * len(simple_chain.links)
        
        fk_numpy = simple_chain.forward_kinematics(joints, backend="numpy")
        fk_jax = simple_chain.forward_kinematics(joints, backend="jax")
        
        np.testing.assert_allclose(fk_numpy, fk_jax, rtol=1e-5, atol=1e-5)
    
    def test_fk_with_nonzero_joints(self, simple_chain):
        """Test FK with non-zero joint values"""
        joints = [0.0] * len(simple_chain.links)
        joints[4] = 0.5
        joints[5] = -0.3
        joints[6] = 0.8
        
        fk_numpy = simple_chain.forward_kinematics(joints, backend="numpy")
        fk_jax = simple_chain.forward_kinematics(joints, backend="jax")
        
        np.testing.assert_allclose(fk_numpy, fk_jax, rtol=1e-5, atol=1e-5)
    
    def test_fk_full_kinematics(self, simple_chain):
        """Test full kinematics mode"""
        joints = [0.0] * len(simple_chain.links)
        
        fk_full_numpy = simple_chain.forward_kinematics(joints, full_kinematics=True, backend="numpy")
        fk_full_jax = simple_chain.forward_kinematics(joints, full_kinematics=True, backend="jax")
        
        assert len(fk_full_numpy) == len(fk_full_jax)
        
        for i, (fn, fj) in enumerate(zip(fk_full_numpy, fk_full_jax)):
            np.testing.assert_allclose(fn, fj, rtol=1e-5, atol=1e-5, 
                                      err_msg=f"Mismatch at link {i}")


class TestInverseKinematics:
    """Tests for inverse kinematics with JAX backend"""
    
    def test_ik_default(self, simple_chain):
        """Test IK with default settings (scipy + x_scale='jac')"""
        target = [-0.25, -0.05, 0.25]  # Target in workspace
        
        ik_result = simple_chain.inverse_kinematics(
            target_position=target, 
            backend="jax"
        )
        
        # Verify the result reaches the target
        fk_result = simple_chain.forward_kinematics(ik_result, backend="jax")[:3, 3]
        error = np.linalg.norm(fk_result - target)
        
        assert error < 0.01, f"IK error too high: {error}"
    
    def test_ik_without_analytical_jacobian(self, simple_chain):
        """Test IK with finite differences jacobian"""
        target = [-0.25, -0.05, 0.25]
        
        ik_result = simple_chain.inverse_kinematics(
            target_position=target, 
            backend="jax",
            use_analytical_jacobian=False
        )
        
        fk_result = simple_chain.forward_kinematics(ik_result, backend="jax")[:3, 3]
        error = np.linalg.norm(fk_result - target)
        
        assert error < 0.01, f"IK error too high: {error}"
    
    def test_ik_dogbox_method(self, simple_chain):
        """Test IK with dogbox method"""
        target = [-0.25, -0.05, 0.25]
        
        ik_result = simple_chain.inverse_kinematics(
            target_position=target, 
            backend="jax",
            scipy_method="dogbox"
        )
        
        fk_result = simple_chain.forward_kinematics(ik_result, backend="jax")[:3, 3]
        error = np.linalg.norm(fk_result - target)
        
        assert error < 0.01, f"IK error too high: {error}"


class TestJaxCache:
    """Tests for the JAX kinematics cache"""
    
    def test_cache_creation(self, simple_chain):
        """Test that the cache is created correctly"""
        cache = simple_chain.jax_cache
        
        assert cache is not None
        assert cache.chain is simple_chain
    
    def test_cache_reuse(self, simple_chain):
        """Test that the cache is reused on subsequent calls"""
        cache1 = simple_chain.jax_cache
        cache2 = simple_chain.jax_cache
        
        assert cache1 is cache2


class TestAOTCompilation:
    """Tests for AOT (Ahead-of-Time) compilation using lower/compile"""
    
    def test_fk_aot_compiled(self, simple_chain):
        """Test that FK functions are AOT compiled"""
        cache = simple_chain.jax_cache
        
        # Check that compiled functions exist (not None)
        assert cache._fk_compiled is not None
        assert cache._fk_full_compiled is not None
        
        # Check they are not JIT functions (they should be CompiledFunction)
        # CompiledFunction doesn't have the 'lower' method that jit functions have
        assert not hasattr(cache._fk_compiled, 'lower')
        assert not hasattr(cache._fk_full_compiled, 'lower')
    
    def test_ik_variants_precompiled(self, simple_chain):
        """Test that IK functions are pre-compiled for all orientation modes"""
        cache = simple_chain.jax_cache
        
        # Expected combinations (excluding no_position=True with orient_mode=None)
        expected_keys = [
            (None, False),       # position only
            ("X", False),        # position + X orientation
            ("X", True),         # X orientation only
            ("Y", False),        # position + Y orientation
            ("Y", True),         # Y orientation only
            ("Z", False),        # position + Z orientation
            ("Z", True),         # Z orientation only
            ("all", False),      # position + all orientation
            ("all", True),       # all orientation only
        ]
        
        for key in expected_keys:
            assert key in cache._ik_residuals, f"Missing compiled residuals for {key}"
            assert key in cache._ik_jacobian, f"Missing compiled jacobian for {key}"
            # Check it's a CompiledFunction (no 'lower' method)
            assert not hasattr(cache._ik_residuals[key], 'lower')
            assert not hasattr(cache._ik_jacobian[key], 'lower')
        
        # The invalid combination should not exist
        assert (None, True) not in cache._ik_residuals
    
    def test_compilation_happens_once(self, simple_chain):
        """Test that compilation is done once at cache creation, not on each call"""
        import time
        
        # Force cache creation (this does the compilation)
        cache = simple_chain.jax_cache
        
        # Get references to compiled functions
        fk_compiled_ref = cache._fk_compiled
        ik_residuals_ref = cache._ik_residuals[(None, False)]
        
        # Run multiple FK and IK calls
        joints = [0.0] * len(simple_chain.links)
        target = np.eye(4)
        target[:3, 3] = [-0.25, -0.05, 0.25]
        
        for _ in range(10):
            _ = cache.forward_kinematics(joints)
            _ = cache.inverse_kinematics(target)
        
        # Verify the same compiled functions are still being used
        assert cache._fk_compiled is fk_compiled_ref
        assert cache._ik_residuals[(None, False)] is ik_residuals_ref
    
    def test_lazy_compilation_fallback(self):
        """Test that lazy compilation works when precompile=False"""
        from ikpy import chain as chain_module
        
        # Create chain with precompile=False
        lazy_chain = chain_module.Chain.from_urdf_file(
            "../resources/poppy_torso/poppy_torso.URDF",
            base_elements=[
                "base", "abs_z", "spine", "bust_y", "bust_motors", "bust_x",
                "chest", "r_shoulder_y"
            ],
            last_link_vector=[0, 0.18, 0],
            active_links_mask=[
                False, False, False, False, True, True, True, True, False
            ],
            jax_precompile=False
        )
        
        cache = lazy_chain.jax_cache
        
        # With lazy compilation, _fk_compiled should be None
        assert cache._fk_compiled is None
        assert cache._fk_full_compiled is None
        
        # But _fk_jit should exist for lazy compilation
        assert cache._fk_jit is not None
        assert cache._fk_full_jit is not None
        
        # FK should still work (using JIT fallback)
        joints = [0.0] * len(lazy_chain.links)
        fk_result = lazy_chain.forward_kinematics(joints, backend="jax")
        assert fk_result.shape == (4, 4)
    
    def test_ik_uses_precompiled_functions(self, simple_chain):
        """Test that IK actually uses the pre-compiled functions"""
        cache = simple_chain.jax_cache
        
        # Get the pre-compiled function for position-only mode
        residuals_fn = cache._ik_residuals[(None, False)]
        
        # Run IK
        target = np.eye(4)
        target[:3, 3] = [-0.25, -0.05, 0.25]
        
        result = simple_chain.inverse_kinematics(
            target_position=[-0.25, -0.05, 0.25],
            backend="jax"
        )
        
        # Verify result is valid
        assert result.shape == (len(simple_chain.links),)
        
        # The compiled function should still be the same object
        assert cache._ik_residuals[(None, False)] is residuals_fn
    
    def test_ik_orientation_modes_all_work(self, simple_chain):
        """Test that IK works for all pre-compiled orientation modes"""
        target = np.eye(4)
        target[:3, 3] = [-0.25, -0.05, 0.25]
        target[:3, 0] = [1, 0, 0]  # X axis
        target[:3, 1] = [0, 1, 0]  # Y axis
        target[:3, 2] = [0, 0, 1]  # Z axis
        
        # Test each orientation mode
        for orient_mode in [None, "X", "Y", "Z", "all"]:
            result = simple_chain.inverse_kinematics(
                target_position=[-0.25, -0.05, 0.25],
                target_orientation=target[:3, 0] if orient_mode == "X" else 
                                   target[:3, 1] if orient_mode == "Y" else
                                   target[:3, 2] if orient_mode == "Z" else
                                   target[:3, :3] if orient_mode == "all" else None,
                orientation_mode=orient_mode,
                backend="jax"
            )
            assert result.shape == (len(simple_chain.links),), f"Failed for orientation_mode={orient_mode}"


class TestPerformance:
    """Performance comparison tests between NumPy and JAX backends"""

    def test_forward_kinematics_speed(self, simple_chain):
        """Compare FK speed between NumPy and JAX"""
        import time

        joints = [0.0] * len(simple_chain.links)
        joints[4] = 0.5
        joints[5] = -0.3

        n_iterations = 1000

        # Warm-up JAX (compilation)
        _ = simple_chain.forward_kinematics(joints, backend="jax")

        # Benchmark NumPy
        start = time.perf_counter()
        for _ in range(n_iterations):
            _ = simple_chain.forward_kinematics(joints, backend="numpy")
        numpy_time = time.perf_counter() - start

        # Benchmark JAX
        start = time.perf_counter()
        for _ in range(n_iterations):
            _ = simple_chain.forward_kinematics(joints, backend="jax")
        jax_time = time.perf_counter() - start

        print(f"\n{'='*60}")
        print(f"Forward Kinematics Benchmark ({n_iterations} iterations)")
        print(f"{'='*60}")
        print(f"NumPy: {numpy_time*1000:.2f} ms total, {numpy_time/n_iterations*1000:.4f} ms/call")
        print(f"JAX:   {jax_time*1000:.2f} ms total, {jax_time/n_iterations*1000:.4f} ms/call")
        print(f"Speedup: {numpy_time/jax_time:.2f}x")
        print(f"{'='*60}")

        # Verify both produce same results
        fk_numpy = simple_chain.forward_kinematics(joints, backend="numpy")
        fk_jax = simple_chain.forward_kinematics(joints, backend="jax")
        np.testing.assert_allclose(fk_numpy, fk_jax, rtol=1e-5, atol=1e-5)


class TestTrajectoryTracking:
    """Test trajectory tracking performance: NumPy+SciPy vs JAX+SciPy"""

    @pytest.fixture
    def baxter_chain(self):
        """Load Baxter left arm chain"""
        import json
        
        json_path = os.path.join(os.path.dirname(__file__), "../resources/baxter/baxter_left_arm.json")
        urdf_path = os.path.join(os.path.dirname(__file__), "../resources/baxter/baxter.urdf")
        
        with open(json_path) as f:
            config = json.load(f)
        
        return chain.Chain.from_urdf_file(
            urdf_path,
            base_elements=config["elements"],
            active_links_mask=config["active_links_mask"],
            last_link_vector=config["last_link_vector"],
            name=config["name"]
        )

    def generate_circular_trajectory(self, center, radius, n_points):
        """Generate a circular trajectory in 3D space"""
        trajectory = []
        for i in range(n_points):
            angle = 2 * np.pi * i / n_points
            x = center[0] + radius * np.cos(angle)
            y = center[1] + radius * np.sin(angle)
            z = center[2]
            trajectory.append([x, y, z])
        return trajectory

    def generate_figure8_trajectory(self, center, radius, n_points):
        """Generate a figure-8 (lemniscate) trajectory"""
        trajectory = []
        for i in range(n_points):
            t = 2 * np.pi * i / n_points
            # Lemniscate of Bernoulli
            scale = 1 / (1 + np.sin(t)**2)
            x = center[0] + radius * np.cos(t) * scale
            y = center[1] + radius * np.sin(t) * np.cos(t) * scale
            z = center[2] + 0.05 * np.sin(2*t)  # Small Z variation
            trajectory.append([x, y, z])
        return trajectory

    def test_trajectory_numpy_vs_jax_scipy(self, baxter_chain):
        """
        Compare trajectory tracking performance between:
        - NumPy backend with SciPy least_squares (finite differences Jacobian)
        - JAX backend with SciPy least_squares (analytical Jacobian via JAX)
        """
        import time
        
        # Generate a challenging figure-8 trajectory
        # Baxter's workspace is roughly centered around x=0.5-0.8, y=0.2-0.5
        center = [0.6, 0.3, 0.2]
        radius = 0.15
        n_points = 50
        
        trajectory = self.generate_figure8_trajectory(center, radius, n_points)
        
        print(f"\n{'='*70}")
        print(f"Trajectory Tracking Benchmark: Baxter ({len(baxter_chain.links)} links, 7 active joints)")
        print(f"Trajectory: Figure-8 with {n_points} waypoints")
        print(f"{'='*70}")
        
        # Warm-up JAX (ensure AOT compilation is done)
        initial_position = baxter_chain.inverse_kinematics(
            target_position=trajectory[0],
            backend="jax"
        )
        
        # ========== NumPy + SciPy (finite differences) ==========
        numpy_results = []
        numpy_errors = []
        current_joints = None
        
        start = time.perf_counter()
        for target in trajectory:
            result = baxter_chain.inverse_kinematics(
                target_position=target,
                initial_position=current_joints,
                backend="numpy"  # Uses scipy least_squares with finite diff Jacobian
            )
            numpy_results.append(result)
            current_joints = result
            
            # Compute error
            fk = baxter_chain.forward_kinematics(result)
            error = np.linalg.norm(fk[:3, 3] - target)
            numpy_errors.append(error)
        numpy_time = time.perf_counter() - start
        
        # ========== JAX + SciPy (analytical Jacobian) ==========
        jax_results = []
        jax_errors = []
        current_joints = None
        
        start = time.perf_counter()
        for target in trajectory:
            result = baxter_chain.inverse_kinematics(
                target_position=target,
                initial_position=current_joints,
                backend="jax"  # Uses scipy least_squares with JAX Jacobian
            )
            jax_results.append(result)
            current_joints = result
            
            # Compute error
            fk = baxter_chain.forward_kinematics(result)
            error = np.linalg.norm(fk[:3, 3] - target)
            jax_errors.append(error)
        jax_time = time.perf_counter() - start
        
        # ========== Results ==========
        print(f"\n--- NumPy + SciPy (finite differences Jacobian) ---")
        print(f"Total time: {numpy_time*1000:.2f} ms")
        print(f"Time per waypoint: {numpy_time/n_points*1000:.2f} ms")
        print(f"Mean position error: {np.mean(numpy_errors)*1000:.4f} mm")
        print(f"Max position error: {np.max(numpy_errors)*1000:.4f} mm")
        
        print(f"\n--- JAX + SciPy (analytical Jacobian) ---")
        print(f"Total time: {jax_time*1000:.2f} ms")
        print(f"Time per waypoint: {jax_time/n_points*1000:.2f} ms")
        print(f"Mean position error: {np.mean(jax_errors)*1000:.4f} mm")
        print(f"Max position error: {np.max(jax_errors)*1000:.4f} mm")
        
        speedup = numpy_time / jax_time
        print(f"\n{'='*70}")
        print(f"SPEEDUP: JAX+SciPy is {speedup:.2f}x {'faster' if speedup > 1 else 'slower'} than NumPy+SciPy")
        print(f"{'='*70}")
        
        # Verify both methods achieve good precision
        assert np.mean(numpy_errors) < 0.01, f"NumPy mean error too high: {np.mean(numpy_errors)}"
        assert np.mean(jax_errors) < 0.01, f"JAX mean error too high: {np.mean(jax_errors)}"

    def test_trajectory_with_warm_start(self, baxter_chain):
        """
        Test trajectory tracking with warm start optimization.
        Shows the benefit of using previous solution as initial guess.
        """
        import time
        
        center = [0.6, 0.3, 0.2]
        radius = 0.1
        n_points = 30
        
        trajectory = self.generate_circular_trajectory(center, radius, n_points)
        
        print(f"\n{'='*70}")
        print(f"Warm Start Comparison on Baxter (circular trajectory, {n_points} points)")
        print(f"{'='*70}")
        
        # ========== Without warm start (JAX) ==========
        jax_cold_results = []
        start = time.perf_counter()
        for target in trajectory:
            result = baxter_chain.inverse_kinematics(
                target_position=target,
                initial_position=None,  # No warm start
                backend="jax"
            )
            jax_cold_results.append(result)
        cold_time = time.perf_counter() - start
        
        # ========== With warm start (JAX) ==========
        jax_warm_results = []
        current_joints = None
        start = time.perf_counter()
        for target in trajectory:
            result = baxter_chain.inverse_kinematics(
                target_position=target,
                initial_position=current_joints,  # Warm start
                backend="jax"
            )
            jax_warm_results.append(result)
            current_joints = result
        warm_time = time.perf_counter() - start
        
        print(f"\n--- JAX+SciPy without warm start ---")
        print(f"Total time: {cold_time*1000:.2f} ms ({cold_time/n_points*1000:.2f} ms/point)")
        
        print(f"\n--- JAX+SciPy with warm start ---")
        print(f"Total time: {warm_time*1000:.2f} ms ({warm_time/n_points*1000:.2f} ms/point)")
        
        print(f"\nWarm start speedup: {cold_time/warm_time:.2f}x")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
