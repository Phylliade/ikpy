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
        """Test IK with default optimizer (adam)"""
        target = [0.1, -0.2, 0.1]
        
        ik_result = simple_chain.inverse_kinematics(
            target_position=target, 
            backend="jax",
            max_iter=500
        )
        
        # Verify the result reaches the target
        fk_result = simple_chain.forward_kinematics(ik_result, backend="jax")[:3, 3]
        error = np.linalg.norm(fk_result - target)
        
        assert error < 0.2, f"IK error too high: {error}"
    
    def test_ik_gradient_descent(self, simple_chain):
        """Test IK with gradient descent optimizer"""
        target = [0.0, -0.3, 0.2]
        
        ik_result = simple_chain.inverse_kinematics(
            target_position=target, 
            backend="jax",
            optimizer="gradient_descent",
            max_iter=1000,
            learning_rate=0.1
        )
        
        fk_result = simple_chain.forward_kinematics(ik_result, backend="jax")[:3, 3]
        error = np.linalg.norm(fk_result - target)
        
        # Gradient descent may need more iterations
        assert error < 0.3, f"IK error too high: {error}"
    
    def test_ik_adam(self, simple_chain):
        """Test IK with Adam optimizer"""
        target = [0.0, -0.3, 0.2]
        
        ik_result = simple_chain.inverse_kinematics(
            target_position=target, 
            backend="jax",
            optimizer="adam",
            max_iter=500,
            learning_rate=0.05
        )
        
        fk_result = simple_chain.forward_kinematics(ik_result, backend="jax")[:3, 3]
        error = np.linalg.norm(fk_result - target)
        
        assert error < 0.2, f"IK error too high: {error}"


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
            assert key in cache._ik_compiled, f"Missing compiled IK for {key}"
            # Check it's a CompiledFunction (no 'lower' method)
            assert not hasattr(cache._ik_compiled[key], 'lower')
        
        # The invalid combination should not exist
        assert (None, True) not in cache._ik_compiled
    
    def test_compilation_happens_once(self, simple_chain):
        """Test that compilation is done once at cache creation, not on each call"""
        import time
        
        # Force cache creation (this does the compilation)
        cache = simple_chain.jax_cache
        
        # Get references to compiled functions
        fk_compiled_ref = cache._fk_compiled
        ik_compiled_ref = cache._ik_compiled[(None, False)]
        
        # Run multiple FK and IK calls
        joints = [0.0] * len(simple_chain.links)
        target = np.eye(4)
        target[:3, 3] = [0.1, -0.2, 0.1]
        
        for _ in range(10):
            _ = cache.forward_kinematics(joints)
            _ = cache.inverse_kinematics(target, max_iter=10)
        
        # Verify the same compiled functions are still being used
        assert cache._fk_compiled is fk_compiled_ref
        assert cache._ik_compiled[(None, False)] is ik_compiled_ref
    
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
        compiled_fn = cache._ik_compiled[(None, False)]
        
        # Run IK
        target = np.eye(4)
        target[:3, 3] = [0.1, -0.2, 0.1]
        
        result = simple_chain.inverse_kinematics(
            target_position=[0.1, -0.2, 0.1],
            backend="jax",
            optimizer="gradient_descent",
            max_iter=50
        )
        
        # Verify result is valid
        assert result.shape == (len(simple_chain.links),)
        
        # The compiled function should still be the same object
        assert cache._ik_compiled[(None, False)] is compiled_fn
    
    def test_ik_orientation_modes_all_work(self, simple_chain):
        """Test that IK works for all pre-compiled orientation modes"""
        target = np.eye(4)
        target[:3, 3] = [0.1, -0.2, 0.1]
        target[:3, 0] = [1, 0, 0]  # X axis
        target[:3, 1] = [0, 1, 0]  # Y axis
        target[:3, 2] = [0, 0, 1]  # Z axis
        
        # Test each orientation mode
        for orient_mode in [None, "X", "Y", "Z", "all"]:
            result = simple_chain.inverse_kinematics(
                target_position=[0.1, -0.2, 0.1],
                target_orientation=target[:3, 0] if orient_mode == "X" else 
                                   target[:3, 1] if orient_mode == "Y" else
                                   target[:3, 2] if orient_mode == "Z" else
                                   target[:3, :3] if orient_mode == "all" else None,
                orientation_mode=orient_mode,
                backend="jax",
                optimizer="gradient_descent",
                max_iter=50
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
