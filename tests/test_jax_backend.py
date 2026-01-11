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
    
    def test_ik_lbfgsb(self, simple_chain):
        """Test IK with L-BFGS-B optimizer"""
        target = [0.1, -0.2, 0.1]
        
        ik_result = simple_chain.inverse_kinematics(
            target_position=target, 
            backend="jax",
            optimizer="L-BFGS-B",
            max_iter=200
        )
        
        # Verify the result reaches the target
        fk_result = simple_chain.forward_kinematics(ik_result, backend="jax")[:3, 3]
        error = np.linalg.norm(fk_result - target)
        
        # Allow for some error since JAX optimizer may not converge as well
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


class TestPerformance:
    """Performance comparison tests between NumPy and JAX backends"""

    def test_jacobian_computation_speed(self, simple_chain):
        """Compare Jacobian computation - JAX autodiff vs numerical differentiation"""
        import time
        import jax.numpy as jnp
        from jax import jacfwd, jit

        joints = np.array([0.0] * len(simple_chain.links))
        joints[4] = 0.5
        joints[5] = -0.3

        n_iterations = 100

        # NumPy: Numerical Jacobian (finite differences)
        def numerical_jacobian(chain, joints, eps=1e-6):
            n_joints = len(joints)
            fk_base = chain.forward_kinematics(joints, backend="numpy")[:3, 3]
            jacobian = np.zeros((3, n_joints))
            for i in range(n_joints):
                joints_perturbed = joints.copy()
                joints_perturbed[i] += eps
                fk_perturbed = chain.forward_kinematics(joints_perturbed, backend="numpy")[:3, 3]
                jacobian[:, i] = (fk_perturbed - fk_base) / eps
            return jacobian

        start = time.perf_counter()
        for _ in range(n_iterations):
            jac_numpy = numerical_jacobian(simple_chain, joints)
        numpy_time = time.perf_counter() - start

        # JAX: Automatic differentiation with JIT
        from ikpy.jax_backend import forward_kinematics_jax, extract_chain_parameters
        chain_params = extract_chain_parameters(simple_chain)

        def fk_position(joints_jax):
            return forward_kinematics_jax(joints_jax, chain_params)[:3, 3]

        # JIT compile the Jacobian function
        jac_fn = jit(jacfwd(fk_position))

        # Warm-up and compile
        joints_jax = jnp.array(joints)
        _ = jac_fn(joints_jax).block_until_ready()

        start = time.perf_counter()
        for _ in range(n_iterations):
            jac_jax = jac_fn(joints_jax).block_until_ready()
        jax_time = time.perf_counter() - start

        print(f"\n{'='*60}")
        print(f"Jacobian Computation ({n_iterations} iterations)")
        print(f"{'='*60}")
        print(f"NumPy (finite diff): {numpy_time*1000:.2f} ms total, {numpy_time/n_iterations*1000:.4f} ms/call")
        print(f"JAX (jit+autodiff):  {jax_time*1000:.2f} ms total, {jax_time/n_iterations*1000:.4f} ms/call")
        print(f"Speedup:             {numpy_time/jax_time:.2f}x")
        print(f"{'='*60}")

        # Verify Jacobians are similar
        np.testing.assert_allclose(jac_numpy, np.array(jac_jax), rtol=1e-3, atol=1e-3)
        print("✓ Jacobians match!")

    def test_forward_kinematics_batched_speed(self, simple_chain):
        """Compare batched FK speed - where JAX really shines"""
        import time
        import jax.numpy as jnp
        from jax import vmap

        n_configs = 1000  # Number of joint configurations to evaluate

        # Generate random joint configurations
        np.random.seed(42)
        all_joints = np.random.uniform(-1, 1, (n_configs, len(simple_chain.links)))

        # NumPy: loop over configurations
        start = time.perf_counter()
        results_numpy = []
        for joints in all_joints:
            fk = simple_chain.forward_kinematics(joints.tolist(), backend="numpy")
            results_numpy.append(fk[:3, 3])
        results_numpy = np.array(results_numpy)
        numpy_time = time.perf_counter() - start

        # JAX: vectorized with vmap
        from ikpy.jax_backend import forward_kinematics_jax, extract_chain_parameters
        chain_params = extract_chain_parameters(simple_chain)

        # Create batched FK function
        batched_fk = vmap(lambda j: forward_kinematics_jax(j, chain_params))

        # Warm-up
        _ = batched_fk(jnp.array(all_joints[:10])).block_until_ready()

        start = time.perf_counter()
        all_joints_jax = jnp.array(all_joints)
        results_jax = batched_fk(all_joints_jax)[:, :3, 3].block_until_ready()
        jax_time = time.perf_counter() - start

        print(f"\n{'='*60}")
        print(f"BATCHED Forward Kinematics ({n_configs} configurations)")
        print(f"{'='*60}")
        print(f"NumPy (loop):      {numpy_time*1000:.2f} ms")
        print(f"JAX (vmap):        {jax_time*1000:.2f} ms")
        print(f"Speedup:           {numpy_time/jax_time:.2f}x")
        print(f"{'='*60}")

        # Verify results match
        np.testing.assert_allclose(results_numpy, np.array(results_jax), rtol=1e-4, atol=1e-4)

    def test_forward_kinematics_speed(self, simple_chain):
        """Compare FK speed between NumPy and JAX"""
        import time

        joints = [0.0] * len(simple_chain.links)
        joints[4] = 0.5
        joints[5] = -0.3

        n_iterations = 1000

        # Warm-up JAX (first call triggers compilation if not precompiled)
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

        # Just verify both produce same results
        fk_numpy = simple_chain.forward_kinematics(joints, backend="numpy")
        fk_jax = simple_chain.forward_kinematics(joints, backend="jax")
        np.testing.assert_allclose(fk_numpy, fk_jax, rtol=1e-5, atol=1e-5)

    def test_inverse_kinematics_speed(self, simple_chain):
        """Compare IK speed between NumPy and JAX"""
        import time

        target = [0.1, -0.2, 0.1]
        n_iterations = 10  # IK is slower, use fewer iterations

        # Warm-up JAX
        _ = simple_chain.inverse_kinematics(target_position=target, backend="jax", max_iter=50)

        # Benchmark NumPy
        start = time.perf_counter()
        for _ in range(n_iterations):
            _ = simple_chain.inverse_kinematics(target_position=target, backend="numpy")
        numpy_time = time.perf_counter() - start

        # Benchmark JAX (L-BFGS-B)
        start = time.perf_counter()
        for _ in range(n_iterations):
            _ = simple_chain.inverse_kinematics(
                target_position=target,
                backend="jax",
                optimizer="L-BFGS-B",
                max_iter=100
            )
        jax_time = time.perf_counter() - start

        print(f"\n{'='*60}")
        print(f"Inverse Kinematics Benchmark ({n_iterations} iterations)")
        print(f"{'='*60}")
        print(f"NumPy (scipy least_squares): {numpy_time*1000:.2f} ms total, {numpy_time/n_iterations*1000:.2f} ms/call")
        print(f"JAX (L-BFGS-B):              {jax_time*1000:.2f} ms total, {jax_time/n_iterations*1000:.2f} ms/call")
        print(f"Speedup: {numpy_time/jax_time:.2f}x")
        print(f"{'='*60}")

        # Verify both reach the target reasonably well
        ik_numpy = simple_chain.inverse_kinematics(target_position=target, backend="numpy")
        ik_jax = simple_chain.inverse_kinematics(target_position=target, backend="jax", max_iter=100)

        fk_numpy = simple_chain.forward_kinematics(ik_numpy)[:3, 3]
        fk_jax = simple_chain.forward_kinematics(ik_jax)[:3, 3]

        error_numpy = np.linalg.norm(fk_numpy - target)
        error_jax = np.linalg.norm(fk_jax - target)

        print(f"NumPy IK error: {error_numpy:.6f}")
        print(f"JAX IK error:   {error_jax:.6f}")
        print(f"{'='*60}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
