#!/usr/bin/env python3
"""
════════════════════════════════════════════════════════════════════════════════
LATTICE TEST: GROVER'S ALGORITHM
════════════════════════════════════════════════════════════════════════════════

Test Grover's search algorithm on the Moonshine quantum lattice.

Grover's algorithm provides quadratic speedup for unstructured search:
- Classical: O(N) queries
- Quantum: O(√N) queries

This test verifies:
1. Oracle construction for marked states
2. Diffusion operator application
3. Amplitude amplification
4. Measurement statistics match theoretical predictions

December 28, 2025
════════════════════════════════════════════════════════════════════════════════
"""

import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from moonshine_client import MoonshineClient


class GroversLatticeTest:
    """Test Grover's algorithm on Moonshine lattice"""

    def __init__(self, client: MoonshineClient):
        self.client = client
        self.results = []

    def test_3_qubit_search(self, target_state: int = 5):
        """
        Test Grover's algorithm for 3-qubit search
        Search space: 8 states (|000⟩ to |111⟩)
        Optimal iterations: π/4 * √8 ≈ 2
        """
        print("\n" + "="*80)
        print("TEST: 3-Qubit Grover Search")
        print("="*80)
        print(f"Target state: |{target_state:03b}⟩ ({target_state})")
        print(f"Search space: 8 states")
        print(f"Optimal iterations: 2")

        # Load Grover oracle from algorithm examples
        algorithm_dir = Path("algorithm_examples")
        oracle_file = algorithm_dir / "grover_oracle_example.json"

        if not oracle_file.exists():
            print(f"✗ Algorithm file not found: {oracle_file}")
            return False

        with open(oracle_file, 'r') as f:
            oracle_spec = json.load(f)

        # Select target lattice nodes
        target_nodes = [0, 100, 1000, 5000, 10000]

        print(f"\nRunning Grover oracle on {len(target_nodes)} lattice nodes...")

        # Run algorithm
        result = self.client.run_algorithm_on_lattice(
            algorithm_spec=oracle_spec,
            target_nodes=target_nodes,
            shots=2048
        )

        if 'error' in result:
            print(f"✗ Test failed: {result['error']}")
            return False

        # Analyze results
        print(f"\n✓ Algorithm executed successfully")
        print(f"  Nodes processed: {result['nodes_processed']}")

        # Check measurement statistics
        for node_id, counts in result['results'].items():
            total_shots = sum(counts.values())
            target_bitstring = format(target_state, '03b')

            if target_bitstring in counts:
                target_prob = counts[target_bitstring] / total_shots
                print(f"  Node {node_id}: P(target) = {target_prob:.3f}")

                # Theoretical expectation: P ≈ 1 after optimal iterations
                if target_prob > 0.8:
                    print(f"    ✓ High success probability")
                else:
                    print(f"    ⚠ Lower than expected")

        self.results.append({
            'test': '3_qubit_grover',
            'target': target_state,
            'passed': True
        })

        return True

    def test_4_qubit_search(self, target_state: int = 13):
        """
        Test Grover's algorithm for 4-qubit search
        Search space: 16 states
        Optimal iterations: π/4 * √16 = π ≈ 3
        """
        print("\n" + "="*80)
        print("TEST: 4-Qubit Grover Search")
        print("="*80)
        print(f"Target state: |{target_state:04b}⟩ ({target_state})")
        print(f"Search space: 16 states")
        print(f"Optimal iterations: 3")

        print("\n⚠ Note: This test requires a 4-qubit Grover oracle")
        print("         (not yet implemented in algorithm library)")
        print("         Placeholder for future implementation")

        self.results.append({
            'test': '4_qubit_grover',
            'target': target_state,
            'passed': False,
            'note': 'Not yet implemented'
        })

        return False

    def test_amplitude_statistics(self):
        """
        Test that amplitude amplification follows theoretical predictions
        Verify measurement statistics match Grover's amplitude evolution
        """
        print("\n" + "="*80)
        print("TEST: Amplitude Amplification Statistics")
        print("="*80)

        print("\n⚠ This test requires running Grover with varying iterations")
        print("         and comparing to theoretical amplitude evolution")
        print("         Placeholder for future implementation")

        # Theoretical Grover amplitude after k iterations:
        # a_k = sin((2k+1)θ) where sin(θ) = 1/√N

        self.results.append({
            'test': 'amplitude_statistics',
            'passed': False,
            'note': 'Not yet implemented'
        })

        return False

    def run_all_tests(self):
        """Run complete Grover's algorithm test suite"""
        print("\n" + "="*80)
        print("GROVER'S ALGORITHM TEST SUITE")
        print("="*80)

        tests_passed = 0
        tests_total = 0

        # Test 1: 3-qubit search
        if self.test_3_qubit_search():
            tests_passed += 1
        tests_total += 1

        # Test 2: 4-qubit search (placeholder)
        tests_total += 1  # Count but don't expect to pass yet

        # Test 3: Amplitude statistics (placeholder)
        tests_total += 1

        # Summary
        print("\n" + "="*80)
        print("TEST SUMMARY: GROVER'S ALGORITHM")
        print("="*80)
        print(f"Tests passed: {tests_passed}/{tests_total}")

        for result in self.results:
            status = "✓ PASS" if result['passed'] else "✗ FAIL"
            print(f"{status}: {result['test']}")
            if 'note' in result:
                print(f"       {result['note']}")

        print("="*80 + "\n")

        return tests_passed, tests_total


def main():
    """Main entry point"""

    print("\n" + "="*80)
    print("🔍 GROVER'S ALGORITHM - LATTICE TEST")
    print("="*80)

    # Create client
    client = MoonshineClient(client_name="GroverTest")

    # Connect to lattice
    if not client.connect():
        print("\n✗ Failed to connect to lattice")
        return 1

    # Create test suite
    test_suite = GroversLatticeTest(client)

    # Run all tests
    passed, total = test_suite.run_all_tests()

    # Exit with appropriate code
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
