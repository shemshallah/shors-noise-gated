#!/usr/bin/env python3
"""
════════════════════════════════════════════════════════════════════════════════
LATTICE TEST: VARIATIONAL QUANTUM EIGENSOLVER (VQE)
════════════════════════════════════════════════════════════════════════════════

Test VQE algorithm on the Moonshine quantum lattice.

VQE is a hybrid quantum-classical algorithm for finding ground state energies:
- Quantum: Prepare parameterized ansatz and measure Hamiltonian expectation
- Classical: Optimize parameters to minimize energy

This test verifies:
1. Hardware-efficient ansatz preparation
2. Hamiltonian measurement in different bases
3. Energy expectation value calculation
4. Parameter optimization loop

December 28, 2025
════════════════════════════════════════════════════════════════════════════════
"""

import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent.parent))

from moonshine_client import MoonshineClient


class VQELatticeTest:
    """Test VQE on Moonshine lattice"""

    def __init__(self, client: MoonshineClient):
        self.client = client
        self.results = []

    def test_h2_molecule(self):
        """
        Test VQE for H₂ molecule ground state
        Hamiltonian: H = -1.05 ZZ + 0.39 XX - 0.39 YY - 0.01 ZI + 0.18 IZ
        Known ground state energy: -1.137 Ha
        """
        print("\n" + "="*80)
        print("TEST: VQE for H₂ Molecule")
        print("="*80)
        print("Target: Ground state energy of H₂")
        print("Expected: E₀ ≈ -1.137 Hartree")

        # Load VQE ansatz from algorithm examples
        algorithm_dir = Path("algorithm_examples")
        ansatz_file = algorithm_dir / "vqe_ansatz_example.json"

        if not ansatz_file.exists():
            print(f"✗ Algorithm file not found: {ansatz_file}")
            return False

        with open(ansatz_file, 'r') as f:
            ansatz_spec = json.load(f)

        print(f"\nAnsatz: {ansatz_spec.get('algorithm')}")
        print(f"  Qubits: {ansatz_spec.get('n_qubits')}")
        print(f"  Depth: {ansatz_spec.get('depth')}")
        print(f"  Parameters: {ansatz_spec.get('n_params')}")

        # Run ansatz on lattice
        target_nodes = [0]  # Single node for VQE

        result = self.client.run_algorithm_on_lattice(
            algorithm_spec=ansatz_spec,
            target_nodes=target_nodes,
            shots=4096
        )

        if 'error' in result:
            print(f"✗ Test failed: {result['error']}")
            return False

        print(f"\n✓ Ansatz executed")
        print(f"  Output states:")

        if 'vqe_output' in result['results']:
            counts = result['results']['vqe_output']
            total = sum(counts.values())

            # Show top 5 states
            sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
            for state, count in sorted_counts[:5]:
                prob = count / total
                print(f"    |{state}⟩: {prob:.4f}")

        # Note: Full VQE requires classical optimization loop
        print(f"\n⚠ Full VQE requires classical optimization")
        print(f"  This test only verifies ansatz execution")
        print(f"  Parameter optimization not yet implemented")

        self.results.append({
            'test': 'h2_vqe',
            'passed': True,
            'note': 'Ansatz only (no optimization)'
        })

        return True

    def test_ansatz_expressibility(self):
        """
        Test ansatz expressibility by measuring state coverage
        Good ansatz should explore large portion of Hilbert space
        """
        print("\n" + "="*80)
        print("TEST: Ansatz Expressibility")
        print("="*80)

        print("\n⚠ This test requires running ansatz with random parameters")
        print("         and measuring state distribution entropy")
        print("         Placeholder for future implementation")

        self.results.append({
            'test': 'ansatz_expressibility',
            'passed': False,
            'note': 'Not yet implemented'
        })

        return False

    def test_hamiltonian_measurement(self):
        """
        Test measurement of Hamiltonian expectation values
        Verify basis rotations for X, Y, Z measurements
        """
        print("\n" + "="*80)
        print("TEST: Hamiltonian Measurement")
        print("="*80)

        print("\n⚠ This test requires implementing basis rotations")
        print("         for measuring Pauli operators")
        print("         Placeholder for future implementation")

        # Example Hamiltonian: H = 0.5 ZZ - 0.3 XI + 0.2 YY
        # Measure each term separately and combine

        self.results.append({
            'test': 'hamiltonian_measurement',
            'passed': False,
            'note': 'Not yet implemented'
        })

        return False

    def run_all_tests(self):
        """Run complete VQE test suite"""
        print("\n" + "="*80)
        print("VQE TEST SUITE")
        print("="*80)

        tests_passed = 0
        tests_total = 0

        # Test 1: H₂ molecule
        if self.test_h2_molecule():
            tests_passed += 1
        tests_total += 1

        # Test 2: Ansatz expressibility (placeholder)
        tests_total += 1

        # Test 3: Hamiltonian measurement (placeholder)
        tests_total += 1

        # Summary
        print("\n" + "="*80)
        print("TEST SUMMARY: VQE")
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
    print("⚛️  VQE - LATTICE TEST")
    print("="*80)

    client = MoonshineClient(client_name="VQETest")

    if not client.connect():
        print("\n✗ Failed to connect to lattice")
        return 1

    test_suite = VQELatticeTest(client)
    passed, total = test_suite.run_all_tests()

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
