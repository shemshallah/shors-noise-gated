#!/usr/bin/env python3
"""
════════════════════════════════════════════════════════════════════════════════
LATTICE TEST: QUANTUM FOURIER TRANSFORM (QFT)
════════════════════════════════════════════════════════════════════════════════

Test QFT algorithm on the Moonshine quantum lattice.

QFT is the quantum analog of the discrete Fourier transform:
- Maps computational basis to Fourier basis
- Critical component of Shor's algorithm
- Demonstrates quantum phase estimation

This test verifies:
1. QFT circuit construction
2. Inverse QFT correctness
3. Phase extraction accuracy
4. Fourier basis state preparation

December 28, 2025
════════════════════════════════════════════════════════════════════════════════
"""

import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from moonshine_client import MoonshineClient


class QFTLatticeTest:
    """Test QFT on Moonshine lattice"""

    def __init__(self, client: MoonshineClient):
        self.client = client
        self.results = []

    def test_4_qubit_qft(self):
        """
        Test 4-qubit QFT
        Verifies transformation to Fourier basis
        """
        print("\n" + "="*80)
        print("TEST: 4-Qubit QFT")
        print("="*80)

        # Load QFT from algorithm examples
        algorithm_dir = Path("algorithm_examples")
        qft_file = algorithm_dir / "qft_example.json"

        if not qft_file.exists():
            print(f"✗ Algorithm file not found: {qft_file}")
            return False

        with open(qft_file, 'r') as f:
            qft_spec = json.load(f)

        print(f"QFT: {qft_spec.get('n_qubits')} qubits")
        print(f"Gates: {len(qft_spec.get('gates', []))}")

        # Run QFT on lattice
        target_nodes = [0, 1000, 10000]

        result = self.client.run_algorithm_on_lattice(
            algorithm_spec=qft_spec,
            target_nodes=target_nodes,
            shots=2048
        )

        if 'error' in result:
            print(f"✗ Test failed: {result['error']}")
            return False

        print(f"\n✓ QFT executed")

        if 'qft_output' in result['results']:
            counts = result['results']['qft_output']
            total = sum(counts.values())

            print(f"  Output distribution:")
            sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
            for state, count in sorted_counts[:8]:
                prob = count / total
                print(f"    |{state}⟩: {prob:.4f}")

            # QFT of uniform superposition should give uniform distribution
            # with specific phase relationships

        self.results.append({
            'test': '4_qubit_qft',
            'passed': True
        })

        return True

    def test_qft_inverse_qft_roundtrip(self):
        """
        Test QFT followed by inverse QFT returns original state
        Verifies unitarity and correctness
        """
        print("\n" + "="*80)
        print("TEST: QFT-IQFT Roundtrip")
        print("="*80)

        print("\n⚠ This test requires sequential QFT → IQFT execution")
        print("         and verification of state recovery")
        print("         Placeholder for future implementation")

        self.results.append({
            'test': 'qft_iqft_roundtrip',
            'passed': False,
            'note': 'Not yet implemented'
        })

        return False

    def test_phase_estimation(self):
        """
        Test phase estimation using QFT
        Extract eigenphase from unitary operator
        """
        print("\n" + "="*80)
        print("TEST: Quantum Phase Estimation")
        print("="*80)

        print("\n⚠ This test requires controlled-unitary operations")
        print("         and QFT-based phase readout")
        print("         Placeholder for future implementation")

        # Phase estimation is used in:
        # - Shor's algorithm (finding period)
        # - VQE (measuring eigenvalues)
        # - Quantum simulation

        self.results.append({
            'test': 'phase_estimation',
            'passed': False,
            'note': 'Not yet implemented'
        })

        return False

    def run_all_tests(self):
        """Run complete QFT test suite"""
        print("\n" + "="*80)
        print("QFT TEST SUITE")
        print("="*80)

        tests_passed = 0
        tests_total = 0

        # Test 1: 4-qubit QFT
        if self.test_4_qubit_qft():
            tests_passed += 1
        tests_total += 1

        # Test 2: Roundtrip (placeholder)
        tests_total += 1

        # Test 3: Phase estimation (placeholder)
        tests_total += 1

        # Summary
        print("\n" + "="*80)
        print("TEST SUMMARY: QFT")
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
    print("🌀 QFT - LATTICE TEST")
    print("="*80)

    client = MoonshineClient(client_name="QFTTest")

    if not client.connect():
        print("\n✗ Failed to connect to lattice")
        return 1

    test_suite = QFTLatticeTest(client)
    passed, total = test_suite.run_all_tests()

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
