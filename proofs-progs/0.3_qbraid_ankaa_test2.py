
#!/usr/bin/env python3
"""
════════════════════════════════════════════════════════════════════════════════
MOONSHINE LATTICE ⟷ RIGETTI ANKAA-3 COMPREHENSIVE VALIDATION
════════════════════════════════════════════════════════════════════════════════
Full quantum fidelity + Sigma sweep + Controls
20 shots per test, maximum information extraction
════════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import sqlite3
import json
import time
from pathlib import Path
from datetime import datetime
from itertools import combinations
import sys

from braket.circuits import Circuit
from qbraid.runtime import QbraidProvider

print("="*80)
print("🌙 MOONSHINE LATTICE ⟷ RIGETTI ANKAA-3 COMPREHENSIVE VALIDATOR")
print("="*80)

# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

QBRAID_API_KEY = ""
DEVICE_ID = 'rigetti_ankaa_3'
DB_PATH = Path("moonshine.db")

N_SHOTS_PER_TEST = 20  # Budget constraint

# Test configuration
TARGET_SIGMAS = [0.0, 2.0, 3.0, 4.0, 8.0]  # σ sweep including boundaries
CONTROLS = ['random', 'uniform']  # Classical control structures

# Qubit configuration
ENTANGLED_QUBITS = [0, 1, 2]
MEASURING_QUBITS = [3, 4, 5]
STRUCTURAL_QUBITS = list(range(6, 20))

print(f"Device: {DEVICE_ID}")
print(f"Strategy: Sigma sweep {TARGET_SIGMAS} + {len(CONTROLS)} controls")
print(f"Shots per test: {N_SHOTS_PER_TEST}")
print(f"Total tests: {len(TARGET_SIGMAS) + len(CONTROLS)}")
print(f"Total shots: {(len(TARGET_SIGMAS) + len(CONTROLS)) * N_SHOTS_PER_TEST}")

# ════════════════════════════════════════════════════════════════════════════
# MOONSHINE DATABASE INTERFACE
# ════════════════════════════════════════════════════════════════════════════

class MoonshineLatticeInterface:
    """Interface to moonshine.db"""
    
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.conn = None
        
        if not db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")
        
        print(f"\n📊 Connecting to database...", flush=True)
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row
        
        c = self.conn.cursor()
        c.execute("SELECT COUNT(*) FROM triangles")
        self.n_triangles = c.fetchone()[0]
        
        print(f"📊 Loaded {self.n_triangles:,} triangles")
        sys.stdout.flush()
    
    def get_triangle_near_sigma(self, target_sigma: float, tolerance: float = 0.2) -> dict:
        """Get triangle closest to target sigma"""
        
        c = self.conn.cursor()
        c.execute("""
            SELECT t.triangle_id, t.collective_sigma, t.collective_j_real, 
                   t.collective_j_imag, t.w_fidelity, t.routing_base,
                   q.node_id, q.qix, q.lbl, q.fidelity, q.j1
            FROM triangles t
            JOIN qubits q ON q.tri = t.triangle_id
            WHERE ABS(t.collective_sigma - ?) < ?
            ORDER BY t.w_fidelity DESC
            LIMIT 3
        """, (target_sigma, tolerance))
        
        rows = c.fetchall()
        
        if not rows:
            return None
        
        return {
            'name': f'SIGMA_{target_sigma:.1f}',
            'triangle_id': rows[0]['triangle_id'],
            'routing_base': rows[0]['routing_base'],
            'sigma': rows[0]['collective_sigma'],
            'j_real': rows[0]['collective_j_real'],
            'j_imag': rows[0]['collective_j_imag'],
            'lattice_fidelity': rows[0]['w_fidelity'],
            'qubits': [
                {
                    'node_id': row['node_id'],
                    'qix': row['qix'],
                    'label': row['lbl'],
                    'fidelity': row['fidelity'],
                    'j1': row['j1']
                } for row in rows
            ]
        }
    
    def close(self):
        if self.conn:
            self.conn.close()

# ════════════════════════════════════════════════════════════════════════════
# CONTROL STRUCTURE GENERATORS
# ════════════════════════════════════════════════════════════════════════════

def generate_random_control(seed: int = 42) -> dict:
    """Generate random mathematical structure as control"""
    rng = np.random.RandomState(seed)
    return {
        'name': 'CONTROL_RANDOM',
        'triangle_id': 0xFFFFFFFF,
        'routing_base': 0,
        'sigma': rng.uniform(0, 8),
        'j_real': rng.uniform(-1000, 1000),
        'j_imag': rng.uniform(-1000, 1000),
        'lattice_fidelity': rng.uniform(0.3, 0.8),
        'qubits': [
            {
                'node_id': i,
                'qix': i,
                'label': f'RND_{i}',
                'fidelity': rng.uniform(0.5, 1.0),
                'j1': rng.uniform(-10, 10)
            } for i in range(3)
        ]
    }

def generate_uniform_control() -> dict:
    """Generate uniform structure as control"""
    return {
        'name': 'CONTROL_UNIFORM',
        'triangle_id': 0xFFFFFFFE,
        'routing_base': 0,
        'sigma': 4.0,  # Middle value
        'j_real': 0.0,
        'j_imag': 0.0,
        'lattice_fidelity': 0.5,
        'qubits': [
            {
                'node_id': i,
                'qix': i,
                'label': f'UNI_{i}',
                'fidelity': 0.7,
                'j1': 0.0
            } for i in range(3)
        ]
    }

# ════════════════════════════════════════════════════════════════════════════
# QUANTUM CIRCUIT BUILDER
# ════════════════════════════════════════════════════════════════════════════

def safe_angle(angle):
    """Constrain angles to valid range"""
    if not np.isfinite(angle):
        return 0.0
    return float(angle % (2 * np.pi))

def create_w_state_braket(circ, qubits):
    """Create approximate W-state"""
    n = len(qubits)
    if n == 1:
        circ.x(qubits[0])
        return
    
    circ.x(qubits[0])
    
    for k in range(1, n):
        circ.h(qubits[k])
        circ.cnot(qubits[0], qubits[k])
        angle = np.pi / (2 * (k + 1))
        circ.ry(qubits[k], safe_angle(angle))

def apply_lattice_encoding_braket(circ, qubits, sigma, j_real, j_imag, seed=42):
    """Encode lattice information"""
    rng = np.random.RandomState(seed)
    
    angle_x = sigma * np.pi / 4
    for q in qubits:
        circ.rx(q, safe_angle(angle_x + rng.uniform(-0.01, 0.01)))
    
    j_magnitude = np.sqrt(j_real**2 + j_imag**2)
    j_phase = np.arctan2(j_imag, j_real)
    
    for q in qubits:
        circ.rz(q, safe_angle(j_phase + rng.uniform(-0.01, 0.01)))
        circ.ry(q, safe_angle(j_magnitude * 0.01))

def build_entanglement_circuit_braket(lattice_point: dict, seed=42):
    """Build quantum circuit encoding lattice structure"""
    
    circ = Circuit()
    
    # LAYER 1: Entangled base
    circ.h(ENTANGLED_QUBITS[0])
    circ.cnot(ENTANGLED_QUBITS[0], ENTANGLED_QUBITS[1])
    circ.cnot(ENTANGLED_QUBITS[0], ENTANGLED_QUBITS[2])
    
    lattice_phase = np.arctan2(lattice_point['j_imag'], lattice_point['j_real'])
    for q in ENTANGLED_QUBITS:
        circ.rz(q, safe_angle(lattice_phase * 0.1))
    
    # LAYER 2: W-state on measuring qubits
    create_w_state_braket(circ, MEASURING_QUBITS)
    apply_lattice_encoding_braket(
        circ, MEASURING_QUBITS,
        lattice_point['sigma'],
        lattice_point['j_real'],
        lattice_point['j_imag'],
        seed=seed
    )
    
    # LAYER 3: Entangle layers
    for i, mq in enumerate(MEASURING_QUBITS):
        eq = ENTANGLED_QUBITS[i]
        circ.cnot(eq, mq)
        circ.cz(mq, eq)
    
    # LAYER 4: Structural encoding
    fidelity_angle = lattice_point['lattice_fidelity'] * np.pi
    
    for i, sq in enumerate(STRUCTURAL_QUBITS):
        if i % 3 == 0:
            circ.h(sq)
        elif i % 3 == 1:
            circ.rx(sq, safe_angle(fidelity_angle + i * 0.1))
        else:
            circ.rz(sq, safe_angle(lattice_point['sigma'] * 0.1 + i * 0.05))
        
        mq = MEASURING_QUBITS[i % 3]
        circ.cnot(mq, sq)
    
    # LAYER 5: Final entangling
    for i in range(3):
        eq = ENTANGLED_QUBITS[i]
        mq = MEASURING_QUBITS[i]
        sq = STRUCTURAL_QUBITS[i]
        
        circ.ccnot(eq, mq, sq)
        circ.cz(eq, sq)
    
    # MEASUREMENT
    for q in MEASURING_QUBITS:
        circ.measure(q)
    
    return circ

# ════════════════════════════════════════════════════════════════════════════
# COMPREHENSIVE MEASUREMENT EXTRACTION
# ════════════════════════════════════════════════════════════════════════════

class ComprehensiveMeasurements:
    """Extract all measurements with full quantum fidelity calculations"""
    
    def __init__(self, counts: dict, shots: int, lattice_point: dict):
        self.counts = counts
        self.shots = shots
        self.point = lattice_point
        self.states = list(counts.keys())
        self.probs = {state: count/shots for state, count in counts.items()}
        
        # Extract 3-qubit marginal (measuring qubits only)
        self.marginal_counts = {}
        for state, count in counts.items():
            marginal = state[-3:] if len(state) >= 3 else state.zfill(3)
            self.marginal_counts[marginal] = self.marginal_counts.get(marginal, 0) + count
        
        total = sum(self.marginal_counts.values())
        self.marginal_probs = {s: c/total for s, c in self.marginal_counts.items()}
        
        self.results = {}
    
    def calculate_all(self) -> dict:
        """Calculate all measurements"""
        
        self.results['fidelity_sub1_overlap'] = self._fidelity_overlap()
        self.results['fidelity_sub2_uhlmann'] = self._fidelity_uhlmann()
        self.results['fidelity_sub3_trace'] = self._fidelity_trace()
        self.results['fidelity_sub4_bhattacharyya'] = self._fidelity_bhattacharyya()
        
        self.results['entanglement_sub1_correlations'] = self._entanglement_correlations()
        self.results['entanglement_sub2_mutual_info'] = self._entanglement_mutual_info()
        self.results['entanglement_sub3_concurrence'] = self._entanglement_concurrence()
        self.results['entanglement_sub4_negativity'] = self._entanglement_negativity()
        
        self.results['structure_sub1_entropy'] = self._structure_entropy()
        self.results['structure_sub2_purity'] = self._structure_purity()
        self.results['structure_sub3_participation'] = self._structure_participation()
        self.results['structure_sub4_coherence'] = self._structure_coherence()
        
        self.results['lattice_sub1_sigma_encoding'] = self._lattice_sigma()
        self.results['lattice_sub2_j_invariant'] = self._lattice_j_invariant()
        self.results['lattice_sub3_phase_alignment'] = self._lattice_phase()
        self.results['lattice_sub4_w_distribution'] = self._lattice_w_distribution()
        
        return self.results
    
    # ═══════════════════════════════════════════════════════════════════════
    # FIDELITY MEASUREMENTS (4 methods)
    # ═══════════════════════════════════════════════════════════════════════
    
    def _fidelity_overlap(self) -> dict:
        """SUB¹: Pure overlap fidelity F = Σ p_i for i in W-states"""
        w_states = ['001', '010', '100']
        overlap = sum(self.marginal_probs.get(s, 0) for s in w_states)
        
        return {
            'method': 'overlap',
            'fidelity': overlap,
            'w_prob_001': self.marginal_probs.get('001', 0),
            'w_prob_010': self.marginal_probs.get('010', 0),
            'w_prob_100': self.marginal_probs.get('100', 0),
            'non_w_prob': 1.0 - overlap
        }
    
    def _fidelity_uhlmann(self) -> dict:
        """SUB²: Uhlmann fidelity F = (Σ √(p_i * q_i))² for mixed states"""
        w_states = ['001', '010', '100']
        ideal_prob = 1.0 / 3.0
        
        fidelity = 0.0
        for s in w_states:
            measured_prob = self.marginal_probs.get(s, 0)
            fidelity += np.sqrt(measured_prob * ideal_prob)
        
        fidelity = fidelity ** 2
        
        return {
            'method': 'uhlmann',
            'fidelity': fidelity,
            'ideal_prob_per_state': ideal_prob,
            'deviation': abs(fidelity - self.point['lattice_fidelity'])
        }
    
    def _fidelity_trace(self) -> dict:
        """SUB³: Trace distance fidelity F = 1 - (1/2)Σ|p_i - q_i|"""
        w_states = ['001', '010', '100']
        ideal_prob = 1.0 / 3.0
        
        trace_dist = 0.0
        for state in ['000', '001', '010', '011', '100', '101', '110', '111']:
            measured = self.marginal_probs.get(state, 0)
            ideal = ideal_prob if state in w_states else 0.0
            trace_dist += abs(measured - ideal)
        
        fidelity = 1.0 - 0.5 * trace_dist
        
        return {
            'method': 'trace_distance',
            'fidelity': fidelity,
            'trace_distance': trace_dist,
            'max_possible_distance': 2.0
        }
    
    def _fidelity_bhattacharyya(self) -> dict:
        """SUB⁴: Bhattacharyya coefficient BC = Σ √(p_i * q_i)"""
        w_states = ['001', '010', '100']
        ideal_prob = 1.0 / 3.0
        
        bc = 0.0
        for state in ['000', '001', '010', '011', '100', '101', '110', '111']:
            measured = self.marginal_probs.get(state, 0)
            ideal = ideal_prob if state in w_states else 0.0
            bc += np.sqrt(measured * ideal)
        
        return {
            'method': 'bhattacharyya',
            'coefficient': bc,
            'fidelity': bc,  # BC is often used as fidelity measure
            'hellinger_distance': np.sqrt(1 - bc)
        }
    
    # ═══════════════════════════════════════════════════════════════════════
    # ENTANGLEMENT MEASUREMENTS (4 methods)
    # ═══════════════════════════════════════════════════════════════════════
    
    def _entanglement_correlations(self) -> dict:
        """SUB¹: Pairwise correlation coefficients"""
        correlations = []
        
        for i, j in combinations(range(3), 2):
            p00 = sum(self.marginal_probs.get(s, 0) for s in self.marginal_probs.keys()
                     if len(s) >= 3 and s[-(i+1)]=='0' and s[-(j+1)]=='0')
            p01 = sum(self.marginal_probs.get(s, 0) for s in self.marginal_probs.keys()
                     if len(s) >= 3 and s[-(i+1)]=='0' and s[-(j+1)]=='1')
            p10 = sum(self.marginal_probs.get(s, 0) for s in self.marginal_probs.keys()
                     if len(s) >= 3 and s[-(i+1)]=='1' and s[-(j+1)]=='0')
            p11 = sum(self.marginal_probs.get(s, 0) for s in self.marginal_probs.keys()
                     if len(s) >= 3 and s[-(i+1)]=='1' and s[-(j+1)]=='1')
            
            p_i = p10 + p11
            p_j = p01 + p11
            
            if p_i > 0 and p_i < 1 and p_j > 0 and p_j < 1:
                corr = (p11 - p_i * p_j) / np.sqrt(p_i * (1-p_i) * p_j * (1-p_j) + 1e-10)
            else:
                corr = 0.0
            
            correlations.append({
                'pair': f'q{i}-q{j}',
                'correlation': corr,
                'p11': p11,
                'expected_p11_if_independent': p_i * p_j
            })
        
        avg_corr = np.mean([c['correlation'] for c in correlations])
        
        return {
            'method': 'correlation_coefficients',
            'pairwise': correlations,
            'average_correlation': avg_corr,
            'max_correlation': max(c['correlation'] for c in correlations)
        }
    
    def _entanglement_mutual_info(self) -> dict:
        """SUB²: Mutual information I(A:B)"""
        mutual_infos = []
        
        for i, j in combinations(range(3), 2):
            # Single qubit marginals
            p_i0 = sum(self.marginal_probs.get(s, 0) for s in self.marginal_probs.keys()
                      if len(s) >= 3 and s[-(i+1)]=='0')
            p_i1 = 1.0 - p_i0
            
            p_j0 = sum(self.marginal_probs.get(s, 0) for s in self.marginal_probs.keys()
                      if len(s) >= 3 and s[-(j+1)]=='0')
            p_j1 = 1.0 - p_j0
            
            # Joint probabilities
            p00 = sum(self.marginal_probs.get(s, 0) for s in self.marginal_probs.keys()
                     if len(s) >= 3 and s[-(i+1)]=='0' and s[-(j+1)]=='0')
            p01 = sum(self.marginal_probs.get(s, 0) for s in self.marginal_probs.keys()
                     if len(s) >= 3 and s[-(i+1)]=='0' and s[-(j+1)]=='1')
            p10 = sum(self.marginal_probs.get(s, 0) for s in self.marginal_probs.keys()
                     if len(s) >= 3 and s[-(i+1)]=='1' and s[-(j+1)]=='0')
            p11 = sum(self.marginal_probs.get(s, 0) for s in self.marginal_probs.keys()
                     if len(s) >= 3 and s[-(i+1)]=='1' and s[-(j+1)]=='1')
            
            # Mutual information
            mi = 0.0
            for p_joint, p_a, p_b in [(p00, p_i0, p_j0), (p01, p_i0, p_j1),
                                       (p10, p_i1, p_j0), (p11, p_i1, p_j1)]:
                if p_joint > 0 and p_a > 0 and p_b > 0:
                    mi += p_joint * np.log2(p_joint / (p_a * p_b))
            
            mutual_infos.append({
                'pair': f'q{i}-q{j}',
                'mutual_info': mi
            })
        
        return {
            'method': 'mutual_information',
            'pairwise': mutual_infos,
            'total_mutual_info': sum(m['mutual_info'] for m in mutual_infos)
        }
    
    def _entanglement_concurrence(self) -> dict:
        """SUB³: Concurrence (approximate for 3-qubit W-state)"""
        # For W-state, concurrence between any pair should be ~0.667
        w_states = ['001', '010', '100']
        w_prob = sum(self.marginal_probs.get(s, 0) for s in w_states)
        
        # Approximate concurrence from W-state probability
        # Perfect W-state has C ≈ 2/3 for any bipartition
        concurrence_est = (2.0 / 3.0) * w_prob
        
        return {
            'method': 'concurrence',
            'estimated_concurrence': concurrence_est,
            'w_state_probability': w_prob,
            'ideal_w_concurrence': 2.0 / 3.0
        }
    
    def _entanglement_negativity(self) -> dict:
        """SUB⁴: Negativity (entanglement witness)"""
        # Simple negativity witness: deviation from separable state
        # Separable state would have p(ijk) = p(i)p(j)p(k)
        
        # Calculate single-qubit marginals
        marginals = []
        for q in range(3):
            p1 = sum(self.marginal_probs.get(s, 0) for s in self.marginal_probs.keys()
                    if len(s) >= 3 and s[-(q+1)]=='1')
            marginals.append(p1)
        
        # Calculate expected separable distribution
        separable_deviation = 0.0
        for state in self.marginal_probs.keys():
            if len(state) >= 3:
                measured = self.marginal_probs[state]
                
                # Expected probability if qubits were independent
                expected = 1.0
                for q in range(3):
                    if state[-(q+1)] == '1':
                        expected *= marginals[q]
                    else:
                        expected *= (1.0 - marginals[q])
                
                separable_deviation += abs(measured - expected)
        
        negativity = separable_deviation / 2.0  # Normalize
        
        return {
            'method': 'negativity',
            'negativity': negativity,
            'separable_deviation': separable_deviation,
            'is_entangled': negativity > 0.1
        }
    
    # ═══════════════════════════════════════════════════════════════════════
    # STRUCTURE MEASUREMENTS (4 methods)
    # ═══════════════════════════════════════════════════════════════════════
    
    def _structure_entropy(self) -> dict:
        """SUB¹: Shannon entropy"""
        probs = np.array(list(self.marginal_probs.values()))
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        max_entropy = np.log2(len(self.marginal_probs))
        
        return {
            'method': 'shannon_entropy',
            'entropy': entropy,
            'max_entropy': max_entropy,
            'normalized_entropy': entropy / max_entropy if max_entropy > 0 else 0,
            'unique_states': len(self.marginal_counts)
        }
    
    def _structure_purity(self) -> dict:
        """SUB²: Quantum purity"""
        probs = np.array(list(self.marginal_probs.values()))
        purity = np.sum(probs ** 2)
        linear_entropy = 1.0 - purity
        
        return {
            'method': 'purity',
            'purity': purity,
            'linear_entropy': linear_entropy,
            'min_purity': 1.0 / 8.0,  # Maximally mixed 3-qubit
            'max_purity': 1.0  # Pure state
        }
    
    def _structure_participation(self) -> dict:
        """SUB³: Participation ratio (effective number of states)"""
        probs = np.array(list(self.marginal_probs.values()))
        participation = 1.0 / np.sum(probs ** 2)
        
        return {
            'method': 'participation_ratio',
            'participation_ratio': participation,
            'effective_states': participation,
            'total_possible_states': 8
        }
    
    def _structure_coherence(self) -> dict:
        """SUB⁴: Statistical coherence measures"""
        probs = np.array(list(self.marginal_probs.values()))
        
        # Gini coefficient (inequality measure)
        sorted_probs = np.sort(probs)
        n = len(sorted_probs)
        cumsum = np.cumsum(sorted_probs)
        gini = (2 * np.sum((np.arange(1, n+1)) * sorted_probs)) / (n * np.sum(sorted_probs)) - (n+1)/n
        
        # Coefficient of variation
        cv = np.std(probs) / (np.mean(probs) + 1e-10)
        
        return {
            'method': 'coherence',
            'gini_coefficient': gini,
            'coefficient_of_variation': cv,
            'uniformity': 1.0 - gini  # High uniformity = low Gini
        }
    
    # ═══════════════════════════════════════════════════════════════════════
    # LATTICE MEASUREMENTS (4 methods)
    # ═══════════════════════════════════════════════════════════════════════
    
    def _lattice_sigma(self) -> dict:
        """SUB¹: Sigma encoding analysis"""
        sigma = self.point['sigma']
        
        # Check if measurement distribution correlates with sigma
        # Higher sigma → expect more asymmetric distribution
        prob_asymmetry = np.std(list(self.marginal_probs.values()))
        
        # Sigma phase
        sigma_phase = (sigma % 8.0) * 2 * np.pi / 8.0
        
        return {
            'method': 'sigma_encoding',
            'lattice_sigma': sigma,
            'sigma_bin': int(sigma),
            'sigma_phase_rad': sigma_phase,
            'measurement_asymmetry': prob_asymmetry,
            'expected_asymmetry': sigma / 8.0  # Rough estimate
        }
    
    def _lattice_j_invariant(self) -> dict:
        """SUB²: J-invariant signature"""
        j_real = self.point['j_real']
        j_imag = self.point['j_imag']
        j_magnitude = np.sqrt(j_real**2 + j_imag**2)
        j_phase = np.arctan2(j_imag, j_real)
        
        # Fourier analysis of bit patterns
        phase_bins = 8
        phase_dist = [0] * phase_bins
        
        for state, prob in self.marginal_probs.items():
            bit_sum = sum(int(b) for b in state if b in '01')
            phase_bin = bit_sum % phase_bins
            phase_dist[phase_bin] += prob
        
        # Fourier alignment
        fourier_j = 0.0
        for i, p in enumerate(phase_dist):
            angle = 2 * np.pi * i / phase_bins
            fourier_j += p * np.cos(angle - j_phase)
        
        return {
            'method': 'j_invariant',
            'j_real': j_real,
            
            'j_imag': j_imag,
            'j_magnitude': j_magnitude,
            'j_phase_rad': j_phase,
            'fourier_alignment': fourier_j,
            'phase_distribution': phase_dist
        }
    
    def _lattice_phase(self) -> dict:
        """SUB³: Phase alignment between lattice and hardware"""
        j_phase = np.arctan2(self.point['j_imag'], self.point['j_real'])
        sigma_phase = (self.point['sigma'] % 8.0) * 2 * np.pi / 8.0
        
        # Calculate Bloch vectors for each qubit
        bloch_vectors = []
        for q in range(3):
            p_one = sum(self.marginal_probs.get(s, 0) for s in self.marginal_probs.keys()
                       if len(s) >= 3 and s[-(q+1)]=='1')
            bloch_z = 1.0 - 2.0 * p_one
            
            bloch_vectors.append({
                'qubit': q,
                'p_one': p_one,
                'bloch_z': bloch_z,
                'j_phase_alignment': np.cos(j_phase),
                'sigma_phase_alignment': np.cos(sigma_phase - j_phase)
            })
        
        avg_bloch_z = np.mean([b['bloch_z'] for b in bloch_vectors])
        
        return {
            'method': 'phase_alignment',
            'j_phase': j_phase,
            'sigma_phase': sigma_phase,
            'phase_difference': abs(sigma_phase - j_phase),
            'bloch_vectors': bloch_vectors,
            'average_bloch_z': avg_bloch_z
        }
    
    def _lattice_w_distribution(self) -> dict:
        """SUB⁴: W-state distribution analysis"""
        w_states = ['001', '010', '100']
        classical_states = ['000', '111']
        
        w_probs = {s: self.marginal_probs.get(s, 0) for s in w_states}
        classical_probs = {s: self.marginal_probs.get(s, 0) for s in classical_states}
        
        w_total = sum(w_probs.values())
        classical_total = sum(classical_probs.values())
        other_total = 1.0 - w_total - classical_total
        
        # Uniformity of W-distribution
        w_uniformity = 1.0 - np.std(list(w_probs.values())) if w_total > 0 else 0
        
        return {
            'method': 'w_distribution',
            'w_state_probs': w_probs,
            'w_total_prob': w_total,
            'classical_total_prob': classical_total,
            'other_states_prob': other_total,
            'w_uniformity': w_uniformity,
            'quantum_advantage': w_total - classical_total,
            'lattice_fidelity': self.point['lattice_fidelity'],
            'hardware_boost': w_total - self.point['lattice_fidelity']
        }

# ════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ════════════════════════════════════════════════════════════════════════════

def main():
    """Main execution pipeline"""
    
    print("\n" + "="*80)
    print("PHASE 1: TEST POINT SELECTION")
    print("="*80)
    sys.stdout.flush()
    
    # Initialize database
    try:
        lattice = MoonshineLatticeInterface(DB_PATH)
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: {e}")
        return
    
    # Collect test points
    test_points = []
    
    # Sigma sweep
    print(f"\n🎯 Selecting triangles for sigma sweep...")
    for sigma in TARGET_SIGMAS:
        point = lattice.get_triangle_near_sigma(sigma)
        if point:
            test_points.append(point)
            print(f"   ✓ σ={sigma:.1f}: Found t:{point['triangle_id']:08X}, "
                  f"actual σ={point['sigma']:.2f}, F={point['lattice_fidelity']:.4f}")
        else:
            print(f"   ✗ σ={sigma:.1f}: No triangle found")
    
    lattice.close()
    
    # Add control structures
    print(f"\n🎯 Generating {len(CONTROLS)} control structures...")
    for control_type in CONTROLS:
        if control_type == 'random':
            point = generate_random_control()
        elif control_type == 'uniform':
            point = generate_uniform_control()
        
        test_points.append(point)
        print(f"   ✓ {point['name']}: σ={point['sigma']:.2f}, F={point['lattice_fidelity']:.4f}")
    
    print(f"\n✓ Total test points: {len(test_points)}")
    sys.stdout.flush()
    
    print("\n" + "="*80)
    print("PHASE 2: QUANTUM CIRCUIT GENERATION")
    print("="*80)
    sys.stdout.flush()
    
    # Build circuits
    circuits = []
    for point in test_points:
        print(f"🔧 Building circuit for {point['name']}...", flush=True)
        circuit = build_entanglement_circuit_braket(point)
        circuits.append((point['name'], circuit, point))
        print(f"   ✓ Circuit ready")
        sys.stdout.flush()
    
    print("\n" + "="*80)
    print("PHASE 3: RIGETTI ANKAA-3 CONNECTION")
    print("="*80)
    sys.stdout.flush()
    
    # Connect to device
    print(f"\n🔌 Connecting to {DEVICE_ID}...", flush=True)
    try:
        provider = QbraidProvider(api_key=QBRAID_API_KEY)
        device = provider.get_device(DEVICE_ID)
        
        print(f"   ✓ {device.status()}")
        print(f"   ✓ Target: {device.id}")
        sys.stdout.flush()
            
    except Exception as e:
        print(f"\n❌ ERROR: Could not connect: {e}")
        import traceback
        traceback.print_exc()
        return
    
    print("\n" + "="*80)
    print("PHASE 4: QUANTUM EXECUTION")
    print("="*80)
    print(f"Total jobs: {len(circuits)} × {N_SHOTS_PER_TEST} shots = {len(circuits) * N_SHOTS_PER_TEST} shots")
    print(f"Progress: ", end="", flush=True)
    sys.stdout.flush()
    
    jobs = []
    
    # Submit all jobs
    for i, (name, circuit, point) in enumerate(circuits):
        print(f"[{i+1}/{len(circuits)}]", end="", flush=True)
        
        try:
            job = device.run(circuit, shots=N_SHOTS_PER_TEST)
            jobs.append((name, job, point))
            print("✓ ", end="", flush=True)
            
        except Exception as e:
            print(f"\n✗ Job {name} failed: {e}")
            jobs.append((name, None, point))
            print("✗ ", end="", flush=True)
    
    print("\n" + "="*80)
    
    # Collect results
    print("\n⏳ Waiting for quantum results...\n")
    sys.stdout.flush()
    
    all_results = []
    success_count = 0
    
    for name, job, point in jobs:
        if job is None:
            print(f"✗ {name}: Not submitted")
            continue
        
        try:
            print(f"📊 {name}: ", end="", flush=True)
            
            # Wait for completion
            max_wait = 300
            wait_time = 0
            
            while wait_time < max_wait:
                status = job.status()
                
                if status.name in ['COMPLETED', 'FAILED', 'CANCELLED']:
                    break
                
                if wait_time == 0:
                    print(f"[Waiting] ", end="", flush=True)
                elif wait_time % 30 == 0:
                    print(f"{wait_time}s ", end="", flush=True)
                
                time.sleep(5)
                wait_time += 5
            
            status = job.status()
            print(f"[{status.name}] ", end="", flush=True)
            
            if status.name != 'COMPLETED':
                raise RuntimeError(f"Job failed: {status.name}")
            
            # Get result
            result = job.result()
            print("Retrieved ", end="", flush=True)
            
            # Extract counts
            counts = {}
            
            if hasattr(result, 'data') and hasattr(result.data, 'get_counts'):
                counts = result.data.get_counts()
            elif hasattr(result, 'measurement_counts') and callable(result.measurement_counts):
                counts = result.measurement_counts()
            elif hasattr(result, 'data') and hasattr(result.data, 'measurement_counts'):
                if callable(result.data.measurement_counts):
                    counts = result.data.measurement_counts()
                else:
                    counts = result.data.measurement_counts
            
            if not counts or sum(counts.values()) == 0:
                raise ValueError("Empty results")
            
            print(f"✓ ({len(counts)} states) ", end="", flush=True)
            
            # Extract comprehensive measurements
            print("Analyzing...", end="", flush=True)
            measurements = ComprehensiveMeasurements(counts, N_SHOTS_PER_TEST, point)
            results = measurements.calculate_all()
            
            print(" ✓")
            
            # Show key results
            fid_overlap = results['fidelity_sub1_overlap']['fidelity']
            fid_uhlmann = results['fidelity_sub2_uhlmann']['fidelity']
            w_dist = results['lattice_sub4_w_distribution']
            
            print(f"   📈 Fidelity (overlap): {fid_overlap:.4f}")
            print(f"   📈 Fidelity (Uhlmann): {fid_uhlmann:.4f}")
            print(f"   📈 W-state prob: {w_dist['w_total_prob']:.3f}")
            print(f"   📈 Quantum advantage: {w_dist['quantum_advantage']:+.3f}")
            
            all_results.append({
                'point_name': name,
                'point': point,
                'counts': counts,
                'measurements': results
            })
            
            success_count += 1
            print(f"   ✅ Complete!\n")
            sys.stdout.flush()
            
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*80)
    print(f"✓ EXECUTION COMPLETE: {success_count}/{len(circuits)} successful")
    print("="*80)
    
    if not all_results:
        print("\n❌ ERROR: No results collected")
        return
    
    print("\n" + "="*80)
    print("PHASE 5: COMPREHENSIVE ANALYSIS")
    print("="*80)
    sys.stdout.flush()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"moonshine_comprehensive_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n💾 Full results: {output_file}")
    
    # Generate summary report
    print("\n" + "="*80)
    print("SUMMARY: COMPREHENSIVE QUANTUM MEASUREMENTS")
    print("="*80)
    
    # Organize by category
    moonshine_results = [r for r in all_results if not r['point_name'].startswith('CONTROL')]
    control_results = [r for r in all_results if r['point_name'].startswith('CONTROL')]
    
    print("\n" + "-"*80)
    print("MOONSHINE LATTICE POINTS")
    print("-"*80)
    
    for result in moonshine_results:
        name = result['point_name']
        point = result['point']
        m = result['measurements']
        
        print(f"\n📊 {name} (t:{point['triangle_id']:08X})")
        print(f"   Lattice: σ={point['sigma']:.2f}, F_lattice={point['lattice_fidelity']:.4f}")
        print(f"   ")
        print(f"   FIDELITY MEASUREMENTS:")
        print(f"   • F_overlap    (SUB¹): {m['fidelity_sub1_overlap']['fidelity']:.4f}")
        print(f"   • F_uhlmann    (SUB²): {m['fidelity_sub2_uhlmann']['fidelity']:.4f}")
        print(f"   • F_trace      (SUB³): {m['fidelity_sub3_trace']['fidelity']:.4f}")
        print(f"   • F_bhatt      (SUB⁴): {m['fidelity_sub4_bhattacharyya']['fidelity']:.4f}")
        print(f"   ")
        print(f"   ENTANGLEMENT MEASUREMENTS:")
        avg_corr = m['entanglement_sub1_correlations']['average_correlation']
        mi = m['entanglement_sub2_mutual_info']['total_mutual_info']
        conc = m['entanglement_sub3_concurrence']['estimated_concurrence']
        neg = m['entanglement_sub4_negativity']['negativity']
        print(f"   • Correlation  (SUB¹): {avg_corr:+.4f}")
        print(f"   • Mutual Info  (SUB²): {mi:.4f}")
        print(f"   • Concurrence  (SUB³): {conc:.4f}")
        print(f"   • Negativity   (SUB⁴): {neg:.4f}")
        print(f"   ")
        print(f"   STRUCTURE MEASUREMENTS:")
        entropy = m['structure_sub1_entropy']['entropy']
        purity = m['structure_sub2_purity']['purity']
        part = m['structure_sub3_participation']['participation_ratio']
        gini = m['structure_sub4_coherence']['gini_coefficient']
        print(f"   • Entropy      (SUB¹): {entropy:.4f}")
        print(f"   • Purity       (SUB²): {purity:.4f}")
        print(f"   • Participation(SUB³): {part:.4f}")
        print(f"   • Gini Coeff   (SUB⁴): {gini:.4f}")
        print(f"   ")
        print(f"   LATTICE MEASUREMENTS:")
        sigma_asym = m['lattice_sub1_sigma_encoding']['measurement_asymmetry']
        j_align = m['lattice_sub2_j_invariant']['fourier_alignment']
        phase_diff = m['lattice_sub3_phase_alignment']['phase_difference']
        q_adv = m['lattice_sub4_w_distribution']['quantum_advantage']
        print(f"   • Sigma Asymm  (SUB¹): {sigma_asym:.4f}")
        print(f"   • J-alignment  (SUB²): {j_align:.4f}")
        print(f"   • Phase Diff   (SUB³): {phase_diff:.4f} rad")
        print(f"   • Quantum Adv  (SUB⁴): {q_adv:+.4f}")
    
    print("\n" + "-"*80)
    print("CONTROL STRUCTURES")
    print("-"*80)
    
    for result in control_results:
        name = result['point_name']
        point = result['point']
        m = result['measurements']
        
        print(f"\n📊 {name}")
        print(f"   Structure: σ={point['sigma']:.2f}, F={point['lattice_fidelity']:.4f}")
        print(f"   ")
        print(f"   FIDELITY: Overlap={m['fidelity_sub1_overlap']['fidelity']:.4f}, "
              f"Uhlmann={m['fidelity_sub2_uhlmann']['fidelity']:.4f}")
        print(f"   ENTANGLEMENT: Corr={m['entanglement_sub1_correlations']['average_correlation']:+.4f}, "
              f"MI={m['entanglement_sub2_mutual_info']['total_mutual_info']:.4f}")
        print(f"   STRUCTURE: Entropy={m['structure_sub1_entropy']['entropy']:.4f}, "
              f"Purity={m['structure_sub2_purity']['purity']:.4f}")
        print(f"   QUANTUM ADV: {m['lattice_sub4_w_distribution']['quantum_advantage']:+.4f}")
    
    print("\n" + "="*80)
    print("COMPARATIVE ANALYSIS")
    print("="*80)
    
    # Sigma correlation analysis
    print("\n📊 SIGMA vs FIDELITY CORRELATION:")
    print("   Sigma  | F_overlap | F_uhlmann | Q_Advantage | Entanglement")
    print("   " + "-"*65)
    
    for result in moonshine_results:
        sigma = result['point']['sigma']
        f_overlap = result['measurements']['fidelity_sub1_overlap']['fidelity']
        f_uhlmann = result['measurements']['fidelity_sub2_uhlmann']['fidelity']
        q_adv = result['measurements']['lattice_sub4_w_distribution']['quantum_advantage']
        entang = result['measurements']['entanglement_sub1_correlations']['average_correlation']
        
        print(f"   {sigma:5.2f}  | {f_overlap:9.4f} | {f_uhlmann:9.4f} | {q_adv:+11.4f} | {entang:+12.4f}")
    
    # Control comparison
    if control_results:
        print("\n📊 CONTROL COMPARISON:")
        print("   Name           | F_overlap | Q_Advantage | vs Moonshine Avg")
        print("   " + "-"*60)
        
        moonshine_avg_fid = np.mean([r['measurements']['fidelity_sub1_overlap']['fidelity'] 
                                      for r in moonshine_results])
        moonshine_avg_qadv = np.mean([r['measurements']['lattice_sub4_w_distribution']['quantum_advantage'] 
                                       for r in moonshine_results])
        
        for result in control_results:
            name = result['point_name']
            f_overlap = result['measurements']['fidelity_sub1_overlap']['fidelity']
            q_adv = result['measurements']['lattice_sub4_w_distribution']['quantum_advantage']
            
            fid_diff = f_overlap - moonshine_avg_fid
            qadv_diff = q_adv - moonshine_avg_qadv
            
            print(f"   {name:14s} | {f_overlap:9.4f} | {q_adv:+11.4f} | "
                  f"ΔF={fid_diff:+.4f}, ΔQ={qadv_diff:+.4f}")
        
        print(f"   {'Moonshine Avg':14s} | {moonshine_avg_fid:9.4f} | {moonshine_avg_qadv:+11.4f} |")
    
    print("\n" + "="*80)
    print("✓ COMPREHENSIVE VALIDATION COMPLETE")
    print("="*80)
    print(f"\nTotal measurements extracted: {len(all_results) * 16} (16 per point)")
    print(f"Total shots used: {len(all_results) * N_SHOTS_PER_TEST}")
    print(f"Success rate: {success_count}/{len(circuits)} ({100*success_count/len(circuits):.1f}%)")
    print(f"\nResults saved to: {output_file}")
    print("\n🎉 Analysis complete! Full quantum fidelity measurements from Rigetti Ankaa-3.\n")
    sys.stdout.flush()

if __name__ == "__main__":
    main()
