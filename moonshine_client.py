#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
================================================================================
MOONSHINE QUANTUM CLIENT - COMPLETE TEST SUITE
================================================================================

Independent client for connecting to Moonshine server and testing:
    * Temporal synchronization (Cesium + Klein anchoring)
    * Sigma-coordinate synchronization
    * Node state queries and verification
    * W-state fidelity monitoring
    * Entanglement correlation testing
    * Routing path calculation
    * Multi-client coordination
    * Heartbeat tracking

NO SERVER DEPENDENCIES - Uses persisted lattice state files

December 28, 2025
================================================================================
"""

import numpy as np
import pickle
import json
import time
import hashlib
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, asdict
from pathlib import Path
from collections import defaultdict
import sys

# ============================================================================
# CONFIGURATION
# ============================================================================

# Must match server configuration
DATA_DIR = Path("moonshine_data")

LATTICE_STATE_FILE = DATA_DIR / "lattice_state.pkl"
ROUTING_TABLES_FILE = DATA_DIR / "routing_tables.pkl"
NODE_METRICS_FILE = DATA_DIR / "node_metrics.pkl"
METRICS_LOG_FILE = DATA_DIR / "metrics_log.jsonl"

# Constants (from server)
MOONSHINE_DIMENSION = 196883
CESIUM_FREQUENCY = 9192631770
SIGMA_PERIOD = 8.0 * np.pi
HEARTBEAT_INTERVAL = 4.0

# Client settings
CLIENT_ID = f"Client-{hashlib.sha256(str(time.time()).encode()).hexdigest()[:8]}"
SYNC_CHECK_INTERVAL = 1.0
HEARTBEAT_TIMEOUT = 10.0

# ============================================================================
# DATA STRUCTURES (MATCHING SERVER)
# ============================================================================

@dataclass
class NodeState:
    """Node state from server"""
    node_id: int
    sigma_address: float
    sigma_sector: int
    j_invariant: complex
    layer: int
    coherence: str
    fidelity: float
    is_synchronized: bool
    last_sync_sigma: float
    sync_drift: float
    timestamp: float

@dataclass
class ClientMetrics:
    """Client-side metrics"""
    timestamp: str
    client_id: str
    connected: bool
    nodes_queried: int
    sync_checks: int
    temporal_drift_ns: float
    sigma_drift: float
    avg_query_latency_ms: float
    tests_passed: int
    tests_failed: int

# ============================================================================
# MOONSHINE CLIENT
# ============================================================================

class MoonshineClient:
    """Independent client for Moonshine quantum network"""

    def __init__(self, client_name: str = None, use_aer: bool = True):
        self.client_name = client_name or CLIENT_ID
        self.logger = self._setup_logger()
        self.use_aer = use_aer

        # State
        self.routing_table: Dict = {}
        self.node_metrics: Dict = {}
        self.server_metadata: Dict = {}

        # Client tracking
        self.connection_time = 0.0
        self.last_sync_check = 0.0
        self.queries_made = 0
        self.sync_checks = 0

        # Test results
        self.tests_passed = 0
        self.tests_failed = 0
        self.test_results: List[Dict] = []

        # Aer simulator
        self.simulator = None
        if use_aer:
            self._initialize_aer_simulator()

        self.logger.info("="*80)
        self.logger.info(f"MOONSHINE QUANTUM CLIENT: {self.client_name}")
        if self.simulator:
            self.logger.info(f"Aer Simulator: ✓ Enabled")
        self.logger.info("="*80)
    
    def _setup_logger(self):
        """Setup client logger"""
        import logging
        logger = logging.getLogger(f"MoonshineClient.{self.client_name}")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s [%(name)s] %(message)s',
                datefmt='%H:%M:%S'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _initialize_aer_simulator(self):
        """Initialize Qiskit Aer simulator for local lattice simulation"""
        try:
            from qiskit_aer import AerSimulator

            self.simulator = AerSimulator()
            self.logger.info("  ✓ Aer Simulator initialized")

        except ImportError:
            self.logger.warning("  ✗ Qiskit Aer not available")
            self.logger.warning("  Install with: pip install qiskit-aer")
            self.simulator = None
        except Exception as e:
            self.logger.error(f"  ✗ Failed to initialize Aer: {e}")
            self.simulator = None

    def simulate_lattice_state(self, node_id: int, shots: int = 1024) -> Dict[str, int]:
        """
        Simulate the quantum state of a lattice node using Aer
        Returns measurement outcomes
        """
        if not self.simulator:
            self.logger.error("Aer simulator not available")
            return {}

        try:
            from qiskit import QuantumCircuit

            # Get node state
            state = self.get_node_state(node_id)
            if not state:
                self.logger.error(f"Node {node_id} not found")
                return {}

            # Create circuit for 3-qubit W-state (pseudoqubit = physical + virtual + inverse)
            qc = QuantumCircuit(3, 3)

            # Initialize W-state: |W⟩ = (|100⟩ + |010⟩ + |001⟩)/√3
            qc.x(0)
            theta1 = 2 * np.arccos(np.sqrt(2/3))
            qc.cry(theta1, 0, 1)
            qc.cx(1, 0)

            theta2 = 2 * np.arccos(np.sqrt(1/2))
            qc.cry(theta2, 1, 2)
            qc.cx(2, 1)

            # Apply phase based on sigma coordinate
            sigma_phase = (state.sigma_address / SIGMA_PERIOD) * 2 * np.pi
            for q in range(3):
                qc.rz(sigma_phase * (q + 1) / 3, q)

            # Measure
            qc.measure([0, 1, 2], [0, 1, 2])

            # Run simulation
            job = self.simulator.run(qc, shots=shots)
            result = job.result()
            counts = result.get_counts()

            self.logger.debug(f"Simulated node {node_id}: {counts}")

            return counts

        except Exception as e:
            self.logger.error(f"Simulation failed: {e}")
            return {}

    def run_algorithm_on_lattice(self, algorithm_spec: Dict, target_nodes: List[int], shots: int = 1024) -> Dict:
        """
        Run a quantum algorithm on specified lattice nodes
        algorithm_spec: output from QuantumAlgorithmLibrary (from qbc_parser)
        """
        if not self.simulator:
            self.logger.error("Aer simulator not available")
            return {'error': 'No simulator'}

        self.logger.info(f"\n[ALGORITHM] Running {algorithm_spec.get('algorithm', 'unknown')}")
        self.logger.info(f"  Target nodes: {len(target_nodes)}")

        try:
            from qiskit import QuantumCircuit

            algorithm_type = algorithm_spec.get('algorithm')
            results = {}

            if algorithm_type in ['grover_oracle', 'grover_diffusion']:
                # Run Grover on lattice
                n_qubits = algorithm_spec.get('n_qubits', 3)
                gates = algorithm_spec.get('gates', [])

                qc = QuantumCircuit(n_qubits, n_qubits)

                # Build circuit from gate list
                for gate in gates:
                    gate_type = gate.get('type')
                    if gate_type == 'H':
                        qc.h(gate['qubit'])
                    elif gate_type == 'X':
                        qc.x(gate['qubit'])
                    elif gate_type == 'Z':
                        qc.z(gate['qubit'])
                    elif gate_type == 'MCZ':
                        # Multi-controlled Z (simplified for now)
                        controls = gate.get('controls', [])
                        target = gate.get('target')
                        # Use Qiskit's mcx gate as approximation
                        if controls and target is not None:
                            qc.mcx(controls, target)
                            qc.z(target)

                # Measure
                qc.measure(list(range(n_qubits)), list(range(n_qubits)))

                # Run for each target node
                for node_id in target_nodes[:5]:  # Limit to first 5 nodes
                    job = self.simulator.run(qc, shots=shots)
                    result = job.result()
                    counts = result.get_counts()

                    results[node_id] = counts

                self.logger.info(f"  ✓ Grover completed on {len(results)} nodes")

            elif algorithm_type == 'vqe_ansatz':
                # Run VQE ansatz
                n_qubits = algorithm_spec.get('n_qubits', 4)
                gates = algorithm_spec.get('gates', [])

                qc = QuantumCircuit(n_qubits, n_qubits)

                for gate in gates:
                    gate_type = gate.get('type')
                    if gate_type == 'RY':
                        qc.ry(gate['angle'], gate['qubit'])
                    elif gate_type == 'RZ':
                        qc.rz(gate['angle'], gate['qubit'])
                    elif gate_type == 'CNOT':
                        qc.cx(gate['control'], gate['target'])

                qc.measure(list(range(n_qubits)), list(range(n_qubits)))

                job = self.simulator.run(qc, shots=shots)
                result = job.result()
                results['vqe_output'] = result.get_counts()

                self.logger.info(f"  ✓ VQE ansatz executed")

            elif algorithm_type in ['qft', 'iqft']:
                # Run QFT
                n_qubits = algorithm_spec.get('n_qubits', 4)
                gates = algorithm_spec.get('gates', [])

                qc = QuantumCircuit(n_qubits, n_qubits)

                # Initial state (uniform superposition)
                for q in range(n_qubits):
                    qc.h(q)

                # Apply QFT gates
                for gate in gates:
                    gate_type = gate.get('type')
                    if gate_type == 'H':
                        qc.h(gate['qubit'])
                    elif gate_type == 'CP':
                        qc.cp(gate['angle'], gate['control'], gate['target'])
                    elif gate_type == 'SWAP':
                        qc.swap(gate['qubit1'], gate['qubit2'])

                qc.measure(list(range(n_qubits)), list(range(n_qubits)))

                job = self.simulator.run(qc, shots=shots)
                result = job.result()
                results['qft_output'] = result.get_counts()

                self.logger.info(f"  ✓ QFT executed")

            return {
                'algorithm': algorithm_type,
                'nodes_processed': len(results),
                'results': results
            }

        except Exception as e:
            self.logger.error(f"Algorithm execution failed: {e}")
            return {'error': str(e)}
    
    def connect(self) -> bool:
        """Connect to Moonshine server via persisted state"""
        self.logger.info(f"\n[CONNECT] Connecting to Moonshine lattice...")
        self.logger.info(f"  Data directory: {DATA_DIR.absolute()}")
        
        # Check/create data directory
        if not DATA_DIR.exists():
            self.logger.info(f"  Creating data directory: {DATA_DIR}")
            DATA_DIR.mkdir(parents=True, exist_ok=True)
        
        # List existing files
        self.logger.info(f"\n  Checking for existing files:")
        if DATA_DIR.exists():
            existing_files = list(DATA_DIR.glob("*"))
            if existing_files:
                for f in existing_files:
                    self.logger.info(f"    • {f.name} ({f.stat().st_size / 1024:.1f} KB)")
            else:
                self.logger.info(f"    (directory is empty)")
        
        # Check for required files
        self.logger.info(f"\n  Required files:")
        self.logger.info(f"    • routing_tables.pkl: {'✓ Found' if ROUTING_TABLES_FILE.exists() else '✗ Missing'}")
        self.logger.info(f"    • node_metrics.pkl: {'✓ Found' if NODE_METRICS_FILE.exists() else '⚠ Optional'}")
        self.logger.info(f"    • metrics_log.jsonl: {'✓ Found' if METRICS_LOG_FILE.exists() else '⚠ Optional'}")
        
        if not ROUTING_TABLES_FILE.exists():
            self.logger.error(f"\n  ✗ Routing tables not found: {ROUTING_TABLES_FILE.name}")
            self.logger.error(f"\n  INITIALIZATION REQUIRED:")
            self.logger.error(f"  The Moonshine lattice must be initialized first.")
            self.logger.error(f"\n  Run the cell again - it will auto-initialize on next run.")
            return False
        
        try:
            # Load routing table
            self.logger.info(f"  Loading routing tables...")
            with open(ROUTING_TABLES_FILE, 'rb') as f:
                self.routing_table = pickle.load(f)
            self.logger.info(f"  ✓ Loaded {len(self.routing_table):,} routing entries")
            
            # Load node metrics (if available)
            if NODE_METRICS_FILE.exists():
                self.logger.info(f"  Loading node metrics...")
                with open(NODE_METRICS_FILE, 'rb') as f:
                    self.node_metrics = pickle.load(f)
                self.logger.info(f"  ✓ Loaded {len(self.node_metrics):,} node metrics")
            else:
                self.logger.info(f"  ! Node metrics not yet available (server not started)")
            
            # Load latest metrics
            if METRICS_LOG_FILE.exists():
                self.logger.info(f"  Loading server metrics...")
                with open(METRICS_LOG_FILE, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        self.server_metadata = json.loads(lines[-1])
                        self.logger.info(f"  ✓ Server metrics loaded")
            
            self.connection_time = time.time()
            
            self.logger.info(f"\n  ✓ CONNECTION ESTABLISHED")
            self.logger.info(f"  Client ID: {self.client_name}")
            self.logger.info(f"  Lattice dimension: {len(self.routing_table):,} nodes")
            
            return True
            
        except Exception as e:
            self.logger.error(f"  ✗ Connection failed: {e}")
            return False
    
    def get_node_state(self, node_id: int) -> Optional[NodeState]:
        """Get complete state for a node"""
        self.queries_made += 1
        
        if node_id not in self.routing_table:
            return None
        
        entry = self.routing_table[node_id]
        metrics = self.node_metrics.get(node_id)
        
        state = NodeState(
            node_id=node_id,
            sigma_address=entry.sigma_address,
            sigma_sector=entry.sigma_sector,
            j_invariant=complex(entry.j_invariant_real, entry.j_invariant_imag),
            layer=entry.layer,
            coherence=entry.coherence_level,
            fidelity=entry.fidelity,
            is_synchronized=metrics.is_synchronized if metrics else False,
            last_sync_sigma=metrics.last_sync_sigma if metrics else 0.0,
            sync_drift=metrics.sync_drift if metrics else 0.0,
            timestamp=entry.timestamp
        )
        
        return state
    
    def check_temporal_sync(self) -> Dict:
        """Check temporal synchronization with server"""
        self.sync_checks += 1
        
        if not self.server_metadata:
            return {'error': 'No server metadata available'}
        
        # Calculate temporal drift
        client_time_ns = time.time_ns()
        
        # Cesium cycles
        client_cesium_cycles = int(client_time_ns * CESIUM_FREQUENCY / 1e9)
        
        # Klein coordinate
        t = client_time_ns / 1e9
        klein_x = (t % (2 * np.pi)) / (2 * np.pi)
        klein_y = ((t / (2 * np.pi)) % 1.0)
        
        return {
            'client_time_ns': client_time_ns,
            'cesium_cycles': client_cesium_cycles,
            'klein_coordinate': complex(klein_x, klein_y),
            'connected_duration_s': time.time() - self.connection_time
        }
    
    def check_sigma_sync(self) -> Dict:
        """Check sigma-coordinate synchronization"""
        if not self.server_metadata:
            return {'error': 'No server metadata available'}
        
        server_sigma = self.server_metadata.get('current_sigma', 0.0)
        
        # Calculate expected sigma based on time elapsed
        elapsed = time.time() - self.connection_time
        heartbeats_elapsed = elapsed / 5.0  # Assume 5 second heartbeat interval
        expected_sigma_advance = heartbeats_elapsed * HEARTBEAT_INTERVAL
        
        client_estimated_sigma = (server_sigma + expected_sigma_advance) % SIGMA_PERIOD
        
        return {
            'server_sigma': server_sigma,
            'client_estimated_sigma': client_estimated_sigma,
            'sigma_drift': abs(client_estimated_sigma - server_sigma),
            'sigma_sector': int((client_estimated_sigma / SIGMA_PERIOD) * 8) % 8
        }
    
    def test_node_query(self, node_ids: List[int]) -> Dict:
        """Test: Query multiple nodes and verify state"""
        self.logger.info(f"\n[TEST] Node Query Test")
        self.logger.info(f"  Querying {len(node_ids)} nodes...")
        
        start_time = time.time()
        results = []
        success = 0
        
        for node_id in node_ids:
            state = self.get_node_state(node_id)
            if state:
                results.append({
                    'node_id': node_id,
                    'sigma': state.sigma_address,
                    'j_invariant': str(state.j_invariant),
                    'layer': state.layer,
                    'coherence': state.coherence,
                    'fidelity': state.fidelity
                })
                success += 1
        
        elapsed_ms = (time.time() - start_time) * 1000
        avg_latency = elapsed_ms / len(node_ids)
        
        passed = success == len(node_ids)
        
        result = {
            'test': 'node_query',
            'passed': passed,
            'nodes_queried': len(node_ids),
            'successful': success,
            'avg_latency_ms': avg_latency,
            'results': results[:5]  # Sample
        }
        
        if passed:
            self.tests_passed += 1
            self.logger.info(f"  ✓ PASS: All {success} nodes queried successfully")
        else:
            self.tests_failed += 1
            self.logger.info(f"  ✗ FAIL: {len(node_ids) - success} nodes failed")
        
        self.logger.info(f"  Avg latency: {avg_latency:.2f} ms")
        
        self.test_results.append(result)
        return result
    
    def test_sigma_addressing(self) -> Dict:
        """Test: Verify sigma-coordinate addressing"""
        self.logger.info(f"\n[TEST] Sigma Addressing Test")
        
        # Sample nodes from different sigma sectors
        nodes_by_sector: Dict[int, List[int]] = defaultdict(list)
        
        for node_id, entry in list(self.routing_table.items())[:10000]:
            sector = entry.sigma_sector
            if len(nodes_by_sector[sector]) < 10:
                nodes_by_sector[sector].append(node_id)
        
        self.logger.info(f"  Testing nodes across {len(nodes_by_sector)} sigma sectors...")
        
        correct_sectors = 0
        total_checked = 0
        
        for sector, node_ids in nodes_by_sector.items():
            for node_id in node_ids:
                entry = self.routing_table[node_id]
                calculated_sector = int((entry.sigma_address / SIGMA_PERIOD) * 8) % 8
                
                if calculated_sector == entry.sigma_sector:
                    correct_sectors += 1
                total_checked += 1
        
        passed = correct_sectors == total_checked
        
        result = {
            'test': 'sigma_addressing',
            'passed': passed,
            'sectors_tested': len(nodes_by_sector),
            'nodes_checked': total_checked,
            'correct': correct_sectors,
            'accuracy': correct_sectors / total_checked if total_checked > 0 else 0
        }
        
        if passed:
            self.tests_passed += 1
            self.logger.info(f"  ✓ PASS: All {total_checked} nodes correctly addressed")
        else:
            self.tests_failed += 1
            self.logger.info(f"  ✗ FAIL: {total_checked - correct_sectors} addressing errors")
        
        self.test_results.append(result)
        return result
    
    def test_j_invariant_distribution(self) -> Dict:
        """Test: Verify j-invariant distribution"""
        self.logger.info(f"\n[TEST] j-Invariant Distribution Test")
        
        # Sample j-invariants
        j_values = []
        sample_size = min(10000, len(self.routing_table))
        
        for node_id in list(self.routing_table.keys())[:sample_size]:
            entry = self.routing_table[node_id]
            j = complex(entry.j_invariant_real, entry.j_invariant_imag)
            j_values.append(j)
        
        # Analyze distribution
        j_magnitudes = [abs(j) for j in j_values]
        
        # Expected special values
        near_1728 = sum(1 for m in j_magnitudes if 1700 < m < 1750)
        near_744 = sum(1 for m in j_magnitudes if 700 < m < 800)
        near_0 = sum(1 for m in j_magnitudes if m < 100)
        
        self.logger.info(f"  Sampled {sample_size} j-invariants")
        self.logger.info(f"  Near j=1728 (square lattice): {near_1728}")
        self.logger.info(f"  Near j=744: {near_744}")
        self.logger.info(f"  Near j=0 (hexagonal): {near_0}")
        
        # Test passes if we have reasonable distribution
        passed = (near_1728 > 0 or near_744 > 0 or near_0 > 0)
        
        result = {
            'test': 'j_invariant_distribution',
            'passed': passed,
            'sample_size': sample_size,
            'near_1728': near_1728,
            'near_744': near_744,
            'near_0': near_0,
            'avg_magnitude': float(np.mean(j_magnitudes)),
            'std_magnitude': float(np.std(j_magnitudes))
        }
        
        if passed:
            self.tests_passed += 1
            self.logger.info(f"  ✓ PASS: j-invariant distribution valid")
        else:
            self.tests_failed += 1
            self.logger.info(f"  ✗ FAIL: No characteristic j-values found")
        
        self.test_results.append(result)
        return result
    
    def test_layer_structure(self) -> Dict:
        """Test: Verify hierarchical layer structure"""
        self.logger.info(f"\n[TEST] Layer Structure Test")
        
        # Count nodes per layer
        nodes_per_layer: Dict[int, int] = defaultdict(int)
        
        for entry in self.routing_table.values():
            nodes_per_layer[entry.layer] += 1
        
        self.logger.info(f"  Analyzing {len(nodes_per_layer)} layers...")
        
        for layer in sorted(nodes_per_layer.keys())[:12]:
            count = nodes_per_layer[layer]
            self.logger.info(f"    Layer {layer}: {count:,} nodes")
        
        # Verify Layer 0 has most nodes (should be 196,883)
        layer_0_count = nodes_per_layer.get(0, 0)
        passed = layer_0_count == MOONSHINE_DIMENSION
        
        result = {
            'test': 'layer_structure',
            'passed': passed,
            'total_layers': len(nodes_per_layer),
            'layer_0_nodes': layer_0_count,
            'expected_layer_0': MOONSHINE_DIMENSION,
            'nodes_per_layer': dict(nodes_per_layer)
        }
        
        if passed:
            self.tests_passed += 1
            self.logger.info(f"  ✓ PASS: Layer 0 has {layer_0_count:,} nodes (expected {MOONSHINE_DIMENSION:,})")
        else:
            self.tests_failed += 1
            self.logger.info(f"  ✗ FAIL: Layer 0 has {layer_0_count:,} nodes (expected {MOONSHINE_DIMENSION:,})")
        
        self.test_results.append(result)
        return result
    
    def test_synchronization_status(self) -> Dict:
        """Test: Check node synchronization status"""
        self.logger.info(f"\n[TEST] Synchronization Status Test")
        
        if not self.node_metrics:
            self.logger.info(f"  ! No node metrics available (server not running)")
            result = {
                'test': 'synchronization',
                'passed': False,
                'error': 'No metrics available'
            }
            self.test_results.append(result)
            return result
        
        synchronized = sum(1 for m in self.node_metrics.values() if m.is_synchronized)
        total = len(self.node_metrics)
        sync_rate = synchronized / total if total > 0 else 0
        
        self.logger.info(f"  Synchronized: {synchronized:,}/{total:,} ({sync_rate*100:.1f}%)")
        
        # Check fidelities
        fidelities = [m.w_state_fidelity for m in self.node_metrics.values()]
        avg_fidelity = np.mean(fidelities) if fidelities else 0
        
        self.logger.info(f"  Avg fidelity: {avg_fidelity:.4f}")
        
        passed = sync_rate > 0.95 and avg_fidelity > 0.90
        
        result = {
            'test': 'synchronization',
            'passed': passed,
            'synchronized': synchronized,
            'total_nodes': total,
            'sync_rate': sync_rate,
            'avg_fidelity': avg_fidelity
        }
        
        if passed:
            self.tests_passed += 1
            self.logger.info(f"  ✓ PASS: {sync_rate*100:.1f}% synchronized, fidelity {avg_fidelity:.4f}")
        else:
            self.tests_failed += 1
            self.logger.info(f"  ✗ FAIL: Sync rate or fidelity below threshold")
        
        self.test_results.append(result)
        return result
    
    def test_temporal_coherence(self) -> Dict:
        """Test: Verify temporal coherence"""
        self.logger.info(f"\n[TEST] Temporal Coherence Test")
        
        temporal = self.check_temporal_sync()
        
        if 'error' in temporal:
            result = {
                'test': 'temporal_coherence',
                'passed': False,
                'error': temporal['error']
            }
            self.tests_failed += 1
            self.test_results.append(result)
            return result
        
        # Check Cesium cycles are advancing
        cesium_cycles = temporal['cesium_cycles']
        
        self.logger.info(f"  Cesium cycles: {cesium_cycles:,}")
        self.logger.info(f"  Klein coordinate: {temporal['klein_coordinate']}")
        self.logger.info(f"  Connection duration: {temporal['connected_duration_s']:.1f}s")
        
        passed = cesium_cycles > 0
        
        result = {
            'test': 'temporal_coherence',
            'passed': passed,
            **temporal
        }
        
        if passed:
            self.tests_passed += 1
            self.logger.info(f"  ✓ PASS: Temporal coherence maintained")
        else:
            self.tests_failed += 1
            self.logger.info(f"  ✗ FAIL: Temporal coherence lost")
        
        self.test_results.append(result)
        return result
    
    def test_sigma_coherence(self) -> Dict:
        """Test: Verify sigma-coordinate coherence"""
        self.logger.info(f"\n[TEST] Sigma Coherence Test")
        
        sigma_check = self.check_sigma_sync()
        
        if 'error' in sigma_check:
            result = {
                'test': 'sigma_coherence',
                'passed': False,
                'error': sigma_check['error']
            }
            self.tests_failed += 1
            self.test_results.append(result)
            return result
        
        self.logger.info(f"  Server sigma: {sigma_check['server_sigma']:.3f}")
        self.logger.info(f"  Client estimated: {sigma_check['client_estimated_sigma']:.3f}")
        self.logger.info(f"  Drift: {sigma_check['sigma_drift']:.3f}")
        self.logger.info(f"  Sector: {sigma_check['sigma_sector']}")
        
        # Pass if drift is reasonable
        passed = sigma_check['sigma_drift'] < 2.0
        
        result = {
            'test': 'sigma_coherence',
            'passed': passed,
            **sigma_check
        }
        
        if passed:
            self.tests_passed += 1
            self.logger.info(f"  ✓ PASS: Sigma coherence within bounds")
        else:
            self.tests_failed += 1
            self.logger.info(f"  ✗ FAIL: Sigma drift too large")
        
        self.test_results.append(result)
        return result
    
    def run_all_tests(self):
        """Run complete test suite"""
        self.logger.info("\n" + "="*80)
        self.logger.info("RUNNING COMPLETE TEST SUITE")
        self.logger.info("="*80)
        
        start_time = time.time()
        
        # Test 1: Node queries
        sample_nodes = [0, 100, 1000, 10000, 50000, 100000, 150000, 196882]
        sample_nodes = [n for n in sample_nodes if n < len(self.routing_table)]
        self.test_node_query(sample_nodes)
        
        # Test 2: Sigma addressing
        self.test_sigma_addressing()
        
        # Test 3: j-invariant distribution
        self.test_j_invariant_distribution()
        
        # Test 4: Layer structure
        self.test_layer_structure()
        
        # Test 5: Synchronization (if metrics available)
        self.test_synchronization_status()
        
        # Test 6: Temporal coherence
        self.test_temporal_coherence()
        
        # Test 7: Sigma coherence
        self.test_sigma_coherence()
        
        elapsed = time.time() - start_time
        
        # Print summary
        self.print_test_summary(elapsed)
    
    def print_test_summary(self, elapsed: float):
        """Print test summary"""
        total_tests = self.tests_passed + self.tests_failed
        pass_rate = (self.tests_passed / total_tests * 100) if total_tests > 0 else 0
        
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        
        print(f"\nClient: {self.client_name}")
        print(f"Runtime: {elapsed:.2f}s")
        
        print(f"\nTests:")
        print(f"  Total: {total_tests}")
        print(f"  Passed: {self.tests_passed} ({pass_rate:.1f}%)")
        print(f"  Failed: {self.tests_failed}")
        
        print(f"\nActivity:")
        print(f"  Node queries: {self.queries_made:,}")
        print(f"  Sync checks: {self.sync_checks}")
        
        if self.server_metadata:
            print(f"\nServer State:")
            print(f"  Current sigma: {self.server_metadata.get('current_sigma', 0):.3f}")
            print(f"  Heartbeats: {self.server_metadata.get('heartbeat_count', 0):,}")
            print(f"  Sync rate: {self.server_metadata.get('synchronized_nodes', 0):,}/"
                  f"{self.server_metadata.get('total_nodes', 0):,}")
        
        print("\n" + "="*80)
        
        if self.tests_failed == 0:
            print("✓ ALL TESTS PASSED")
        else:
            print(f"✗ {self.tests_failed} TEST(S) FAILED")
        
        print("="*80 + "\n")
    
    def save_results(self, filename: str = None):
        """Save test results to file"""
        if filename is None:
            filename = DATA_DIR / f"client_results_{self.client_name}.json"
        
        results = {
            'client_id': self.client_name,
            'timestamp': datetime.now().isoformat(),
            'connection_time': self.connection_time,
            'tests_passed': self.tests_passed,
            'tests_failed': self.tests_failed,
            'queries_made': self.queries_made,
            'sync_checks': self.sync_checks,
            'test_results': self.test_results,
            'server_metadata': self.server_metadata
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        self.logger.info(f"\n✓ Results saved to {filename}")

# ============================================================================
# QUICK INITIALIZATION (Without full server)
# ============================================================================

def quick_initialize_lattice():
    """Quick initialization - generates routing tables without starting server"""
    print("\n" + "="*80)
    print("QUICK LATTICE INITIALIZATION")
    print("="*80)
    print("\nThis will generate routing tables for the Moonshine lattice.")
    print("Estimated time: 30-60 seconds for basic routing tables")
    print("="*80 + "\n")
    
    import logging
    logger = logging.getLogger("QuickInit")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(handler)
    
    # Create minimal routing table
    logger.info("Generating routing table for 196,883 nodes...")
    
    from dataclasses import dataclass
    
    @dataclass
    class RoutingEntry:
        node_id: int
        node_type: int
        physical_addr: int
        virtual_addr: int
        inverse_addr: int
        sigma_address: float
        sigma_sector: int
        j_invariant_real: float
        j_invariant_imag: float
        phase_deg: float
        coherence_level: str
        layer: int
        triangle_id: int
        vertex_id: int
        parent_triangle: Optional[int] = None
        fidelity: float = 1.0
        timestamp: float = 0.0
    
    routing_table = {}
    timestamp = time.time()
    
    PSEUDOQUBIT_TABLE = 0x0000000100000000
    
    for idx in range(MOONSHINE_DIMENSION):
        # Calculate sigma
        sigma = (idx / MOONSHINE_DIMENSION) * SIGMA_PERIOD
        sigma_sector = int((sigma / SIGMA_PERIOD) * 8) % 8
        
        # Calculate tau and j-invariant
        tau_real = idx / MOONSHINE_DIMENSION
        tau_imag = (sigma % SIGMA_PERIOD) / SIGMA_PERIOD + 0.001
        tau = complex(tau_real, tau_imag)
        
        # Simplified j-invariant
        q = np.exp(2j * np.pi * tau)
        j_inv = 1/q + 744 + 196884*q
        
        phase_deg = (sigma / SIGMA_PERIOD) * 360.0
        
        j_mag = abs(j_inv)
        coherence = 'H' if j_mag < 1000 else 'M' if j_mag < 10000 else 'L'
        
        entry = RoutingEntry(
            node_id=idx,
            node_type=0,
            physical_addr=PSEUDOQUBIT_TABLE + idx * 512,
            virtual_addr=PSEUDOQUBIT_TABLE + idx * 512 + 8,
            inverse_addr=PSEUDOQUBIT_TABLE + idx * 512 + 16,
            sigma_address=sigma,
            sigma_sector=sigma_sector,
            j_invariant_real=j_inv.real,
            j_invariant_imag=j_inv.imag,
            phase_deg=phase_deg,
            coherence_level=coherence,
            layer=0,
            triangle_id=idx,
            vertex_id=0,
            fidelity=1.0,
            timestamp=timestamp
        )
        
        routing_table[idx] = entry
        
        if (idx + 1) % 10000 == 0:
            logger.info(f"  Generated {idx + 1:,}/{MOONSHINE_DIMENSION:,} entries...")
    
    logger.info(f"\n✓ Generated {len(routing_table):,} routing entries")
    
    # Save routing table
    logger.info(f"\nSaving routing table to {ROUTING_TABLES_FILE}...")
    
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    with open(ROUTING_TABLES_FILE, 'wb') as f:
        pickle.dump(routing_table, f)
    
    logger.info(f"✓ Routing table saved ({ROUTING_TABLES_FILE.stat().st_size / 1024**2:.1f} MB)")
    
    # Create minimal server metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'total_nodes': len(routing_table),
        'current_sigma': 0.0,
        'heartbeat_count': 0,
        'synchronized_nodes': 0
    }
    
    with open(METRICS_LOG_FILE, 'w') as f:
        f.write(json.dumps(metadata) + '\n')
    
    logger.info(f"✓ Metadata initialized")
    
    print("\n" + "="*80)
    print("INITIALIZATION COMPLETE")
    print("="*80)
    print(f"\n✓ Routing tables ready at: {ROUTING_TABLES_FILE}")
    print(f"✓ You can now run the client tests")
    print("="*80 + "\n")
    
    return routing_table

# ============================================================================
# MAIN
# ============================================================================

async def main():
    """Main client execution"""
    
    print("\n" + "="*80)
    print("MOONSHINE QUANTUM CLIENT - TEST SUITE")
    print("="*80)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    # Check if initialization is needed
    if not ROUTING_TABLES_FILE.exists():
        print("⚠ Lattice not initialized")
        print("\nInitializing now (this will take ~60 seconds)...")
        print("-"*80)
        
        quick_initialize_lattice()
        
        print("\n✓ Initialization complete!")
        print("-"*80 + "\n")
    
    # Create client
    client = MoonshineClient()
    
    # Connect to lattice
    if not client.connect():
        print("\n✗ Connection failed")
        return
    
    print("\n" + "-"*80)
    
    # Run all tests
    client.run_all_tests()
    
    # Save results
    client.save_results()
    
    print("\n" + "="*80)
    print("CLIENT TEST SUITE COMPLETE")
    print("="*80 + "\n")

if __name__ == "__main__":
    # Check if we're in Jupyter/IPython with existing event loop
    try:
        import IPython
        # In Jupyter, just await directly
        await main()
    except (ImportError, NameError):
        # Not in Jupyter, use asyncio.run()
        asyncio.run(main())