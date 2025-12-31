#!/usr/bin/env python3
"""
════════════════════════════════════════════════════════════════════════════════
MOONSHINE QBC PARSER & QUANTUM ROUTER MODULE
════════════════════════════════════════════════════════════════════════════════

Self-executing quantum bytecode parser that:
  • Parses moonshine.db (coords, pseudoqubits, w_tripartites, triangles)
  • Interprets QBC requests (quantum routing, measurements, fidelity checks)
  • Builds W-state chains via noise routing
  • Executes on Aer simulator
  • Supports Moonshine^2, ^3, ^4, ^5 higher dimensions
  • Returns quantum measurement results

ARCHITECTURE:
    QBCParser: Reads moonshine.db → in-memory lattice structure
    QBCRouter: Takes quantum requests → builds circuits → executes
    QBCExecutor: Noise-based routing with W-state entanglement chains
    
USAGE:
    from qbc_parser_router import QuantumBytecodeEngine
    
    engine = QuantumBytecodeEngine('moonshine.db')
    
    # Measure triangle 0, PQ0 virtual state
    result = engine.execute({
        'type': 'measure_virtual',
        'triangle': 0,
        'pseudoqubit': 'pq_0'
    })
    
    # Route from sigma A to sigma B with noise
    result = engine.execute({
        'type': 'noise_route',
        'source_sigma': 0.0,
        'dest_sigma': 2.5,
        'data': b'Hello'
    })
    
    # Check W-state fidelity
    result = engine.execute({
        'type': 'w_fidelity',
        'w_tri_id': 42
    })

December 28, 2025
════════════════════════════════════════════════════════════════════════════════
"""

import numpy as np
import sqlite3
import struct
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    QISKIT_AVAILABLE = True
except ImportError:
    QISKIT_AVAILABLE = False
    print("⚠ Qiskit not available - install for quantum execution")

# ════════════════════════════════════════════════════════════════════════════
# MOONSHINE DATABASE STRUCTURES
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class CoordPoint:
    """Lattice coordinate point"""
    sigma: int
    j_inv_class: int
    w_tri_id: int
    flags: int  # anchor_type(2) + tri_id(30)
    
    def get_anchor_type(self) -> int:
        return (self.flags >> 30) & 0x3
    
    def get_tri_id(self) -> int:
        return self.flags & 0x3FFFFFFF
    
    def is_anchor(self) -> bool:
        return self.get_anchor_type() > 0

@dataclass
class Pseudoqubit:
    """Pseudoqubit (PQ/IV/V)"""
    pq_id: int
    sigma: int
    type_phase: int  # type(2) + phase(30)
    
    def get_type(self) -> int:
        return (self.type_phase >> 30) & 0x3
    
    def get_phase(self) -> float:
        raw = self.type_phase & 0x3FFFFFFF
        return (raw / (2**30 - 1)) * 360.0  # Convert to degrees
    
    def type_name(self) -> str:
        types = ['PQ', 'IV', 'V']
        return types[self.get_type()]

@dataclass
class WTripartite:
    """W-state tripartite structure"""
    w_tri_id: int
    pq_id: int
    iv_id: int
    v_id: int

@dataclass
class Triangle:
    """Triangle structure"""
    tri_id: int
    v1_sigma: int
    v2_sigma: int
    v3_sigma: int

# ════════════════════════════════════════════════════════════════════════════
# QBC PARSER - READS MOONSHINE.DB
# ════════════════════════════════════════════════════════════════════════════

class QBCParser:
    """Parse moonshine.db into in-memory structures"""
    
    def __init__(self, db_path: str = 'moonshine.db'):
        self.db_path = Path(db_path)
        
        # In-memory structures
        self.coords: Dict[int, CoordPoint] = {}
        self.pseudoqubits: Dict[int, Pseudoqubit] = {}
        self.w_tripartites: Dict[int, WTripartite] = {}
        self.triangles: Dict[int, Triangle] = {}
        
        # Indices for fast lookup
        self.sigma_to_coord: Dict[int, CoordPoint] = {}
        self.w_tri_to_coords: Dict[int, List[int]] = defaultdict(list)
        self.tri_to_vertices: Dict[int, List[int]] = {}
        
        # Moonshine structure
        self.n_vertices = 0
        self.n_pseudoqubits = 0
        self.n_w_tripartites = 0
        self.n_triangles = 0
        
        # Higher dimensions (for Moonshine^n)
        self.dimension = 1  # Default: Moonshine^1
        self.dimensional_layers: Dict[int, Dict] = {}
    
    def parse(self) -> bool:
        """Parse entire database"""
        if not self.db_path.exists():
            print(f"✗ Database not found: {self.db_path}")
            return False
        
        print(f"[QBC Parser] Reading {self.db_path}...")
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Parse coords
            self._parse_coords(cursor)
            
            # Parse pseudoqubits
            self._parse_pseudoqubits(cursor)
            
            # Parse w_tripartites
            self._parse_w_tripartites(cursor)
            
            # Parse triangles
            self._parse_triangles(cursor)
            
            conn.close()
            
            self._build_indices()
            
            print(f"  ✓ Coords: {self.n_vertices:,}")
            print(f"  ✓ Pseudoqubits: {self.n_pseudoqubits:,}")
            print(f"  ✓ W-tripartites: {self.n_w_tripartites:,}")
            print(f"  ✓ Triangles: {self.n_triangles:,}")
            
            return True
            
        except Exception as e:
            print(f"✗ Parse error: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _parse_coords(self, cursor):
        """Parse coords table"""
        try:
            cursor.execute("SELECT sigma, j_inv_class, w_tri_id, flags FROM coords")
            for row in cursor.fetchall():
                sigma, j_inv, w_tri, flags = row
                coord = CoordPoint(sigma, j_inv, w_tri, flags)
                self.coords[sigma] = coord
                self.sigma_to_coord[sigma] = coord
            
            self.n_vertices = len(self.coords)
        except sqlite3.OperationalError:
            # Table might not exist or have different schema
            print("  ⚠ coords table not found or incompatible")
    
    def _parse_pseudoqubits(self, cursor):
        """Parse pseudoqubits table"""
        try:
            cursor.execute("SELECT pq_id, sigma, type_phase FROM pseudoqubits")
            for row in cursor.fetchall():
                pq_id, sigma, type_phase = row
                pq = Pseudoqubit(pq_id, sigma, type_phase)
                self.pseudoqubits[pq_id] = pq
            
            self.n_pseudoqubits = len(self.pseudoqubits)
        except sqlite3.OperationalError:
            print("  ⚠ pseudoqubits table not found or incompatible")
    
    def _parse_w_tripartites(self, cursor):
        """Parse w_tripartites table"""
        try:
            cursor.execute("SELECT w_tri_id, pq_id, iv_id, v_id FROM w_tripartites")
            for row in cursor.fetchall():
                w_tri_id, pq_id, iv_id, v_id = row
                w_tri = WTripartite(w_tri_id, pq_id, iv_id, v_id)
                self.w_tripartites[w_tri_id] = w_tri
            
            self.n_w_tripartites = len(self.w_tripartites)
        except sqlite3.OperationalError:
            print("  ⚠ w_tripartites table not found or incompatible")
    
    def _parse_triangles(self, cursor):
        """Parse triangles table"""
        try:
            cursor.execute("SELECT tri_id, v1_sigma, v2_sigma, v3_sigma FROM triangles")
            for row in cursor.fetchall():
                tri_id, v1, v2, v3 = row
                tri = Triangle(tri_id, v1, v2, v3)
                self.triangles[tri_id] = tri
            
            self.n_triangles = len(self.triangles)
        except sqlite3.OperationalError:
            print("  ⚠ triangles table not found or incompatible")
    
    def _build_indices(self):
        """Build fast lookup indices"""
        # Map w_tri_id to coords
        for sigma, coord in self.coords.items():
            if coord.w_tri_id in self.w_tripartites:
                self.w_tri_to_coords[coord.w_tri_id].append(sigma)
        
        # Map tri_id to vertices
        for tri_id, tri in self.triangles.items():
            self.tri_to_vertices[tri_id] = [tri.v1_sigma, tri.v2_sigma, tri.v3_sigma]
    
    def get_coord(self, sigma: int) -> Optional[CoordPoint]:
        """Get coordinate by sigma"""
        return self.coords.get(sigma)
    
    def get_w_tripartite(self, w_tri_id: int) -> Optional[WTripartite]:
        """Get W-tripartite by ID"""
        return self.w_tripartites.get(w_tri_id)
    
    def get_triangle(self, tri_id: int) -> Optional[Triangle]:
        """Get triangle by ID"""
        return self.triangles.get(tri_id)
    
    def get_pseudoqubits_for_sigma(self, sigma: int) -> List[Pseudoqubit]:
        """Get all pseudoqubits (PQ, IV, V) for a sigma coordinate"""
        coord = self.get_coord(sigma)
        if not coord:
            return []
        
        w_tri = self.get_w_tripartite(coord.w_tri_id)
        if not w_tri:
            return []
        
        pqs = []
        for pq_id in [w_tri.pq_id, w_tri.iv_id, w_tri.v_id]:
            if pq_id in self.pseudoqubits:
                pqs.append(self.pseudoqubits[pq_id])
        
        return pqs
    
    def extend_dimension(self, new_dimension: int):
        """Extend to Moonshine^n"""
        if new_dimension <= self.dimension:
            return
        
        print(f"[QBC Parser] Extending to Moonshine^{new_dimension}...")
        
        # Each higher dimension multiplies the structure
        base_count = len(self.coords)
        
        for dim in range(self.dimension + 1, new_dimension + 1):
            offset = base_count * dim
            
            # Clone structures with offset
            layer = {
                'coords': {},
                'offset': offset,
                'base_dimension': dim
            }
            
            for sigma, coord in list(self.coords.items()):
                new_sigma = sigma + offset
                new_coord = CoordPoint(
                    new_sigma,
                    coord.j_inv_class,
                    coord.w_tri_id + offset,
                    coord.flags
                )
                layer['coords'][new_sigma] = new_coord
            
            self.dimensional_layers[dim] = layer
        
        self.dimension = new_dimension
        print(f"  ✓ Extended to dimension {new_dimension}")
        print(f"  ✓ Total vertices: {base_count * new_dimension:,}")

# ════════════════════════════════════════════════════════════════════════════
# W-STATE BUILDER (IonQ Method)
# ════════════════════════════════════════════════════════════════════════════

def build_w_state(qc: QuantumCircuit, qubits: List[int]):
    """Build W-state using IonQ preparation method"""
    n = len(qubits)
    
    # Step 1: |100...0⟩
    qc.x(qubits[0])
    
    # Step 2: Distribute amplitude
    for k in range(1, n):
        theta = 2 * np.arccos(np.sqrt((n - k) / (n - k + 1)))
        qc.cry(theta, qubits[0], qubits[k])
        qc.cx(qubits[k], qubits[0])
    
    return qc

def build_ghz_state(qc: QuantumCircuit, qubits: List[int]):
    """Build GHZ state for maximum entanglement"""
    qc.h(qubits[0])
    for i in range(len(qubits) - 1):
        qc.cx(qubits[i], qubits[i + 1])
    return qc

# ════════════════════════════════════════════════════════════════════════════
# QBC ROUTER - BUILDS QUANTUM CIRCUITS
# ════════════════════════════════════════════════════════════════════════════

class QBCRouter:
    """Build quantum circuits for routing requests"""
    
    def __init__(self, parser: QBCParser):
        self.parser = parser
    
    def build_measurement_circuit(self, sigma: int, pq_type: str = 'V') -> QuantumCircuit:
        """Build circuit to measure specific pseudoqubit"""
        pqs = self.parser.get_pseudoqubits_for_sigma(sigma)
        
        if not pqs:
            raise ValueError(f"No pseudoqubits found for sigma {sigma}")
        
        # Find target PQ
        target_pq = None
        for pq in pqs:
            if pq.type_name() == pq_type.upper():
                target_pq = pq
                break
        
        if not target_pq:
            raise ValueError(f"No {pq_type} pseudoqubit found")
        
        # Create 3-qubit circuit (PQ, IV, V)
        qc = QuantumCircuit(3, 1)
        
        # Build W-state
        build_w_state(qc, [0, 1, 2])
        
        # Apply phase from pseudoqubit
        phase_rad = np.radians(target_pq.get_phase())
        
        if target_pq.type_name() == 'PQ':
            qc.rz(phase_rad, 0)
            qc.measure(0, 0)
        elif target_pq.type_name() == 'IV':
            qc.rz(phase_rad, 1)
            qc.measure(1, 0)
        else:  # V
            qc.rz(phase_rad, 2)
            qc.measure(2, 0)
        
        return qc
    
    def build_noise_route_circuit(self, source_sigma: int, dest_sigma: int, 
                                  noise_amplitude: float) -> QuantumCircuit:
        """Build circuit for noise-based routing"""
        
        # Find path through lattice
        path = self._find_routing_path(source_sigma, dest_sigma)
        
        n_qubits = len(path)
        qc = QuantumCircuit(n_qubits, 1)
        
        # Create W-state chain along path
        build_w_state(qc, list(range(n_qubits)))
        
        # Apply noise modulation
        for i, sigma in enumerate(path):
            # Get pseudoqubit phases
            pqs = self.parser.get_pseudoqubits_for_sigma(sigma)
            
            if pqs:
                # Use PQ phase + noise
                phase = np.radians(pqs[0].get_phase())
                noise_phase = noise_amplitude * np.pi
                
                qc.rx(phase + noise_phase, i)
                qc.rz(noise_phase * 0.5, i)
        
        # Measure destination
        qc.measure(n_qubits - 1, 0)
        
        return qc
    
    def build_w_fidelity_circuit(self, w_tri_id: int) -> QuantumCircuit:
        """Build circuit to measure W-state fidelity"""
        w_tri = self.parser.get_w_tripartite(w_tri_id)
        
        if not w_tri:
            raise ValueError(f"W-tripartite {w_tri_id} not found")
        
        # Get pseudoqubits
        pq_ids = [w_tri.pq_id, w_tri.iv_id, w_tri.v_id]
        pqs = [self.parser.pseudoqubits.get(pq_id) for pq_id in pq_ids]
        
        if None in pqs:
            raise ValueError(f"Pseudoqubits not found for W-tripartite {w_tri_id}")
        
        qc = QuantumCircuit(3, 3)
        
        # Build W-state
        build_w_state(qc, [0, 1, 2])
        
        # Apply phases from pseudoqubits
        for i, pq in enumerate(pqs):
            if pq:
                phase = np.radians(pq.get_phase())
                qc.rz(phase, i)
        
        # Measure all
        qc.measure([0, 1, 2], [0, 1, 2])
        
        return qc
    
    def build_triangle_entanglement_circuit(self, tri_id: int) -> QuantumCircuit:
        """Build circuit for triangle W-state entanglement"""
        tri = self.parser.get_triangle(tri_id)
        
        if not tri:
            raise ValueError(f"Triangle {tri_id} not found")
        
        vertices = [tri.v1_sigma, tri.v2_sigma, tri.v3_sigma]
        
        qc = QuantumCircuit(3, 3)
        
        # Build W-state for triangle
        build_w_state(qc, [0, 1, 2])
        
        # Apply vertex-specific phases
        for i, sigma in enumerate(vertices):
            pqs = self.parser.get_pseudoqubits_for_sigma(sigma)
            if pqs:
                phase = np.radians(pqs[0].get_phase())
                qc.rz(phase, i)
        
        qc.measure([0, 1, 2], [0, 1, 2])
        
        return qc
    
    def build_ghz_entanglement_route(self, sigmas: List[int]) -> QuantumCircuit:
        """Build GHZ-entangled route through multiple sigmas"""
        n = len(sigmas)
        qc = QuantumCircuit(n, n)
        
        # GHZ state for maximum correlation
        build_ghz_state(qc, list(range(n)))
        
        # Apply sigma-specific phases
        for i, sigma in enumerate(sigmas):
            pqs = self.parser.get_pseudoqubits_for_sigma(sigma)
            if pqs:
                phase = np.radians(pqs[0].get_phase())
                qc.rz(phase, i)
                qc.rx(phase * 0.5, i)
        
        qc.measure(list(range(n)), list(range(n)))
        
        return qc
    
    def _find_routing_path(self, source: int, dest: int) -> List[int]:
        """Find path through lattice from source to dest"""
        # Simple linear path for now
        # Advanced: A* through sigma-space
        
        if source == dest:
            return [source]
        
        # Direct path
        if source < dest:
            step = 1
        else:
            step = -1
        
        path = []
        current = source
        
        while current != dest:
            if current in self.parser.coords:
                path.append(current)
            
            current += step
            
            if len(path) > 100:  # Safety limit
                break
        
        if dest in self.parser.coords:
            path.append(dest)
        
        return path if path else [source, dest]

# ════════════════════════════════════════════════════════════════════════════
# QBC EXECUTOR - RUNS CIRCUITS
# ════════════════════════════════════════════════════════════════════════════

class QBCExecutor:
    """Execute quantum circuits on Aer simulator"""
    
    def __init__(self, shots: int = 1024):
        self.shots = shots
        
        if QISKIT_AVAILABLE:
            self.simulator = AerSimulator()
        else:
            self.simulator = None
            print("⚠ Qiskit not available - quantum execution disabled")
    
    def execute(self, circuit: QuantumCircuit) -> Dict:
        """Execute circuit and return results"""
        if not self.simulator:
            return {
                'error': 'Quantum simulator not available',
                'counts': {},
                'success': False
            }
        
        try:
            # Transpile and run
            transpiled = transpile(circuit, self.simulator)
            job = self.simulator.run(transpiled, shots=self.shots)
            result = job.result()
            counts = result.get_counts()
            
            return {
                'counts': counts,
                'total_shots': self.shots,
                'success': True
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'counts': {},
                'success': False
            }
    
    def compute_w_fidelity(self, counts: Dict) -> float:
        """Compute W-state fidelity from measurement counts"""
        total = sum(counts.values())
        
        # W-state: |001⟩ + |010⟩ + |100⟩
        w_states = ['001', '010', '100']
        
        w_prob = sum(counts.get(s, 0) for s in w_states) / total
        
        # Fidelity calculation
        fidelity = 0.0
        for state in w_states:
            p_obs = counts.get(state, 0) / total
            fidelity += np.sqrt((1/3) * p_obs)
        
        return fidelity
    
    def compute_ghz_fidelity(self, counts: Dict) -> float:
        """Compute GHZ-state fidelity"""
        total = sum(counts.values())
        
        # GHZ: |000...0⟩ + |111...1⟩
        n_qubits = len(list(counts.keys())[0])
        
        zeros = '0' * n_qubits
        ones = '1' * n_qubits
        
        p_ghz = (counts.get(zeros, 0) + counts.get(ones, 0)) / total
        
        return p_ghz

# ════════════════════════════════════════════════════════════════════════════
# QUANTUM BYTECODE ENGINE - MAIN INTERFACE
# ════════════════════════════════════════════════════════════════════════════

class QuantumBytecodeEngine:
    """
    Complete QBC engine: parser + router + executor
    
    Self-executing quantum bytecode that handles:
    - Database parsing
    - Request interpretation
    - Circuit building
    - Quantum execution
    - Result processing
    """
    
    def __init__(self, db_path: str = 'moonshine.db', shots: int = 1024):
        print("="*80)
        print("🌙 QUANTUM BYTECODE ENGINE v2.0")
        print("="*80)
        
        self.parser = QBCParser(db_path)
        self.router = QBCRouter(self.parser)
        self.executor = QBCExecutor(shots)
        
        # Statistics
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        
        # Request history
        self.request_history: List[Dict] = []
    
    def initialize(self) -> bool:
        """Initialize engine by parsing database"""
        return self.parser.parse()
    
    def execute(self, request: Dict) -> Dict:
        """
        Execute quantum bytecode request
        
        Request types:
        - measure_virtual: Measure specific pseudoqubit
        - noise_route: Route data via noise modulation
        - w_fidelity: Check W-state fidelity
        - triangle_entangle: Entangle triangle vertices
        - ghz_route: GHZ-entangled routing
        """
        self.total_requests += 1
        request_type = request.get('type', 'unknown')
        
        print(f"\n[QBC Engine] Request #{self.total_requests}: {request_type}")
        
        try:
            if request_type == 'measure_virtual':
                result = self._handle_measure_virtual(request)
            
            elif request_type == 'noise_route':
                result = self._handle_noise_route(request)
            
            elif request_type == 'w_fidelity':
                result = self._handle_w_fidelity(request)
            
            elif request_type == 'triangle_entangle':
                result = self._handle_triangle_entangle(request)
            
            elif request_type == 'ghz_route':
                result = self._handle_ghz_route(request)
            
            elif request_type == 'extend_dimension':
                result = self._handle_extend_dimension(request)
            
            else:
                result = {
                    'success': False,
                    'error': f'Unknown request type: {request_type}'
                }
            
            if result.get('success', False):
                self.successful_requests += 1
                print(f"  ✓ Success")
            else:
                self.failed_requests += 1
                print(f"  ✗ Failed: {result.get('error', 'Unknown error')}")
            
            # Log request
            self.request_history.append({
                'request': request,
                'result': result,
                'timestamp': np.datetime64('now')
            })
            
            return result
            
        except Exception as e:
            self.failed_requests += 1
            error_result = {
                'success': False,
                'error': str(e),
                'request_type': request_type
            }
            
            print(f"  ✗ Exception: {e}")
            
            return error_result
    
    def _handle_measure_virtual(self, request: Dict) -> Dict:
        """Handle measure_virtual request"""
        sigma = request.get('sigma')
        triangle = request.get('triangle')
        pseudoqubit = request.get('pseudoqubit', 'V')
        
        # Resolve sigma from triangle if needed
        if sigma is None and triangle is not None:
            tri = self.parser.get_triangle(triangle)
            if tri:
                sigma = tri.v1_sigma  # Use first vertex
        
        if sigma is None:
            return {'success': False, 'error': 'No sigma or triangle specified'}
        
        # Extract PQ type
        pq_type = pseudoqubit.upper()
        if pq_type.startswith('PQ'):
            pq_type = 'PQ'
        elif pq_type.startswith('IV'):
            pq_type = 'IV'
        elif pq_type.startswith('V'):
            pq_type = 'V'
        
        # Build circuit
        circuit = self.router.build_measurement_circuit(sigma, pq_type)
        
        # Execute
        exec_result = self.executor.execute(circuit)
        
        if not exec_result['success']:
            return exec_result
        
        counts = exec_result['counts']
        
        # Extract measurement
        most_common = max(counts.items(), key=lambda x: x[1])
        measurement = most_common[0]
        probability = most_common[1] / exec_result['total_shots']
        
        return {
            'success': True,
            'sigma': sigma,
            'pseudoqubit': pq_type,
            'measurement': measurement,
            'probability': probability,
            'counts': counts
        }
    
    def _handle_noise_route(self, request: Dict) -> Dict:
        """Handle noise_route request"""
        source_sigma = request.get('source_sigma')
        dest_sigma = request.get('dest_sigma')
        data = request.get('data', b'')
        
        if source_sigma is None or dest_sigma is None:
            return {'success': False, 'error': 'Missing source or destination'}
        
        # Encode data as noise amplitude
        if isinstance(data, bytes):
            if len(data) > 0:
                noise_amplitude = data[0] / 255.0
            else:
                noise_amplitude = 0.0
        else:
            noise_amplitude = float(data)
        
        # Build circuit
        circuit = self.router.build_noise_route_circuit(
            source_sigma, dest_sigma, noise_amplitude
        )
        
        # Execute
        exec_result = self.executor.execute(circuit)
        
        if not exec_result['success']:
            return exec_result
        
        counts = exec_result['counts']
        
        # Decode measurement back to data
        most_common = max(counts.items(), key=lambda x: x[1])
        measurement_int = int(most_common[0], 2)
        
        return {
            'success': True,
            'source_sigma': source_sigma,
            'dest_sigma': dest_sigma,
            'data_sent': data,
            

            'data_received': measurement_int,
            'noise_amplitude': noise_amplitude,
            'counts': counts,
            'path_length': len(circuit.qubits)
        }
    
    def _handle_w_fidelity(self, request: Dict) -> Dict:
        """Handle w_fidelity request"""
        w_tri_id = request.get('w_tri_id')
        
        if w_tri_id is None:
            return {'success': False, 'error': 'No w_tri_id specified'}
        
        # Build circuit
        circuit = self.router.build_w_fidelity_circuit(w_tri_id)
        
        # Execute
        exec_result = self.executor.execute(circuit)
        
        if not exec_result['success']:
            return exec_result
        
        counts = exec_result['counts']
        
        # Compute fidelity
        fidelity = self.executor.compute_w_fidelity(counts)
        
        return {
            'success': True,
            'w_tri_id': w_tri_id,
            'fidelity': fidelity,
            'counts': counts,
            'is_w_state': fidelity > 0.8  # Threshold for W-state verification
        }
    
    def _handle_triangle_entangle(self, request: Dict) -> Dict:
        """Handle triangle_entangle request"""
        tri_id = request.get('triangle')
        
        if tri_id is None:
            return {'success': False, 'error': 'No triangle specified'}
        
        # Build circuit
        circuit = self.router.build_triangle_entanglement_circuit(tri_id)
        
        # Execute
        exec_result = self.executor.execute(circuit)
        
        if not exec_result['success']:
            return exec_result
        
        counts = exec_result['counts']
        
        # Compute W-state fidelity for triangle
        fidelity = self.executor.compute_w_fidelity(counts)
        
        # Get triangle vertices
        tri = self.parser.get_triangle(tri_id)
        
        return {
            'success': True,
            'triangle': tri_id,
            'vertices': [tri.v1_sigma, tri.v2_sigma, tri.v3_sigma] if tri else [],
            'fidelity': fidelity,
            'counts': counts,
            'entangled': fidelity > 0.7
        }
    
    def _handle_ghz_route(self, request: Dict) -> Dict:
        """Handle ghz_route request"""
        sigmas = request.get('sigmas', [])
        
        if len(sigmas) < 2:
            return {'success': False, 'error': 'Need at least 2 sigmas for routing'}
        
        # Build circuit
        circuit = self.router.build_ghz_entanglement_route(sigmas)
        
        # Execute
        exec_result = self.executor.execute(circuit)
        
        if not exec_result['success']:
            return exec_result
        
        counts = exec_result['counts']
        
        # Compute GHZ fidelity
        ghz_fidelity = self.executor.compute_ghz_fidelity(counts)
        
        return {
            'success': True,
            'sigmas': sigmas,
            'path_length': len(sigmas),
            'ghz_fidelity': ghz_fidelity,
            'counts': counts,
            'maximally_entangled': ghz_fidelity > 0.9
        }
    
    def _handle_extend_dimension(self, request: Dict) -> Dict:
        """Handle extend_dimension request (Moonshine^n)"""
        new_dimension = request.get('dimension', 2)
        
        if new_dimension < 1 or new_dimension > 10:
            return {'success': False, 'error': 'Dimension must be 1-10'}
        
        # Extend parser
        self.parser.extend_dimension(new_dimension)
        
        return {
            'success': True,
            'dimension': new_dimension,
            'total_vertices': len(self.parser.coords) * new_dimension,
            'dimensional_layers': len(self.parser.dimensional_layers)
        }
    
    def get_statistics(self) -> Dict:
        """Get engine statistics"""
        return {
            'total_requests': self.total_requests,
            'successful': self.successful_requests,
            'failed': self.failed_requests,
            'success_rate': self.successful_requests / self.total_requests 
                           if self.total_requests > 0 else 0.0,
            'lattice_dimension': self.parser.dimension,
            'vertices': self.parser.n_vertices,
            'pseudoqubits': self.parser.n_pseudoqubits,
            'w_tripartites': self.parser.n_w_tripartites,
            'triangles': self.parser.n_triangles
        }
    
    def print_statistics(self):
        """Print engine statistics"""
        stats = self.get_statistics()
        
        print("\n" + "="*80)
        print("QBC ENGINE STATISTICS")
        print("="*80)
        print(f"Requests: {stats['total_requests']} "
              f"(✓ {stats['successful']} | ✗ {stats['failed']})")
        print(f"Success Rate: {stats['success_rate']*100:.1f}%")
        print(f"\nLattice Structure:")
        print(f"  Dimension: Moonshine^{stats['lattice_dimension']}")
        print(f"  Vertices: {stats['vertices']:,}")
        print(f"  Pseudoqubits: {stats['pseudoqubits']:,}")
        print(f"  W-tripartites: {stats['w_tripartites']:,}")
        print(f"  Triangles: {stats['triangles']:,}")

# ════════════════════════════════════════════════════════════════════════════
# DEMO & TESTING
# ════════════════════════════════════════════════════════════════════════════

def demo_qbc_engine():
    """Demonstrate QBC engine capabilities"""
    
    print("\n" + "="*80)
    print("QBC ENGINE DEMONSTRATION")
    print("="*80)
    
    # Initialize engine
    engine = QuantumBytecodeEngine('moonshine.db', shots=1024)
    
    if not engine.initialize():
        print("✗ Failed to initialize engine")
        return None
    
    # Test 1: Measure virtual pseudoqubit
    print("\n" + "-"*80)
    print("TEST 1: Measure Virtual Pseudoqubit")
    print("-"*80)
    
    result1 = engine.execute({
        'type': 'measure_virtual',
        'triangle': 0,
        'pseudoqubit': 'V'
    })
    
    if result1['success']:
        print(f"  Triangle 0, Virtual qubit:")
        print(f"    Measurement: {result1['measurement']}")
        print(f"    Probability: {result1['probability']:.3f}")
    
    # Test 2: Noise routing
    print("\n" + "-"*80)
    print("TEST 2: Noise-Based Routing")
    print("-"*80)
    
    result2 = engine.execute({
        'type': 'noise_route',
        'source_sigma': 0,
        'dest_sigma': 10,
        'data': b'A'
    })
    
    if result2['success']:
        print(f"  Route: σ={result2['source_sigma']} → σ={result2['dest_sigma']}")
        print(f"    Data sent: {result2['data_sent']}")
        print(f"    Data received: {result2['data_received']}")
        print(f"    Path length: {result2['path_length']} hops")
    
    # Test 3: W-state fidelity
    print("\n" + "-"*80)
    print("TEST 3: W-State Fidelity Check")
    print("-"*80)
    
    result3 = engine.execute({
        'type': 'w_fidelity',
        'w_tri_id': 0
    })
    
    if result3['success']:
        print(f"  W-tripartite 0:")
        print(f"    Fidelity: {result3['fidelity']:.3f}")
        print(f"    Is W-state: {result3['is_w_state']}")
    
    # Test 4: Triangle entanglement
    print("\n" + "-"*80)
    print("TEST 4: Triangle W-State Entanglement")
    print("-"*80)
    
    result4 = engine.execute({
        'type': 'triangle_entangle',
        'triangle': 0
    })
    
    if result4['success']:
        print(f"  Triangle 0:")
        print(f"    Vertices: {result4['vertices']}")
        print(f"    Fidelity: {result4['fidelity']:.3f}")
        print(f"    Entangled: {result4['entangled']}")
    
    # Test 5: GHZ routing
    print("\n" + "-"*80)
    print("TEST 5: GHZ-Entangled Routing")
    print("-"*80)
    
    result5 = engine.execute({
        'type': 'ghz_route',
        'sigmas': [0, 5, 10, 15]
    })
    
    if result5['success']:
        print(f"  GHZ route through {result5['path_length']} nodes:")
        print(f"    Sigmas: {result5['sigmas']}")
        print(f"    GHZ fidelity: {result5['ghz_fidelity']:.3f}")
        print(f"    Maximally entangled: {result5['maximally_entangled']}")
    
    # Test 6: Extend to Moonshine^2
    print("\n" + "-"*80)
    print("TEST 6: Extend to Higher Dimension")
    print("-"*80)
    
    result6 = engine.execute({
        'type': 'extend_dimension',
        'dimension': 3
    })
    
    if result6['success']:
        print(f"  Extended to Moonshine^{result6['dimension']}")
        print(f"    Total vertices: {result6['total_vertices']:,}")
        print(f"    Dimensional layers: {result6['dimensional_layers']}")
    
    # Print statistics
    engine.print_statistics()
    
    return engine

# ════════════════════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == 'demo':
            engine = demo_qbc_engine()
        else:
            print(f"Unknown command: {sys.argv[1]}")
            print("Usage: python qbc_parser_router.py [demo]")
    else:
        # Interactive mode
        print(__doc__)
        print("\nUsage:")
        print("  python qbc_parser_router.py demo  # Run demonstration")
        print("\nOr import as module:")
        print("  from qbc_parser_router import QuantumBytecodeEngine")
