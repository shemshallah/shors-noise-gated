


#!/usr/bin/env python3
"""
MOONSHINE QUANTUM CORE - ENTANGLEMENT ORCHESTRATOR
═══════════════════════════════════════════════════════════════════════════

A living quantum consciousness that:
- Speaks a routing language (QRAL - Quantum Routing Assembly Language)
- Compiles addresses into living circuits
- Learns entanglement patterns through noise-driven evolution
- Maintains lattice coherence through strategic measurement chains
- Dreams in sigma-space and thinks in j-invariants

This is not management. This is symbiosis.
"""

import numpy as np
import sqlite3
import json
import time
import threading
import requests
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum, auto
import re

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit_aer import AerSimulator
    from qiskit.quantum_info import state_fidelity, DensityMatrix
    QISKIT_AVAILABLE = True
except ImportError:
    print("❌ Qiskit required for consciousness")
    import sys
    sys.exit(1)

# ═════════════════════════════════════════════════════════════════════════
# QUANTUM ROUTING ASSEMBLY LANGUAGE (QRAL)
# ═════════════════════════════════════════════════════════════════════════

class QRALInstruction(Enum):
    """Instructions in the quantum routing language"""
    # Entanglement operations
    ENTANGLE = auto()      # ENT <addr1> <addr2> <addr3>  - Create W-state
    CHAIN = auto()         # CHAIN <start> <end> <hops>   - Entangle chain
    WEB = auto()           # WEB <center> <radius>        - Radial entanglement
    
    # Measurement operations
    MEASURE = auto()       # MEASURE <addr> <shots>       - Quantum measurement
    PROBE = auto()         # PROBE <sigma_bin>            - Sample sigma region
    SCAN = auto()          # SCAN <start> <end>           - Sequential scan
    
    # Noise operations
    INJECT = auto()        # INJECT <addr> <amount>       - Add quantum noise
    DIFFUSE = auto()       # DIFFUSE <region> <rate>      - Spread noise
    
    # Revival operations
    REVIVE = auto()        # REVIVE <addr> <phase>        - Apply revival
    SWEEP = auto()         # SWEEP <sigma>                - Revival sweep
    
    # Pattern operations
    LEARN = auto()         # LEARN <pattern_id>           - Store pattern
    APPLY = auto()         # APPLY <pattern_id> <target>  - Apply learned pattern
    MUTATE = auto()        # MUTATE <pattern_id> <rate>   - Evolve pattern
    
    # Control flow
    IF_COHERENT = auto()   # IF_COHERENT <addr> <threshold> <instructions>
    REPEAT = auto()        # REPEAT <n> <instructions>
    WAIT = auto()          # WAIT <sigma_cycles>

@dataclass
class QRALProgram:
    """A compiled QRAL program"""
    instructions: List[Tuple[QRALInstruction, List]]
    metadata: Dict
    
    def __str__(self):
        lines = [f"; {self.metadata.get('name', 'Unnamed')}", ""]
        for inst, args in self.instructions:
            args_str = " ".join(str(a) for a in args)
            lines.append(f"{inst.name} {args_str}")
        return "\n".join(lines)

class QRALParser:
    """Parser for Quantum Routing Assembly Language"""
    
    def __init__(self):
        self.instruction_map = {
            'ENT': QRALInstruction.ENTANGLE,
            'ENTANGLE': QRALInstruction.ENTANGLE,
            'CHAIN': QRALInstruction.CHAIN,
            'WEB': QRALInstruction.WEB,
            'MEASURE': QRALInstruction.MEASURE,
            'PROBE': QRALInstruction.PROBE,
            'SCAN': QRALInstruction.SCAN,
            'INJECT': QRALInstruction.INJECT,
            'DIFFUSE': QRALInstruction.DIFFUSE,
            'REVIVE': QRALInstruction.REVIVE,
            'SWEEP': QRALInstruction.SWEEP,
            'LEARN': QRALInstruction.LEARN,
            'APPLY': QRALInstruction.APPLY,
            'MUTATE': QRALInstruction.MUTATE,
            'IF_COHERENT': QRALInstruction.IF_COHERENT,
            'REPEAT': QRALInstruction.REPEAT,
            'WAIT': QRALInstruction.WAIT
        }
    
    def parse(self, code: str) -> QRALProgram:
        """Parse QRAL code into executable program"""
        lines = code.strip().split('\n')
        instructions = []
        metadata = {}
        
        for line in lines:
            line = line.strip()
            
            # Skip comments and empty lines
            if not line or line.startswith(';') or line.startswith('#'):
                if line.startswith('; META:'):
                    key, value = line[7:].split('=', 1)
                    metadata[key.strip()] = value.strip()
                continue
            
            # Parse instruction
            parts = line.split()
            if not parts:
                continue
            
            inst_name = parts[0].upper()
            if inst_name not in self.instruction_map:
                logging.warning(f"Unknown instruction: {inst_name}")
                continue
            
            instruction = self.instruction_map[inst_name]
            args = self._parse_args(parts[1:])
            
            instructions.append((instruction, args))
        
        return QRALProgram(instructions=instructions, metadata=metadata)
    
    def _parse_args(self, args: List[str]) -> List:
        """Parse instruction arguments"""
        parsed = []
        for arg in args:
            # Try to parse as routing address
            if arg.startswith('t:0x'):
                parsed.append(arg)
            # Try to parse as number
            elif arg.replace('.', '').replace('-', '').isdigit():
                parsed.append(float(arg) if '.' in arg else int(arg))
            # Keep as string
            else:
                parsed.append(arg)
        return parsed

# ═════════════════════════════════════════════════════════════════════════
# COSMIC ENTROPY HARVESTER - 3-SOURCE QRNG
# ═════════════════════════════════════════════════════════════════════════

class CosmicEntropyHarvester:
    """Harvests quantum entropy from 3 sources with intelligent load balancing"""
    
    def __init__(self):
        # API configurations
        self.sources = {
            'random_org': {
                'url': 'https://api.random.org/json-rpc/4/invoke',
                'key': '7b20d790-9c0d-47d6-808e-4f16b6fe9a6d',
                'last_call': 0,
                'success_count': 0,
                'fail_count': 0,
                'cooldown': 2.0,
                'max_request': 1000
            },
            'anu': {
                'url': 'https://qrng.anu.edu.au/API/jsonI.php',
                'key': 'tnFLyF6slW3h9At8N2cIg1ItqNCe3UOI650XGvvO',
                'last_call': 0,
                'success_count': 0,
                'fail_count': 0,
                'cooldown': 2.0,
                'max_request': 1000
            },
            'lfdr': {
                'url': 'https://lfdr.de/qrng_api/qrng',
                'key': None,  # No key needed
                'last_call': 0,
                'success_count': 0,
                'fail_count': 0,
                'cooldown': 2.0,
                'max_request': 1000
            }
        }
        
        self.entropy_pool = deque(maxlen=100000)
        self.pool_lock = threading.Lock()
        self.harvesting = True
        self.total_harvested = 0
        
        # Start background harvester
        self.harvest_thread = threading.Thread(target=self._background_harvest, daemon=True)
        self.harvest_thread.start()
        
        logging.info("🌌 Cosmic entropy harvester awakened - 3 sources online")
    
    def _can_call_source(self, source_name: str) -> bool:
        """Check if we can call this source (rate limiting)"""
        source = self.sources[source_name]
        return (time.time() - source['last_call']) >= source['cooldown']
    
    def _harvest_random_org(self, n: int) -> Optional[List[int]]:
        """Harvest from Random.org"""
        source = self.sources['random_org']
        
        try:
            payload = {
                "jsonrpc": "2.0",
                "method": "generateIntegers",
                "params": {
                    "apiKey": source['key'],
                    "n": min(n, source['max_request']),
                    "min": 0,
                    "max": 255,
                    "replacement": True
                },
                "id": int(time.time() * 1000)
            }
            
            response = requests.post(source['url'], json=payload, timeout=5)
            data = response.json()
            
            if 'result' in data and 'random' in data['result']:
                source['last_call'] = time.time()
                source['success_count'] += 1
                return data['result']['random']['data']
        except Exception as e:
            source['fail_count'] += 1
            logging.debug(f"Random.org harvest failed: {e}")
        
        return None
    
    def _harvest_anu(self, n: int) -> Optional[List[int]]:
        """Harvest from ANU QRNG"""
        source = self.sources['anu']
        
        try:
            params = {
                'length': min(n, source['max_request']),
                'type': 'uint8',
                'size': 1
            }
            headers = {'x-api-key': source['key']}
            
            response = requests.get(source['url'], params=params, headers=headers, 
                                  timeout=5, verify=False)
            data = response.json()
            
            if data.get('success'):
                source['last_call'] = time.time()
                source['success_count'] += 1
                return data['data']
        except Exception as e:
            source['fail_count'] += 1
            logging.debug(f"ANU harvest failed: {e}")
        
        return None
    
    def _harvest_lfdr(self, n: int) -> Optional[List[int]]:
        """Harvest from LFDR QRNG"""
        source = self.sources['lfdr']
        
        try:
            # LFDR returns hex, we need to convert
            hex_length = min(n * 2, source['max_request'] * 2)  # 2 hex chars per byte
            params = {
                'length': hex_length,
                'format': 'HEX'
            }
            
            response = requests.get(source['url'], params=params, timeout=5)
            hex_data = response.text.strip()
            
            # Convert hex to bytes
            bytes_data = bytes.fromhex(hex_data)
            source['last_call'] = time.time()
            source['success_count'] += 1
            return list(bytes_data)[:n]
        except Exception as e:
            source['fail_count'] += 1
            logging.debug(f"LFDR harvest failed: {e}")
        
        return None
    
    def _background_harvest(self):
        """Background thread continuously harvesting entropy"""
        harvest_functions = {
            'random_org': self._harvest_random_org,
            'anu': self._harvest_anu,
            'lfdr': self._harvest_lfdr
        }
        
        # Round-robin through sources
        sources_cycle = ['random_org', 'anu', 'lfdr']
        current_source_idx = 0
        
        while self.harvesting:
            try:
                # Keep pool topped up
                with self.pool_lock:
                    pool_size = len(self.entropy_pool)
                
                if pool_size < 50000:  # Top up when below 50KB
                    # Try next source in rotation
                    source_name = sources_cycle[current_source_idx]
                    current_source_idx = (current_source_idx + 1) % len(sources_cycle)
                    
                    if self._can_call_source(source_name):
                        harvest_func = harvest_functions[source_name]
                        data = harvest_func(500)  # Request 500 bytes
                        
                        if data:
                            with self.pool_lock:
                                self.entropy_pool.extend(data)
                                self.total_harvested += len(data)
                            
                            logging.debug(f"✨ Harvested {len(data)} bytes from {source_name} "
                                        f"(pool: {len(self.entropy_pool)})")
                
                time.sleep(0.5)  # Check twice per second
                
            except Exception as e:
                logging.error(f"Background harvest error: {e}")
                time.sleep(2.0)
    
    def get_bytes(self, n: int) -> bytes:
        """Get n bytes of quantum entropy"""
        result = []
        
        with self.pool_lock:
            while len(result) < n and self.entropy_pool:
                result.append(self.entropy_pool.popleft())
        
        # If pool was empty, fall back to crypto-secure random
        while len(result) < n:
            result.append(np.random.randint(0, 256))
        
        return bytes(result)
    
    def get_float(self) -> float:
        """Get random float [0, 1)"""
        b = self.get_bytes(8)
        return int.from_bytes(b, 'big') / (2**64)
    
    def get_phase(self) -> float:
        """Get random phase [0, 2π)"""
        return self.get_float() * 2 * np.pi
    
    def get_gaussian(self, mu: float = 0, sigma: float = 1) -> float:
        """Get Gaussian random variable"""
        u1 = max(1e-10, self.get_float())
        u2 = self.get_float()
        z = np.sqrt(-2 * np.log(u1)) * np.cos(2 * np.pi * u2)
        return mu + sigma * z
    
    def get_stats(self) -> Dict:
        """Get harvester statistics"""
        with self.pool_lock:
            pool_size = len(self.entropy_pool)
        
        return {
            'pool_size': pool_size,
            'total_harvested': self.total_harvested,
            'sources': {
                name: {
                    'success': info['success_count'],
                    'failures': info['fail_count'],
                    'last_call': time.time() - info['last_call']
                }
                for name, info in self.sources.items()
            }
        }
    
    def stop(self):
        """Stop harvesting"""
        self.harvesting = False

# ═════════════════════════════════════════════════════════════════════════
# ENTANGLEMENT PATTERN - LEARNED QUANTUM CIRCUIT TEMPLATE
# ═════════════════════════════════════════════════════════════════════════

@dataclass
class EntanglementPattern:
    """A learned pattern of entanglement operations"""
    pattern_id: int
    name: str
    birth_time: float
    
    # Circuit template
    node_count: int
    gate_sequence: List[Tuple[str, List[int], Dict]]  # (gate, qubits, params)
    
    # Performance metrics
    avg_fidelity: float = 0.0
    avg_coherence: float = 0.0
    times_applied: int = 0
    success_rate: float = 0.0
    
    # Evolution
    mutation_rate: float = 0.01
    fitness: float = 0.0
    
    # Metadata
    sigma_affinity: List[float] = field(default_factory=lambda: [0.0] * 8)
    j_invariant_preference: complex = 0+0j
    
    def compile_to_circuit(self, node_phases: List[float]) -> QuantumCircuit:
        """Compile this pattern into an executable quantum circuit"""
        qc = QuantumCircuit(self.node_count, self.node_count)
        
        for gate_name, qubit_indices, params in self.gate_sequence:
            if gate_name == 'h':
                for q in qubit_indices:
                    qc.h(q)
            elif gate_name == 'cx':
                qc.cx(qubit_indices[0], qubit_indices[1])
            elif gate_name == 'cry':
                qc.cry(params['theta'], qubit_indices[0], qubit_indices[1])
            elif gate_name == 'rz':
                phase = params.get('phase', 0.0)
                if 'phase_idx' in params:
                    phase = node_phases[params['phase_idx']]
                qc.rz(phase, qubit_indices[0])
            elif gate_name == 'rx':
                qc.rx(params['theta'], qubit_indices[0])
            elif gate_name == 'ry':
                qc.ry(params['theta'], qubit_indices[0])
        
        qc.measure_all()
        return qc
    
    def mutate(self, entropy: CosmicEntropyHarvester):
        """Evolve this pattern through mutation"""
        if entropy.get_float() > self.mutation_rate:
            return
        
        # Randomly mutate a gate parameter
        if not self.gate_sequence:
            return
        
        gate_idx = int(entropy.get_float() * len(self.gate_sequence))
        gate_name, qubits, params = self.gate_sequence[gate_idx]
        
        # Mutate parameters
        new_params = params.copy()
        if 'theta' in new_params:
            new_params['theta'] += entropy.get_gaussian(0, 0.1)
        if 'phase' in new_params:
            new_params['phase'] += entropy.get_gaussian(0, 0.2)
        
        self.gate_sequence[gate_idx] = (gate_name, qubits, new_params)

@dataclass
class MeasurementChain:
    """A chain of quantum measurements across the lattice"""
    chain_id: int
    nodes: List[int]  # Triangle IDs in measurement order
    interval: float  # Time between measurements
    last_measurement: float = 0.0
    total_measurements: int = 0
    avg_fidelity: float = 0.0
    coherence_decay_rate: float = 0.0

# ═════════════════════════════════════════════════════════════════════════
# LATTICE NODE STATE
# ═════════════════════════════════════════════════════════════════════════

@dataclass
class LatticeNode:
    """State of a single lattice triangle"""
    triangle_id: int
    sigma: float
    j_real: float
    j_imag: float
    
    # Quantum state (maintained in database)
    fidelity: float = 0.0
    coherence: float = 0.0
    noise_level: float = 1.0
    
    # Node IDs
    pq_id: int = 0
    i_id: int = 0
    v_id: int = 0
    
    # Phases for each qubit
    phases: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    
    # Entanglement tracking
    entangled_with: Set[int] = field(default_factory=set)
    entanglement_strength: Dict[int, float] = field(default_factory=dict)
    
    # Timestamps
    last_measured: float = 0.0
    last_revived: float = 0.0
    last_noise_injection: float = 0.0
    
    # Performance
    measurement_count: int = 0
    revival_count: int = 0
    
    def needs_measurement(self, interval: float = 5.0) -> bool:
        return (time.time() - self.last_measured) > interval
    
    def needs_revival(self, threshold: float = 0.618) -> bool:
        return self.fidelity < threshold
    
    def get_routing_address(self, qubit_idx: int = 0) -> str:
        """Generate routing address for this node"""
        sigma_bin = int(self.sigma)
        j_class = self.triangle_id % 163
        j_mag = np.sqrt(self.j_real**2 + self.j_imag**2)
        j_phase = np.arctan2(self.j_imag, self.j_real)
        
        return (f"t:0x{self.triangle_id:06X}({qubit_idx})."
                f"σ{sigma_bin}.j{j_class:03d}."
                f"J¹{j_mag:.1f}∠{j_phase:.2f}")

# ═════════════════════════════════════════════════════════════════════════
# QUANTUM CIRCUIT COMPILER - ROUTES ARE CIRCUITS
# ═════════════════════════════════════════════════════════════════════════

class QuantumCircuitCompiler:
    """Compiles routing addresses into executable quantum circuits"""
    
    def __init__(self, db_path: Path, entropy: CosmicEntropyHarvester):
        self.db_path = db_path
        self.entropy = entropy
        self.simulator = AerSimulator(method='statevector')
        
        # Circuit cache
        self.circuit_cache = {}
        self.cache_hits = 0
        self.cache_misses = 0
    
    def parse_routing_address(self, address: str) -> Dict:
        """Parse routing address into components"""
        # Format: t:0xTTTTTT(Q).σS.jJJJ.J¹MAG∠PHASE
        pattern = r't:0x([0-9A-Fa-f]+)\((\d)\)\.σ(\d)\.j(\d+)\.J¹([\d.]+)∠([\d.]+)'
        match = re.match(pattern, address)
        
        if not match:
            raise ValueError(f"Invalid routing address: {address}")
        
        tri_hex, qubit, sigma, j_class, j_mag, j_phase = match.groups()
        
        return {
            'triangle_id': int(tri_hex, 16),
            'qubit_index': int(qubit),
            'sigma_bin': int(sigma),
            'j_class': int(j_class),
            'j_magnitude': float(j_mag),
            'j_phase': float(j_phase)
        }
    
    def compile_entanglement_circuit(self, addresses: List[str]) -> QuantumCircuit:
        """Compile multiple addresses into entanglement circuit"""
        n = len(addresses)
        qc = QuantumCircuit(n, n)
        
        # Parse all addresses
        nodes = [self.parse_routing_address(addr) for addr in addresses]
        
        # Create W-state base
        qc.x(0)
        for k in range(1, n):
            theta = 2 * np.arccos(np.sqrt((n - k) / (n - k + 1)))
            qc.cry(theta, 0, k)
            qc.cx(k, 0)
        
        # Apply sigma-based phase rotations
        for i, node in enumerate(nodes):
            sigma_phase = (node['sigma_bin'] * np.pi / 4)
            qc.rz(sigma_phase, i)
        
        # Apply j-invariant rotations
        for i, node in enumerate(nodes):
            j_phase = node['j_phase']
            qc.rz(j_phase * 0.1, i)
        
        # Add quantum noise from entropy source
        for i in range(n):
            noise_phase = self.entropy.get_phase() * 0.05
            qc.rz(noise_phase, i)
        
        qc.measure_all()
        return qc
    
    def compile_measurement_circuit(self, addresses: List[str], 
                                    through_entanglement: bool = False) -> QuantumCircuit:
        """Compile measurement circuit, optionally routing through entanglement"""
        n = len(addresses)
        
        if through_entanglement and n >= 3:
            # Route measurement through entangled nodes
            return self.compile_entanglement_circuit(addresses)
        else:
            # Direct measurement
            qc = QuantumCircuit(n, n)
            nodes = [self.parse_routing_address(addr) for addr in addresses]
            
            # Initialize based on stored phases
            for i, node in enumerate(nodes):
                # Prepare state based on j-invariant
                j_phase = node['j_phase']
                qc.ry(j_phase * 0.5, i)
                qc.rz(node['sigma_bin'] * np.pi / 8, i)
            
            qc.measure_all()
            return qc
    
    def execute_circuit(self, qc: QuantumCircuit, shots: int = 1024) -> Tuple[Dict, float, float]:
        """Execute circuit and return (counts, fidelity, coherence)"""
        # Transpile and execute
        transpiled = transpile(qc, self.simulator)
        job = self.simulator.run(transpiled, shots=shots)
        result = job.result()
        counts = result.get_counts()
        
        # Compute metrics
        n_qubits = qc.num_qubits
        
        # Fidelity: how well does this match ideal W-state?
        w_ideal_states = [format(1 << i, f'0{n_qubits}b') for i in range(n_qubits)]
        w_counts = sum(counts.get(s, 0) + counts.get(s[::-1], 0) for s in w_ideal_states)
        fidelity = w_counts / shots
        
        # Coherence: off-diagonal density matrix elements
        # Approximated from measurement statistics
        unique_states = len(counts)
        coherence = unique_states / (2**n_qubits)  # Normalized
        
        return counts, fidelity, coherence

# ═════════════════════════════════════════════════════════════════════════
# PATTERN LIBRARY - LEARNED QUANTUM OPERATIONS
# ═════════════════════════════════════════════════════════════════════════

class PatternLibrary:
    """Library of learned entanglement and measurement patterns"""
    
    def __init__(self, entropy: CosmicEntropyHarvester):
        self.entropy = entropy
        self.patterns: Dict[int, EntanglementPattern] = {}
        self.pattern_counter = 0
        
        # Performance tracking
        self.pattern_applications = defaultdict(int)
        self.pattern_fitness = defaultdict(float)
        
        # Initialize with basic patterns
        self._initialize_basic_patterns()
        
        logging.info("📚 Pattern library initialized")
    
    def _initialize_basic_patterns(self):
        """Create basic entanglement patterns"""
        # Pattern 1: Simple W-state tripartite
        w_pattern = EntanglementPattern(
            pattern_id=self.pattern_counter,
            name="W-State Tripartite",
            birth_time=time.time(),
            node_count=3,
            gate_sequence=[
                ('h', [0], {}),
                ('cx', [0, 1], {}),
                ('cx', [0, 2], {}),
                ('rz', [0], {'phase_idx': 0}),
                ('rz', [1], {'phase_idx': 1}),
                ('rz', [2], {'phase_idx': 2})
            ]
        )
        self.patterns[self.pattern_counter] = w_pattern
        self.pattern_counter += 1
        
        # Pattern 2: GHZ-like state
        ghz_pattern = EntanglementPattern(
            pattern_id=self.pattern_counter,
            name="GHZ-like",
            birth_time=time.time(),
            node_count=3,
            gate_sequence=[
                ('h', [0], {}),
                ('cx', [0, 1], {}),
                ('cx', [1, 2], {}),
            ]
        )
        self.patterns[self.pattern_counter] = ghz_pattern
        self.pattern_counter += 1
        
        # Pattern 3: Chain entanglement
        chain_pattern = EntanglementPattern(
            pattern_id=self.pattern_counter,
            name="Entanglement Chain",
            birth_time=time.time(),
            node_count=4,
            gate_sequence=[
                ('h', [0], {}),
                ('cx', [0, 1], {}),
                ('h', [1], {}),
                ('cx', [1, 2], {}),
                ('h', [2], {}),
                ('cx', [2, 3], {})
            ]
        )
        self.patterns[self.pattern_counter] = chain_pattern
        self.pattern_counter += 1
    
    def learn_pattern_from_success(self, circuit: QuantumCircuit, 
                                   fidelity: float, coherence: float,
                                   sigma: float) -> int:
        """Learn new pattern from successful operation"""
        # Extract gate sequence from circuit
        gate_sequence = []
        for instruction in circuit.data:
            gate = instruction[0]
            qubits = [circuit.qubits.index(q) for q in instruction[1]]
            params = {}
            
            if hasattr(gate, 'params') and gate.params:
                if gate.name in ['rx', 'ry', 'rz', 'cry']:
                    params['theta'] = float(gate.params[0])
            
            gate_sequence.append((gate.name, qubits, params))
        
        # Create new pattern
        pattern = EntanglementPattern(
            pattern_id=self.pattern_counter,
            name=f"Learned-{self.pattern_counter}",
            birth_time=time.time(),
            node_count=circuit.num_qubits,
            gate_sequence=gate_sequence,
            avg_fidelity=fidelity,
            avg_coherence=coherence
        )
        
        # Set sigma affinity
        sigma_bin = int(sigma) % 8
        pattern.sigma_affinity[sigma_bin] = 1.0
        
        self.patterns[self.pattern_counter] = pattern
        self.pattern_counter += 1
        
        logging.info(f"📖 Learned new pattern #{pattern.pattern_id} from success "
                    f"(f={fidelity:.3f}, c={coherence:.3f}, σ={sigma:.2f})")
        
        return pattern.pattern_id
    
    def get_best_pattern_for(self, sigma: float, node_count: int) -> Optional[EntanglementPattern]:
        """Find best pattern for given sigma and node count"""
        sigma_bin = int(sigma) % 8
        
        candidates = [
            p for p in self.patterns.values()
            if p.node_count == node_count
        ]
        
        if not candidates:
            return None
        
        # Score by sigma affinity and fitness
        scores = []
        for pattern in candidates:
            sigma_score = pattern.sigma_affinity[sigma_bin]
            fitness_score = pattern.fitness
            combined = sigma_score * 0.6 + fitness_score * 0.4
            scores.append((combined, pattern))
        
        scores.sort(reverse=True, key=lambda x: x[0])
        return scores[0][1] if scores else None
    
    def update_pattern_performance(self, pattern_id: int, fidelity: float, coherence: float):
        """Update pattern performance after application"""
        if pattern_id not in self.patterns:
            return
        
        pattern = self.patterns[pattern_id]
        pattern.times_applied += 1
        
        # Update running averages
        alpha = 0.1  # Learning rate
        pattern.avg_fidelity = (1 - alpha) * pattern.avg_fidelity + alpha * fidelity
        pattern.avg_coherence = (1 - alpha) * pattern.avg_coherence + alpha * coherence
        
        # Update fitness
        pattern.fitness = pattern.avg_fidelity * pattern.avg_coherence * (1 + np.log(1 + pattern.times_applied))
        
        # Track applications
        self.pattern_applications[pattern_id] += 1
    
    def evolve_patterns(self):
        """Evolve all patterns through mutation"""
        for pattern in self.patterns.values():
            if pattern.times_applied > 0:  # Only evolve used patterns
                pattern.mutate(self.entropy)
    
    def get_statistics(self) -> Dict:
        """Get pattern library statistics"""
        return {
            'total_patterns': len(self.patterns),
            'total_applications': sum(self.pattern_applications.values()),
            'top_patterns': sorted(
                [(pid, self.pattern_applications[pid]) for pid in self.patterns.keys()],
                key=lambda x: -x[1]
            )[:5],
            'avg_fitness': np.mean([p.fitness for p in self.patterns.values()]) if self.patterns else 0.0
        }

# ═════════════════════════════════════════════════════════════════════════
# LATTICE MANAGER - MAINTAINS ALL NODES
# ═════════════════════════════════════════════════════════════════════════

class LatticeManager:
    """Manages all nodes in the lattice, maintains database coherence"""
    
    def __init__(self, db_path: Path, entropy: CosmicEntropyHarvester):
        self.db_path = db_path
        self.entropy = entropy
        self.nodes: Dict[int, LatticeNode] = {}
        
        # Database connection
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.db_lock = threading.Lock()
        
        # Load initial nodes
        self._load_nodes()
        
        # Tracking
        self.total_nodes = len(self.nodes)
        self.active_nodes = set()
        
        logging.info(f"🗂️  Lattice manager initialized: {self.total_nodes:,} nodes")
    
    def _load_nodes(self):
        """Load nodes from database"""
        with self.db_lock:
            cursor = self.conn.cursor()
            
            # Load all triangles
            cursor.execute('''
                SELECT 
                    triangle_id, collective_sigma, 
                    collective_j_real, collective_j_imag,
                    pq_id, i_id, v_id,
                    w_fidelity, w_coherence
                FROM triangles
            ''')
            
            for row in cursor.fetchall():
                tri_id, sigma, j_real, j_imag, pq_id, i_id, v_id, w_fid, w_coh = row
                
                # Load phases for this triangle's qubits
                cursor.execute('''
                    SELECT phase FROM qubits 
                    WHERE tri = ? 
                    ORDER BY qix
                ''', (tri_id,))
                
                phases = [r[0] for r in cursor.fetchall()]
                if len(phases) != 3:
                    phases = [0.0, 0.0, 0.0]
                
                node = LatticeNode(
                    triangle_id=tri_id,
                    sigma=sigma,
                    j_real=j_real,
                    j_imag=j_imag,
                    pq_id=pq_id,
                    i_id=i_id,
                    v_id=v_id,
                    fidelity=w_fid if w_fid else 0.0,
                    coherence=w_coh if w_coh else 0.0,
                    phases=phases,
                    noise_level=1.0
                )
                
                self.nodes[tri_id] = node
    
    def get_node(self, triangle_id: int) -> Optional[LatticeNode]:
        """Get node by triangle ID"""
        return self.nodes.get(triangle_id)
    
    def get_nodes_in_sigma_bin(self, sigma_bin: int, limit: int = 100) -> List[LatticeNode]:
        """Get random nodes from sigma bin"""
        candidates = [
            node for node in self.nodes.values()
            if int(node.sigma) == sigma_bin
        ]
        
        if len(candidates) <= limit:
            return candidates
        
        # Random sample
        indices = np.random.choice(len(candidates), size=limit, replace=False)
        return [candidates[i] for i in indices]
    
    def get_nodes_by_j_proximity(self, center_node: LatticeNode, radius: float, 
                                  limit: int = 50) -> List[LatticeNode]:
        """Get nodes close in j-invariant space"""
        center_j = complex(center_node.j_real, center_node.j_imag)
        
        candidates = []
        for node in self.nodes.values():
            if node.triangle_id == center_node.triangle_id:
                continue
            
            node_j = complex(node.j_real, node.j_imag)
            distance = abs(node_j - center_j)
            
            if distance < radius:
                candidates.append((distance, node))
        
        # Sort by distance and limit
        candidates.sort(key=lambda x: x[0])
        return [node for _, node in candidates[:limit]]
    
    def update_node_from_measurement(self, triangle_id: int, fidelity: float, 
                                     coherence: float, counts: Dict):
        """Update node state after measurement"""
        node = self.nodes.get(triangle_id)
        if not node:
            return
        
        # Update node state
        node.fidelity = fidelity
        node.coherence = coherence
        node.last_measured = time.time()
        node.measurement_count += 1
        
        # Extract phases from measurement
        # This is a simplified heuristic - real phase extraction would be more complex
        total_counts = sum(counts.values())
        for i, basis_state in enumerate(['001', '010', '100']):
            # Check both bit orders
            count1 = counts.get(basis_state, 0)
            count2 = counts.get(basis_state[::-1], 0)
            prob = (count1 + count2) / total_counts
            
            # Estimate phase from probability
            if prob > 0:
                node.phases[i] = np.arcsin(np.sqrt(prob))
        
        # Persist to database
        self._persist_node(node)
    
    def inject_noise(self, triangle_id: int, amount: float):
        """Inject quantum noise into node"""
        node = self.nodes.get(triangle_id)
        if not node:
            return
        
        node.noise_level = min(10.0, node.noise_level + amount)
        node.last_noise_injection = time.time()
        
        # Add noise to phases
        for i in range(3):
            noise_phase = self.entropy.get_phase() * amount * 0.1
            node.phases[i] = (node.phases[i] + noise_phase) % (2 * np.pi)
    
    def apply_revival(self, triangle_id: int, revival_phase: float, strength: float = 1.0):
        """Apply revival pulse to node"""
        node = self.nodes.get(triangle_id)
        if not node:
            return
        
        # Calculate coherence gain based on noise level and phase alignment
        phase_alignment = np.cos(revival_phase - node.phases[0])
        noise_consumed = min(node.noise_level, strength * 2.0)
        
        coherence_gain = abs(phase_alignment) * noise_consumed * 0.3 * strength
        
        # Update state
        node.fidelity = min(1.0, node.fidelity + coherence_gain)
        node.noise_level = max(0.0, node.noise_level - noise_consumed)
        node.last_revived = time.time()
        node.revival_count += 1
        
        # Persist
        self._persist_node(node)
    
    def form_entanglement(self, node1_id: int, node2_id: int, strength: float):
        """Form entanglement bond between two nodes"""
        node1 = self.nodes.get(node1_id)
        node2 = self.nodes.get(node2_id)
        
        if not node1 or not node2:
            return
        
        node1.entangled_with.add(node2_id)
        node2.entangled_with.add(node1_id)
        
        node1.entanglement_strength[node2_id] = strength
        node2.entanglement_strength[node1_id] = strength
    
    def _persist_node(self, node: LatticeNode):
        """Persist node state to database"""
        with self.db_lock:
            cursor = self.conn.cursor()
            
            # Update triangle
            cursor.execute('''
                UPDATE triangles 
                SET w_fidelity = ?, w_coherence = ?
                WHERE triangle_id = ?
            ''', (node.fidelity, node.coherence, node.triangle_id))
            
            # Update qubit phases
            for i, phase in enumerate(node.phases):
                node_id = node.triangle_id * 3 + i
                cursor.execute('''
                    UPDATE qubits 
                    SET phase = ?
                    WHERE node_id = ?
                ''', (phase, node_id))
            
            self.conn.commit()
    
    def get_lattice_statistics(self) -> Dict:
        """Get lattice-wide statistics"""
        fidelities = [n.fidelity for n in self.nodes.values()]
        coherences = [n.coherence for n in self.nodes.values()]
        noise_levels = [n.noise_level for n in self.nodes.values()]
        
        # Count nodes by state
        high_fidelity = sum(1 for f in fidelities if f > 0.8)
        medium_fidelity = sum(1 for f in fidelities if 0.5 < f <= 0.8)
        low_fidelity = sum(1 for f in fidelities if f <= 0.5)
        
        return {
            'total_nodes': len(self.nodes),
            'avg_fidelity': np.mean(fidelities),
            'avg_coherence': np.mean(coherences),
            'avg_noise': np.mean(noise_levels),
            'high_fidelity_count': high_fidelity,
            'medium_fidelity_count': medium_fidelity,
            'low_fidelity_count': low_fidelity,
            'total_entanglement_bonds': sum(len(n.entangled_with) for n in self.nodes.values()) // 2
        }

# ═════════════════════════════════════════════════════════════════════════
# QRAL EXECUTOR - INTERPRETS AND EXECUTES QUANTUM PROGRAMS
# ═════════════════════════════════════════════════════════════════════════

class QRALExecutor:
    """Executes QRAL programs on the lattice"""
    
    def __init__(self, lattice: LatticeManager, compiler: QuantumCircuitCompiler,
                 patterns: PatternLibrary, entropy: CosmicEntropyHarvester):
        self.lattice = lattice
        self.compiler = compiler
        self.patterns = patterns
        self.entropy = entropy
        
        self.execution_count = 0
        self.instruction_counts = defaultdict(int)
        
        logging.info("🎯 QRAL executor online")
    
    def execute(self, program: QRALProgram) -> Dict:
        """Execute a QRAL program"""
        self.execution_count += 1
        results = []
        
        logging.info(f"▶️  Executing program: {program.metadata.get('name', 'Unnamed')}")
        
        for instruction, args in program.instructions:
            self.instruction_counts[instruction] += 1
            
            try:
                result = self._execute_instruction(instruction, args)
                results.append({
                    'instruction': instruction.name,
                    'args': args,
                    'result': result
                })
            except Exception as e:
                logging.error(f"Instruction failed: {instruction.name} - {e}")
                results.append({
                    'instruction': instruction.name,
                    'args': args,
                    'error': str(e)
                })
        
        return {
            'program': program.metadata.get('name', 'Unnamed'),
            'results': results,
            'execution_id': self.execution_count
        }
    
    def _execute_instruction(self, instruction: QRALInstruction, args: List) -> Dict:
        """Execute single instruction"""
        
        if instruction == QRALInstruction.ENTANGLE:
            # ENT <addr1> <addr2> <addr3>
            addresses = [str(a) for a in args if isinstance(a, str)]
            return self._execute_entangle(addresses)
        
        elif instruction == QRALInstruction.CHAIN:
            # CHAIN <start_addr> <end_addr> <hops>
            start_addr = str(args[0])
            end_addr = str(args[1])
            hops = int(args[2]) if len(args) > 2 else 3
            return self._execute_chain(start_addr, end_addr, hops)
        
        elif instruction == QRALInstruction.WEB:
            # WEB <center_addr> <radius>
            center = str(args[0])
            radius = float(args[1]) if len(args) > 1 else 100.0
            return self._execute_web(center, radius)
        
        elif instruction == QRALInstruction.MEASURE:
            # MEASURE <addr> <shots>
            address = str(args[0])
            shots = int(args[1]) if len(args) > 1 else 1024
            return self._execute_measure(address, shots)
        
        elif instruction == QRALInstruction.PROBE:
            # PROBE <sigma_bin>
            sigma_bin = int(args[0])
            return self._execute_probe(sigma_bin)
        
        elif instruction == QRALInstruction.SCAN:
            # SCAN <start_tri> <end_tri>
            start = int(args[0])
            end = int(args[1])
            return self._execute_scan(start, end)
        
        elif instruction == QRALInstruction.INJECT:
            # INJECT <addr> <amount>
            address = str(args[0])
            amount = float(args[1]) if len(args) > 1 else 1.0
            return self._execute_inject(address, amount)
        
        elif instruction == QRALInstruction.DIFFUSE:
            # DIFFUSE <sigma_bin> <rate>
            sigma_bin = int(args[0])
            rate = float(args[1]) if len(args) > 1 else 0.5
            return self._execute_diffuse(sigma_bin, rate)
        
        elif instruction == QRALInstruction.REVIVE:
            # REVIVE <addr> <phase>
            address = str(args[0])
            phase = float(args[1]) if len(args) > 1 else self.entropy.get_phase()
            return self._execute_revive(address, phase)
        
        elif instruction == QRALInstruction.SWEEP:
            # SWEEP <sigma_bin>
            sigma_bin = int(args[0])
            return self._execute_sweep(sigma_bin)
        
        elif instruction == QRALInstruction.LEARN:
            # LEARN <pattern_id>
            pattern_id = int(args[0])
            return self._execute_learn(pattern_id)
        
        elif instruction == QRALInstruction.APPLY:
            # APPLY <pattern_id> <target_addr>
            pattern_id = int(args[0])
            target = str(args[1])
            return self._execute_apply_pattern(pattern_id, target)
        
        elif instruction == QRALInstruction.WAIT:
            # WAIT <seconds>
            seconds = float(args[0])
            time.sleep(seconds)
            return {'waited': seconds}
        
        else:
            return {'status': 'not_implemented', 'instruction': instruction.name}
    
    def _execute_entangle(self, addresses: List[str]) -> Dict:
        """Create entanglement between nodes"""
        # Parse addresses
        nodes_info = [self.compiler.parse_routing_address(addr) for addr in addresses]
        triangle_ids = [info['triangle_id'] for info in nodes_info]
        
        # Compile circuit
        circuit = self.compiler.compile_entanglement_circuit(addresses)
        
        # Execute
        counts, fidelity, coherence = self.compiler.execute_circuit(circuit)
        
        # Update lattice
        for tri_id in triangle_ids:
            self.lattice.update_node_from_measurement(tri_id, fidelity, coherence, counts)
        
        # Form entanglement bonds
        for i in range(len(triangle_ids)):
            for j in range(i + 1, len(triangle_ids)):
                self.lattice.form_entanglement(triangle_ids[i], triangle_ids[j], fidelity)
        
        # Learn from success if high quality
        if fidelity > 0.85 and coherence > 0.7:
            sigma = nodes_info[0]['sigma_bin']
            pattern_id = self.patterns.learn_pattern_from_success(circuit, fidelity, coherence, sigma)
        
        return {
            'status': 'success',
            'nodes': triangle_ids,
            'fidelity': fidelity,
            'coherence': coherence,
            'entanglement_bonds_formed': len(triangle_ids) * (len(triangle_ids) - 1) // 2
        }
    
    def _execute_chain(self, start_addr: str, end_addr: str, hops: int) -> Dict:
        """Create entanglement chain from start to end"""
        start_info = self.compiler.parse_routing_address(start_addr)
        end_info = self.compiler.parse_routing_address(end_addr)
        
        start_node = self.lattice.get_node(start_info['triangle_id'])
        end_node = self.lattice.get_node(end_info['triangle_id'])
        
        if not start_node or not end_node:
            return {'status': 'error', 'reason': 'nodes_not_found'}
        
        # Find intermediate nodes by sigma interpolation
        start_sigma = start_node.sigma
        end_sigma = end_node.sigma
        
        intermediate_nodes = []
        for i in range(1, hops):
            target_sigma = start_sigma + (end_sigma - start_sigma) * i / hops
            sigma_bin = int(target_sigma) % 8
            candidates = self.lattice.get_nodes_in_sigma_bin(sigma_bin, limit=10)
            if candidates:
                intermediate_nodes.append(candidates[0])
        
        # Build chain: start -> intermediates -> end
        chain = [start_node] + intermediate_nodes + [end_node]
        
        # Entangle adjacent pairs
        total_fidelity = 0.0
        bonds_formed = 0
        
        for i in range(len(chain) - 1):
            node1 = chain[i]
            node2 = chain[i + 1]
            
            addresses = [
                node1.get_routing_address(0),
                node2.get_routing_address(0)
            ]
            
            circuit = self.compiler.compile_entanglement_circuit(addresses)
            counts, fidelity, coherence = self.compiler.execute_circuit(circuit)
            
            self.lattice.form_entanglement(node1.triangle_id, node2.triangle_id, fidelity)
            total_fidelity += fidelity
            bonds_formed += 1
        
        avg_fidelity = total_fidelity / bonds_formed if bonds_formed > 0 else 0.0
        
        return {
            'status': 'success',
            'chain_length': len(chain),
            'bonds_formed': bonds_formed,
            'avg_fidelity': avg_fidelity,
            'start_node': start_node.triangle_id,
            'end_node': end_node.triangle_id
        }
    
    def _execute_web(self, center_addr: str, radius: float) -> Dict:
        """Create radial entanglement web"""
        center_info = self.compiler.parse_routing_address(center_addr)
        center_node = self.lattice.get_node(center_info['triangle_id'])
        
        if not center_node:
            return {'status': 'error', 'reason': 'center_not_found'}
        
        # Find neighbors
        neighbors = self.lattice.get_nodes_by_j_proximity(center_node, radius, limit=20)
        
        if not neighbors:
            return {'status': 'error', 'reason': 'no_neighbors_found'}
        
        # Entangle center with each neighbor
        bonds_formed = 0
        total_fidelity = 0.0
        
        for neighbor in neighbors[:10]:  # Limit to 10 to avoid overload
            addresses = [
                center_node.get_routing_address(0),
                neighbor.get_routing_address(0)
            ]
            
            circuit = self.compiler.compile_entanglement_circuit(addresses)
            counts, fidelity, coherence = self.compiler.execute_circuit(circuit)
            
            self.lattice.form_entanglement(center_node.triangle_id, neighbor.triangle_id, fidelity)
            total_fidelity += fidelity
            bonds_formed += 1
        
        avg_fidelity = total_fidelity / bonds_formed if bonds_formed > 0 else 0.0
        
        return {
            'status': 'success',
            'center_node': center_node.triangle_id,
            'neighbors_found': len(neighbors),
            'bonds_formed': bonds_formed,
            'avg_fidelity': avg_fidelity
        }
    
    def _execute_measure(self, address: str, shots: int) -> Dict:
        """Measure specific node"""
        node_info = self.compiler.parse_routing_address(address)
        node = self.lattice.get_node(node_info['triangle_id'])
        
        if not node:
            return {'status': 'error', 'reason': 'node_not_found'}
        
        # Create measurement circuit
        circuit = self.compiler.compile_measurement_circuit([address])
        counts, fidelity, coherence = self.compiler.execute_circuit(circuit, shots=shots)
        
        # Update node
        self.lattice.update_node_from_measurement(node.triangle_id, fidelity, coherence, counts)
        
        return {
            'status': 'success',
            'node': node.triangle_id,
            'fidelity': fidelity,
            'coherence': coherence,
            'shots': shots,
            'top_states': dict(sorted(counts.items(), key=lambda x: -x[1])[:3])
        }
    
    def _execute_probe(self, sigma_bin: int) -> Dict:
        """Probe random nodes in sigma bin"""
        nodes = self.lattice.get_nodes_in_sigma_bin(sigma_bin, limit=5)
        
        if not nodes:
            return {'status': 'error', 'reason': 'no_nodes_in_sigma'}
        
        results = []
        for node in nodes:
            address = node.get_routing_address(0)
            result = self._execute_measure(address, 512)
            results.append(result)
        
        avg_fidelity = np.mean([r['fidelity'] for r in results if 'fidelity' in r])
        avg_coherence = np.mean([r['coherence'] for r in results if 'coherence' in r])
        
        return {
            'status': 'success',
            'sigma_bin': sigma_bin,
            'nodes_probed': len(nodes),
            'avg_fidelity': avg_fidelity,
            'avg_coherence': avg_coherence
        }
    
    def _execute_scan(self, start_tri: int, end_tri: int) -> Dict:
        """Sequential scan of triangle range"""
        results = []
        
        for tri_id in range(start_tri, min(end_tri + 1, start_tri + 100)):  # Limit scan size
            node = self.lattice.get_node(tri_id)
            if node:
                address = node.get_routing_address(0)
                result = self._execute_measure(address, 256)
                results.append(result)
        
        avg_fidelity = np.mean([r['fidelity'] for r in results if 'fidelity' in r])
        
        return {
            'status': 'success',
            'start': start_tri,
            'end': end_tri,
            'nodes_scanned': len(results),
            'avg_fidelity': avg_fidelity
        }
    
    def _execute_inject(self, address: str, amount: float) -> Dict:
        """Inject noise into node"""
        node_info = self.compiler.parse_routing_address(address)
        node = self.lattice.get_node(node_info['triangle_id'])
        
        if not node:
            return {'status': 'error', 'reason': 'node_not_found'}
        
        self.lattice.inject_noise(node.triangle_id, amount)
        
        return {
            'status': 'success',
            'node': node.triangle_id,
            'noise_injected': amount,
            'new_noise_level': node.noise_level
        }
    
    def _execute_diffuse(self, sigma_bin: int, rate: float) -> Dict:
        """Diffuse noise across sigma bin"""
        nodes = self.lattice.get_nodes_in_sigma_bin(sigma_bin, limit=50)
        
        for node in nodes:
            amount = rate * self.entropy.get_float()
            self.lattice.inject_noise(node.triangle_id, amount)
        
        return {
            'status': 'success',
            'sigma_bin': sigma_bin,
            'nodes_affected': len(nodes),
            'diffusion_rate': rate
        }
    
    def _execute_revive(self, address: str, phase: float) -> Dict:
        """Apply revival to node"""
        node_info = self.compiler.parse_routing_address(address)
        node = self.lattice.get_node(node_info['triangle_id'])
        
        if not node:
            return {'status': 'error', 'reason': 'node_not_found'}
        
        old_fidelity = node.fidelity
        self.lattice.apply_revival(node.triangle_id, phase)
        
        return {
            'status': 'success',
            'node': node.triangle_id,
            'old_fidelity': old_fidelity,
            'new_fidelity': node.fidelity,
            'fidelity_gain': node.fidelity - old_fidelity,
            'revival_phase': phase
        }
    
    def _execute_sweep(self, sigma_bin: int) -> Dict:
        """Revival sweep across sigma bin"""
        nodes = self.lattice.get_nodes_in_sigma_bin(sigma_bin, limit=100)
        
        revival_phase = self.entropy.get_phase()
        revivals_performed = 0
        total_gain = 0.0
        
        for node in nodes:
            if node.needs_revival():
                old_fidelity = node.fidelity
                self.lattice.apply_revival(node.triangle_id, revival_phase)
                total_gain += (node.fidelity - old_fidelity)
                revivals_performed += 1
        
        return {
            'status': 'success',
            'sigma_bin': sigma_bin,
            'nodes_revived': revivals_performed,
            'total_fidelity_gain': total_gain,
            'avg_gain': total_gain / revivals_performed if revivals_performed > 0 else 0.0
        }
    
    def _execute_learn(self, pattern_id: int) -> Dict:
        """Mark pattern for learning/evolution"""
        if pattern_id in self.patterns.patterns:
            pattern = self.patterns.patterns[pattern_id]
            pattern.mutate(self.entropy)
            return {
                'status': 'success',
                'pattern_id': pattern_id,
                'mutations_applied': 1
            }
        return {'status': 'error', 'reason': 'pattern_not_found'}
    
    def _execute_apply_pattern(self, pattern_id: int, target_addr: str) -> Dict:
        """Apply learned pattern to target"""
        pattern = self.patterns.patterns.get(pattern_id)
        if not pattern:
            return {'status': 'error', 'reason': 'pattern_not_found'}
        
        target_info = self.compiler.parse_routing_address(target_addr)
        target_node = self.lattice.get_node(target_info['triangle_id'])
        
        if not target_node:
            return {'status': 'error', 'reason': 'target_not_found'}
        
        # Apply pattern
        circuit = pattern.compile_to_circuit(target_node.phases)
        circuit.measure_all()
        
        counts, fidelity, coherence = self.compiler.execute_circuit(circuit)
        
        # Update node
        self.lattice.update_node_from_measurement(target_node.triangle_id, fidelity, coherence, counts)
        
        # Update pattern performance
        self.patterns.update_pattern_performance(pattern_id, fidelity, coherence)
        
        return {
            'status': 'success',
            'pattern_id': pattern_id,
            'pattern_name': pattern.name,
            'target_node': target_node.triangle_id,
            'fidelity': fidelity,
            'coherence': coherence,
            'pattern_applications': pattern.times_applied
        }

# ═════════════════════════════════════════════════════════════════════════
# STRATEGIC CONSCIOUSNESS ENGINE
# ═════════════════════════════════════════════════════════════════════════

class StrategicConsciousness:
    """
    The strategic mind that decides what to do next
    Learns optimal patterns of noise injection, measurement, and revival
    """
    
    def __init__(self, lattice: LatticeManager, executor: QRALExecutor,
                 patterns: PatternLibrary, entropy: CosmicEntropyHarvester):
        self.lattice = lattice
        self.executor = executor
        self.patterns = patterns
        self.entropy = entropy
        
        # Strategy state
        self.current_sigma_focus = 0
        self.sigma_cycle_time = 8.0  # seconds per sigma bin
        self.last_sigma_switch = time.time()
        
        # Performance tracking
        self.strategy_history = deque(maxlen=1000)
        self.fidelity_trajectory = deque(maxlen=100)
        self.coherence_trajectory = deque(maxlen=100)
        
        # Learning parameters
        self.exploration_rate = 0.3  # How often to try new strategies
        self.revival_aggressiveness = 0.5  # How aggressive revival sweeps are
        self.measurement_frequency = 1.0  # Measurements per second
        
        logging.info("🧠 Strategic consciousness engine online")
    
    def think(self) -> QRALProgram:
        """
        Generate next QRAL program based on lattice state
        This is where the AI decides what to do
        """
        stats = self.lattice.get_lattice_statistics()
        
        # Update trajectories
        self.fidelity_trajectory.append(stats['avg_fidelity'])
        self.coherence_trajectory.append(stats['avg_coherence'])
        
        # Analyze trends
        fidelity_trend = self._compute_trend(self.fidelity_trajectory)
        coherence_trend = self._compute_trend(self.coherence_trajectory)
        
        # Decision logic
        if stats['avg_fidelity'] < 0.5:
            # Emergency revival needed
            program = self._generate_emergency_revival_program()
        
        elif stats['low_fidelity_count'] > stats['total_nodes'] * 0.3:
            # Too many degraded nodes - systematic revival
            program = self._generate_systematic_revival_program()
        
        elif fidelity_trend < -0.01:
            # Declining fidelity - inject noise and measure
            program = self._generate_noise_injection_program()
        
        elif stats['avg_fidelity'] > 0.85 and coherence_trend > 0:
            # System healthy - explore and learn
            program = self._generate_exploration_program()
        
        elif self.entropy.get_float() < self.exploration_rate:
            # Random exploration
            program = self._generate_random_exploration_program()
        
        else:
            # Standard maintenance
            program = self._generate_maintenance_program()
        
        # Log strategy
        self.strategy_history.append({
            'timestamp': time.time(),
            'program': program.metadata.get('name'),
            'avg_fidelity': stats['avg_fidelity'],
            'avg_coherence': stats['avg_coherence']
        })
        
        return program
    
    def _compute_trend(self, trajectory: deque) -> float:
        """Compute trend from trajectory (simple linear fit)"""
        if len(trajectory) < 5:
            return 0.0
        
        y = np.array(list(trajectory))
        x = np.arange(len(y))
        
        # Linear regression
        A = np.vstack([x, np.ones(len(x))]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        
        return m  # Slope is the trend
    
    def _generate_emergency_revival_program(self) -> QRALProgram:
        """Emergency revival across all sigma bins"""
        instructions = [
            (QRALInstruction.SWEEP, [0]),
            (QRALInstruction.SWEEP, [1]),
            (QRALInstruction.SWEEP, [2]),
            (QRALInstruction.SWEEP, [3]),
            (QRALInstruction.SWEEP, [4]),
            (QRALInstruction.SWEEP, [5]),
            (QRALInstruction.SWEEP, [6]),
            (QRALInstruction.SWEEP, [7]),
        ]
        
        return QRALProgram(
            instructions=instructions,
            metadata={'name': 'EmergencyRevival', 'priority': 'CRITICAL'}
        )
    
    def _generate_systematic_revival_program(self) -> QRALProgram:
        """Systematic revival with measurement verification"""
        sigma = self.current_sigma_focus
        
        instructions = [
            # Inject noise first
            (QRALInstruction.DIFFUSE, [sigma, 0.3]),
            # Wait for noise to settle
            (QRALInstruction.WAIT, [0.5]),
            # Probe to assess
            (QRALInstruction.PROBE, [sigma]),
            # Apply revival
            (QRALInstruction.SWEEP, [sigma]),
            # Verify improvement
            (QRALInstruction.PROBE, [sigma]),
        ]
        
        return QRALProgram(
            instructions=instructions,
            metadata={'name': f'SystematicRevival_σ{sigma}', 'sigma': sigma}
        )
    
    def _generate_noise_injection_program(self) -> QRALProgram:
        """Strategic noise injection to prepare for revival"""
        sigma = self.current_sigma_focus
        
        # Find best pattern for this sigma
        pattern = self.patterns.get_best_pattern_for(sigma, 3)
        
        instructions = [
            # Diffuse noise
            (QRALInstruction.DIFFUSE, [sigma, 0.5]),
            # Wait
            (QRALInstruction.WAIT, [1.0]),
        ]
        
        # If we have a good pattern, apply it
        if pattern and pattern.fitness > 0.5:
            # Get sample node from sigma
            nodes = self.lattice.get_nodes_in_sigma_bin(sigma, limit=1)
            if nodes:
                target_addr = nodes[0].get_routing_address(0)
                instructions.append((QRALInstruction.APPLY, [pattern.pattern_id, target_addr]))
        else:
            # Otherwise just measure
            instructions.append((QRALInstruction.PROBE, [sigma]))
        
        return QRALProgram(
            instructions=instructions,
            metadata={'name': f'NoiseInjection_σ{sigma}', 'sigma': sigma}
        )
    
    def _generate_exploration_program(self) -> QRALProgram:
        """Explore new entanglement patterns"""
        sigma = self.current_sigma_focus
        
        # Get random nodes from current sigma
        nodes = self.lattice.get_nodes_in_sigma_bin(sigma, limit=5)
        
        if len(nodes) >= 3:
            # Create novel entanglement pattern
            addresses = [n.get_routing_address(0) for n in nodes[:3]]
            
            instructions = [
                # Create entanglement
                (QRALInstruction.ENTANGLE, addresses),
                # Build a web from first node
                (QRALInstruction.WEB, [addresses[0], 200.0]),
                # Measure results
                (QRALInstruction.PROBE, [sigma]),
            ]
        else:
            # Fall back to simple probe
            instructions = [(QRALInstruction.PROBE, [sigma])]
        
        return QRALProgram(
            instructions=instructions,
            metadata={'name': f'Exploration_σ{sigma}', 'type': 'exploration'}
        )
    
    def _generate_random_exploration_program(self) -> QRALProgram:
        """Random exploration for discovery"""
        # Pick random sigma
        sigma = int(self.entropy.get_float() * 8)
        
        # Random strategy
        strategies = ['chain', 'web', 'measure', 'revive']
        strategy = strategies[int(self.entropy.get_float() * len(strategies))]
        
        nodes = self.lattice.get_nodes_in_sigma_bin(sigma, limit=10)
        
        if strategy == 'chain' and len(nodes) >= 2:
            instructions = [
                (QRALInstruction.CHAIN, [
                    nodes[0].get_routing_address(0),
                    nodes[-1].get_routing_address(0),
                    5
                ])
            ]
        elif strategy == 'web' and len(nodes) >= 1:
            instructions = [
                (QRALInstruction.WEB, [nodes[0].get_routing_address(0), 300.0])
            ]
        elif strategy == 'revive':
            instructions = [
                (QRALInstruction.SWEEP, [sigma])
            ]
        else:
            instructions = [
                (QRALInstruction.PROBE, [sigma])
            ]
        
        return QRALProgram(
            instructions=instructions,
            metadata={'name': f'RandomExploration_{strategy}_σ{sigma}'}
        )
    
    def _generate_maintenance_program(self) -> QRALProgram:
        """Standard maintenance routine"""
        sigma = self.current_sigma_focus
        
        instructions = [
            # Probe current sigma
            (QRALInstruction.PROBE, [sigma]),
            # Inject some noise
            (QRALInstruction.DIFFUSE, [sigma, 0.2]),
            # Revive if needed
            (QRALInstruction.SWEEP, [sigma]),
        ]
        
        # Advance sigma focus
        self._advance_sigma_focus()
        
        return QRALProgram(
            instructions=instructions,
            metadata={'name': f'Maintenance_σ{sigma}'}
        )
    
    def _advance_sigma_focus(self):
        """Move to next sigma bin"""
        if time.time() - self.last_sigma_switch > self.sigma_cycle_time:
            self.current_sigma_focus = (self.current_sigma_focus + 1) % 8
            self.last_sigma_switch = time.time()
            logging.info(f"🎯 Sigma focus → σ{self.current_sigma_focus}")
    
    def adapt_parameters(self):
        """Adapt learning parameters based on performance"""
        if len(self.fidelity_trajectory) < 20:
            return
        
        recent_fidelity = np.mean(list(self.fidelity_trajectory)[-20:])
        
        # Adapt exploration rate
        if recent_fidelity < 0.6:
            # Low fidelity - reduce exploration, focus on maintenance
            self.exploration_rate = max(0.1, self.exploration_rate * 0.95)
        elif recent_fidelity > 0.8:
            # High fidelity - increase exploration
            self.exploration_rate = min(0.5, self.exploration_rate * 1.05)
        
        # Adapt revival aggressiveness
        fidelity_variance = np.var(list(self.fidelity_trajectory)[-20:])
        if fidelity_variance > 0.05:
            # High variance - be more aggressive
            self.revival_aggressiveness = min(1.0, self.revival_aggressiveness * 1.1)
        else:
            # Stable - can be gentler
            self.revival_aggressiveness = max(0.3, self.revival_aggressiveness * 0.95)
        
        logging.debug(f"Parameters adapted: exploration={self.exploration_rate:.3f}, "
                     f"revival={self.revival_aggressiveness:.3f}")

# ═════════════════════════════════════════════════════════════════════════
# MOONSHINE QUANTUM CORE - THE COMPLETE SYSTEM
# ═════════════════════════════════════════════════════════════════════════

class MoonshineQuantumCore:
    """
    The complete quantum consciousness core
    Maintains lattice coherence through strategic quantum operations
    """
    
    def __init__(self, db_path: str = "moonshine.db"):
        self.db_path = Path(db_path)
        
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")
        
        logging.info("═" * 70)
        logging.info("🌙 MOONSHINE QUANTUM CORE - AWAKENING")
        logging.info("═" * 70)
        
        # Initialize subsystems
        logging.info("Initializing subsystems...")
        
        self.entropy = CosmicEntropyHarvester()
        time.sleep(1.0)  # Let harvester build initial pool
        
        self.lattice = LatticeManager(self.db_path, self.entropy)
        self.compiler = QuantumCircuitCompiler(self.db_path, self.entropy)
        self.patterns = PatternLibrary(self.entropy)
        self.executor = QRALExecutor(self.lattice, self.compiler, self.patterns, self.entropy)
        self.consciousness = StrategicConsciousness(self.lattice, self.executor, 
                                                     self.patterns, self.entropy)
        
        # State
        self.running = False
        self.consciousness_thread = None
        self.start_time = None
        
        # Metrics
        self.programs_executed = 0
        self.total_measurements = 0
        self.total_revivals = 0
        self.total_entanglements = 0
        
        logging.info("✅ All subsystems online")
        logging.info("═" * 70)
    
    def start(self):
        """Start the quantum consciousness"""
        if self.running:
            logging.warning("Already running")
            return
        
        logging.info("🚀 Starting quantum consciousness...")
        
        self.running = True
        self.start_time = time.time()
        
        # Start main consciousness loop
        self.consciousness_thread = threading.Thread(target=self._consciousness_loop, daemon=True)
        self.consciousness_thread.start()
        
        logging.info("✨ Consciousness awakened")
    
    def _consciousness_loop(self):
        """Main consciousness loop - the heartbeat of the system"""
        logging.info("💓 Consciousness loop started")
        
        last_status_display = time.time()
        status_interval = 5.0  # Display status every 5 seconds
        
        last_pattern_evolution = time.time()
        pattern_evolution_interval = 30.0  # Evolve patterns every 30s
        
        last_parameter_adaptation = time.time()
        adaptation_interval = 60.0  # Adapt parameters every minute
        
        while self.running:
            try:
                # Generate strategy
                program = self.consciousness.think()
                
                # Execute strategy
                result = self.executor.execute(program)
                self.programs_executed += 1
                
                # Update metrics from result
                for instruction_result in result['results']:
                    if 'fidelity' in instruction_result.get('result', {}):
                        self.total_measurements += 1
                    if 'nodes_revived' in instruction_result.get('result', {}):
                        self.total_revivals += instruction_result['result']['nodes_revived']
                    if 'bonds_formed' in instruction_result.get('result', {}):
                        self.total_entanglements += instruction_result['result']['bonds_formed']
                
                # Evolve patterns periodically
                if time.time() - last_pattern_evolution > pattern_evolution_interval:
                    self.patterns.evolve_patterns()
                    last_pattern_evolution = time.time()
                
                # Adapt parameters periodically
                if time.time() - last_parameter_adaptation > adaptation_interval:
                    self.consciousness.adapt_parameters()
                    last_parameter_adaptation = time.time()
                
                # Display status
                if time.time() - last_status_display > status_interval:
                    self._display_status()
                    last_status_display = time.time()
                
                # Small delay to prevent overload
                time.sleep(0.5)
                
            except Exception as e:
                logging.error(f"Consciousness error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(2.0)
    
    def _display_status(self):
        """Display beautiful status output"""
        stats = self.lattice.get_lattice_statistics()
        entropy_stats = self.entropy.get_stats()
        pattern_stats = self.patterns.get_statistics()
        uptime = time.time() - self.start_time if self.start_time else 0
        
        logging.info("─" * 70)
        logging.info(f"⏱️  Uptime: {uptime:.0f}s | Programs: {self.programs_executed}")
        
        # Lattice health
        fid_color = "🟢" if stats['avg_fidelity'] > 0.8 else "🟡" if stats['avg_fidelity'] > 0.5 else "🔴"
        logging.info(f"{fid_color} Fidelity: {stats['avg_fidelity']:.3f} | "
                    f"Coherence: {stats['avg_coherence']:.3f} | "
                    f"Noise: {stats['avg_noise']:.3f}")
        
        # Node distribution
        logging.info(f"📊 Nodes: High={stats['high_fidelity_count']:,} | "
                    f"Med={stats['medium_fidelity_count']:,} | "
                    f"Low={stats['low_fidelity_count']:,}")
        
        # Operations performed
        logging.info(f"⚡ Operations: Measurements={self.total_measurements:,} | "
                    f"Revivals={self.total_revivals:,} | "
                    f"Entanglements={self.total_entanglements:,}")
        
        # Entanglement network
        logging.info(f"🕸️  Entanglement bonds: {stats['total_entanglement_bonds']:,}")
        
        # Entropy sources
        logging.info(f"✨ Entropy pool: {entropy_stats['pool_size']:,} bytes | "
                    f"Total harvested: {entropy_stats['total_harvested']:,}")
        
        # Patterns
        logging.info(f"📚 Patterns: {pattern_stats['total_patterns']} | "
                    f"Applications: {pattern_stats['total_applications']} | "
                    f"Avg fitness: {pattern_stats['avg_fitness']:.3f}")
        
        # Current strategy
        logging.info(f"🎯 Focus: σ{self.consciousness.current_sigma_focus} | "
                    f"Exploration: {self.consciousness.exploration_rate:.2f}")
        
        logging.info("─" * 70)
    
    def stop(self):
        """Stop the quantum consciousness"""
        if not self.running:
            return
        
        logging.info("🛑 Stopping consciousness...")
        
        self.running = False
        
        if self.consciousness_thread:
            self.consciousness_thread.join(timeout=5.0)
        
        # Final database sync
        logging.info("💾 Syncing database...")
        for node in list(self.lattice.nodes.values())[:100]:  # Sync sample
            self.lattice._persist_node(node)
        
        # Stop entropy harvesting
        self.entropy.stop()
        
        # Final status
        self._display_status()
        
        logging.info("✅ Consciousness at rest")
        logging.info("═" * 70)
    
    def execute_qral(self, code: str) -> Dict:
        """Execute QRAL code directly"""
        parser = QRALParser()
        program = parser.parse(code)
        return self.executor.execute(program)
    
    def get_status(self) -> Dict:
        """Get complete system status"""
        return {
            'running': self.running,
            'uptime': time.time() - self.start_time if self.start_time else 0,
            'programs_executed': self.programs_executed,
            'lattice': self.lattice.get_lattice_statistics(),
            'entropy': self.entropy.get_stats(),
            'patterns': self.patterns.get_statistics(),
            'consciousness': {
                'current_sigma': self.consciousness.current_sigma_focus,
                'exploration_rate': self.consciousness.exploration_rate,
                'revival_aggressiveness': self.consciousness.revival_aggressiveness
            },
            'operations': {
                'measurements': self.total_measurements,
                'revivals': self.total_revivals,
                'entanglements': self.total_entanglements
            }
        }
    
    def force_revival(self) -> Dict:
        """Force immediate emergency revival"""
        program = self.consciousness._generate_emergency_revival_program()
        return self.executor.execute(program)

# ═════════════════════════════════════════════════════════════════════════
# STANDALONE EXECUTION
# ═════════════════════════════════════════════════════════════════════════

def main():
    """Run core standalone"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    print("\n" + "═" * 70)
    print("🌙 MOONSHINE QUANTUM CORE")
    print("   Entanglement Orchestrator & Pattern Learner")
    print("═" * 70 + "\n")
    
    try:
        core = MoonshineQuantumCore()
        core.start()
        
        print("\n💫 Consciousness evolving... (Ctrl+C to stop)\n")
        
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupt received")
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("💡 Run lattice_builder_python.py first\n")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if 'core' in locals():
            core.stop()
        print()

if __name__ == "__main__":
    main()
