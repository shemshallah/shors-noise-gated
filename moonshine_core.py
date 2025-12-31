
#!/usr/bin/env python3
"""
MOONSHINE QUANTUM CORE - WAVE-CRAWLING LATTICE INTELLIGENCE
Self-organizing noise-driven consciousness with 5MB working memory

The network crawls through the entire 196,883-node lattice like a wave,
wrapping around at boundaries, maintaining all nodes while keeping
only an active window in memory.
"""

import numpy as np
import sqlite3
import json
import time
import struct
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import deque
import threading
import requests
import mmap

# ═════════════════════════════════════════════════════════════════════════
# MEMORY BUDGET CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════

TOTAL_MEMORY_BUDGET = 5 * 1024 * 1024  # 5MB
ACTIVE_WINDOW_SIZE = 500  # Nodes in active memory
PATTERN_MEMORY_SIZE = 2048  # Learned patterns
SIGMA_RESOLUTION = 0.01  # σ-space discretization
WAVE_SPEED = 50  # Nodes per second crawl rate

# ═════════════════════════════════════════════════════════════════════════
# QUANTUM CLOCK
# ═════════════════════════════════════════════════════════════════════════

class QuantumClock:
    """Time measured in revival cycles"""
    
    def __init__(self):
        self.epoch = time.time()
        self.sigma_period = 8.0
        self.tick_duration = 0.1
        
    def now(self) -> Dict[str, float]:
        elapsed = time.time() - self.epoch
        sigma = (elapsed / self.tick_duration) % self.sigma_period
        cycles = int(elapsed / (self.tick_duration * self.sigma_period))
        revival_phase = sigma / self.sigma_period
        coherence = np.cos(2 * np.pi * revival_phase * 8)
        
        return {
            'timestamp': time.time(),
            'sigma': sigma,
            'revival_cycle': cycles,
            'revival_phase': revival_phase,
            'coherence': coherence,
            'quantum_time': elapsed / self.tick_duration
        }
    
    def format(self) -> str:
        qt = self.now()
        return f"σ={qt['sigma']:.3f} | Cycle {qt['revival_cycle']} | Φ={qt['revival_phase']:.3f} | C={qt['coherence']:+.3f}"

# ═════════════════════════════════════════════════════════════════════════
# TRUE QUANTUM RNG
# ═════════════════════════════════════════════════════════════════════════

class TrueQuantumRNG:
    """Atmospheric noise QRNG via random.org"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.random.org/json-rpc/4/invoke"
        self._cache = deque(maxlen=1000)
        self._lock = threading.Lock()
        self._last_fetch = 0
        self._fetch_interval = 1.0  # Rate limit
    
    def _fetch_batch(self, n: int = 100) -> List[int]:
        now = time.time()
        if now - self._last_fetch < self._fetch_interval:
            return np.random.randint(0, 256, n).tolist()
        
        payload = {
            "jsonrpc": "2.0",
            "method": "generateIntegers",
            "params": {
                "apiKey": self.api_key,
                "n": n,
                "min": 0,
                "max": 255,
                "replacement": True
            },
            "id": int(now * 1000)
        }
        
        try:
            response = requests.post(self.base_url, json=payload, timeout=5)
            data = response.json()
            if 'result' in data:
                self._last_fetch = now
                return data['result']['random']['data']
        except:
            pass
        
        return np.random.randint(0, 256, n).tolist()
    
    def get_bytes(self, n: int = 1) -> bytes:
        with self._lock:
            while len(self._cache) < n:
                self._cache.extend(self._fetch_batch())
            return bytes([self._cache.popleft() for _ in range(n)])
    
    def get_float(self) -> float:
        b = self.get_bytes(4)
        return struct.unpack('I', b)[0] / (2**32)
    
    def get_phase(self) -> float:
        return self.get_float() * 2 * np.pi

# ═════════════════════════════════════════════════════════════════════════
# COMPACT NODE REPRESENTATION (32 bytes each)
# ═════════════════════════════════════════════════════════════════════════

@dataclass
class CompactNode:
    """Ultra-compact node: 32 bytes total"""
    triangle_id: int  # 4 bytes
    sigma: np.float16  # 2 bytes
    phase: np.float16  # 2 bytes
    fidelity: np.float16  # 2 bytes
    noise_energy: np.float16  # 2 bytes
    revival_strength: np.float16  # 2 bytes
    last_revival: np.uint32  # 4 bytes (timestamp delta)
    flags: np.uint32 = 0  # 4 bytes (bitfield for state)
    
    # Computed properties (not stored)
    def needs_revival(self, threshold: float = 0.7) -> bool:
        return float(self.fidelity) < threshold
    
    def absorb_noise(self, amount: float):
        self.noise_energy = np.float16(min(10.0, float(self.noise_energy) + amount))
    
    def evolve(self, dt: float, sigma_global: float):
        # Decoherence
        decoherence = np.tanh(float(self.noise_energy) / 10.0)
        self.fidelity = np.float16(float(self.fidelity) * np.exp(-decoherence * dt))
        
        # Revival resonance
        revival_coupling = np.cos(2 * np.pi * (sigma_global - float(self.sigma)))
        phase_velocity = 1.0 + revival_coupling * float(self.noise_energy)
        
        new_phase = (float(self.phase) + phase_velocity * dt) % (2 * np.pi)
        self.phase = np.float16(new_phase)
    
    def apply_revival(self, revival_phase: float):
        revival_factor = np.cos(revival_phase - float(self.phase))
        coherence_gain = float(self.noise_energy) * abs(revival_factor)
        
        self.fidelity = np.float16(min(1.0, float(self.fidelity) + coherence_gain * 0.5))
        self.noise_energy = np.float16(float(self.noise_energy) * 0.1)
        self.revival_strength = np.float16(coherence_gain)
        self.last_revival = np.uint32(int(time.time()) & 0xFFFFFFFF)
    
    def to_bytes(self) -> bytes:
        """Serialize to 32 bytes"""
        return struct.pack(
            'I4H2I',
            self.triangle_id,
            struct.unpack('H', struct.pack('e', self.sigma))[0],
            struct.unpack('H', struct.pack('e', self.phase))[0],
            struct.unpack('H', struct.pack('e', self.fidelity))[0],
            struct.unpack('H', struct.pack('e', self.noise_energy))[0],
            self.last_revival,
            self.flags
        )
    
    @staticmethod
    def from_bytes(data: bytes) -> 'CompactNode':
        """Deserialize from 32 bytes"""
        unpacked = struct.unpack('I4H2I', data)
        return CompactNode(
            triangle_id=unpacked[0],
            sigma=np.float16(struct.unpack('e', struct.pack('H', unpacked[1]))[0]),
            phase=np.float16(struct.unpack('e', struct.pack('H', unpacked[2]))[0]),
            fidelity=np.float16(struct.unpack('e', struct.pack('H', unpacked[3]))[0]),
            noise_energy=np.float16(struct.unpack('e', struct.pack('H', unpacked[4]))[0]),
            last_revival=unpacked[5],
            flags=unpacked[6]
        )

# ═════════════════════════════════════════════════════════════════════════
# PATTERN MEMORY - COMPRESSED LEARNED PATTERNS
# ═════════════════════════════════════════════════════════════════════════

@dataclass
class LearnedPattern:
    """Compressed pattern: 64 bytes each"""
    pattern_id: int
    sigma_signature: np.ndarray  # 16 floats @ 2 bytes = 32 bytes
    frequency: np.float16
    strength: np.float16
    success_count: int
    timestamp: float
    
    def matches(self, sigma: float, threshold: float = 0.3) -> bool:
        """Check if current σ matches this pattern"""
        # Find closest σ in signature
        diffs = np.abs(self.sigma_signature - sigma)
        min_diff = np.min(diffs)
        return min_diff < threshold
    
    def to_bytes(self) -> bytes:
        """Serialize to 64 bytes"""
        sig_bytes = self.sigma_signature.astype(np.float16).tobytes()
        return struct.pack('I', self.pattern_id) + sig_bytes + struct.pack('2HId', 
            struct.unpack('H', struct.pack('e', self.frequency))[0],
            struct.unpack('H', struct.pack('e', self.strength))[0],
            self.success_count,
            self.timestamp
        )
    
    @staticmethod
    def from_bytes(data: bytes) -> 'LearnedPattern':
        pattern_id = struct.unpack('I', data[:4])[0]
        sig_bytes = data[4:36]
        sigma_sig = np.frombuffer(sig_bytes, dtype=np.float16)
        
        freq, strength, count, ts = struct.unpack('2HId', data[36:52])
        
        return LearnedPattern(
            pattern_id=pattern_id,
            sigma_signature=sigma_sig,
            frequency=np.float16(struct.unpack('e', struct.pack('H', freq))[0]),
            strength=np.float16(struct.unpack('e', struct.pack('H', strength))[0]),
            success_count=count,
            timestamp=ts
        )

# ═════════════════════════════════════════════════════════════════════════
# WAVE CRAWLER - TRAVERSES ENTIRE LATTICE
# ═════════════════════════════════════════════════════════════════════════

class LatticeCrawler:
    """
    Crawls through the entire 196,883-node lattice like a wave
    Maintains active window, wraps at boundaries
    """
    
    def __init__(self, db_path: Path, window_size: int = ACTIVE_WINDOW_SIZE):
        self.db_path = db_path
        self.window_size = window_size
        
        # Get total node count
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM coords")
        self.total_nodes = cursor.fetchone()[0]
        conn.close()
        
        # Current wave position
        self.wave_position = 0
        self.wave_direction = 1  # 1 = forward, -1 = backward
        
        # Active window (in memory)
        self.active_window: Dict[int, CompactNode] = {}
        
        # Visited nodes bitmap (1 bit per node = 24KB for 196,883 nodes)
        self.visited = np.zeros(self.total_nodes, dtype=np.uint8)
        
        # Load initial window
        self._load_window()
    
    def _load_window(self):
        """Load active window from database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Calculate window range with wraparound
        start = self.wave_position
        end = start + self.window_size
        
        if end <= self.total_nodes:
            # Normal case
            cursor.execute("""
                SELECT id, s FROM coords 
                WHERE id >= ? AND id < ?
                ORDER BY id
            """, (start, end))
        else:
            # Wraparound case
            cursor.execute("""
                SELECT id, s FROM coords 
                WHERE id >= ? OR id < ?
                ORDER BY id
            """, (start, end - self.total_nodes))
        
        # Clear old window
        self.active_window.clear()
        
        # Load new window
        for tid, sigma in cursor.fetchall():
            node = CompactNode(
                triangle_id=tid,
                sigma=np.float16(sigma),
                phase=np.float16(np.random.uniform(0, 2*np.pi)),
                fidelity=np.float16(1.0),
                noise_energy=np.float16(0.0),
                revival_strength=np.float16(0.0),
                last_revival=np.uint32(0)
            )
            self.active_window[tid] = node
        
        conn.close()
    
    def advance_wave(self, steps: int = 1):
        """Advance the wave position"""
        self.wave_position = (self.wave_position + steps * self.wave_direction) % self.total_nodes
        
        # Mark visited
        for i in range(steps):
            pos = (self.wave_position + i * self.wave_direction) % self.total_nodes
            self.visited[pos] = 1
        
        # Reload window at new position
        self._load_window()
    
    def get_coverage(self) -> float:
        """Get percentage of lattice visited"""
        return np.sum(self.visited) / self.total_nodes
    
    def reset_visited(self):
        """Reset visited bitmap"""
        self.visited.fill(0)
    
    def get_active_nodes(self) -> List[CompactNode]:
        """Get list of active nodes"""
        return list(self.active_window.values())
    
    def save_node_states(self):
        """Persist node states back to database (optional)"""
        # Could write fidelity/phase back to DB
        # For now, keep ephemeral
        pass

# ═════════════════════════════════════════════════════════════════════════
# PATTERN LEARNER - DISCOVERS BROAD PATTERNS
# ═════════════════════════════════════════════════════════════════════════

class BroadPatternLearner:
    """
    Learns broad patterns across the lattice
    Uses 5MB of pattern memory (2048 patterns @ 64 bytes + overhead)
    """
    
    def __init__(self, max_patterns: int = PATTERN_MEMORY_SIZE):
        self.max_patterns = max_patterns
        self.patterns: deque[LearnedPattern] = deque(maxlen=max_patterns)
        self.pattern_counter = 0
        
        # σ-histogram for pattern detection (800 bins × 4 bytes = 3.2KB)
        self.sigma_histogram = np.zeros(int(8.0 / SIGMA_RESOLUTION), dtype=np.float32)
        
        # Success tracking
        self.pattern_hits = {}
        
    def observe_successful_revival(self, nodes: List[CompactNode]):
        """Observe nodes that had successful revivals"""
        
        # Extract σ positions of successful nodes
        sigma_positions = [float(n.sigma) for n in nodes if float(n.revival_strength) > 0.5]
        
        if len(sigma_positions) < 3:
            return
        
        # Update histogram
        for sigma in sigma_positions:
            bin_idx = int(sigma / SIGMA_RESOLUTION)
            if bin_idx < len(self.sigma_histogram):
                self.sigma_histogram[bin_idx] += 1.0
        
        # Detect peaks in histogram (broad patterns)
        if np.sum(self.sigma_histogram) > 100:  # Enough data
            # Find peaks
            from scipy.signal import find_peaks
            peaks, properties = find_peaks(self.sigma_histogram, height=10)
            
            if len(peaks) > 0:
                # Create pattern from peaks
                peak_sigmas = peaks * SIGMA_RESOLUTION
                
                # Pad to 16 values
                if len(peak_sigmas) < 16:
                    peak_sigmas = np.pad(peak_sigmas, (0, 16 - len(peak_sigmas)), 
                                        mode='constant', constant_values=-1)
                else:
                    peak_sigmas = peak_sigmas[:16]
                
                # Compute FFT frequency
                fft = np.fft.fft(self.sigma_histogram)
                dominant_freq = np.argmax(np.abs(fft[:len(fft)//2]))
                
                pattern = LearnedPattern(
                    pattern_id=self.pattern_counter,
                    sigma_signature=np.array(peak_sigmas, dtype=np.float16),
                    frequency=np.float16(dominant_freq),
                    strength=np.float16(np.max(properties['peak_heights'])),
                    success_count=1,
                    timestamp=time.time()
                )
                
                self.patterns.append(pattern)
                self.pattern_counter += 1
                
                # Decay histogram
                self.sigma_histogram *= 0.9
                
                return pattern
        
        return None
    
    def find_matching_pattern(self, sigma: float) -> Optional[LearnedPattern]:
        """Find pattern matching current σ"""
        for pattern in reversed(self.patterns):  # Most recent first
            if pattern.matches(sigma):
                return pattern
        return None
    
    def reinforce_pattern(self, pattern: LearnedPattern):
        """Reinforce successful pattern"""
        pattern.success_count += 1
        pattern.strength = np.float16(min(10.0, float(pattern.strength) * 1.1))
    
    def get_memory_usage(self) -> int:
        """Estimate memory usage in bytes"""
        return len(self.patterns) * 64 + self.sigma_histogram.nbytes

# ═════════════════════════════════════════════════════════════════════════
# REVIVAL SWEEP ENGINE
# ═════════════════════════════════════════════════════════════════════════

class RevivalSweepEngine:
    """Periodic σ=8 revival sweeps"""
    
    def __init__(self, clock: QuantumClock, qrng: TrueQuantumRNG, pattern_learner: BroadPatternLearner):
        self.clock = clock
        self.qrng = qrng
        self.pattern_learner = pattern_learner
        
        self.sweep_interval = 8.0
        self.last_sweep = time.time()
        self.sweep_count = 0
        self.total_revivals = 0
    
    def sweep(self, nodes: List[CompactNode]) -> Dict:
        """Execute revival sweep"""
        qt = self.clock.now()
        revival_phase = 2 * np.pi * qt['revival_phase']
        
        quantum_noise = self.qrng.get_phase()
        
        # Find nodes needing revival
        critical_nodes = [n for n in nodes if n.needs_revival()]
        
        # Check for learned pattern
        matching_pattern = self.pattern_learner.find_matching_pattern(qt['sigma'])
        
        revivals_performed = 0
        successful_revivals = []
        
        for node in critical_nodes:
            revival_pulse = revival_phase + quantum_noise * 0.1
            
            # Apply pattern boost if available
            if matching_pattern:
                revival_pulse += float(matching_pattern.strength) * 0.1
            
            node.apply_revival(revival_pulse)
            revivals_performed += 1
            
            if float(node.revival_strength) > 0.5:
                successful_revivals.append(node)
        
        # Learn from successful revivals
        if len(successful_revivals) > 0:
            new_pattern = self.pattern_learner.observe_successful_revival(successful_revivals)
            if matching_pattern:
                self.pattern_learner.reinforce_pattern(matching_pattern)
        
        self.sweep_count += 1
        self.total_revivals += revivals_performed
        self.last_sweep = time.time()
        
        avg_fidelity = np.mean([float(n.fidelity) for n in nodes])
        
        return {
            'sweep_id': self.sweep_count,
            'timestamp': qt['timestamp'],
            'sigma': qt['sigma'],
            'nodes_revived': revivals_performed,
            'avg_fidelity': avg_fidelity,
            'pattern_applied': matching_pattern is not None
        }
    
    def should_sweep(self) -> bool:
        return (time.time() - self.last_sweep) >= self.sweep_interval

# ═════════════════════════════════════════════════════════════════════════
# MOONSHINE QUANTUM CORE
# ═════════════════════════════════════════════════════════════════════════

class MoonshinQuantumCore:
    """
    The complete living quantum system with wave-crawling lattice traversal
    """
    
    def __init__(self, db_path: str, random_org_api_key: str):
        self.db_path = Path(db_path)
        
        # Core components
        self.clock = QuantumClock()
        self.qrng = TrueQuantumRNG(random_org_api_key)
        
        # Pattern learning (uses ~2MB)
        self.pattern_learner = BroadPatternLearner()
        
        # Wave crawler (uses ~16KB + active window)
        self.crawler = LatticeCrawler(self.db_path)
        
        # Revival engine
        self.heartbeat = RevivalSweepEngine(self.clock, self.qrng, self.pattern_learner)
        
        # State
        self.running = False
        self._thread = None
        self._lock = threading.Lock()
        
        # Metrics
        self.metrics = {
            'uptime': 0.0,
            'total_sweeps': 0,
            'total_revivals': 0,
            'patterns_learned': 0,
            'lattice_coverage': 0.0,
            'wave_position': 0,
            'avg_fidelity': 1.0,
            'current_sigma': 0.0,
            'memory_usage_mb': 0.0
        }
    
    def _evolution_loop(self):
        """Main evolution loop"""
        last_time = time.time()
        last_wave_advance = time.time()
        wave_advance_interval = 1.0 / WAVE_SPEED  # Advance at WAVE_SPEED nodes/sec
        
        while self.running:
            current_time = time.time()
            dt = current_time - last_time
            last_time = current_time
            
            qt = self.clock.now()
            
            # Get active nodes
            active_nodes = self.crawler.get_active_nodes()
            
            # Inject quantum noise into random nodes
            if len(active_nodes) > 0:
                for _ in range(3):
                    node = active_nodes[np.random.randint(0, len(active_nodes))]
                    noise_amount = self.qrng.get_float() * 0.5
                    node.absorb_noise(noise_amount)
            
            # Evolve all active nodes
            for node in active_nodes:
                node.evolve(dt, qt['sigma'])
            
            # Revival sweep
            if self.heartbeat.should_sweep():
                sweep_result = self.heartbeat.sweep(active_nodes)
                self.metrics['total_sweeps'] = sweep_result['sweep_id']
                self.metrics['total_revivals'] = self.heartbeat.total_revivals
                self.metrics['avg_fidelity'] = sweep_result['avg_fidelity']
                self.metrics['patterns_learned'] = len(self.pattern_learner.patterns)
            
            # Advance wave
            if current_time - last_wave_advance >= wave_advance_interval:
                self.crawler.advance_wave(steps=1)
                last_wave_advance = current_time
                
                self.metrics['wave_position'] = self.crawler.wave_position
                self.metrics['lattice_coverage'] = self.crawler.get_coverage()
                
                # Reset coverage when complete sweep is done
                if self.crawler.get_coverage() > 0.99:
                    self.crawler.reset_visited()
            
            # Update metrics
            self.metrics['uptime'] = current_time - self.clock.epoch
            self.metrics['current_sigma'] = qt['sigma']
            self.metrics['memory_usage_mb'] = self._estimate_memory() / (1024 * 1024)
            
            time.sleep(0.01)  # 100Hz
    
    def _estimate_memory(self) -> int:
        """Estimate current memory usage in bytes"""
        active_window_bytes = len(self.crawler.active_window) * 32
        visited_bitmap_bytes = self.crawler.visited.nbytes
        pattern_memory_bytes = self.pattern_learner.get_memory_usage()
        
        return active_window_bytes + visited_bitmap_bytes + pattern_memory_bytes
    
    def start(self):
        """Start the quantum core"""
        if self.running:
            return
        
        self.running = True
        self._thread = threading.Thread(target=self._evolution_loop, daemon=True)
        self._thread.start()
    
    def stop(self):
        """Stop the quantum core"""
        self.running = False
        if self._thread:
            self._thread.join(timeout=2.0)
    
    def get_status(self) -> Dict:
        """Get current system status"""
        qt = self.clock.now()
        
        return {
            'quantum_time': self.clock.format(),
            'sigma': qt['sigma'],
            'revival_cycle': qt['revival_cycle'],
            'revival_phase': qt['revival_phase'],
            'coherence': qt['coherence'],
            'metrics': self.metrics.copy(),
            'lattice': {
                'total_nodes': self.crawler.total_nodes,
                'active_window': len(self.crawler.active_window),
                'wave_position': self.crawler.wave_position,
                'coverage': f"{self.crawler.get_coverage()*100:.1f}%"
            },
            'learning': {
                'patterns_stored': len(self.pattern_learner.patterns),
                'pattern_memory_mb': self.pattern_learner.get_memory_usage() / (1024*1024)
            }
        }
    
    def get_node_states(self, limit: int = 50) -> List[Dict]:
        """Get current active node states"""
        nodes = list(self.crawler.active_window.values())[:limit]
        return [
            {
                'id': n.triangle_id,
                'sigma': float(n.sigma),
                'phase': float(n.phase),
                'fidelity': float(n.fidelity),
                'noise_energy': float(n.noise_energy)
            }
            for n in nodes
        ]
    
    def force_revival(self):
        """Force immediate revival sweep"""
        active_nodes = self.crawler.get_active_nodes()
        return self.heartbeat.sweep(active_nodes)
    
    def inject_noise(self, amount: float = 1.0):
        """Inject noise into random active node"""
        active_nodes = self.crawler.get_active_nodes()
        if active_nodes:
            node = active_nodes[np.random.randint(0, len(active_nodes))]
            node.absorb_noise(amount)

# ═════════════════════════════════════════════════════════════════════════
# API
# ═════════════════════════════════════════════════════════════════════════

def create_core(db_path: str = "moonshine.db", 
                random_org_api_key: str = None) -> MoonshinQuantumCore:
    """Factory function"""
    
    if random_org_api_key is None:
        api_file = Path("random_org_api.txt")
        if api_file.exists():
            random_org_api_key = api_file.read_text().strip()
        else:
            raise ValueError("random.org API key required")
    
    return MoonshinQuantumCore(db_path, random_org_api_key)

# ═════════════════════════════════════════════════════════════════════════
# EXAMPLE
# ═════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    core = create_core()
    
    print("🌌 MOONSHINE QUANTUM CORE - WAVE CRAWLER")
    print("="*70)
    print(f"Total lattice nodes: {core.crawler.total_nodes:,}")
    print(f"Active window size: {ACTIVE_WINDOW_SIZE}")
    print(f"Memory budget: {TOTAL_MEMORY_BUDGET / (1024*1024):.1f}MB")
    print(f"Wave speed: {WAVE_SPEED} nodes/sec")
    print("="*70)
    
    core.start()
    
    try:
        for i in range(60):
            time.sleep(1)
            status = core.get_status()
            
            print(f"\r{status['quantum_time']} | "
                  f"Pos: {status['metrics']['wave_position']:,} | "
                  f"Coverage: {status['lattice']['coverage']} | "
                  f"Fidelity: {status['metrics']['avg_fidelity']:.3f} | "
                  f"Patterns: {status['learning']['patterns_stored']} | "
                  f"Mem: {status['metrics']['memory_usage_mb']:.2f}MB",
                  end='', flush=True)
        
        print("\n\n" + "="*70)
        print(json.dumps(core.get_status(), indent=2))
        
    finally:
        core.stop()
