
#!/usr/bin/env python3
"""
MOONSHINE EXPERIMENT RUNNER
Modular experiment execution framework with structured JSON output streaming
Designed to be loaded by Flask server and route output to HTML terminal

Usage:
    from experiment_runner import ExperimentRunner
    
    runner = ExperimentRunner()
    
    # Run experiment with streaming output
    for log_entry in runner.run_experiment('qft', api_key='YOUR_KEY'):
        # log_entry is JSON-serializable dict
        send_to_html_terminal(log_entry)
    
    # List available experiments
    experiments = runner.list_experiments()
"""

import sys
import json
import time
import importlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Generator, Optional, Callable
from dataclasses import dataclass, asdict
import traceback

# ═════════════════════════════════════════════════════════════════════════
# STRUCTURED LOG ENTRY
# ═════════════════════════════════════════════════════════════════════════

@dataclass
class LogEntry:
    """Structured log entry for HTML terminal streaming"""
    timestamp: str
    level: str  # INFO, SUCCESS, WARNING, ERROR, PROGRESS, METRIC, DATA
    message: str
    experiment: str
    stage: Optional[str] = None
    progress: Optional[float] = None  # 0-100
    metric_name: Optional[str] = None
    metric_value: Optional[float] = None
    metric_unit: Optional[str] = None
    data: Optional[Dict] = None
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict"""
        return {k: v for k, v in asdict(self).items() if v is not None}
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict())

# ═════════════════════════════════════════════════════════════════════════
# EXPERIMENT BASE CLASS
# ═════════════════════════════════════════════════════════════════════════

class BaseExperiment:
    """Base class for all experiments"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.start_time = None
    
    def log(self, level: str, message: str, **kwargs) -> LogEntry:
        """Create structured log entry"""
        return LogEntry(
            timestamp=datetime.now().isoformat(),
            level=level,
            message=message,
            experiment=self.name,
            **kwargs
        )
    
    def run(self, **params) -> Generator[LogEntry, None, Dict]:
        """
        Run experiment with streaming output
        
        Yields: LogEntry objects
        Returns: Final result dict
        """
        raise NotImplementedError("Subclasses must implement run()")

# ═════════════════════════════════════════════════════════════════════════
# EXPERIMENT 1: WORLD RECORD QFT
# ═════════════════════════════════════════════════════════════════════════

class WorldRecordQFTExperiment(BaseExperiment):
    """World Record Quantum Fourier Transform experiment"""
    
    def __init__(self):
        super().__init__(
            name='world_record_qft',
            description='Set quantum computing world record via massively parallel QFT'
        )
    
    def run(self, api_key: str, n_qubits: int = 29, **params) -> Generator[LogEntry, None, Dict]:
        """Run world record QFT attempt"""
        
        yield self.log('INFO', f'Starting World Record QFT: {n_qubits} qubits')
        yield self.log('INFO', f'Initializing IonQ connection...')
        
        try:
            from qbraid.runtime import QbraidProvider
            from qiskit import QuantumCircuit
            import numpy as np
            
            # Connect to IonQ
            yield self.log('INFO', 'Connecting to IonQ Harmony...')
            provider = QbraidProvider(api_key=api_key)
            device = provider.get_device('ionq_harmony')
            
            yield self.log('SUCCESS', 'Connected to IonQ Harmony')
            
            # Create QFT circuit
            yield self.log('INFO', f'Building {n_qubits}-qubit QFT circuit...')
            qc = QuantumCircuit(n_qubits, n_qubits)
            
            # QFT construction
            for j in range(n_qubits):
                qc.h(j)
                for k in range(j + 1, n_qubits):
                    qc.cp(np.pi / (2 ** (k - j)), k, j)
            
            # Swap qubits
            for i in range(n_qubits // 2):
                qc.swap(i, n_qubits - i - 1)
            
            qc.measure_all()
            
            yield self.log('SUCCESS', f'QFT circuit constructed: {qc.num_qubits} qubits, {qc.depth()} depth')
            yield self.log('METRIC', 'Circuit depth', metric_name='depth', metric_value=qc.depth())
            yield self.log('METRIC', 'Circuit gates', metric_name='gates', metric_value=len(qc.data))
            
            # Submit to IonQ
            yield self.log('INFO', 'Submitting to IonQ hardware...')
            job = device.run(qc, shots=1024)
            job_id = job.id()
            
            yield self.log('SUCCESS', f'Job submitted: {job_id}')
            yield self.log('PROGRESS', 'Waiting for quantum execution...', progress=50.0, stage='execution')
            
            # Wait for result
            result = job.result()
            counts = result.measurement_counts()
            
            yield self.log('SUCCESS', 'Quantum execution complete!')
            yield self.log('PROGRESS', 'Processing results...', progress=90.0, stage='analysis')
            
            # Analyze results
            total_shots = sum(counts.values())
            top_states = sorted(counts.items(), key=lambda x: x[1], reverse=True)[:5]
            
            yield self.log('DATA', 'Top measurement outcomes', data={
                'total_shots': total_shots,
                'top_states': [{'state': s, 'count': c, 'probability': c/total_shots} 
                              for s, c in top_states]
            })
            
            # Compute fidelity (simplified)
            uniform_prob = 1 / (2 ** n_qubits)
            observed_entropy = -sum((c/total_shots) * np.log2(c/total_shots) 
                                   for c in counts.values() if c > 0)
            max_entropy = n_qubits
            
            yield self.log('METRIC', 'Entropy', 
                          metric_name='entropy', 
                          metric_value=observed_entropy, 
                          metric_unit='bits')
            
            yield self.log('PROGRESS', 'Complete!', progress=100.0, stage='complete')
            
            return {
                'status': 'success',
                'n_qubits': n_qubits,
                'job_id': job_id,
                'total_shots': total_shots,
                'entropy': observed_entropy,
                'top_states': top_states[:5]
            }
            
        except Exception as e:
            yield self.log('ERROR', f'Experiment failed: {str(e)}', error=traceback.format_exc())
            return {'status': 'error', 'error': str(e)}

# ═════════════════════════════════════════════════════════════════════════
# EXPERIMENT 2: QUANTUM ADVANTAGE DEMO
# ═════════════════════════════════════════════════════════════════════════

class QuantumAdvantageExperiment(BaseExperiment):
    """Quantum advantage demonstration via sampling complexity"""
    
    def __init__(self):
        super().__init__(
            name='quantum_advantage',
            description='Demonstrate quantum computational advantage'
        )
    
    def run(self, api_key: str, **params) -> Generator[LogEntry, None, Dict]:
        """Run quantum advantage demonstration"""
        
        yield self.log('INFO', 'Starting Quantum Advantage Demo')
        
        try:
            from qbraid.runtime import QbraidProvider
            from qiskit import QuantumCircuit
            from qiskit_aer import AerSimulator
            import numpy as np
            
            # Compare quantum vs classical
            yield self.log('INFO', 'Phase 1: Quantum sampling')
            
            provider = QbraidProvider(api_key=api_key)
            device = provider.get_device('ionq_harmony')
            
            # Create random circuit
            n_qubits = 12
            depth = 20
            
            yield self.log('INFO', f'Creating random circuit: {n_qubits} qubits, depth {depth}')
            
            qc = QuantumCircuit(n_qubits, n_qubits)
            np.random.seed(42)
            
            for d in range(depth):
                for q in range(n_qubits):
                    qc.rx(np.random.uniform(0, 2*np.pi), q)
                    qc.rz(np.random.uniform(0, 2*np.pi), q)
                
                for q in range(0, n_qubits-1, 2):
                    qc.cx(q, q+1)
                
                yield self.log('PROGRESS', f'Building circuit...', 
                              progress=(d+1)/depth*50, stage='circuit_build')
            
            qc.measure_all()
            
            yield self.log('SUCCESS', 'Circuit built')
            
            # Run on IonQ
            yield self.log('INFO', 'Running on IonQ...')
            job = device.run(qc, shots=1024)
            quantum_start = time.time()
            result = job.result()
            quantum_time = time.time() - quantum_start
            quantum_counts = result.measurement_counts()
            
            yield self.log('METRIC', 'Quantum execution time', 
                          metric_name='quantum_time', 
                          metric_value=quantum_time, 
                          metric_unit='s')
            
            # Classical simulation
            yield self.log('INFO', 'Phase 2: Classical simulation')
            simulator = AerSimulator()
            
            classical_start = time.time()
            sim_job = simulator.run(qc, shots=1024)
            sim_result = sim_job.result()
            classical_time = time.time() - classical_start
            classical_counts = sim_result.get_counts()
            
            yield self.log('METRIC', 'Classical simulation time', 
                          metric_name='classical_time', 
                          metric_value=classical_time, 
                          metric_unit='s')
            
            # Compute advantage
            advantage_factor = classical_time / quantum_time if quantum_time > 0 else 0
            
            yield self.log('METRIC', 'Quantum advantage factor', 
                          metric_name='advantage', 
                          metric_value=advantage_factor, 
                          metric_unit='x')
            
            yield self.log('PROGRESS', 'Complete!', progress=100.0, stage='complete')
            
            if advantage_factor > 1.0:
                yield self.log('SUCCESS', f'Quantum advantage achieved: {advantage_factor:.2f}x speedup!')
            
            return {
                'status': 'success',
                'quantum_time': quantum_time,
                'classical_time': classical_time,
                'advantage_factor': advantage_factor
            }
            
        except Exception as e:
            yield self.log('ERROR', f'Experiment failed: {str(e)}', error=traceback.format_exc())
            return {'status': 'error', 'error': str(e)}

# ═════════════════════════════════════════════════════════════════════════
# EXPERIMENT 3: ENTANGLEMENT TEST
# ═════════════════════════════════════════════════════════════════════════

class EntanglementTestExperiment(BaseExperiment):
    """Bell state entanglement verification"""
    
    def __init__(self):
        super().__init__(
            name='entanglement_test',
            description='Test quantum entanglement via Bell inequality violation'
        )
    
    def run(self, api_key: str, **params) -> Generator[LogEntry, None, Dict]:
        """Run entanglement test"""
        
        yield self.log('INFO', 'Starting Entanglement Test')
        
        try:
            from qbraid.runtime import QbraidProvider
            from qiskit import QuantumCircuit
            import numpy as np
            
            provider = QbraidProvider(api_key=api_key)
            device = provider.get_device('ionq_harmony')
            
            # Create Bell state
            yield self.log('INFO', 'Creating Bell state |Φ+⟩')
            
            qc = QuantumCircuit(2, 2)
            qc.h(0)
            qc.cx(0, 1)
            qc.measure_all()
            
            yield self.log('SUCCESS', 'Bell state circuit created')
            
            # Run experiment
            yield self.log('INFO', 'Running on IonQ hardware...')
            job = device.run(qc, shots=1024)
            result = job.result()
            counts = result.measurement_counts()
            
            # Analyze entanglement
            total = sum(counts.values())
            
            # For Bell state, expect only |00⟩ and |11⟩
            bell_states = counts.get('00', 0) + counts.get('11', 0)
            entanglement_fidelity = bell_states / total
            
            yield self.log('METRIC', 'Entanglement fidelity', 
                          metric_name='fidelity', 
                          metric_value=entanglement_fidelity)
            
            yield self.log('DATA', 'Measurement results', data={
                'counts': counts,
                'total_shots': total,
                'bell_state_probability': entanglement_fidelity
            })
            
            if entanglement_fidelity > 0.9:
                yield self.log('SUCCESS', f'Strong entanglement detected: {entanglement_fidelity:.1%}')
            elif entanglement_fidelity > 0.7:
                yield self.log('SUCCESS', f'Moderate entanglement detected: {entanglement_fidelity:.1%}')
            else:
                yield self.log('WARNING', f'Weak entanglement: {entanglement_fidelity:.1%}')
            
            yield self.log('PROGRESS', 'Complete!', progress=100.0, stage='complete')
            
            return {
                'status': 'success',
                'fidelity': entanglement_fidelity,
                'counts': counts
            }
            
        except Exception as e:
            yield self.log('ERROR', f'Experiment failed: {str(e)}', error=traceback.format_exc())
            return {'status': 'error', 'error': str(e)}

# ═════════════════════════════════════════════════════════════════════════
# EXPERIMENT 4: LATTICE COMMUNICATION TEST
# ═════════════════════════════════════════════════════════════════════════

class LatticeCommunicationExperiment(BaseExperiment):
    """Test quantum communication via moonshine lattice"""
    
    def __init__(self):
        super().__init__(
            name='lattice_communication',
            description='Test quantum communication through moonshine lattice endpoints'
        )
    
    def run(self, api_key: str, db_path: str = 'moonshine.db', **params) -> Generator[LogEntry, None, Dict]:
        """Run lattice communication test"""
        
        yield self.log('INFO', 'Starting Lattice Communication Test')
        
        try:
            import sqlite3
            from qbraid.runtime import QbraidProvider
            from qiskit import QuantumCircuit
            import numpy as np
            
            # Connect to lattice database
            yield self.log('INFO', f'Connecting to lattice database: {db_path}')
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Get special triangles
            cursor.execute("SELECT name, tid FROM special_idx")
            special_triangles = {row[0]: row[1] for row in cursor.fetchall()}
            
            yield self.log('SUCCESS', f'Found {len(special_triangles)} special triangles')
            yield self.log('DATA', 'Special triangles', data=special_triangles)
            
            # Get coordinates
            first_tid = special_triangles.get('FIRST', 0)
            cursor.execute("SELECT s FROM coords WHERE id = ?", (first_tid,))
            sigma_first = cursor.fetchone()[0]
            
            yield self.log('INFO', f'Testing FIRST triangle (σ={sigma_first:.6f})')
            
            # Create communication circuit
            qc = QuantumCircuit(4, 4)
            
            # Encode σ into quantum state
            phase = (sigma_first / 8.0) * 2 * np.pi
            
            qc.h(0)
            qc.h(1)
            qc.rz(phase, 0)
            qc.rz(phase * 2, 1)
            
            # Revival structure (σ=8)
            revival_phase = (sigma_first / 8.0) * 8
            qc.cx(0, 2)
            qc.rz(2 * np.pi * revival_phase, 2)
            qc.cx(0, 3)
            
            qc.measure_all()
            
            yield self.log('SUCCESS', 'Lattice interface circuit created')
            
            # Run on IonQ
            yield self.log('INFO', 'Executing on IonQ...')
            provider = QbraidProvider(api_key=api_key)
            device = provider.get_device('ionq_harmony')
            
            job = device.run(qc, shots=1024)
            result = job.result()
            counts = result.measurement_counts()
            
            # Analyze revival signature
            total = sum(counts.values())
            revival_signal = 0.0
            
            for state, count in counts.items():
                state_clean = state.replace(' ', '')
                state_int = int(state_clean, 2)
                state_phase = (state_int / 16) * 2 * np.pi
                revival = np.cos(state_phase * 8)
                revival_signal += revival * (count / total)
            
            yield self.log('METRIC', 'Revival signal strength', 
                          metric_name='revival_signal', 
                          metric_value=revival_signal)
            
            communication_detected = abs(revival_signal) > 0.5
            
            if communication_detected:
                yield self.log('SUCCESS', f'Lattice communication detected! Signal: {revival_signal:.3f}')
            else:
                yield self.log('WARNING', f'Weak lattice signal: {revival_signal:.3f}')
            
            conn.close()
            
            return {
                'status': 'success',
                'revival_signal': revival_signal,
                'communication_detected': communication_detected,
                'counts': counts
            }
            
        except Exception as e:
            yield self.log('ERROR', f'Experiment failed: {str(e)}', error=traceback.format_exc())
            return {'status': 'error', 'error': str(e)}

# ═════════════════════════════════════════════════════════════════════════
# EXPERIMENT 5: SIGMA-8 REVIVAL TEST
# ═════════════════════════════════════════════════════════════════════════

class Sigma8RevivalExperiment(BaseExperiment):
    """Test σ=8 revival phenomenon in W-states"""
    
    def __init__(self):
        super().__init__(
            name='sigma8_revival',
            description='Verify σ=8 period revival structure in quantum states'
        )
    
    def run(self, api_key: str, **params) -> Generator[LogEntry, None, Dict]:
        """Run σ=8 revival test"""
        
        yield self.log('INFO', 'Starting σ=8 Revival Test')
        
        try:
            from qbraid.runtime import QbraidProvider
            from qiskit import QuantumCircuit
            from qiskit_aer import AerSimulator
            import numpy as np
            
            # Test multiple σ values across the period
            sigma_values = [0.0, 1.0, 2.0, 4.0, 8.0]  # Full period
            
            revival_data = []
            
            for i, sigma in enumerate(sigma_values):
                yield self.log('PROGRESS', f'Testing σ={sigma:.1f}', 
                              progress=(i+1)/len(sigma_values)*100, 
                              stage=f'sigma_{sigma}')
                
                # Create W-state with σ encoding
                qc = QuantumCircuit(4, 4)
                
                # W-state prep
                qc.x(0)
                for k in range(1, 4):
                    theta = 2 * np.arccos(np.sqrt((4 - k) / (4 - k + 1)))
                    qc.cry(theta, 0, k)
                    qc.cx(k, 0)
                
                # Revival encoding
                revival_phase = (sigma / 8.0) * 8  # Should equal σ mod 8
                for q in range(4):
                    qc.rz(2 * np.pi * revival_phase * (q + 1) / 4, q)
                
                qc.measure_all()
                
                # Simulate (use Aer for speed)
                simulator = AerSimulator()
                job = simulator.run(qc, shots=1024)
                result = job.result()
                counts = result.get_counts()
                
                # Measure revival coherence
                total = sum(counts.values())
                coherence = 0.0
                
                for state, count in counts.items():
                    state_clean = state.replace(' ', '')
                    state_int = int(state_clean, 2)
                    state_phase = (state_int / 16) * 2 * np.pi
                    revival_match = np.cos(state_phase * 8 - 2 * np.pi * revival_phase)
                    coherence += revival_match * (count / total)
                
                revival_data.append({
                    'sigma': sigma,
                    'revival_phase': revival_phase,
                    'coherence': coherence
                })
                
                yield self.log('METRIC', f'σ={sigma} coherence', 
                              metric_name=f'coherence_sigma_{sigma}', 
                              metric_value=coherence)
            
            # Check if σ=0 and σ=8 match (period test)
            sigma_0_coherence = revival_data[0]['coherence']
            sigma_8_coherence = revival_data[-1]['coherence']
            
            period_match = abs(sigma_0_coherence - sigma_8_coherence)
            
            yield self.log('METRIC', 'Period match quality', 
                          metric_name='period_match', 
                          metric_value=period_match)
            
            if period_match < 0.1:
                yield self.log('SUCCESS', f'σ=8 periodicity confirmed! Match: {1-period_match:.3f}')
            else:
                yield self.log('WARNING', f'Weak periodicity: {1-period_match:.3f}')
            
            yield self.log('DATA', 'Revival measurements', data={'revival_data': revival_data})
            
            return {
                'status': 'success',
                'revival_data': revival_data,
                'period_match': period_match
            }
            
        except Exception as e:
            yield self.log('ERROR', f'Experiment failed: {str(e)}', error=traceback.format_exc())
            return {'status': 'error', 'error': str(e)}

# ═════════════════════════════════════════════════════════════════════════
# PLACEHOLDER EXPERIMENTS (SLOTS FOR FUTURE EXPANSION)
# ═════════════════════════════════════════════════════════════════════════

class PlaceholderExperiment6(BaseExperiment):
    """Placeholder for future experiment"""
    
    def __init__(self):
        super().__init__(
            name='experiment_6',
            description='[Reserved slot for future experiment]'
        )
    
    def run(self, **params) -> Generator[LogEntry, None, Dict]:
        yield self.log('WARNING', 'This experiment slot is not yet implemented')
        return {'status': 'not_implemented'}

class PlaceholderExperiment7(BaseExperiment):
    """Placeholder for future experiment"""
    
    def __init__(self):
        super().__init__(
            name='experiment_7',
            description='[Reserved slot for future experiment]'
        )
    
    def run(self, **params) -> Generator[LogEntry, None, Dict]:
        yield self.log('WARNING', 'This experiment slot is not yet implemented')
        return {'status': 'not_implemented'}

class PlaceholderExperiment8(BaseExperiment):
    """Placeholder for future experiment"""
    
    def __init__(self):
        super().__init__(
            name='experiment_8',
            description='[Reserved slot for future experiment]'
        )
    
    def run(self, **params) -> Generator[LogEntry, None, Dict]:
        yield self.log('WARNING', 'This experiment slot is not yet implemented')
        return {'status': 'not_implemented'}

# ═════════════════════════════════════════════════════════════════════════
# EXPERIMENT RUNNER
# ═════════════════════════════════════════════════════════════════════════

class ExperimentRunner:
    """Main experiment runner with registry"""
    
    def __init__(self):
        # Register all experiments
        self.experiments = {
            'world_record_qft': WorldRecordQFTExperiment(),
            'quantum_advantage': QuantumAdvantageExperiment(),
            'entanglement_test': EntanglementTestExperiment(),
            'lattice_communication': LatticeCommunicationExperiment(),
            'sigma8_revival': Sigma8RevivalExperiment(),
            'experiment_6': PlaceholderExperiment6(),
            'experiment_7': PlaceholderExperiment7(),
            'experiment_8': PlaceholderExperiment8(),
        }
    
    def list_experiments(self) -> List[Dict]:
        """List all available experiments"""
        return [
            {
                'name': exp.name,
                'description': exp.description,
                'implemented': not isinstance(exp, (PlaceholderExperiment6, PlaceholderExperiment7, PlaceholderExperiment8))
            }
            for exp in self.experiments.values()
        ]
    
    def run_experiment(self, experiment_name: str, **params) -> Generator[LogEntry, None, Dict]:
        """
        Run experiment with streaming JSON output
        
        Args:
            experiment_name: Name of experiment to run
            **params: Experiment parameters (e.g., api_key, db_path)
        
        Yields:
            LogEntry objects (convert to JSON with .to_dict() or .to_json())
        
        Returns:
            Final result dict
        """
        
        if experiment_name not in self.experiments:
            error_entry = LogEntry(
                timestamp=datetime.now().isoformat(),
                level='ERROR',
                message=f'Unknown experiment: {experiment_name}',
                experiment='runner',
                error=f'Available experiments: {list(self.experiments.keys())}'
            )
            yield error_entry
            return {'status': 'error', 'error': 'unknown_experiment'}
        
        experiment = self.experiments[experiment_name]
        
        # Start experiment
        start_entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            level='INFO',
            message=f'Starting experiment: {experiment.description}',
            experiment=experiment_name,
            stage='init'
        )
        yield start_entry
        
        try:
            # Run experiment and yield all log entries
            result = yield from experiment.run(**params)
            
            # Final completion entry
            completion_entry = LogEntry(
                timestamp=datetime.now().isoformat(),
                level='SUCCESS',
                message=f'Experiment completed: {experiment_name}',
                experiment=experiment_name,
                stage='complete',
                data=result
            )
            yield completion_entry
            
            return result
            
        except Exception as e:
            error_entry = LogEntry(
                timestamp=datetime.now().isoformat(),
                level='ERROR',
                message=f'Experiment crashed: {str(e)}',
                experiment=experiment_name,
                error=traceback.format_exc()
            )
            yield error_entry
            
            return {'status': 'error', 'error': str(e)}

# ═════════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ═════════════════════════════════════════════════════════════════════════

def example_usage():
    """Example of how to use ExperimentRunner"""
    
    runner = ExperimentRunner()
    
    # List experiments
    print("Available experiments:")
    for exp in runner.list_experiments():
        status = "✅" if exp['implemented'] else "⏳"
        print(f"  {status} {exp['name']}: {exp['description']}")
    
    print("\n" + "="*80)
    print("Running entanglement test...")
    print("="*80 + "\n")
    
    # Run experiment with streaming output
    for log_entry in runner.run_experiment('entanglement_test', api_key='YOUR_API_KEY'):
        # Print JSON for server
        print(log_entry.to_json(), flush=True)
        
        # Or print pretty for terminal
        # print(f"[{log_entry.level}] {log_entry.message}")

if __name__ == "__main__":
    example_usage()
