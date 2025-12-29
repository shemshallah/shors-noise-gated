#!/usr/bin/env python3
"""
MOONSHINE QUANTUM INTERNET - CLIENT (SQLite Version)
=====================================================

Lightweight client using SQLite database.
Perfect for Colab/Jupyter - no pickle issues!

Uses LOCAL Aer simulator - no quantum hardware account needed!
"""

import sqlite3
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

# ════════════════════════════════════════════════════════════════════════════
# MOONSHINE CLIENT (SQLite)
# ════════════════════════════════════════════════════════════════════════════

class MoonshineClient:
    """
    Moonshine Quantum Internet Client (SQLite)
    
    - Loads from SQLite database (works perfectly in Colab!)
    - Uses local Aer simulator (no quantum account needed!)
    - Query by triangle ID or σ-coordinate
    - Run quantum experiments locally
    """
    
    def __init__(self, db_path: str = None):
        """
        Initialize client
        
        Args:
            db_path: Path to moonshine_routes.db
                     If None, looks in standard locations
        """
        print("\n" + "="*80)
        print("🌙 MOONSHINE QUANTUM INTERNET CLIENT")
        print("="*80)
        
        # Connect to database
        self.conn = self._connect_database(db_path)
        
        # Get metadata
        self.metadata = self._load_metadata()
        
        # Initialize Aer simulator
        self.simulator = AerSimulator(method='statevector')
        print(f"✓ Aer simulator initialized")
        
        total_routes = int(self.metadata.get('total_routes', 0))
        print(f"\n✨ Client ready! {total_routes:,} routes available")
        print("="*80 + "\n")
    
    def _connect_database(self, custom_path: Optional[str] = None) -> sqlite3.Connection:
        """Connect to SQLite database"""
        
        # Try custom path first
        if custom_path:
            db_path = Path(custom_path)
            if db_path.exists():
                return self._open_db(db_path)
        
        # Try standard locations
        standard_paths = [
            Path('moonshine_routes.db'),
            Path('data/moonshine_routes.db'),
            Path('/app/data/moonshine_routes.db'),
            Path.home() / 'moonshine_routes.db',
        ]
        
        for path in standard_paths:
            if path.exists():
                print(f"📂 Found database: {path}")
                return self._open_db(path)
        
        raise FileNotFoundError(
            f"Database not found!\n"
            f"Tried locations:\n" + 
            "\n".join(f"  - {p}" for p in standard_paths) +
            f"\n\nDownload moonshine_routes.db from server."
        )
    
    def _open_db(self, path: Path) -> sqlite3.Connection:
        """Open database and verify structure"""
        try:
            conn = sqlite3.connect(str(path))
            conn.row_factory = sqlite3.Row  # Access columns by name
            
            # Verify tables exist
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM routes")
            total = cursor.fetchone()[0]
            
            print(f"✓ Connected to database with {total:,} routes")
            return conn
            
        except Exception as e:
            raise RuntimeError(f"Failed to open database from {path}: {e}")
    
    def _load_metadata(self) -> Dict:
        """Load metadata from database"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT key, value FROM metadata")
        return {row['key']: row['value'] for row in cursor.fetchall()}
    
    def get_route(self, triangle_id: int) -> Optional[Dict]:
        """
        Get routing information for a specific triangle
        
        Args:
            triangle_id: Triangle ID (0 to 196,882)
            
        Returns:
            Dict with routing info or None if not found
        """
        cursor = self.conn.cursor()
        cursor.execute('SELECT * FROM routes WHERE triangle_id = ?', (triangle_id,))
        row = cursor.fetchone()
        
        if row:
            return dict(row)
        
        print(f"⚠️  Triangle {triangle_id} not found")
        return None
    
    def find_by_sigma(self, sigma: float, tolerance: float = 0.01, limit: int = 100) -> List[Dict]:
        """
        Find triangles near a σ-coordinate
        
        Args:
            sigma: Target σ-coordinate
            tolerance: How close σ must be
            limit: Maximum results to return
            
        Returns:
            List of matching routes sorted by distance
        """
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT *, ABS(sigma - ?) as distance
            FROM routes 
            WHERE sigma BETWEEN ? AND ?
            ORDER BY distance
            LIMIT ?
        ''', (sigma, sigma - tolerance, sigma + tolerance, limit))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def find_by_j_invariant(self, j_real: float, j_imag: float, 
                           tolerance: float = 100.0, limit: int = 100) -> List[Dict]:
        """
        Find triangles near a j-invariant coordinate
        
        Args:
            j_real: Target j-invariant real component
            j_imag: Target j-invariant imaginary component
            tolerance: Distance tolerance
            limit: Maximum results
            
        Returns:
            List of matching routes
        """
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT *,
                   SQRT(POWER(j_real - ?, 2) + POWER(j_imag - ?, 2)) as distance
            FROM routes
            WHERE distance < ?
            ORDER BY distance
            LIMIT ?
        ''', (j_real, j_imag, tolerance, limit))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def get_special_triangles(self) -> Dict[str, Dict]:
        """Get the 3 special triangles used in manifold bridge"""
        return {
            'first': self.get_route(0),
            'middle': self.get_route(98441),
            'last': self.get_route(196882)
        }
    
    def query_range(self, start_id: int, end_id: int) -> List[Dict]:
        """Get range of triangles"""
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT * FROM routes 
            WHERE triangle_id BETWEEN ? AND ?
            ORDER BY triangle_id
        ''', (start_id, end_id))
        
        return [dict(row) for row in cursor.fetchall()]
    
    def display_route(self, triangle_id: int):
        """Display detailed routing info for a triangle"""
        route = self.get_route(triangle_id)
        
        if not route:
            return
        
        print(f"\n📍 Triangle {triangle_id} Routing Info:")
        print(f"   σ-address:   {route['sigma']:.6f}")
        print(f"   j-invariant: {route['j_real']:.4f} + {route['j_imag']:.4f}i")
        
        if 'theta' in route:
            print(f"   Phase θ:     {route['theta']:.6f} rad")
        
        print(f"   Addresses:")
        print(f"      Pseudoqubit:     0x{route['pq_addr']:X}")
        print(f"      Virtual:         0x{route['v_addr']:X}")
        print(f"      Inverse-Virtual: 0x{route['iv_addr']:X}")
    
    def create_probe_circuit(self, triangle_id: int, shots: int = 1024) -> Dict:
        """
        Create and run Aer probe circuit for a triangle
        
        This measures virtual + inverse-virtual qubits to probe
        the pseudophysical state through noise routing.
        
        Args:
            triangle_id: Which triangle to probe
            shots: Number of measurements
            
        Returns:
            Dict with measurement results and routing info
        """
        route = self.get_route(triangle_id)
        if not route:
            return {'error': f'Triangle {triangle_id} not found'}
        
        # Create W-state circuit
        qc = QuantumCircuit(3, 2)
        
        # W-state preparation
        qc.x(0)
        theta1 = 2 * np.arccos(np.sqrt(2/3))
        qc.ry(theta1, 0)
        theta2 = 2 * np.arcsin(1/np.sqrt(2))
        qc.ry(theta2/2, 1)
        qc.cx(0, 1)
        qc.ry(-theta2/2, 1)
        qc.cx(0, 1)
        qc.x(0)
        qc.ccx(0, 1, 2)
        qc.x(0)
        qc.x(1)
        qc.ccx(0, 1, 2)
        qc.x(1)
        
        # σ-modulation based on triangle's σ-coordinate
        sigma = route['sigma']
        for qubit in range(3):
            angle_x = sigma * np.pi / 4
            angle_z = sigma * np.pi / 2
            qc.rx(angle_x, qubit)
            qc.rz(angle_z, qubit)
        
        # Measure virtual (q1) and inverse-virtual (q2)
        # Physical (q0) stays unmeasured!
        qc.measure([1, 2], [0, 1])
        
        # Run on Aer
        result = self.simulator.run(qc, shots=shots).result()
        counts = result.get_counts()
        
        # Calculate fidelity
        w_states = ['00', '01', '10']
        w_count = sum(counts.get(state, 0) for state in w_states)
        fidelity = w_count / shots
        
        return {
            'triangle_id': triangle_id,
            'route': route,
            'counts': counts,
            'fidelity': fidelity,
            'shots': shots,
            'measured_qubits': 'Virtual (q1) + Inverse-Virtual (q2)'
        }
    
    def stats(self) -> Dict:
        """Get statistics about the routing table"""
        cursor = self.conn.cursor()
        
        cursor.execute('SELECT COUNT(*), MIN(sigma), MAX(sigma) FROM routes')
        count, min_sigma, max_sigma = cursor.fetchone()
        
        cursor.execute('SELECT MIN(j_real), MAX(j_real) FROM routes')
        min_j, max_j = cursor.fetchone()
        
        return {
            'total_routes': count,
            'sigma_range': (min_sigma, max_sigma),
            'j_real_range': (min_j, max_j),
            'source': 'sqlite_database'
        }
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
    
    def __enter__(self):
        """Context manager support"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager cleanup"""
        self.close()


# ════════════════════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ════════════════════════════════════════════════════════════════════════════

def main():
    """Example client usage"""
    
    # Initialize client (auto-finds database)
    with MoonshineClient() as client:
        
        # Show stats
        stats = client.stats()
        print(f"📊 Routing Table Stats:")
        print(f"   Total routes: {stats['total_routes']:,}")
        print(f"   σ-range: [{stats['sigma_range'][0]:.6f}, {stats['sigma_range'][1]:.6f})")
        print(f"   j-real range: [{stats['j_real_range'][0]:.2f}, {stats['j_real_range'][1]:.2f}]")
        
        # Display special triangles
        print(f"\n🔺 Special Triangles (Manifold Bridge):")
        special = client.get_special_triangles()
        for name, route in special.items():
            if route:
                print(f"\n   {name.upper()}: Triangle {route['triangle_id']}")
                print(f"      σ-address: {route['sigma']:.6f}")
                print(f"      j-invariant: {route['j_real']:.4f} + {route['j_imag']:.4f}i")
        
        # Query specific triangle
        print(f"\n📍 Query Triangle 12345:")
        client.display_route(12345)
        
        # Find triangles near σ=2.5
        print(f"\n🔍 Triangles near σ=2.5:")
        matches = client.find_by_sigma(2.5, tolerance=0.1)
        for match in matches[:5]:  # Show top 5
            print(f"   Triangle {match['triangle_id']:6d}: σ={match['sigma']:.6f} (Δ={match['distance']:.6f})")
        
        # Run probe circuit
        print(f"\n🔬 Probing Triangle 0 with Aer:")
        result = client.create_probe_circuit(triangle_id=0, shots=1024)
        print(f"   Fidelity: {result['fidelity']:.4f}")
        print(f"   Measured: {result['measured_qubits']}")
        print(f"   Top outcomes: {dict(list(result['counts'].items())[:5])}")


if __name__ == '__main__':
    main()
