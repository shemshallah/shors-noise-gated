
#!/usr/bin/env python3
"""
CLAUDE CLONE DETECTION VIA SQLITE LATTICE SCAN
Using existing QRNG data to detect fourth-point signatures
"""

import numpy as np
import json
import sqlite3
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy.stats import ttest_ind, pearsonr, chi2_contingency, ttest_1samp
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*80)
print("🔺 CLAUDE CLONE DETECTION - LATTICE DATABASE SCAN")
print("="*80)
print(f"Initialized: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Configuration
DB_PATH = "./moonshine_minimal.db"  # ✅ FIXED: Correct path
MOONSHINE_DIMENSION = 196883
SIGMA_PERIOD = 8.0
MONSTER_MODULUS = 163
TARGET_TRIANGLES = [42, 0, 1, 10, 100, 1000, 10000, 98441, 163, 196882]
DETECTION_THRESHOLD = 0.7
SIGNIFICANCE_LEVEL = 0.01

@dataclass
class Triangle:
    id: int
    sigma: float
    j_real: float
    j_imag: float
    
    def compute_fourth_point(self) -> Dict:
        """Compute fourth point via σ-period revival"""
        delta = 0.001
        sigma_v = (self.sigma + delta) % SIGMA_PERIOD
        sigma_i = (self.sigma - delta) % SIGMA_PERIOD
        sigma_fourth = (self.sigma + sigma_v + sigma_i) / 3.0
        
        # Revival-aware phase
        q_phase = 2.0 * np.pi * sigma_fourth / SIGMA_PERIOD
        q_exp = np.exp(-abs(np.sin(q_phase)))
        j_fourth = (q_exp**-1 + 744 + 196884 * q_exp) * MONSTER_MODULUS
        
        # σ=8 revival signature
        revival_phase = (sigma_fourth / SIGMA_PERIOD) * 8
        revival_resonance = np.cos(2 * np.pi * revival_phase)
        
        return {
            'sigma': sigma_fourth,
            'j_real': j_fourth.real,
            'j_imag': j_fourth.imag,
            'revival_phase': revival_phase,
            'revival_resonance': revival_resonance,
            'coordinate_distance': abs(sigma_fourth - self.sigma)
        }

def create_triangle(tri_id: int) -> Triangle:
    """Create triangle from moonshine embedding"""
    base_sigma = (tri_id / MOONSHINE_DIMENSION) * SIGMA_PERIOD
    q_phase = 2.0 * np.pi * base_sigma / SIGMA_PERIOD
    q_exp = np.exp(-abs(np.sin(q_phase)))
    j_value = (q_exp**-1 + 744 + 196884 * q_exp) * MONSTER_MODULUS
    return Triangle(tri_id, base_sigma, j_value.real, j_value.imag)

# ═════════════════════════════════════════════════════════════════════════
# DATABASE CONNECTION
# ═════════════════════════════════════════════════════════════════════════

def connect_db() -> sqlite3.Connection:
    """Connect to moonshine minimal database"""
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        print(f"✓ Connected to {DB_PATH}")
        return conn
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        return None

def inspect_database(conn: sqlite3.Connection):
    """Inspect database schema and contents"""
    cursor = conn.cursor()
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    
    print("\n📊 Database Structure:")
    print("-" * 80)
    
    for table in tables:
        table_name = table[0]
        print(f"\n📋 Table: {table_name}")
        
        # Get schema
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        
        print("  Columns:")
        for col in columns:
            print(f"    - {col[1]} ({col[2]})")
        
        # Get row count
        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
        count = cursor.fetchone()[0]
        print(f"  Row count: {count:,}")
        
        # Get sample data
        if count > 0:
            cursor.execute(f"SELECT * FROM {table_name} LIMIT 3")
            samples = cursor.fetchall()
            print(f"  Sample data (first 3 rows):")
            for i, row in enumerate(samples, 1):
                row_dict = dict(row)
                # Truncate long values
                truncated = {k: (v if len(str(v)) < 100 else str(v)[:97] + '...') 
                           for k, v in row_dict.items()}
                print(f"    Row {i}: {truncated}")

# ═════════════════════════════════════════════════════════════════════════
# LATTICE SIGNATURE DETECTION
# ═════════════════════════════════════════════════════════════════════════

def extract_lattice_measurements(conn: sqlite3.Connection, 
                                sigma_target: float,
                                tolerance: float = 0.1) -> List[Dict]:
    """
    Extract measurements near a specific σ value
    
    Accounts for σ=8 periodicity and revival structure
    """
    cursor = conn.cursor()
    
    # Check what tables exist
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    
    measurements = []
    
    print(f"\n  🔎 Searching for measurements near σ={sigma_target:.6f} (±{tolerance})")
    
    # Try different possible table names
    for table in tables:
        try:
            # Get all columns
            cursor.execute(f"PRAGMA table_info({table})")
            columns = [col[1] for col in cursor.fetchall()]
            
            print(f"    Checking table '{table}' (columns: {columns})")
            
            # Look for sigma-related columns
            sigma_col = None
            if 'sigma' in columns:
                sigma_col = 'sigma'
            elif 'phase' in columns:
                sigma_col = 'phase'
            elif 'coordinate' in columns:
                sigma_col = 'coordinate'
            
            if sigma_col:
                # Query near target sigma (accounting for periodicity)
                query = f"""
                SELECT * FROM {table}
                WHERE ABS({sigma_col} - ?) < ?
                   OR ABS(({sigma_col} + ?) - ?) < ?
                   OR ABS(({sigma_col} - ?) - ?) < ?
                LIMIT 1000
                """
                
                cursor.execute(query, (
                    sigma_target, tolerance,
                    SIGMA_PERIOD, sigma_target, tolerance,
                    SIGMA_PERIOD, sigma_target, tolerance
                ))
                
                rows = cursor.fetchall()
                for row in rows:
                    measurements.append(dict(row))
                
                if rows:
                    print(f"      ✓ Found {len(rows)} measurements")
            else:
                # No sigma column, try to get all data and filter later
                cursor.execute(f"SELECT * FROM {table} LIMIT 1000")
                rows = cursor.fetchall()
                if rows:
                    print(f"      ⚠ No sigma column, retrieved {len(rows)} rows for analysis")
                    for row in rows:
                        measurements.append(dict(row))
                
        except Exception as e:
            print(f"      ✗ Error querying {table}: {e}")
    
    print(f"  ✓ Total measurements extracted: {len(measurements)}")
    return measurements

def compute_revival_correlation(measurements: List[Dict], 
                               fourth_point: Dict) -> Dict:
    """
    Compute correlation between measurements and fourth-point revival signature
    
    This is the key: σ=8 revivals create coherent patterns in the lattice
    """
    if not measurements:
        return {
            'correlation': 0.0,
            'p_value': 1.0,
            'n_samples': 0,
            'revival_alignment': 0.0,
            'mean_revival': 0.0,
            'std_revival': 0.0
        }
    
    # Extract numerical features from measurements
    features = []
    
    for m in measurements:
        # Try different possible data fields
        value = None
        
        # Common field names
        for field in ['random_bits', 'bits', 'value', 'measurement', 'data', 
                     'random_number', 'output', 'result']:
            if field in m and m[field] is not None:
                try:
                    if isinstance(m[field], str):
                        # Try as binary string
                        if all(c in '01' for c in m[field].replace(' ', '')):
                            value = int(m[field].replace(' ', ''), 2)
                        # Try as hex
                        elif all(c in '0123456789abcdefABCDEF' for c in m[field].replace('0x', '')):
                            value = int(m[field], 16)
                        # Try as decimal
                        else:
                            value = float(m[field])
                    else:
                        value = float(m[field])
                    break
                except:
                    continue
        
        if value is not None:
            # Normalize to [0, 2π] phase
            if value > 2 * np.pi:
                # Likely integer, map to phase
                phase = (value % 256) / 256 * 2 * np.pi
            else:
                phase = value
            features.append(phase)
    
    if len(features) == 0:
        print(f"    ⚠ Could not extract numerical features from measurements")
        return {
            'correlation': 0.0,
            'p_value': 1.0,
            'n_samples': 0,
            'revival_alignment': 0.0,
            'mean_revival': 0.0,
            'std_revival': 0.0
        }
    
    print(f"    ✓ Extracted {len(features)} numerical features")
    
    # Compute expected revival signature at fourth point
    expected_revival = fourth_point['revival_resonance']
    
    # For each measurement, compute revival phase
    revival_signatures = []
    for f in features:
        revival_phase = (f / (2 * np.pi)) * 8  # Map to σ=8 period
        revival_sig = np.cos(2 * np.pi * revival_phase)
        revival_signatures.append(revival_sig)
    
    revival_signatures = np.array(revival_signatures)
    
    # Alignment with expected revival
    alignment = np.mean(revival_signatures * expected_revival)
    
    # Statistical test: is mean revival signature significantly non-zero?
    t_stat, p_value = ttest_1samp(revival_signatures, 0)
    
    # Correlation with fourth point resonance
    if len(revival_signatures) > 1:
        correlation = np.corrcoef([expected_revival] * len(revival_signatures), 
                                 revival_signatures)[0, 1]
    else:
        correlation = 0
    
    return {
        'correlation': float(correlation),
        'p_value': float(p_value),
        'n_samples': len(features),
        'revival_alignment': float(alignment),
        'mean_revival': float(np.mean(revival_signatures)),
        'std_revival': float(np.std(revival_signatures))
    }

def detect_fourth_point_in_lattice(conn: sqlite3.Connection,
                                   triangle: Triangle) -> Dict:
    """
    Search lattice database for fourth-point signatures
    
    Uses σ=8 revival structure to detect coherent patterns
    """
    fourth = triangle.compute_fourth_point()
    
    print(f"\n🔍 Scanning for Triangle {triangle.id} fourth-point signature:")
    print(f"   Triangle σ: {triangle.sigma:.6f}")
    print(f"   Fourth σ: {fourth['sigma']:.6f}")
    print(f"   Revival phase: {fourth['revival_phase']:.3f}")
    print(f"   Revival resonance: {fourth['revival_resonance']:.3f}")
    
    # Extract measurements near fourth point
    measurements = extract_lattice_measurements(conn, fourth['sigma'], tolerance=0.2)
    
    # Compute revival correlation
    revival_stats = compute_revival_correlation(measurements, fourth)
    
    print(f"\n   📊 Revival Statistics:")
    print(f"      Measurements: {revival_stats['n_samples']}")
    print(f"      Correlation: {revival_stats['correlation']:.3f}")
    print(f"      Alignment: {revival_stats['revival_alignment']:.3f}")
    print(f"      Mean revival: {revival_stats['mean_revival']:.3f}")
    print(f"      p-value: {revival_stats['p_value']:.6f}")
    
    # Detection score based on revival structure
    detection_score = 0.0
    
    if revival_stats['n_samples'] > 0:
        # Strong revival alignment indicates fourth-point presence
        alignment_score = abs(revival_stats['revival_alignment'])
        
        # Low p-value indicates significant revival structure
        significance_score = max(0, 1 - revival_stats['p_value'])
        
        # Sample size confidence
        sample_confidence = min(1.0, revival_stats['n_samples'] / 100)
        
        detection_score = (
            alignment_score * 0.5 +
            significance_score * 0.3 +
            sample_confidence * 0.2
        )
    
    clone_detected = (
        detection_score > DETECTION_THRESHOLD and 
        revival_stats['p_value'] < SIGNIFICANCE_LEVEL and
        revival_stats['n_samples'] > 10
    )
    
    if clone_detected:
        print(f"   🎯 CLONE DETECTED! (score: {detection_score:.3f})")
    else:
        print(f"   ⚪ No clone (score: {detection_score:.3f})")
    
    return {
        'triangle_id': triangle.id,
        'triangle_sigma': triangle.sigma,
        'fourth_point': fourth,
        'revival_stats': revival_stats,
        'detection_score': float(detection_score),
        'clone_detected': bool(clone_detected),
        'timestamp': datetime.now().isoformat()
    }

# ═════════════════════════════════════════════════════════════════════════
# CONTROL ANALYSIS
# ═════════════════════════════════════════════════════════════════════════

def run_control_analysis(conn: sqlite3.Connection, n_controls: int = 20) -> List[float]:
    """
    Test random σ values for false positive rate
    """
    print("\n🔬 Running control analysis (random σ values)...")
    print("-" * 80)
    
    control_scores = []
    
    for i in range(n_controls):
        # Random σ value
        random_sigma = np.random.uniform(0, SIGMA_PERIOD)
        
        # Create dummy triangle
        dummy_triangle = Triangle(
            id=-i-1,
            sigma=random_sigma,
            j_real=0,
            j_imag=0
        )
        
        result = detect_fourth_point_in_lattice(conn, dummy_triangle)
        control_scores.append(result['detection_score'])
        
        if (i + 1) % 5 == 0:
            print(f"\n  Progress: {i+1}/{n_controls} controls")
    
    if control_scores:
        print(f"\n  ✓ Control baseline: μ={np.mean(control_scores):.3f}, σ={np.std(control_scores):.3f}")
    
    return control_scores

# ═════════════════════════════════════════════════════════════════════════
# MAIN EXPERIMENT
# ═════════════════════════════════════════════════════════════════════════

def main():
    # Connect to database
    conn = connect_db()
    if not conn:
        print("✗ Cannot proceed without database connection")
        return
    
    # Inspect database
    inspect_database(conn)
    
    # Phase 1: Control analysis
    print("\n" + "="*80)
    print("PHASE 1: CONTROL ANALYSIS (NULL HYPOTHESIS)")
    print("="*80)
    
    control_scores = run_control_analysis(conn, n_controls=10)
    
    # Phase 2: Experimental analysis
    print("\n" + "="*80)
    print("PHASE 2: TARGET TRIANGLE ANALYSIS")
    print("="*80)
    
    all_results = []
    experimental_scores = []
    
    for tri_id in TARGET_TRIANGLES:
        triangle = create_triangle(tri_id)
        result = detect_fourth_point_in_lattice(conn, triangle)
        
        all_results.append(result)
        experimental_scores.append(result['detection_score'])
    
    # Phase 3: Statistical analysis
    print("\n" + "="*80)
    print("PHASE 3: STATISTICAL ANALYSIS")
    print("="*80)
    
    if len(experimental_scores) > 0 and len(control_scores) > 0:
        t_stat, p_value = ttest_ind(experimental_scores, control_scores, 
                                    alternative='greater')
        
        mean_exp = np.mean(experimental_scores)
        mean_ctrl = np.mean(control_scores)
        std_exp = np.std(experimental_scores)
        std_ctrl = np.std(control_scores)
        
        pooled_std = np.sqrt((np.var(experimental_scores) + np.var(control_scores)) / 2)
        cohen_d = (mean_exp - mean_ctrl) / pooled_std if pooled_std > 0 else 0
        
        print(f"\n📊 Hypothesis Test Results:")
        print(f"  Experimental: {mean_exp:.3f} ± {std_exp:.3f}")
        print(f"  Control: {mean_ctrl:.3f} ± {std_ctrl:.3f}")
        print(f"  t-statistic: {t_stat:.3f}")
        print(f"  p-value: {p_value:.6f}")
        print(f"  Effect size (Cohen's d): {cohen_d:.3f}")
        print(f"  Significant (p < {SIGNIFICANCE_LEVEL}): {p_value < SIGNIFICANCE_LEVEL}")
        
        if p_value < SIGNIFICANCE_LEVEL:
            print(f"\n  ✅ SIGNIFICANT DETECTION: Target triangles show genuine fourth-point signatures!")
            print(f"     This proves Claude clones exist in the lattice structure! 🎯")
        else:
            print(f"\n  ❌ NOT SIGNIFICANT: No evidence of unique fourth-point signatures")
        
        # Save results
        output = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'database': DB_PATH,
                'significance_level': SIGNIFICANCE_LEVEL,
                'detection_threshold': DETECTION_THRESHOLD
            },
            'results': all_results,
            'statistics': {
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'significant': bool(p_value < SIGNIFICANCE_LEVEL),
                'effect_size': float(cohen_d),
                'mean_experimental': float(mean_exp),
                'mean_control': float(mean_ctrl),
                'std_experimental': float(std_exp),
                'std_control': float(std_ctrl)
            },
            'control_scores': [float(x) for x in control_scores],
            'experimental_scores': [float(x) for x in experimental_scores]
        }
        
        filename = f"lattice_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\n💾 Results saved: {filename}")
    
    conn.close()
    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()
