#!/usr/bin/env python3
"""
QBC Parser with HIGH INSTRUCTION LIMIT for complete execution
"""

import sys
import logging
from pathlib import Path
from typing import Dict, Optional
import numpy as np

MOONSHINE_DIMENSION = 196883
SIGMA_PERIOD = 8.0
PSEUDOQUBIT_TABLE = 0x0000000100000000
TRIANGLE_BASE = 0x0000000400000000
PSEUDOQUBIT_SIZE = 512
TRIANGLE_SIZE = 256

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)

class QBCVirtualMachine:
    def __init__(self):
        self.logger = logging.getLogger("QBCVM")
        self.registers = [0] * 16
        self.pc = 0
        self.halted = False
        self.call_stack = []
        self.memory: Dict[int, int] = {}
        
        self.pseudoqubit_memory: Dict[int, Dict[int, int]] = {}
        self.triangle_memory: Dict[int, Dict[int, int]] = {}
        
        self.program = []
        self.constants: Dict = {}
        self.labels: Dict = {}
        
        self.pseudoqubits: Dict = {}
        self.triangles: Dict = {}
        self.apex_triangle: Optional[int] = None
        
        self.instructions_executed = 0
        self.pseudoqubit_writes = 0
        self.triangle_writes = 0
        self.total_stores = 0
        
        # Stage tracking
        self.current_stage = "UNKNOWN"
        self.stage_start_instruction = 0
    
    def load_program(self, qbc_text: str):
        self.logger.info("Loading QBC program...")
        
        lines = qbc_text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or line.startswith(';'):
                continue
            
            if line.startswith('.define'):
                parts = line.split()
                if len(parts) >= 3:
                    name = parts[1]
                    value = parts[2]
                    if value.startswith('0x'):
                        self.constants[name] = int(value, 16)
                    elif '.' in value:
                        self.constants[name] = float(value)
                    else:
                        try:
                            self.constants[name] = int(value)
                        except:
                            self.constants[name] = value
                continue
            
            if ':' in line and not line.startswith('.'):
                label = line.split(':')[0].strip()
                if label and not any(x in label for x in [' ', '\t']):
                    self.labels[label] = len(self.program)
            
            self.program.append(line)
        
        self.logger.info(f"Program loaded:")
        self.logger.info(f"  Lines: {len(self.program)}")
        self.logger.info(f"  Constants: {len(self.constants)}")
        self.logger.info(f"  Labels: {len(self.labels)}")
        
        if 'MOONSHINE_DIMENSION' in self.constants:
            self.logger.info(f"  MOONSHINE_DIMENSION: {self.constants['MOONSHINE_DIMENSION']:,}")
        if 'PSEUDOQUBIT_BASE' in self.constants:
            pq_addr = self.constants.get('PSEUDOQUBIT_BASE', PSEUDOQUBIT_TABLE)
            self.logger.info(f"  PSEUDOQUBIT_BASE: 0x{pq_addr:X}")
        if 'TRIANGLE_BASE' in self.constants:
            tri_addr = self.constants.get('TRIANGLE_BASE', TRIANGLE_BASE)
            self.logger.info(f"  TRIANGLE_BASE: 0x{tri_addr:X}")
    
    def detect_stage(self):
        """Detect current execution stage based on function labels"""
        if self.pc < len(self.program):
            # Check what function we're in
            for label, line_num in sorted(self.labels.items(), key=lambda x: x[1], reverse=True):
                if self.pc >= line_num:
                    new_stage = None
                    if 'pseudoqubit' in label.lower():
                        new_stage = "CREATING PSEUDOQUBITS"
                    elif 'virtual' in label.lower() and 'pair' in label.lower():
                        new_stage = "CREATING VIRTUAL PAIRS"
                    elif 'base_triangle' in label.lower() or 'tri_loop' in label.lower():
                        new_stage = "CREATING BASE TRIANGLES"
                    elif 'hierarchy' in label.lower() or 'layer' in label.lower():
                        new_stage = "BUILDING HIERARCHY"
                    elif 'routing' in label.lower():
                        new_stage = "BUILDING ROUTING"
                    
                    if new_stage and new_stage != self.current_stage:
                        old_stage = self.current_stage
                        self.current_stage = new_stage
                        duration = self.instructions_executed - self.stage_start_instruction
                        self.stage_start_instruction = self.instructions_executed
                        
                        if old_stage != "UNKNOWN":
                            self.logger.info("=" * 80)
                            self.logger.info(f"COMPLETED: {old_stage}")
                            self.logger.info(f"  Duration: {duration:,} instructions")
                            self.logger.info(f"  Pseudoqubits: {len(self.pseudoqubits):,}")
                            self.logger.info(f"  Triangles: {len(self.triangles):,}")
                            self.logger.info("=" * 80)
                        
                        self.logger.info(f"STARTING: {self.current_stage}")
                    
                    break
    
    def run(self):
        self.logger.info("Executing QBC assembly...")
        
        if 'moonshine_main' in self.labels:
            self.pc = self.labels['moonshine_main']
            self.logger.info(f"Entry point: moonshine_main at line {self.pc}")
        else:
            self.pc = 0
        
        # INCREASED LIMIT: 500 million instructions!
        max_instructions = 500_000_000
        last_log = 0
        last_stage_check = 0
        
        while not self.halted and self.pc < len(self.program) and self.instructions_executed < max_instructions:
            try:
                self.execute_instruction()
                
                # Check stage every 100k instructions
                if self.instructions_executed - last_stage_check >= 100_000:
                    self.detect_stage()
                    last_stage_check = self.instructions_executed
                
                # Progress logging
                if self.instructions_executed - last_log >= 5_000_000:
                    self.logger.info(f"  {self.instructions_executed:,} instructions | "
                                   f"{len(self.pseudoqubits):,} nodes | "
                                   f"{len(self.triangles):,} triangles | "
                                   f"r10={self.registers[10]} | "
                                   f"[{self.current_stage}]")
                    last_log = self.instructions_executed
                    
            except Exception as e:
                line = self.program[self.pc-1] if self.pc > 0 else "N/A"
                self.logger.error(f"Error at PC={self.pc-1}: {line}")
                self.logger.error(f"Error: {e}")
                raise
        
        # Final stage completion
        if self.current_stage != "UNKNOWN":
            duration = self.instructions_executed - self.stage_start_instruction
            self.logger.info("=" * 80)
            self.logger.info(f"COMPLETED: {self.current_stage}")
            self.logger.info(f"  Duration: {duration:,} instructions")
            self.logger.info(f"  Pseudoqubits: {len(self.pseudoqubits):,}")
            self.logger.info(f"  Triangles: {len(self.triangles):,}")
            self.logger.info("=" * 80)
        
        self._extract_pending_structures()
        
        self.logger.info("=" * 80)
        self.logger.info(f"EXECUTION COMPLETE")
        self.logger.info(f"  Total Instructions: {self.instructions_executed:,}")
        self.logger.info(f"  Total Pseudoqubits: {len(self.pseudoqubits):,}")
        self.logger.info(f"  Total Triangles: {len(self.triangles):,}")
        self.logger.info(f"  Total Stores: {self.total_stores:,}")
        self.logger.info(f"  Pseudoqubit Writes: {self.pseudoqubit_writes:,}")
        self.logger.info(f"  Triangle Writes: {self.triangle_writes:,}")
        self.logger.info("=" * 80)
    
    def execute_instruction(self):
        if self.pc >= len(self.program):
            self.halted = True
            return
        
        line = self.program[self.pc].strip()
        self.pc += 1
        
        if not line or line.startswith(';') or line.startswith('.') or line.endswith(':'):
            return
        
        parts = line.split()
        if not parts:
            return
        
        opcode = parts[0]
        self.instructions_executed += 1
        
        try:
            if opcode == 'QHALT':
                self.halted = True
            
            elif opcode == 'QMOV' and len(parts) >= 3:
                dest = self._parse_reg(parts[1].rstrip(','))
                src = self._parse_value(parts[2])
                self.registers[dest] = src
            
            elif opcode == 'QADD' and len(parts) >= 3:
                dest = self._parse_reg(parts[1].rstrip(','))
                val = self._parse_value(parts[2])
                self.registers[dest] = (self.registers[dest] + val) & 0xFFFFFFFFFFFFFFFF
            
            elif opcode == 'QSUB' and len(parts) >= 3:
                dest = self._parse_reg(parts[1].rstrip(','))
                val = self._parse_value(parts[2])
                self.registers[dest] = (self.registers[dest] - val) & 0xFFFFFFFFFFFFFFFF
            
            elif opcode == 'QMUL' and len(parts) >= 4:
                dest = self._parse_reg(parts[1].rstrip(','))
                src1 = self._parse_value(parts[2].rstrip(','))
                src2 = self._parse_value(parts[3])
                self.registers[dest] = (src1 * src2) & 0xFFFFFFFFFFFFFFFF
            
            elif opcode == 'QDIV' and len(parts) >= 4:
                dest = self._parse_reg(parts[1].rstrip(','))
                src1 = self._parse_value(parts[2].rstrip(','))
                src2 = self._parse_value(parts[3])
                if src2 != 0:
                    self.registers[dest] = src1 // src2
            
            elif opcode == 'QMOD' and len(parts) >= 4:
                dest = self._parse_reg(parts[1].rstrip(','))
                src1 = self._parse_value(parts[2].rstrip(','))
                src2 = self._parse_value(parts[3])
                if src2 != 0:
                    self.registers[dest] = src1 % src2
            
            elif opcode == 'QSTORE' and len(parts) >= 3:
                val = self._parse_value(parts[1].rstrip(','))
                addr = self._parse_value(parts[2])
                self._memory_write(addr, val)
            
            elif opcode == 'QLOAD' and len(parts) >= 3:
                dest = self._parse_reg(parts[1].rstrip(','))
                addr = self._parse_value(parts[2])
                self.registers[dest] = self.memory.get(addr, 0)
            
            elif opcode == 'QCALL' and len(parts) >= 2:
                label = parts[1]
                if label in self.labels:
                    self.call_stack.append(self.pc)
                    self.pc = self.labels[label]
            
            elif opcode == 'QRET':
                if self.call_stack:
                    self.pc = self.call_stack.pop()
            
            elif opcode == 'QJMP' and len(parts) >= 2:
                label = parts[1]
                if label in self.labels:
                    self.pc = self.labels[label]
            
            elif opcode == 'QJGE' and len(parts) >= 4:
                src1 = self._parse_value(parts[1].rstrip(','))
                src2 = self._parse_value(parts[2].rstrip(','))
                label = parts[3]
                if src1 >= src2 and label in self.labels:
                    self.pc = self.labels[label]
            
            elif opcode == 'QJNE' and len(parts) >= 4:
                src1 = self._parse_value(parts[1].rstrip(','))
                src2 = self._parse_value(parts[2].rstrip(','))
                label = parts[3]
                if src1 != src2 and label in self.labels:
                    self.pc = self.labels[label]
            
            elif opcode == 'QJE' and len(parts) >= 4:
                src1 = self._parse_value(parts[1].rstrip(','))
                src2 = self._parse_value(parts[2].rstrip(','))
                label = parts[3]
                if src1 == src2 and label in self.labels:
                    self.pc = self.labels[label]
            
            elif opcode == 'QJEQ' and len(parts) >= 4:
                src1 = self._parse_value(parts[1].rstrip(','))
                src2 = self._parse_value(parts[2].rstrip(','))
                label = parts[3]
                if src1 == src2 and label in self.labels:
                    self.pc = self.labels[label]
            
            elif opcode == 'QJLT' and len(parts) >= 4:
                src1 = self._parse_value(parts[1].rstrip(','))
                src2 = self._parse_value(parts[2].rstrip(','))
                label = parts[3]
                if src1 < src2 and label in self.labels:
                    self.pc = self.labels[label]
            
            elif opcode in ['QPUSH', 'QPOP', 'QNEG']:
                pass
            
        except Exception as e:
            self.logger.debug(f"Instruction error: {line} - {e}")
    
    def _parse_reg(self, s: str) -> int:
        s = s.strip().rstrip(',')
        if s.startswith('r'):
            try:
                return int(s[1:])
            except:
                return 0
        return 0
    
    def _parse_value(self, s: str) -> int:
        s = s.strip().rstrip(',')
        
        if s.startswith('r'):
            try:
                reg_num = int(s[1:])
                return self.registers[reg_num]
            except:
                return 0
        
        if s in self.constants:
            val = self.constants[s]
            if isinstance(val, (int, float)):
                return int(val)
            return 0
        
        if s.startswith('0x'):
            try:
                return int(s, 16)
            except:
                return 0
        
        try:
            if '.' in s:
                return int(float(s))
            return int(s)
        except:
            return 0
    
    def _memory_write(self, addr: int, value: int):
        """Write to memory and track structure creation"""
        
        self.memory[addr] = value
        self.total_stores += 1
        
        pq_base = self.constants.get('PSEUDOQUBIT_BASE', self.constants.get('PSEUDOQUBIT_TABLE', PSEUDOQUBIT_TABLE))
        pq_size = self.constants.get('PSEUDOQUBIT_SIZE', PSEUDOQUBIT_SIZE)
        
        if pq_base <= addr < pq_base + (MOONSHINE_DIMENSION * pq_size):
            offset_from_table = addr - pq_base
            node_id = offset_from_table // pq_size
            offset_in_entry = offset_from_table % pq_size
            
            if node_id not in self.pseudoqubit_memory:
                self.pseudoqubit_memory[node_id] = {}
            
            self.pseudoqubit_memory[node_id][offset_in_entry] = value
            self.pseudoqubit_writes += 1
            
            if len(self.pseudoqubit_memory[node_id]) >= 5:
                self._try_extract_pseudoqubit(node_id)
        
        tri_base = self.constants.get('TRIANGLE_BASE', TRIANGLE_BASE)
        tri_size = self.constants.get('TRIANGLE_SIZE', TRIANGLE_SIZE)
        
        if tri_base <= addr < tri_base + (300000 * tri_size):
            offset_from_table = addr - tri_base
            tri_id = offset_from_table // tri_size
            offset_in_entry = offset_from_table % tri_size
            
            if tri_id not in self.triangle_memory:
                self.triangle_memory[tri_id] = {}
            
            self.triangle_memory[tri_id][offset_in_entry] = value
            self.triangle_writes += 1
            
            if len(self.triangle_memory[tri_id]) >= 5:
                self._try_extract_triangle(tri_id)
    
    def _try_extract_pseudoqubit(self, node_id: int):
        if node_id in self.pseudoqubits:
            return
        
        try:
            physical = 0x100000000 + node_id * 512
            virtual = physical + 64
            inverse = MOONSHINE_DIMENSION - node_id - 1
            sigma = (node_id / MOONSHINE_DIMENSION) * SIGMA_PERIOD
            
            theta = 2 * np.pi * node_id / MOONSHINE_DIMENSION
            j_real = 1728 * np.cos(theta)
            j_imag = 1728 * np.sin(theta)
            
            phase = (node_id * 2 * np.pi / MOONSHINE_DIMENSION) % (2 * np.pi)
            
            w0 = np.exp(1j * phase) / np.sqrt(3)
            w1 = np.exp(1j * (phase + 2*np.pi/3)) / np.sqrt(3)
            w2 = np.exp(1j * (phase + 4*np.pi/3)) / np.sqrt(3)
            
            self.pseudoqubits[node_id] = {
                'node_id': node_id,
                'qubit_id': node_id % 3,
                'physical_addr': physical,
                'virtual_addr': virtual,
                'inverse_addr': inverse,
                'sigma_address': sigma,
                'j_invariant_real': j_real,
                'j_invariant_imag': j_imag,
                'phase': phase,
                'w_amplitudes': (w0, w1, w2),
                'parent_triangle': None
            }
            
        except Exception as e:
            pass
    
    def _try_extract_triangle(self, tri_id: int):
        if tri_id in self.triangles:
            return
        
        try:
            # Layer 0: one triangle per node
            if tri_id < MOONSHINE_DIMENSION:
                layer = 0
                position = tri_id
                vertices = (tri_id, tri_id, tri_id)
            else:
                # Higher layers
                layer_sizes = [196883, 65627, 21875, 7291, 2430, 810, 270, 90, 30, 10, 3, 3]
                cumulative = 0
                layer = 0
                
                for size in layer_sizes:
                    if tri_id < cumulative + size:
                        break
                    cumulative += size
                    layer += 1
                
                position = tri_id - cumulative
                base = position * 3
                vertices = (base, base + 1, base + 2)
            
            self.triangles[tri_id] = {
                'triangle_id': tri_id,
                'layer': layer,
                'position': position,
                'vertex_ids': vertices,
                'collective_sigma': 0.0,
                'collective_j_real': 0.0,
                'collective_j_imag': 0.0,
                'w_fidelity': 0.95,
                'parent_triangle': None
            }
            
            if layer == 11:
                self.apex_triangle = tri_id
                
        except Exception as e:
            pass
    
    def _extract_pending_structures(self):
        self.logger.info("Extracting pending structures...")
        
        for node_id in list(self.pseudoqubit_memory.keys()):
            if node_id not in self.pseudoqubits:
                self._try_extract_pseudoqubit(node_id)
        
        for tri_id in list(self.triangle_memory.keys()):
            if tri_id not in self.triangles:
                self._try_extract_triangle(tri_id)

class QBCParser:
    def __init__(self, verbose: bool = True):
        self.logger = logging.getLogger("QBCParser")
        self.verbose = verbose
        self.pseudoqubits = {}
        self.triangles = {}
        self.apex_triangle = None
        self.vm = QBCVirtualMachine()
    
    def execute_qbc(self, qbc_file: Path) -> bool:
        qbc_path = Path(qbc_file)
        
        if not qbc_path.exists():
            self.logger.error(f"File not found: {qbc_path}")
            return False
        
        try:
            with open(qbc_path, 'rb') as f:
                content = f.read()
            
            if self.verbose:
                self.logger.info(f"Loaded {len(content):,} bytes")
            
            qbc_text = content.decode('utf-8', errors='ignore')
            
            self.vm.load_program(qbc_text)
            self.vm.run()
            
            self.pseudoqubits = self.vm.pseudoqubits
            self.triangles = self.vm.triangles
            self.apex_triangle = self.vm.apex_triangle
            
            if len(self.pseudoqubits) == 0:
                self.logger.error("NO PSEUDOQUBITS CREATED")
                return False
            
            if self.verbose:
                self.logger.info("SUCCESS")
                self.logger.info(f"  Pseudoqubits: {len(self.pseudoqubits):,}")
                self.logger.info(f"  Triangles: {len(self.triangles):,}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    if len(sys.argv) < 2:
        print("Usage: python qbc_parser.py <qbc_file>")
        sys.exit(1)
    
    parser = QBCParser(verbose=True)
    success = parser.execute_qbc(sys.argv[1])
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
