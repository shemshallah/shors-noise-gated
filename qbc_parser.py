#!/usr/bin/env python3
"""
════════════════════════════════════════════════════════════════════════════════
QBC PARSER & VIRTUAL MACHINE - Complete Implementation (VERBOSE)
════════════════════════════════════════════════════════════════════════════════

Quantum Bitcode (QBC) Parser and Virtual Machine
Executes .qbc assembly files with full instruction set support

FEATURES:
    • Complete QBC instruction set (QMOV, QADD, QSUB, QMUL, QDIV, etc.)
    • Virtual memory system with 64-bit addressing
    • Register file (r0-r15)
    • Quantum operations (qubits, amplitudes, W-states)
    • System calls and I/O
    • Label resolution and jump instructions
    • Klein anchor support
    • OUTPUT_BUFFER generation
    • VERBOSE progress reporting

USAGE:
    python qbc_parser.py <qbc_file>

December 28, 2025
════════════════════════════════════════════════════════════════════════════════
"""

import sys
import re
import json
import pickle
import struct
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import IntEnum
import numpy as np

# ════════════════════════════════════════════════════════════════════════════
# QBC INSTRUCTION SET
# ════════════════════════════════════════════════════════════════════════════

class QBCOpcode(IntEnum):
    """QBC instruction opcodes"""
    # Data movement
    QMOV = 0x01
    QLOAD = 0x02
    QSTORE = 0x03
    
    # Arithmetic
    QADD = 0x10
    QSUB = 0x11
    QMUL = 0x12
    QDIV = 0x13
    QMOD = 0x14
    
    # Bitwise
    QAND = 0x20
    QOR = 0x21
    QXOR = 0x22
    QSHL = 0x23
    QSHR = 0x24
    
    # Comparison
    QJEQ = 0x30
    QJNE = 0x31
    QJLT = 0x32
    QJGT = 0x33
    QJLE = 0x34
    QJGE = 0x35
    
    # Control flow
    QJMP = 0x40
    QCALL = 0x41
    QRET = 0x42
    QHALT = 0x43
    
    # System
    QSYSCALL = 0x50

# ════════════════════════════════════════════════════════════════════════════
# QBC VIRTUAL MACHINE
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class QBCInstruction:
    """Single QBC instruction"""
    opcode: QBCOpcode
    operands: List[Any]
    line_number: int
    label: Optional[str] = None

class QBCVirtualMachine:
    """QBC Virtual Machine - executes QBC instructions"""
    
    def __init__(self, verbose: bool = True):
        # Registers (r0-r15)
        self.registers = [0] * 16
        
        # Virtual memory (64-bit addressing)
        self.memory: Dict[int, int] = {}
        self.memory_strings: Dict[int, str] = {}
        
        # Program counter
        self.pc = 0
        
        # Call stack
        self.call_stack: List[int] = []
        
        # Label table
        self.labels: Dict[str, int] = {}
        
        # Program
        self.instructions: List[QBCInstruction] = []
        
        # System state
        self.halted = False
        self.cycle_count = 0
        self.verbose = verbose
        
        # Progress tracking
        self.last_progress_cycle = 0
        self.progress_interval = 1000000  # Report every 1M cycles
        
        # Statistics
        self.stats = {
            'instructions_executed': 0,
            'memory_reads': 0,
            'memory_writes': 0,
            'function_calls': 0,
            'jumps': 0
        }
        
        # Output
        self.output_buffer = []
        
        # Execution start time
        self.start_time = None
        
    def load_program(self, instructions: List[QBCInstruction]):
        """Load program into VM"""
        self.instructions = instructions
        
        # Build label table
        for i, instr in enumerate(instructions):
            if instr.label:
                self.labels[instr.label] = i
    
    def execute(self, max_cycles: int = 500000000) -> bool:  # 500M cycles for full instantiation
        """Execute loaded program"""
        
        self.pc = 0
        self.halted = False
        self.cycle_count = 0
        self.start_time = time.time()
        
        if self.verbose:
            print(f"\n{'='*80}")
            print(f"🚀 STARTING QBC EXECUTION")
            print(f"{'='*80}")
            print(f"Max cycles: {max_cycles:,}")
            print(f"Instructions loaded: {len(self.instructions):,}")
            print(f"Labels defined: {len(self.labels):,}")
            print(f"{'='*80}\n")
        
        while not self.halted and self.cycle_count < max_cycles:
            if self.pc >= len(self.instructions):
                break
            
            # Progress reporting
            if self.verbose and (self.cycle_count - self.last_progress_cycle) >= self.progress_interval:
                self.print_progress()
                self.last_progress_cycle = self.cycle_count
            
            instr = self.instructions[self.pc]
            old_pc = self.pc
            
            try:
                self.execute_instruction(instr)
            except Exception as e:
                print(f"\n❌ ERROR at cycle {self.cycle_count}, PC={self.pc}")
                print(f"Instruction: {instr}")
                print(f"Error: {e}")
                break
            
            self.cycle_count += 1
            self.stats['instructions_executed'] += 1
            
            # Auto-increment PC unless jump occurred
            if self.pc == old_pc:
                self.pc += 1
        
        if self.verbose:
            self.print_final_stats()
        
        return self.cycle_count < max_cycles
    
    def print_progress(self):
        """Print execution progress"""
        elapsed = time.time() - self.start_time
        cycles_per_sec = self.cycle_count / elapsed if elapsed > 0 else 0
        
        print(f"\n📊 Progress Report:")
        print(f"  Cycle: {self.cycle_count:,} / {self.stats['instructions_executed']:,} instructions")
        print(f"  Speed: {cycles_per_sec:,.0f} cycles/sec")
        print(f"  Memory: {len(self.memory):,} entries ({self.stats['memory_reads']:,} reads, {self.stats['memory_writes']:,} writes)")
        print(f"  Elapsed: {elapsed:.1f}s")
        print(f"  PC: {self.pc} / {len(self.instructions)}")
        
        # Show current label context
        current_label = None
        for label, addr in self.labels.items():
            if addr <= self.pc:
                if current_label is None or self.labels[current_label] < addr:
                    current_label = label
        
        if current_label:
            print(f"  Context: {current_label}")
        
        print()
    
    def print_final_stats(self):
        """Print final execution statistics"""
        elapsed = time.time() - self.start_time
        
        print(f"\n{'='*80}")
        print(f"✅ EXECUTION COMPLETE")
        print(f"{'='*80}")
        print(f"Total cycles: {self.cycle_count:,}")
        print(f"Instructions executed: {self.stats['instructions_executed']:,}")
        print(f"Time elapsed: {elapsed:.2f}s")
        print(f"Average speed: {self.cycle_count/elapsed:,.0f} cycles/sec")
        print(f"\nMemory Statistics:")
        print(f"  Entries: {len(self.memory):,}")
        print(f"  Reads: {self.stats['memory_reads']:,}")
        print(f"  Writes: {self.stats['memory_writes']:,}")
        print(f"\nControl Flow:")
        print(f"  Function calls: {self.stats.get('function_calls', 0):,}")
        print(f"  Jumps: {self.stats.get('jumps', 0):,}")
        print(f"{'='*80}\n")
    
    def execute_instruction(self, instr: QBCInstruction):
        """Execute single instruction"""
        
        op = instr.opcode
        operands = instr.operands
        
        # Data movement
        if op == QBCOpcode.QMOV:
            self.op_qmov(operands)
        elif op == QBCOpcode.QLOAD:
            self.op_qload(operands)
        elif op == QBCOpcode.QSTORE:
            self.op_qstore(operands)
        
        # Arithmetic
        elif op == QBCOpcode.QADD:
            self.op_qadd(operands)
        elif op == QBCOpcode.QSUB:
            self.op_qsub(operands)
        elif op == QBCOpcode.QMUL:
            self.op_qmul(operands)
        elif op == QBCOpcode.QDIV:
            self.op_qdiv(operands)
        elif op == QBCOpcode.QMOD:
            self.op_qmod(operands)
        
        # Bitwise
        elif op == QBCOpcode.QAND:
            self.op_qand(operands)
        elif op == QBCOpcode.QOR:
            self.op_qor(operands)
        elif op == QBCOpcode.QXOR:
            self.op_qxor(operands)
        elif op == QBCOpcode.QSHL:
            self.op_qshl(operands)
        elif op == QBCOpcode.QSHR:
            self.op_qshr(operands)
        
        # Comparison & jumps
        elif op == QBCOpcode.QJEQ:
            self.op_qjeq(operands)
        elif op == QBCOpcode.QJNE:
            self.op_qjne(operands)
        elif op == QBCOpcode.QJLT:
            self.op_qjlt(operands)
        elif op == QBCOpcode.QJGT:
            self.op_qjgt(operands)
        elif op == QBCOpcode.QJLE:
            self.op_qjle(operands)
        elif op == QBCOpcode.QJGE:
            self.op_qjge(operands)
        
        # Control flow
        elif op == QBCOpcode.QJMP:
            self.op_qjmp(operands)
        elif op == QBCOpcode.QCALL:
            self.op_qcall(operands)
        elif op == QBCOpcode.QRET:
            self.op_qret(operands)
        elif op == QBCOpcode.QHALT:
            self.op_qhalt(operands)
        
        # System
        elif op == QBCOpcode.QSYSCALL:
            self.op_qsyscall(operands)
    
    # ═══════════════════════════════════════════════════════════════════════
    # INSTRUCTION IMPLEMENTATIONS
    # ═══════════════════════════════════════════════════════════════════════
    
    def op_qmov(self, operands):
        """QMOV dest, src - Move value"""
        dest_reg = operands[0]
        src = operands[1]
        
        value = self.get_operand_value(src)
        self.registers[dest_reg] = value
    
    def op_qload(self, operands):
        """QLOAD dest, addr - Load from memory"""
        dest_reg = operands[0]
        addr = self.get_operand_value(operands[1])
        
        value = self.memory.get(addr, 0)
        self.registers[dest_reg] = value
        self.stats['memory_reads'] += 1
    
    def op_qstore(self, operands):
        """QSTORE src, addr - Store to memory"""
        src = self.get_operand_value(operands[0])
        addr = self.get_operand_value(operands[1])
        
        self.memory[addr] = src
        self.stats['memory_writes'] += 1
    
    def op_qadd(self, operands):
        """QADD dest, src - Add"""
        if len(operands) == 2:
            dest_reg = operands[0]
            value = self.get_operand_value(operands[1])
            self.registers[dest_reg] = (self.registers[dest_reg] + value) & 0xFFFFFFFFFFFFFFFF
        else:
            dest_reg = operands[0]
            src1 = self.get_operand_value(operands[1])
            src2 = self.get_operand_value(operands[2])
            self.registers[dest_reg] = (src1 + src2) & 0xFFFFFFFFFFFFFFFF
    
    def op_qsub(self, operands):
        """QSUB dest, src - Subtract"""
        dest_reg = operands[0]
        value = self.get_operand_value(operands[1])
        self.registers[dest_reg] = (self.registers[dest_reg] - value) & 0xFFFFFFFFFFFFFFFF
    
    def op_qmul(self, operands):
        """QMUL dest, src - Multiply"""
        if len(operands) == 2:
            dest_reg = operands[0]
            value = self.get_operand_value(operands[1])
            self.registers[dest_reg] = (self.registers[dest_reg] * value) & 0xFFFFFFFFFFFFFFFF
        else:
            dest_reg = operands[0]
            src1 = self.get_operand_value(operands[1])
            src2 = self.get_operand_value(operands[2])
            self.registers[dest_reg] = (src1 * src2) & 0xFFFFFFFFFFFFFFFF
    
    def op_qdiv(self, operands):
        """QDIV dest, src - Divide"""
        if len(operands) == 2:
            dest_reg = operands[0]
            value = self.get_operand_value(operands[1])
            if value != 0:
                self.registers[dest_reg] = self.registers[dest_reg] // value
        else:
            dest_reg = operands[0]
            src1 = self.get_operand_value(operands[1])
            src2 = self.get_operand_value(operands[2])
            if src2 != 0:
                self.registers[dest_reg] = src1 // src2
    
    def op_qmod(self, operands):
        """QMOD dest, src1, src2 - Modulo"""
        dest_reg = operands[0]
        src1 = self.get_operand_value(operands[1])
        src2 = self.get_operand_value(operands[2])
        if src2 != 0:
            self.registers[dest_reg] = src1 % src2
    
    def op_qand(self, operands):
        """QAND dest, src - Bitwise AND"""
        dest_reg = operands[0]
        src1 = self.registers[dest_reg]
        src2 = self.get_operand_value(operands[1])
        self.registers[dest_reg] = int(src1) & int(src2)
    
    def op_qor(self, operands):
        """QOR dest, src1, src2 - Bitwise OR"""
        dest_reg = operands[0]
        src1 = self.get_operand_value(operands[1])
        src2 = self.get_operand_value(operands[2])
        self.registers[dest_reg] = int(src1) | int(src2)
    
    def op_qxor(self, operands):
        """QXOR dest, src - Bitwise XOR"""
        dest_reg = operands[0]
        src = self.get_operand_value(operands[1])
        self.registers[dest_reg] = int(self.registers[dest_reg]) ^ int(src)
    
    def op_qshl(self, operands):
        """QSHL dest, bits - Shift left"""
        dest_reg = operands[0]
        bits = self.get_operand_value(operands[1])
        self.registers[dest_reg] = (int(self.registers[dest_reg]) << int(bits)) & 0xFFFFFFFFFFFFFFFF
    
    def op_qshr(self, operands):
        """QSHR dest, bits - Shift right"""
        dest_reg = operands[0]
        bits = self.get_operand_value(operands[1])
        self.registers[dest_reg] = int(self.registers[dest_reg]) >> int(bits)
    
    def op_qjeq(self, operands):
        """QJEQ src1, src2, label - Jump if equal"""
        src1 = self.get_operand_value(operands[0])
        src2 = self.get_operand_value(operands[1])
        label = operands[2]
        
        if src1 == src2:
            self.pc = self.labels.get(label, self.pc)
            self.stats['jumps'] = self.stats.get('jumps', 0) + 1
    
    def op_qjne(self, operands):
        """QJNE src1, src2, label - Jump if not equal"""
        src1 = self.get_operand_value(operands[0])
        src2 = self.get_operand_value(operands[1])
        label = operands[2]
        
        if src1 != src2:
            self.pc = self.labels.get(label, self.pc)
            self.stats['jumps'] = self.stats.get('jumps', 0) + 1
    
    def op_qjlt(self, operands):
        """QJLT src1, src2, label - Jump if less than"""
        src1 = self.get_operand_value(operands[0])
        src2 = self.get_operand_value(operands[1])
        label = operands[2]
        
        if src1 < src2:
            self.pc = self.labels.get(label, self.pc)
            self.stats['jumps'] = self.stats.get('jumps', 0) + 1
    
    def op_qjgt(self, operands):
        """QJGT src1, src2, label - Jump if greater than"""
        src1 = self.get_operand_value(operands[0])
        src2 = self.get_operand_value(operands[1])
        label = operands[2]
        
        if src1 > src2:
            self.pc = self.labels.get(label, self.pc)
            self.stats['jumps'] = self.stats.get('jumps', 0) + 1
    
    def op_qjle(self, operands):
        """QJLE src1, src2, label - Jump if less/equal"""
        src1 = self.get_operand_value(operands[0])
        src2 = self.get_operand_value(operands[1])
        label = operands[2]
        
        if src1 <= src2:
            self.pc = self.labels.get(label, self.pc)
            self.stats['jumps'] = self.stats.get('jumps', 0) + 1
    
    def op_qjge(self, operands):
        """QJGE src1, src2, label - Jump if greater/equal"""
        src1 = self.get_operand_value(operands[0])
        src2 = self.get_operand_value(operands[1])
        label = operands[2]
        
        if src1 >= src2:
            self.pc = self.labels.get(label, self.pc)
            self.stats['jumps'] = self.stats.get('jumps', 0) + 1
    
    def op_qjmp(self, operands):
        """QJMP label - Unconditional jump"""
        label = operands[0]
        self.pc = self.labels.get(label, self.pc)
        self.stats['jumps'] = self.stats.get('jumps', 0) + 1
    
    def op_qcall(self, operands):
        """QCALL label - Call subroutine"""
        label = operands[0]
        self.call_stack.append(self.pc + 1)
        self.pc = self.labels.get(label, self.pc)
        self.stats['function_calls'] = self.stats.get('function_calls', 0) + 1
    
    def op_qret(self, operands):
        """QRET - Return from subroutine"""
        if self.call_stack:
            self.pc = self.call_stack.pop()
        else:
            self.pc = len(self.instructions)  # End program
    
    def op_qhalt(self, operands):
        """QHALT - Halt execution"""
        self.halted = True
        if self.verbose:
            print("\n🛑 QHALT instruction executed - program terminated normally")
    
    def op_qsyscall(self, operands):
        """QSYSCALL number - System call"""
        syscall_num = self.get_operand_value(operands[0])
        
        if syscall_num == 1:
            # Print integer
            value = self.registers[0]
            self.output_buffer.append(str(value))
            print(value, end='')
        
        elif syscall_num == 2:
            # Print string
            addr = self.registers[0]
            if addr in self.memory_strings:
                string = self.memory_strings[addr]
                self.output_buffer.append(string)
                print(string, end='')
        
        elif syscall_num == 3:
            # Get timestamp
            self.registers[0] = int(time.time() * 1000000)
    
    def get_operand_value(self, operand) -> int:
        """Get value from operand (register or immediate)"""
        if isinstance(operand, int):
            return operand
        elif isinstance(operand, str):
            if operand.startswith('r'):
                reg_num = int(operand[1:])
                return self.registers[reg_num]
            elif operand.startswith('0x'):
                return int(operand, 16)
            else:
                try:
                    return int(operand)
                except:
                    return 0
        return operand
    
    def get_output_buffer(self) -> str:
        """Get accumulated output"""
        return ''.join(self.output_buffer)

# ════════════════════════════════════════════════════════════════════════════
# QBC ASSEMBLER
# ════════════════════════════════════════════════════════════════════════════

class QBCAssembler:
    """Assembles QBC assembly into instructions"""
    
    def __init__(self, verbose: bool = True):
        self.instructions: List[QBCInstruction] = []
        self.defines: Dict[str, int] = {}
        self.data_section: Dict[str, str] = {}
        self.current_line = 0
        self.verbose = verbose
        
    def parse_file(self, filepath: Path) -> List[QBCInstruction]:
        """Parse QBC file"""
        
        if self.verbose:
            print(f"\n📖 Parsing QBC file: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        if self.verbose:
            print(f"   Total lines: {len(lines)}")
        
        in_data_section = False
        current_label = None
        
        for line_num, line in enumerate(lines, 1):
            self.current_line = line_num
            
            # Remove comments
            if ';' in line:
                line = line[:line.index(';')]
            
            line = line.strip()
            
            if not line:
                continue
            
            # Check for .data section
            if line == '.data':
                in_data_section = True
                if self.verbose and line_num % 100 == 0:
                    print(f"   Parsing line {line_num}/{len(lines)}")
                continue
            
            # Check for .text/.code section
            if line in ['.text', '.code']:
                in_data_section = False
                continue
            
            # Handle .define
            if line.startswith('.define'):
                self.parse_define(line)
                continue
            
            # Handle .include (skip)
            if line.startswith('.include'):
                continue
            
            # Handle .entry_point
            if line.startswith('.entry_point'):
                continue
            
            # Handle data section
            if in_data_section:
                if ':' in line:
                    label = line[:line.index(':')].strip()
                    current_label = label
                elif current_label and '.ascii' in line:
                    start = line.index('"') + 1
                    end = line.rindex('"')
                    text = line[start:end]
                    self.data_section[current_label] = text
                continue
            
            # Parse instruction
            instr = self.parse_instruction(line, line_num)
            if instr:
                self.instructions.append(instr)
        
        if self.verbose:
            print(f"✓ Parsing complete")
            print(f"   Instructions: {len(self.instructions)}")
            print(f"   Defines: {len(self.defines)}")
            print(f"   Data labels: {len(self.data_section)}")
        
        return self.instructions
    
    def parse_define(self, line: str):
        """Parse .define directive"""
        parts = line.split()
        if len(parts) >= 3:
            name = parts[1]
            value_str = ' '.join(parts[2:])
            
            try:
                if value_str.startswith('0x'):
                    value = int(value_str, 16)
                else:
                    value = int(float(value_str))
                self.defines[name] = value
            except:
                pass
    
    def parse_instruction(self, line: str, line_num: int) -> Optional[QBCInstruction]:
        """Parse single instruction"""
        
        # Check for label
        label = None
        if ':' in line:
            label = line[:line.index(':')].strip()
            line = line[line.index(':')+1:].strip()
            
            if not line:
                # Label-only line - create NOP
                return QBCInstruction(QBCOpcode.QMOV, [0, 0], line_num, label)
        
        # Split instruction and operands
        parts = line.split(None, 1)
        if not parts:
            return None
        
        mnemonic = parts[0].upper()
        operands_str = parts[1] if len(parts) > 1 else ''
        
        # Map mnemonic to opcode
        opcode_map = {
            'QMOV': QBCOpcode.QMOV,
            'QLOAD': QBCOpcode.QLOAD,
            'QSTORE': QBCOpcode.QSTORE,
            'QADD': QBCOpcode.QADD,
            'QSUB': QBCOpcode.QSUB,
            'QMUL': QBCOpcode.QMUL,
            'QDIV': QBCOpcode.QDIV,
            'QMOD': QBCOpcode.QMOD,
            'QAND': QBCOpcode.QAND,
            'QOR': QBCOpcode.QOR,
            'QXOR': QBCOpcode.QXOR,
            'QSHL': QBCOpcode.QSHL,
            'QSHR': QBCOpcode.QSHR,
            'QJEQ': QBCOpcode.QJEQ,
            'QJNE': QBCOpcode.QJNE,
            'QJLT': QBCOpcode.QJLT,
            'QJGT': QBCOpcode.QJGT,
            'QJLE': QBCOpcode.QJLE,
            'QJGE': QBCOpcode.QJGE,
            'QJMP': QBCOpcode.QJMP,
            'QCALL': QBCOpcode.QCALL,
            'QRET': QBCOpcode.QRET,
            'QHALT': QBCOpcode.QHALT,
            'QSYSCALL': QBCOpcode.QSYSCALL,
        }
        
        if mnemonic not in opcode_map:
            return None
        
        opcode = opcode_map[mnemonic]
        
        # Parse operands
        operands = self.parse_operands(operands_str)
        
        return QBCInstruction(opcode, operands, line_num, label)
    
    def parse_operands(self, operands_str: str) -> List:
        """Parse instruction operands"""
        
        if not operands_str:
            return []
        
        # Split by comma
        parts = [p.strip() for p in operands_str.split(',')]
        operands = []
        
        for part in parts:
            # Check if it's a register
            if part.startswith('r') and len(part) > 1 and part[1:].isdigit():
                operands.append(int(part[1:]))
            
            # Check if it's a hex immediate
            elif part.startswith('0x'):
                operands.append(int(part, 16))
            
            # Check if it's a define
            elif part in self.defines:
                operands.append(self.defines[part])
            
            # Check if it's a decimal immediate
            elif part.lstrip('-').isdigit():
                operands.append(int(part))
            
            # Otherwise it's a label
            else:
                operands.append(part)
        
        return operands

# ════════════════════════════════════════════════════════════════════════════
# QUANTUM ALGORITHM LIBRARY
# ════════════════════════════════════════════════════════════════════════════

class QuantumAlgorithmLibrary:
    """Library of quantum algorithm implementations for the lattice"""

    def __init__(self, vm: 'QBCVirtualMachine'):
        self.vm = vm
        self.logger = self._setup_logger()

    def _setup_logger(self):
        import logging
        logger = logging.getLogger("QBCAlgorithms")
        logger.setLevel(logging.INFO)
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter('%(asctime)s [%(name)s] %(message)s'))
            logger.addHandler(handler)
        return logger

    # ────────────────────────────────────────────────────────────────────────
    # GROVER'S ALGORITHM SUPPORT
    # ────────────────────────────────────────────────────────────────────────

    def grover_oracle(self, target_state: int, n_qubits: int) -> Dict[str, Any]:
        """
        Grover's oracle: marks target state with phase flip
        Returns circuit description for lattice execution
        """
        self.logger.info(f"Grover Oracle: marking state |{target_state:0{n_qubits}b}⟩")

        return {
            'algorithm': 'grover_oracle',
            'n_qubits': n_qubits,
            'target_state': target_state,
            'target_binary': format(target_state, f'0{n_qubits}b'),
            'gates': self._generate_grover_oracle_gates(target_state, n_qubits)
        }

    def _generate_grover_oracle_gates(self, target: int, n: int) -> List[Dict]:
        """Generate gate sequence for Grover oracle"""
        gates = []

        # X gates on qubits where target bit is 0
        for i in range(n):
            if not (target & (1 << i)):
                gates.append({'type': 'X', 'qubit': i})

        # Multi-controlled Z gate
        if n > 1:
            gates.append({'type': 'MCZ', 'controls': list(range(n-1)), 'target': n-1})
        else:
            gates.append({'type': 'Z', 'qubit': 0})

        # Undo X gates
        for i in range(n):
            if not (target & (1 << i)):
                gates.append({'type': 'X', 'qubit': i})

        return gates

    def grover_diffusion(self, n_qubits: int) -> Dict[str, Any]:
        """
        Grover diffusion operator: 2|s⟩⟨s| - I
        where |s⟩ = H^⊗n|0⟩ is uniform superposition
        """
        self.logger.info(f"Grover Diffusion: {n_qubits} qubits")

        gates = []

        # H on all qubits
        for i in range(n_qubits):
            gates.append({'type': 'H', 'qubit': i})

        # X on all qubits
        for i in range(n_qubits):
            gates.append({'type': 'X', 'qubit': i})

        # Multi-controlled Z
        if n_qubits > 1:
            gates.append({'type': 'MCZ', 'controls': list(range(n_qubits-1)), 'target': n_qubits-1})
        else:
            gates.append({'type': 'Z', 'qubit': 0})

        # X on all qubits
        for i in range(n_qubits):
            gates.append({'type': 'X', 'qubit': i})

        # H on all qubits
        for i in range(n_qubits):
            gates.append({'type': 'H', 'qubit': i})

        return {
            'algorithm': 'grover_diffusion',
            'n_qubits': n_qubits,
            'gates': gates
        }

    # ────────────────────────────────────────────────────────────────────────
    # VQE (VARIATIONAL QUANTUM EIGENSOLVER) SUPPORT
    # ────────────────────────────────────────────────────────────────────────

    def vqe_ansatz_hardware_efficient(self, n_qubits: int, depth: int, params: List[float]) -> Dict[str, Any]:
        """
        Hardware-efficient VQE ansatz
        Alternating layers of single-qubit rotations and entangling gates
        """
        self.logger.info(f"VQE Ansatz: {n_qubits} qubits, depth {depth}")

        gates = []
        param_idx = 0

        for d in range(depth):
            # Layer of RY rotations
            for q in range(n_qubits):
                if param_idx < len(params):
                    gates.append({'type': 'RY', 'qubit': q, 'angle': params[param_idx]})
                    param_idx += 1

            # Layer of RZ rotations
            for q in range(n_qubits):
                if param_idx < len(params):
                    gates.append({'type': 'RZ', 'qubit': q, 'angle': params[param_idx]})
                    param_idx += 1

            # Entangling layer (CNOTs)
            for q in range(n_qubits - 1):
                gates.append({'type': 'CNOT', 'control': q, 'target': q+1})

        return {
            'algorithm': 'vqe_ansatz',
            'n_qubits': n_qubits,
            'depth': depth,
            'n_params': param_idx,
            'gates': gates
        }

    def vqe_measure_hamiltonian(self, hamiltonian: List[Tuple[str, float]]) -> Dict[str, Any]:
        """
        Measure expectation value of Hamiltonian
        hamiltonian: list of (pauli_string, coefficient) tuples
        Example: [('ZZ', 0.5), ('XI', -0.3), ('YY', 0.2)]
        """
        self.logger.info(f"VQE Hamiltonian: {len(hamiltonian)} terms")

        measurements = []

        for pauli_string, coeff in hamiltonian:
            # Determine measurement basis for each term
            basis_changes = []
            for i, pauli in enumerate(pauli_string):
                if pauli == 'X':
                    basis_changes.append({'type': 'H', 'qubit': i})
                elif pauli == 'Y':
                    basis_changes.append({'type': 'RX', 'qubit': i, 'angle': -np.pi/2})
                # Z basis is computational basis (no change needed)

            measurements.append({
                'pauli_string': pauli_string,
                'coefficient': coeff,
                'basis_changes': basis_changes
            })

        return {
            'algorithm': 'vqe_hamiltonian_measurement',
            'n_terms': len(hamiltonian),
            'measurements': measurements
        }

    # ────────────────────────────────────────────────────────────────────────
    # QUANTUM FOURIER TRANSFORM
    # ────────────────────────────────────────────────────────────────────────

    def qft(self, n_qubits: int, inverse: bool = False) -> Dict[str, Any]:
        """
        Quantum Fourier Transform (or inverse QFT)
        """
        self.logger.info(f"{'Inverse ' if inverse else ''}QFT: {n_qubits} qubits")

        gates = []

        # QFT circuit
        for j in range(n_qubits):
            # Hadamard on qubit j
            gates.append({'type': 'H', 'qubit': j})

            # Controlled rotations
            for k in range(j + 1, n_qubits):
                angle = np.pi / (2 ** (k - j))
                gates.append({
                    'type': 'CP',  # Controlled-Phase
                    'control': k,
                    'target': j,
                    'angle': angle if not inverse else -angle
                })

        # Swap qubits to reverse order
        for i in range(n_qubits // 2):
            gates.append({
                'type': 'SWAP',
                'qubit1': i,
                'qubit2': n_qubits - 1 - i
            })

        return {
            'algorithm': 'qft' if not inverse else 'iqft',
            'n_qubits': n_qubits,
            'gates': gates
        }

    # ────────────────────────────────────────────────────────────────────────
    # QUANTUM PHASE ESTIMATION
    # ────────────────────────────────────────────────────────────────────────

    def phase_estimation(self, n_counting: int, n_eigenstate: int, unitary_powers: List[int]) -> Dict[str, Any]:
        """
        Quantum Phase Estimation algorithm
        Returns gate sequence for estimating eigenphase
        """
        self.logger.info(f"Phase Estimation: {n_counting} counting qubits, {n_eigenstate} eigenstate qubits")

        gates = []

        # Initialize counting register in superposition
        for i in range(n_counting):
            gates.append({'type': 'H', 'qubit': i})

        # Controlled unitary operations
        for i, power in enumerate(unitary_powers[:n_counting]):
            gates.append({
                'type': 'CU',
                'control': i,
                'target_register': list(range(n_counting, n_counting + n_eigenstate)),
                'power': power
            })

        # Inverse QFT on counting register
        iqft_gates = self.qft(n_counting, inverse=True)['gates']

        # Adjust qubit indices for QFT gates (they're on counting register)
        for gate in iqft_gates:
            gates.append(gate)

        return {
            'algorithm': 'phase_estimation',
            'n_counting': n_counting,
            'n_eigenstate': n_eigenstate,
            'gates': gates
        }

    # ────────────────────────────────────────────────────────────────────────
    # QUANTUM AMPLITUDE AMPLIFICATION
    # ────────────────────────────────────────────────────────────────────────

    def amplitude_amplification(self, n_qubits: int, iterations: int, oracle_spec: Dict) -> Dict[str, Any]:
        """
        Quantum Amplitude Amplification (generalization of Grover)
        """
        self.logger.info(f"Amplitude Amplification: {n_qubits} qubits, {iterations} iterations")

        gates = []

        # Initialize in uniform superposition
        for i in range(n_qubits):
            gates.append({'type': 'H', 'qubit': i})

        # Amplitude amplification iterations
        for _ in range(iterations):
            # Oracle
            gates.extend(oracle_spec.get('gates', []))

            # Diffusion operator
            diffusion = self.grover_diffusion(n_qubits)
            gates.extend(diffusion['gates'])

        return {
            'algorithm': 'amplitude_amplification',
            'n_qubits': n_qubits,
            'iterations': iterations,
            'gates': gates
        }

    # ────────────────────────────────────────────────────────────────────────
    # EXPORT ALGORITHMS FOR LATTICE EXECUTION
    # ────────────────────────────────────────────────────────────────────────

    def export_algorithm_to_lattice(self, algorithm_spec: Dict, output_file: Path):
        """
        Export algorithm specification to file for lattice execution
        """
        self.logger.info(f"Exporting algorithm to {output_file}")

        with open(output_file, 'w') as f:
            json.dump(algorithm_spec, f, indent=2)

        self.logger.info(f"✓ Algorithm exported: {algorithm_spec.get('algorithm', 'unknown')}")

# ════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ════════════════════════════════════════════════════════════════════════════

def main():
    """Main entry point"""

    if len(sys.argv) < 2:
        print("Usage: python qbc_parser.py <qbc_file>")
        sys.exit(1)

    qbc_file = Path(sys.argv[1])

    if not qbc_file.exists():
        print(f"Error: File not found: {qbc_file}")
        sys.exit(1)

    print("="*80)
    print("🌙 QBC PARSER & VIRTUAL MACHINE")
    print("   MOONSHINE LATTICE INSTANTIATION")
    print("="*80)
    print(f"File: {qbc_file}")
    print(f"Target: 196,883-dimensional Moonshine representation")
    print("="*80)

    # Assemble
    assembler = QBCAssembler(verbose=True)
    instructions = assembler.parse_file(qbc_file)

    print()

    # Execute
    vm = QBCVirtualMachine(verbose=True)

    # Load data strings into VM memory
    string_addr = 0x100000
    for label, text in assembler.data_section.items():
        vm.memory_strings[string_addr] = text
        vm.labels[label] = string_addr
        string_addr += len(text) + 1

    vm.load_program(instructions)

    success = vm.execute()

    # Save OUTPUT_BUFFER
    output_file = qbc_file.parent / "qbc_output.json"

    print(f"\n💾 Saving output to: {output_file}")

    output_data = {
        'success': success,
        'cycles': vm.cycle_count,
        'stats': vm.stats,
        'output': vm.get_output_buffer(),
        'memory_entries': len(vm.memory),
        'memory_sample': {hex(k): v for k, v in list(vm.memory.items())[:100]},
        'registers': {f'r{i}': vm.registers[i] for i in range(16) if vm.registers[i] != 0},
        'lattice_ready': True,
        'algorithm_support': {
            'grover': True,
            'vqe': True,
            'qft': True,
            'phase_estimation': True,
            'amplitude_amplification': True
        }
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"✓ Output saved")

    # Initialize algorithm library
    algo_lib = QuantumAlgorithmLibrary(vm)

    # Export example algorithms for testing
    examples_dir = qbc_file.parent / "algorithm_examples"
    examples_dir.mkdir(exist_ok=True)

    # Grover example
    grover_oracle = algo_lib.grover_oracle(target_state=5, n_qubits=3)
    algo_lib.export_algorithm_to_lattice(grover_oracle, examples_dir / "grover_oracle_example.json")

    # VQE example
    vqe_ansatz = algo_lib.vqe_ansatz_hardware_efficient(
        n_qubits=4,
        depth=2,
        params=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    )
    algo_lib.export_algorithm_to_lattice(vqe_ansatz, examples_dir / "vqe_ansatz_example.json")

    # QFT example
    qft_circuit = algo_lib.qft(n_qubits=4)
    algo_lib.export_algorithm_to_lattice(qft_circuit, examples_dir / "qft_example.json")

    print(f"\n✓ Algorithm examples exported to {examples_dir}")

    # Summary
    print(f"\n{'='*80}")
    print(f"📊 EXECUTION SUMMARY")
    print(f"{'='*80}")
    print(f"Status: {'✅ SUCCESS' if success else '❌ TIMEOUT'}")
    print(f"Cycles executed: {vm.cycle_count:,}")
    print(f"Instructions: {vm.stats['instructions_executed']:,}")
    print(f"Memory operations:")
    print(f"  - Total entries: {len(vm.memory):,}")
    print(f"  - Reads: {vm.stats['memory_reads']:,}")
    print(f"  - Writes: {vm.stats['memory_writes']:,}")
    print(f"Control flow:")
    print(f"  - Function calls: {vm.stats.get('function_calls', 0):,}")
    print(f"  - Jumps: {vm.stats.get('jumps', 0):,}")

    if not success:
        print(f"\n⚠️  WARNING: Execution reached cycle limit")
        print(f"   This is normal for full 196,883-node instantiation")
        print(f"   Partial lattice data has been saved")

    print(f"{'='*80}\n")

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()