
#!/usr/bin/env python3
"""
MOONSHINE FLASK SERVER
Complete web server with HTML terminal, QBC parser, and experiment runner
Routes all logs to web interface with real-time streaming
"""

from flask import Flask, render_template, jsonify, request, Response
import json
import time
import threading
import logging
import sys
import io
from pathlib import Path
from datetime import datetime
from queue import Queue, Empty
import traceback

# Import local modules
from qbc_parser import QBCParser
from experiment_runner import ExperimentRunner
from moonshine_core import create_core as create_quantum_core

# ═════════════════════════════════════════════════════════════════════════
# LOGGING SETUP - CAPTURE ALL OUTPUT
# ═════════════════════════════════════════════════════════════════════════

class LogCapture:
    """Captures all log output and routes to web terminal"""
    
    def __init__(self):
        self.log_queue = Queue()
        self.subscribers = []
        
    def write(self, message):
        """Capture stdout/stderr"""
        if message.strip():
            self.emit({
                'timestamp': datetime.now().isoformat(),
                'level': 'INFO',
                'message': message.strip(),
                'source': 'system'
            })
    
    def flush(self):
        pass
    
    def emit(self, log_entry):
        """Emit log entry to queue and subscribers"""
        self.log_queue.put(log_entry)
        
        # Send to all SSE subscribers
        dead_subscribers = []
        for sub in self.subscribers:
            try:
                sub.put(log_entry)
            except:
                dead_subscribers.append(sub)
        
        # Clean up dead subscribers
        for sub in dead_subscribers:
            self.subscribers.remove(sub)
    
    def subscribe(self):
        """Subscribe to log stream"""
        queue = Queue()
        self.subscribers.append(queue)
        return queue

# Create global log capture
log_capture = LogCapture()

# Redirect stdout/stderr
sys.stdout = log_capture
sys.stderr = log_capture

# Setup logging to use our capture
class QueueHandler(logging.Handler):
    def emit(self, record):
        log_capture.emit({
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'message': self.format(record),
            'source': record.name
        })

logging.basicConfig(
    level=logging.INFO,
    handlers=[QueueHandler()],
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)

logger = logging.getLogger("MoonshineServer")

# ═════════════════════════════════════════════════════════════════════════
# FLASK APP
# ═════════════════════════════════════════════════════════════════════════

app = Flask(__name__)

# Global state
server_state = {
    'qbc_parsed': False,
    'lattice_loaded': False,
    'quantum_core_running': False,
    'experiments': {},
    'qbc_parser': None,
    'quantum_core': None,
    'experiment_runner': None
}

# ═════════════════════════════════════════════════════════════════════════
# HTML TERMINAL
# ═════════════════════════════════════════════════════════════════════════

HTML_TERMINAL = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Moonshine Quantum Terminal</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Courier New', monospace;
            background: #0a0a0a;
            color: #00ff00;
            overflow: hidden;
        }
        
        .container {
            display: flex;
            flex-direction: column;
            height: 100vh;
        }
        
        .header {
            background: linear-gradient(135deg, #1a1a2e 0%, #0f0f1e 100%);
            padding: 20px;
            border-bottom: 2px solid #00ff00;
            box-shadow: 0 4px 20px rgba(0, 255, 0, 0.3);
        }
        
        .header h1 {
            font-size: 24px;
            text-shadow: 0 0 10px #00ff00;
            letter-spacing: 2px;
        }
        
        .status-bar {
            display: flex;
            gap: 20px;
            margin-top: 10px;
            font-size: 12px;
        }
        
        .status-item {
            padding: 5px 10px;
            background: rgba(0, 255, 0, 0.1);
            border-radius: 3px;
            border: 1px solid rgba(0, 255, 0, 0.3);
        }
        
        .status-active {
            background: rgba(0, 255, 0, 0.3);
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.5; }
        }
        
        .controls {
            background: #1a1a1a;
            padding: 15px 20px;
            border-bottom: 1px solid #333;
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }
        
        button {
            background: #00ff00;
            color: #000;
            border: none;
            padding: 8px 16px;
            font-family: 'Courier New', monospace;
            font-weight: bold;
            cursor: pointer;
            border-radius: 3px;
            transition: all 0.3s;
        }
        
        button:hover {
            background: #00cc00;
            box-shadow: 0 0 10px #00ff00;
        }
        
        button:disabled {
            background: #333;
            color: #666;
            cursor: not-allowed;
            box-shadow: none;
        }
        
        .terminal {
            flex: 1;
            overflow-y: auto;
            padding: 20px;
            background: #000;
            font-size: 13px;
            line-height: 1.6;
        }
        
        .log-entry {
            margin: 2px 0;
            padding: 4px 8px;
            border-left: 3px solid transparent;
            animation: fadeIn 0.3s;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateX(-10px); }
            to { opacity: 1; transform: translateX(0); }
        }
        
        .log-INFO { 
            color: #00ff00; 
            border-left-color: #00ff00;
        }
        
        .log-SUCCESS { 
            color: #00ffff; 
            border-left-color: #00ffff;
            font-weight: bold;
        }
        
        .log-WARNING { 
            color: #ffaa00; 
            border-left-color: #ffaa00;
        }
        
        .log-ERROR { 
            color: #ff0000; 
            border-left-color: #ff0000;
            font-weight: bold;
        }
        
        .log-PROGRESS {
            color: #ff00ff;
            border-left-color: #ff00ff;
        }
        
        .log-METRIC {
            color: #ffff00;
            border-left-color: #ffff00;
        }
        
        .log-DATA {
            color: #00aaff;
            border-left-color: #00aaff;
        }
        
        .timestamp {
            color: #666;
            margin-right: 10px;
        }
        
        .source {
            color: #888;
            margin-right: 10px;
            font-weight: bold;
        }
        
        .progress-bar {
            width: 200px;
            height: 4px;
            background: #333;
            border-radius: 2px;
            overflow: hidden;
            display: inline-block;
            vertical-align: middle;
            margin-left: 10px;
        }
        
        .progress-fill {
            height: 100%;
            background: #00ff00;
            transition: width 0.3s;
        }
        
        pre {
            margin: 5px 0;
            padding: 10px;
            background: #1a1a1a;
            border-radius: 3px;
            overflow-x: auto;
        }
        
        .metric {
            display: inline-block;
            background: rgba(255, 255, 0, 0.1);
            padding: 2px 8px;
            border-radius: 3px;
            margin: 0 5px;
        }
        
        .experiment-select {
            background: #1a1a1a;
            color: #00ff00;
            border: 1px solid #00ff00;
            padding: 8px 16px;
            font-family: 'Courier New', monospace;
            cursor: pointer;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🌌 MOONSHINE QUANTUM TERMINAL</h1>
            <div class="status-bar">
                <div class="status-item" id="status-server">Server: <span id="server-status">Connecting...</span></div>
                <div class="status-item" id="status-qbc">QBC: <span id="qbc-status">Not Parsed</span></div>
                <div class="status-item" id="status-lattice">Lattice: <span id="lattice-status">Not Loaded</span></div>
                <div class="status-item" id="status-core">Core: <span id="core-status">Stopped</span></div>
            </div>
        </div>
        
        <div class="controls">
            <button id="btn-parse-qbc" onclick="parseQBC()">Parse QBC</button>
            <button id="btn-load-lattice" onclick="loadLattice()" disabled>Load Lattice</button>
            <button id="btn-start-core" onclick="startCore()" disabled>Start Quantum Core</button>
            <button id="btn-stop-core" onclick="stopCore()" disabled>Stop Core</button>
            
            <select class="experiment-select" id="experiment-select">
                <option value="">-- Select Experiment --</option>
            </select>
            <button id="btn-run-experiment" onclick="runExperiment()" disabled>Run Experiment</button>
            
            <button onclick="clearTerminal()">Clear Terminal</button>
        </div>
        
        <div class="terminal" id="terminal"></div>
    </div>
    
    <script>
        const terminal = document.getElementById('terminal');
        let eventSource = null;
        let logCount = 0;
        const MAX_LOGS = 1000;
        
        // Connect to log stream
        function connectLogStream() {
            eventSource = new EventSource('/api/logs/stream');
            
            eventSource.onmessage = function(event) {
                const log = JSON.parse(event.data);
                appendLog(log);
            };
            
            eventSource.onerror = function() {
                updateStatus('server-status', 'Disconnected', false);
                setTimeout(connectLogStream, 5000);
            };
            
            updateStatus('server-status', 'Connected', true);
        }
        
        function appendLog(log) {
            const entry = document.createElement('div');
            entry.className = `log-entry log-${log.level}`;
            
            let html = `<span class="timestamp">${new Date(log.timestamp).toLocaleTimeString()}</span>`;
            
            if (log.source) {
                html += `<span class="source">[${log.source}]</span>`;
            }
            
            html += `<span class="message">${escapeHtml(log.message)}</span>`;
            
            if (log.progress !== undefined) {
                html += `<div class="progress-bar"><div class="progress-fill" style="width: ${log.progress}%"></div></div> ${log.progress.toFixed(1)}%`;
            }
            
            if (log.metric_name && log.metric_value !== undefined) {
                html += `<span class="metric">${log.metric_name}: ${log.metric_value}${log.metric_unit || ''}</span>`;
            }
            
            entry.innerHTML = html;
            terminal.appendChild(entry);
            
            if (log.data) {
                const pre = document.createElement('pre');
                pre.textContent = JSON.stringify(log.data, null, 2);
                terminal.appendChild(pre);
            }
            
            if (log.error) {
                const pre = document.createElement('pre');
                pre.style.color = '#ff0000';
                pre.textContent = log.error;
                terminal.appendChild(pre);
            }
            
            // Auto-scroll
            terminal.scrollTop = terminal.scrollHeight;
            
            // Limit log count
            logCount++;
            if (logCount > MAX_LOGS) {
                terminal.removeChild(terminal.firstChild);
                logCount--;
            }
        }
        
        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }
        
        function updateStatus(elementId, text, active) {
            const span = document.getElementById(elementId);
            span.textContent = text;
            const parent = span.parentElement;
            if (active) {
                parent.classList.add('status-active');
            } else {
                parent.classList.remove('status-active');
            }
        }
        
        async function parseQBC() {
            const btn = document.getElementById('btn-parse-qbc');
            btn.disabled = true;
            btn.textContent = 'Parsing...';
            
            try {
                const response = await fetch('/api/parse-qbc', { method: 'POST' });
                const result = await response.json();
                
                if (result.success) {
                    updateStatus('qbc-status', 'Parsed', true);
                    document.getElementById('btn-load-lattice').disabled = false;
                    btn.textContent = 'Parse QBC ✓';
                } else {
                    btn.disabled = false;
                    btn.textContent = 'Parse QBC (Failed)';
                }
            } catch (error) {
                btn.disabled = false;
                btn.textContent = 'Parse QBC (Error)';
            }
        }
        
        async function loadLattice() {
            const btn = document.getElementById('btn-load-lattice');
            btn.disabled = true;
            btn.textContent = 'Loading...';
            
            try {
                const response = await fetch('/api/load-lattice', { method: 'POST' });
                const result = await response.json();
                
                if (result.success) {
                    updateStatus('lattice-status', 'Loaded', true);
                    document.getElementById('btn-start-core').disabled = false;
                    btn.textContent = 'Load Lattice ✓';
                } else {
                    btn.disabled = false;
                    btn.textContent = 'Load Lattice (Failed)';
                }
            } catch (error) {
                btn.disabled = false;
                btn.textContent = 'Load Lattice (Error)';
            }
        }
        
        async function startCore() {
            const btn = document.getElementById('btn-start-core');
            btn.disabled = true;
            btn.textContent = 'Starting...';
            
            try {
                const response = await fetch('/api/core/start', { method: 'POST' });
                const result = await response.json();
                
                if (result.success) {
                    updateStatus('core-status', 'Running', true);
                    document.getElementById('btn-stop-core').disabled = false;
                    document.getElementById('btn-run-experiment').disabled = false;
                    btn.textContent = 'Start Core ✓';
                } else {
                    btn.disabled = false;
                    btn.textContent = 'Start Core (Failed)';
                }
            } catch (error) {
                btn.disabled = false;
                btn.textContent = 'Start Core (Error)';
            }
        }
        
        async function stopCore() {
            const btn = document.getElementById('btn-stop-core');
            btn.disabled = true;
            
            try {
                const response = await fetch('/api/core/stop', { method: 'POST' });
                const result = await response.json();
                
                updateStatus('core-status', 'Stopped', false);
                document.getElementById('btn-start-core').disabled = false;
                document.getElementById('btn-run-experiment').disabled = true;
                btn.disabled = false;
            } catch (error) {
                btn.disabled = false;
            }
        }
        
        async function runExperiment() {
            const select = document.getElementById('experiment-select');
            const experimentName = select.value;
            
            if (!experimentName) {
                alert('Please select an experiment');
                return;
            }
            
            const btn = document.getElementById('btn-run-experiment');
            btn.disabled = true;
            btn.textContent = 'Running...';
            
            try {
                const response = await fetch('/api/experiments/run', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ experiment: experimentName })
                });
                
                // Logs will stream via SSE
                btn.textContent = 'Run Experiment';
                btn.disabled = false;
            } catch (error) {
                btn.textContent = 'Run Experiment (Error)';
                btn.disabled = false;
            }
        }
        
        function clearTerminal() {
            terminal.innerHTML = '';
            logCount = 0;
        }
        
        // Load experiments
        async function loadExperiments() {
            try {
                const response = await fetch('/api/experiments/list');
                const result = await response.json();
                
                const select = document.getElementById('experiment-select');
                result.experiments.forEach(exp => {
                    if (exp.implemented) {
                        const option = document.createElement('option');
                        option.value = exp.name;
                        option.textContent = `${exp.name} - ${exp.description}`;
                        select.appendChild(option);
                    }
                });
            } catch (error) {
                console.error('Failed to load experiments:', error);
            }
        }
        
        // Initialize
        connectLogStream();
        loadExperiments();
    </script>
</body>
</html>
"""

# ═════════════════════════════════════════════════════════════════════════
# ROUTES
# ═════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    """Serve HTML terminal"""
    return HTML_TERMINAL

@app.route('/api/logs/stream')
def log_stream():
    """Server-sent events for log streaming"""
    def generate():
        queue = log_capture.subscribe()
        try:
            while True:
                try:
                    log_entry = queue.get(timeout=30)
                    yield f"data: {json.dumps(log_entry)}\n\n"
                except Empty:
                    # Send keepalive
                    yield ": keepalive\n\n"
        except GeneratorExit:
            log_capture.subscribers.remove(queue)
    
    return Response(generate(), mimetype='text/event-stream')

@app.route('/api/parse-qbc', methods=['POST'])
def parse_qbc():
    """Parse QBC lattice builder"""
    try:
        logger.info("="*80)
        logger.info("STARTING QBC PARSE")
        logger.info("="*80)
        
        qbc_file = Path('lattice_builder.py')
        if not qbc_file.exists():
            logger.error(f"QBC file not found: {qbc_file}")
            return jsonify({'success': False, 'error': 'QBC file not found'})
        
        parser = QBCParser(verbose=True)
        success = parser.execute_qbc(qbc_file)
        
        if success:
            server_state['qbc_parser'] = parser
            server_state['qbc_parsed'] = True
            logger.info("="*80)
            logger.info("QBC PARSE COMPLETE")
            logger.info(f"Pseudoqubits: {len(parser.pseudoqubits):,}")
            logger.info(f"Triangles: {len(parser.triangles):,}")
            logger.info("="*80)
        
        return jsonify({
            'success': success,
            'pseudoqubits': len(parser.pseudoqubits) if success else 0,
            'triangles': len(parser.triangles) if success else 0
        })
        
    except Exception as e:
        logger.error(f"QBC parse failed: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/load-lattice', methods=['POST'])
def load_lattice():
    """Load lattice database"""
    try:
        logger.info("="*80)
        logger.info("LOADING LATTICE DATABASE")
        logger.info("="*80)
        
        db_path = Path('moonshine.db')
        if not db_path.exists():
            logger.error("moonshine.db not found")
            return jsonify({'success': False, 'error': 'Database not found'})
        
        # Verify database
        import sqlite3
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM coords")
        node_count = cursor.fetchone()[0]
        
        conn.close()
        
        server_state['lattice_loaded'] = True
        logger.info(f"Lattice loaded: {node_count:,} nodes")
        logger.info("="*80)
        
        return jsonify({'success': True, 'nodes': node_count})
        
    except Exception as e:
        logger.error(f"Lattice load failed: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/core/start', methods=['POST'])
def start_core():
    """Start quantum core"""
    try:
        logger.info("="*80)
        logger.info("STARTING QUANTUM CORE")
        logger.info("="*80)
        
        if server_state['quantum_core'] is None:
            # Get API key
            api_key_file = Path('random_org_api.txt')
            if not api_key_file.exists():
                logger.warning("random.org API key not found, using fallback")
                api_key = "DEMO_KEY"
            else:
                api_key = api_key_file.read_text().strip()
            
            core = create_quantum_core(random_org_api_key=api_key)
            server_state['quantum_core'] = core
        
        server_state['quantum_core'].start()
        server_state['quantum_core_running'] = True
        
        logger.info("Quantum core started")
        logger.info("="*80)
        
        return jsonify({'success': True})
        
    except Exception as e:
        logger.error(f"Core start failed: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/core/stop', methods=['POST'])
def stop_core():
    """Stop quantum core"""
    try:
        if server_state['quantum_core']:
            server_state['quantum_core'].stop()
            server_state['quantum_core_running'] = False
            logger.info("Quantum core stopped")
        
        return jsonify({'success': True})
        
    except Exception as e:
        logger.error(f"Core stop failed: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/core/status', methods=['GET'])
def core_status():
    """Get quantum core status"""
    try:
        if server_state['quantum_core']:
            status = server_state['quantum_core'].get_status()
            return jsonify({'success': True, 'status': status})
        else:
            return jsonify({'success': False, 'error': 'Core not started'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/experiments/list', methods=['GET'])
def list_experiments():
    """List available experiments"""
    try:
        if server_state['experiment_runner'] is None:
            runner = ExperimentRunner()
            server_state['experiment_runner'] = runner
        
        experiments = server_state['experiment_runner'].list_experiments()
        return jsonify({'success': True, 'experiments': experiments})
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/experiments/run', methods=['POST'])
def run_experiment():
    """Run experiment with streaming logs"""
    try:
        data = request.get_json()
        experiment_name = data.get('experiment')
        
        if not experiment_name:
            return jsonify({'success': False, 'error': 'No experiment specified'})
        
        logger.info("="*80)
        logger.info(f"RUNNING EXPERIMENT: {experiment_name}")
        logger.info("="*80)
        
        # Get API key
        api_key = request.headers.get('X-API-Key') or data.get('api_key', 'DEMO_KEY')
        
        if server_state['experiment_runner'] is None:
            server_state['experiment_runner'] = ExperimentRunner()
        
        # Run experiment in background thread
        def run_in_background():
            try:
                for log_entry in server_state['experiment_runner'].run_experiment(
                    experiment_name, 
                    api_key=api_key,
                    db_path='moonshine.db'
                ):
                    log_capture.emit(log_entry.to_dict())
            except Exception as e:
                logger.error(f"Experiment failed: {e}")
                logger.error(traceback.format_exc())
        
        thread = threading.Thread(target=run_in_background, daemon=True)
        thread.start()
        
        return jsonify({'success': True})
        
    except Exception as e:
        logger.error(f"Experiment run failed: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/status', methods=['GET'])
def status():
    """Get server status"""
    return jsonify({
        'success': True,
        'status': server_state
    })

# ═════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    logger.info("="*80)
    logger.info("🌌 MOONSHINE QUANTUM SERVER STARTING")
    logger.info("="*80)
    logger.info("Server: http://localhost:5000")
    logger.info("Terminal: http://localhost:5000/")
    logger.info("="*80)
    
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
