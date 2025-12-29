#!/usr/bin/env python3
"""
Moonshine Quantum Server - Web Wrapper for Cloud Hosting

Adds HTTP endpoints for:
- Health checks (required by Render/Railway/Fly)
- Routing table download
- Server status
"""

from flask import Flask, jsonify, send_file
import threading
import time
import sys
import io
from pathlib import Path
from datetime import datetime

# Import the main server
import moonshine_production_server_v3 as server

app = Flask(__name__)

# Server state
SERVER_STATE = {
    'started': False,
    'start_time': None,
    'heartbeat_count': 0,
    'current_sigma': 0.0,
    'routing_table_ready': False
}

# Log capture
LOG_BUFFER = []
MAX_LOGS = 1000

class LogCapture:
    """Capture stdout/stderr to display in web interface"""
    def __init__(self):
        self.terminal = sys.stdout
        self.log = io.StringIO()
        
    def write(self, message):
        self.terminal.write(message)
        self.terminal.flush()
        
        # Add to buffer
        if message.strip():
            timestamp = datetime.now().strftime('%H:%M:%S')
            log_line = f'<div class="log-line">[{timestamp}] {message.strip()}</div>'
            LOG_BUFFER.append(log_line)
            
            # Keep only last MAX_LOGS lines
            if len(LOG_BUFFER) > MAX_LOGS:
                LOG_BUFFER.pop(0)
    
    def flush(self):
        self.terminal.flush()

# Redirect stdout to capture logs
sys.stdout = LogCapture()
sys.stderr = LogCapture()

@app.route('/logs')
def logs():
    """Return captured logs"""
    return jsonify({
        'logs': LOG_BUFFER,
        'count': len(LOG_BUFFER),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/restart', methods=['POST'])
def restart():
    """Restart the server (for debugging)"""
    import os
    import signal
    
    LOG_BUFFER.append('<div class="log-error">🔄 SERVER RESTART REQUESTED</div>')
    
    # Schedule restart after response
    def do_restart():
        time.sleep(1)
        os.kill(os.getpid(), signal.SIGTERM)
    
    threading.Thread(target=do_restart, daemon=True).start()
    
    return jsonify({
        'success': True,
        'message': 'Server restarting...'
    })

@app.route('/health')
def health():
    """Health check endpoint (required by cloud platforms)"""
    return jsonify({
        'status': 'healthy',
        'server': 'moonshine-quantum-internet',
        'uptime': time.time() - SERVER_STATE['start_time'] if SERVER_STATE['start_time'] else 0,
        'heartbeat': SERVER_STATE['heartbeat_count']
    })

@app.route('/')
def index():
    """Landing page with live terminal output"""
    return f"""
    <html>
    <head>
        <title>Moonshine Quantum Internet</title>
        <style>
            body {{ 
                font-family: 'Courier New', monospace; 
                background: #000; 
                color: #0f0; 
                padding: 20px;
                margin: 0;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
            }}
            h1 {{ 
                color: #0ff; 
                text-align: center;
                font-size: 2em;
                margin-bottom: 5px;
            }}
            .author {{
                text-align: center;
                color: #0ff;
                font-size: 0.9em;
                margin-bottom: 20px;
            }}
            .status {{ 
                background: #111; 
                padding: 15px; 
                margin: 10px 0; 
                border: 1px solid #0f0;
            }}
            .terminal {{
                background: #000;
                border: 2px solid #0f0;
                padding: 15px;
                height: 500px;
                overflow-y: scroll;
                margin: 20px 0;
                font-family: 'Courier New', monospace;
                font-size: 12px;
                line-height: 1.4;
            }}
            .terminal::-webkit-scrollbar {{
                width: 10px;
            }}
            .terminal::-webkit-scrollbar-track {{
                background: #111;
            }}
            .terminal::-webkit-scrollbar-thumb {{
                background: #0f0;
            }}
            .log-line {{
                color: #0f0;
                margin: 2px 0;
            }}
            .log-error {{
                color: #f00;
            }}
            .log-success {{
                color: #0ff;
            }}
            .heartbeat {{
                color: #f0f;
                font-weight: bold;
            }}
            a {{ color: #0ff; }}
            .blink {{
                animation: blink 1s infinite;
            }}
            @keyframes blink {{
                50% {{ opacity: 0.5; }}
            }}
            .grid {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
                margin: 20px 0;
            }}
            @media (max-width: 768px) {{
                .grid {{ grid-template-columns: 1fr; }}
            }}
        </style>
        <script>
            let logLines = [];
            let autoScroll = true;
            
            // Fetch server logs
            async function fetchLogs() {{
                try {{
                    const response = await fetch('/logs');
                    const data = await response.json();
                    const terminal = document.getElementById('terminal');
                    
                    if (data.logs && data.logs.length > logLines.length) {{
                        // New logs available
                        logLines = data.logs;
                        terminal.innerHTML = logLines.join('<br>');
                        
                        if (autoScroll) {{
                            terminal.scrollTop = terminal.scrollHeight;
                        }}
                    }}
                }} catch (e) {{
                    console.error('Failed to fetch logs:', e);
                }}
            }}
            
            // Update status
            async function updateStatus() {{
                try {{
                    const response = await fetch('/status');
                    const data = await response.json();
                    
                    document.getElementById('heartbeat').textContent = data.heartbeat_count || 0;
                    document.getElementById('sigma').textContent = (data.current_sigma || 0).toFixed(4);
                    document.getElementById('uptime').textContent = (data.uptime_seconds / 60 || 0).toFixed(1);
                    document.getElementById('status-indicator').textContent = data.started ? '🟢 ONLINE' : '🟡 STARTING...';
                    
                    // Update metrics from latest_metrics in status response
                    if (data.latest_metrics) {{
                        document.getElementById('fidelity').textContent = (data.latest_metrics.fidelity || 0).toFixed(4);
                        document.getElementById('chsh').textContent = (data.latest_metrics.chsh || 0).toFixed(3);
                        document.getElementById('coherence').textContent = (data.latest_metrics.coherence || 0).toFixed(4);
                        document.getElementById('triangle').textContent = data.latest_metrics.triangle_id || '---';
                        document.getElementById('qubit').textContent = data.latest_metrics.measured_qubit >= 0 ? 
                            'q' + data.latest_metrics.measured_qubit : '---';
                        document.getElementById('wcount').textContent = data.latest_metrics.w_count || 0;
                    }}
                }} catch (e) {{
                    console.error('Status update error:', e);
                    document.getElementById('status-indicator').textContent = '🔴 ERROR';
                }}
            }}
            
            // Auto-scroll toggle
            function toggleAutoScroll() {{
                autoScroll = !autoScroll;
                document.getElementById('scroll-btn').textContent = autoScroll ? '📜 Auto-scroll: ON' : '📜 Auto-scroll: OFF';
            }}
            
            // Clear terminal
            function clearTerminal() {{
                document.getElementById('terminal').innerHTML = '<div class="log-line">Terminal cleared by user...</div>';
                logLines = [];
            }}
            
            // Restart server
            async function restartServer() {{
                if (!confirm('⚠️ This will restart the server and clear all state. Continue?')) {{
                    return;
                }}
                
                try {{
                    document.getElementById('terminal').innerHTML += '<div class="log-error">🔄 RESTART REQUESTED BY USER...</div>';
                    const response = await fetch('/restart', {{ method: 'POST' }});
                    const data = await response.json();
                    
                    if (data.success) {{
                        document.getElementById('terminal').innerHTML += '<div class="log-success">✓ Server restarting...</div>';
                        setTimeout(() => location.reload(), 3000);
                    }} else {{
                        document.getElementById('terminal').innerHTML += '<div class="log-error">✗ Restart failed</div>';
                    }}
                }} catch (e) {{
                    document.getElementById('terminal').innerHTML += '<div class="log-error">✗ Restart error: ' + e.message + '</div>';
                }}
            }}
            
            // Initial load
            updateStatus();
            fetchLogs();
            
            // Update every second
            setInterval(updateStatus, 1000);
            setInterval(fetchLogs, 1000);
        </script>
    </head>
    <body>
        <div class="container">
            <h1>🌙 Moonshine Quantum Internet</h1>
            <div class="author">
                Made with 💙 by <strong>shemshallah::justin.howard-stanley</strong>
            </div>
            
            <div class="status">
                <strong>Status:</strong> <span id="status-indicator" class="blink">🟡 STARTING...</span><br>
                <strong>Heartbeat:</strong> <span id="heartbeat" class="heartbeat">0</span> beats<br>
                <strong>σ-coordinate:</strong> <span id="sigma">0.0000</span><br>
                <strong>Uptime:</strong> <span id="uptime">0.0</span> minutes
            </div>
            
            <div class="status">
                <h3>🔬 Live Quantum Metrics (Aer)</h3>
                <strong>W-State Fidelity:</strong> <span id="fidelity" style="color: #0ff;">0.0000</span><br>
                <strong>CHSH Parameter:</strong> <span id="chsh" style="color: #0ff;">0.000</span> (max: 2.828)<br>
                <strong>Coherence (Ψ):</strong> <span id="coherence" style="color: #0ff;">0.0000</span><br>
                <strong>Triangle ID:</strong> <span id="triangle" style="color: #0ff;">---</span><br>
                <strong>Measured Qubit:</strong> <span id="qubit" style="color: #0ff;">---</span><br>
                <strong>W-Count:</strong> <span id="wcount" style="color: #0ff;">0</span>/1024<br>
                <small style="color: #888;">Lattice sync every heartbeat (~1s)</small>
            </div>
            
            <div style="margin: 20px 0;">
                <button onclick="toggleAutoScroll()" id="scroll-btn" style="background: #0f0; color: #000; padding: 10px; border: none; cursor: pointer; margin-right: 10px;">📜 Auto-scroll: ON</button>
                <button onclick="clearTerminal()" style="background: #f00; color: #fff; padding: 10px; border: none; cursor: pointer; margin-right: 10px;">🗑️ Clear Terminal</button>
                <button onclick="restartServer()" style="background: #ff0; color: #000; padding: 10px; border: none; cursor: pointer;">🔄 Restart Server</button>
            </div>
            
            <div style="background: #111; padding: 10px; margin: 10px 0; border: 1px solid #0f0;">
                <strong>📺 LIVE SERVER OUTPUT</strong>
            </div>
            
            <div id="terminal" class="terminal">
                <div class="log-line">Connecting to Moonshine server...</div>
                <div class="log-line">Waiting for logs...</div>
            </div>
            
            <div class="grid">
                <div class="status">
                    <h3>🗺️ Network Details</h3>
                    <ul style="margin: 0; padding-left: 20px;">
                        <li><strong>Nodes:</strong> 196,883 triangles</li>
                        <li><strong>Hierarchy:</strong> 11 layers → 3 apex pillars</li>
                        <li><strong>Backend:</strong> IonQ Simulator + Aer</li>
                        <li><strong>σ-Period:</strong> 8.0 (revivals at 0, 4, 8)</li>
                    </ul>
                </div>
                
                <div class="status">
                    <h3>📥 Downloads</h3>
                    <ul style="margin: 0; padding-left: 20px;">
                        <li><a href="/routing-table">Routing Table (SQLite DB - 15MB)</a></li>
                        <li><a href="/special-triangles.csv">Special Triangles (CSV)</a></li>
                        <li><a href="/status">Server Status (JSON)</a></li>
                        <li><a href="https://huggingface.co/spaces/shemshallah/moonshine-quantum-internet/resolve/main/moonshine_client.py">Client (HuggingFace)</a></li>
                        <li><a href="https://github.com/shemshallah/moonshine-quantum-internet/blob/main/moonshine_client.py">Client (GitHub)</a></li>
                        <li><a href="https://github.com/shemshallah/moonshine-quantum-internet">GitHub Repository</a></li>
                    </ul>
                </div>
            </div>
            
            <div class="status">
                <h3>🔗 Connect to the Manifold</h3>
                <pre style="background: #000; padding: 10px; overflow-x: auto;">pip install qiskit qiskit-aer
wget https://shemshallah-moonshine-quantum-internet.hf.space/routing-table -O moonshine_routes.db
wget https://huggingface.co/spaces/shemshallah/moonshine-quantum-internet/resolve/main/moonshine_client.py
python moonshine_client.py</pre>
            </div>
            
            <div style="text-align: center; margin-top: 30px; color: #666; font-size: 0.8em;">
                <p>Moonshine Quantum Internet v3.0 | Research Project | 2025</p>
                <p>Based on the Monster group's 196,883-dimensional Moonshine module</p>
            </div>
        </div>
    </body>
    </html>
    """

@app.route('/status')
def status():
    """Detailed server status"""
    # Get latest metrics if available
    latest_metrics = None
    has_keep_alive = hasattr(server, 'keep_alive') and server.keep_alive is not None
    
    if has_keep_alive:
        keep_alive_obj = server.keep_alive
        has_metrics = hasattr(keep_alive_obj, 'metrics_history')
        metrics_count = len(keep_alive_obj.metrics_history) if has_metrics and keep_alive_obj.metrics_history else 0
        
        print(f"[STATUS DEBUG] keep_alive exists, metrics_count={metrics_count}, beats={keep_alive_obj.beats}")
        
        if has_metrics and keep_alive_obj.metrics_history:
            latest_metrics = keep_alive_obj.metrics_history[-1]
            print(f"[STATUS DEBUG] latest_metrics: {latest_metrics}")
    else:
        print(f"[STATUS DEBUG] keep_alive not available yet")
    
    return jsonify({
        'server': 'moonshine-quantum-internet',
        'version': '3.0',
        'started': SERVER_STATE['started'],
        'start_time': SERVER_STATE['start_time'],
        'uptime_seconds': time.time() - SERVER_STATE['start_time'] if SERVER_STATE['start_time'] else 0,
        'heartbeat_count': SERVER_STATE['heartbeat_count'],
        'current_sigma': SERVER_STATE['current_sigma'],
        'routing_table_ready': SERVER_STATE['routing_table_ready'],
        'latest_metrics': latest_metrics,
        'network': {
            'total_nodes': 196883,
            'hierarchy_layers': 11,
            'apex_pillars': 3,
            'sigma_period': 8.0
        }
    })

@app.route('/metrics')
def metrics():
    """Get recent quantum metrics history"""
    try:
        # Get last N metrics
        from flask import request
        n = request.args.get('n', default=100, type=int)
        n = min(n, 1000)  # Cap at 1000
        
        # Debug: Check what we have access to
        has_keep_alive = hasattr(server, 'keep_alive')
        keep_alive_obj = server.keep_alive if has_keep_alive else None
        has_metrics = hasattr(keep_alive_obj, 'metrics_history') if keep_alive_obj else False
        
        print(f"[METRICS DEBUG] has_keep_alive={has_keep_alive}, "
              f"keep_alive_obj={keep_alive_obj is not None}, "
              f"has_metrics={has_metrics}")
        
        if keep_alive_obj and has_metrics:
            history = keep_alive_obj.metrics_history[-n:] if keep_alive_obj.metrics_history else []
            print(f"[METRICS DEBUG] history length: {len(history)}")
            return jsonify({
                'count': len(history),
                'metrics': history,
                'current_beat': keep_alive_obj.beats,
                'current_sigma': keep_alive_obj.sigma
            })
        else:
            return jsonify({
                'error': 'Metrics not yet available',
                'debug': {
                    'has_keep_alive': has_keep_alive,
                    'has_metrics': has_metrics
                }
            }), 503
    except Exception as e:
        import traceback
        print(f"[METRICS ERROR] {e}")
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/routing-table')
def routing_table():
    """Download routing table (SQLite database)"""
    db_path = Path('/app/data/moonshine_routes.db')
    if not db_path.exists():
        return jsonify({'error': 'Routing table not ready yet'}), 503
    
    return send_file(db_path, 
                     as_attachment=True,
                     download_name='moonshine_routes.db')

@app.route('/special-triangles.csv')
def special_triangles_csv():
    """Download special triangles CSV"""
    csv_path = Path('/app/data/special_triangles.csv')
    if not csv_path.exists():
        return jsonify({'error': 'Special triangles not ready yet'}), 503
    
    return send_file(csv_path,
                     as_attachment=True,
                     download_name='special_triangles.csv')

def run_server_background():
    """Run Moonshine server in background thread"""
    print("🌙 Starting Moonshine Quantum Server in background...")
    
    SERVER_STATE['started'] = True
    SERVER_STATE['start_time'] = time.time()
    
    # Run the main server
    # Note: This would need modification to server code to update SERVER_STATE
    server.main()

def monitor_server_state():
    """Monitor server state and update SERVER_STATE"""
    while True:
        time.sleep(10)
        
        # Check if routing table exists (SQLite database)
        db_path = Path('/app/data/moonshine_routes.db')
        if db_path.exists():
            SERVER_STATE['routing_table_ready'] = True
            
            # Get heartbeat info from keep_alive directly
            try:
                if hasattr(server, 'keep_alive'):
                    SERVER_STATE['heartbeat_count'] = server.keep_alive.beats
                    SERVER_STATE['current_sigma'] = server.keep_alive.sigma
            except:
                pass

if __name__ == '__main__':
    # Start server in background thread
    server_thread = threading.Thread(target=run_server_background, daemon=True)
    server_thread.start()
    
    # Start monitor thread
    monitor_thread = threading.Thread(target=monitor_server_state, daemon=True)
    monitor_thread.start()
    
    # Give server time to start
    time.sleep(5)
    
    # Start Flask web server
    # Hugging Face Spaces uses PORT 7860, others use environment variable
    import os
    port = int(os.environ.get('PORT', 7860))
    
    print(f"🌐 Starting web interface on port {port}...")
    app.run(host='0.0.0.0', port=port)
