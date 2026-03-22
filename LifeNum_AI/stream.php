<?php
/**
 * stream.php
 * Runs main.py in the background and streams its stdout/stderr
 * as Server-Sent Events (text/event-stream) to the browser.
 *
 * The HTML dashboard connects here via EventSource.
 * No API logic here — that all lives in main.py.
 */

// ── Abort / kill handler ──────────────────────────────────────────────────
if (isset($_GET['action']) && $_GET['action'] === 'stop') {
    // Read the PID we stored and kill the process tree
    $pidFile = sys_get_temp_dir() . '/plaiades_main.pid';
    if (file_exists($pidFile)) {
        $pid = (int) file_get_contents($pidFile);
        if ($pid > 0) {
            // Kill entire process group (works on Linux/macOS)
            posix_kill(-$pid, SIGTERM);
            // fallback: kill just the pid
            posix_kill($pid, SIGTERM);
        }
        unlink($pidFile);
    }
    header('Content-Type: application/json');
    echo json_encode(['status' => 'stopped']);
    exit;
}

// ── SSE headers ──────────────────────────────────────────────────────────
header('Content-Type: text/event-stream');
header('Cache-Control: no-cache');
header('X-Accel-Buffering: no'); // disable nginx buffering
header('Access-Control-Allow-Origin: *');

// Disable output buffering
if (ob_get_level()) ob_end_clean();
ini_set('output_buffering', 'off');
ini_set('implicit_flush', true);

// ── Helper: send one SSE event ────────────────────────────────────────────
function sse(string $event, string $data): void {
    echo "event: {$event}\n";
    // Newlines inside data must be escaped as multiple data: lines
    foreach (explode("\n", $data) as $line) {
        echo "data: {$line}\n";
    }
    echo "\n";
    flush();
}

// ── Locate main.py ────────────────────────────────────────────────────────
// Adjust this path if main.py lives elsewhere relative to stream.php
$scriptDir  = __DIR__;
$mainPy     = $scriptDir . '/main.py';

if (!file_exists($mainPy)) {
    sse('error', json_encode(['message' => "main.py not found at: {$mainPy}"]));
    exit;
}

// ── Launch main.py ────────────────────────────────────────────────────────
$python = trim(shell_exec('which python3') ?: shell_exec('which python') ?: 'python3');

$descriptors = [
    0 => ['pipe', 'r'],  // stdin
    1 => ['pipe', 'w'],  // stdout
    2 => ['pipe', 'w'],  // stderr
];

$env = array_merge($_ENV, [
    'PYTHONUNBUFFERED' => '1',   // crucial: no Python output buffering
    'PYTHONIOENCODING' => 'utf-8',
]);

$process = proc_open(
    escapeshellcmd($python) . ' -u ' . escapeshellarg($mainPy),
    $descriptors,
    $pipes,
    $scriptDir,
    $env
);

if (!is_resource($process)) {
    sse('error', json_encode(['message' => 'Failed to launch main.py']));
    exit;
}

// Store PID so the stop endpoint can kill it
$status = proc_get_status($process);
$pid    = $status['pid'];
file_put_contents(sys_get_temp_dir() . '/plaiades_main.pid', $pid);

// Close stdin; set stdout/stderr to non-blocking
fclose($pipes[0]);
stream_set_blocking($pipes[1], false);
stream_set_blocking($pipes[2], false);

sse('start', json_encode(['pid' => $pid, 'script' => $mainPy]));

// ── Stream loop ───────────────────────────────────────────────────────────
$timeout   = 0;       // run until process ends or client disconnects
$lastAlive = time();

while (true) {
    // Client disconnected?
    if (connection_aborted()) {
        proc_terminate($process, SIGTERM);
        break;
    }

    $status = proc_get_status($process);

    // Read stdout
    $line = fgets($pipes[1]);
    if ($line !== false && $line !== '') {
        sse('log', json_encode(['stream' => 'stdout', 'line' => rtrim($line, "\r\n")]));
        $lastAlive = time();
    }

    // Read stderr
    $err = fgets($pipes[2]);
    if ($err !== false && $err !== '') {
        sse('log', json_encode(['stream' => 'stderr', 'line' => rtrim($err, "\r\n")]));
        $lastAlive = time();
    }

    // Process finished?
    if (!$status['running']) {
        // Drain remaining output
        while (($line = fgets($pipes[1])) !== false) {
            if ($line !== '') sse('log', json_encode(['stream' => 'stdout', 'line' => rtrim($line)]));
        }
        while (($err = fgets($pipes[2])) !== false) {
            if ($err !== '') sse('log', json_encode(['stream' => 'stderr', 'line' => rtrim($err)]));
        }
        sse('exit', json_encode(['code' => $status['exitcode']]));
        break;
    }

    // Keep-alive ping every 15 s so the browser doesn't time out
    if (time() - $lastAlive > 15) {
        sse('ping', '{}');
        $lastAlive = time();
    }

    usleep(30000); // 30 ms poll interval — low CPU, responsive output
}

fclose($pipes[1]);
fclose($pipes[2]);
proc_close($process);
@unlink(sys_get_temp_dir() . '/plaiades_main.pid');