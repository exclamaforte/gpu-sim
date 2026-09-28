#!/usr/bin/env python3
"""Data-generation throughput instrumentation and experiment log.

Tracks examples/hour through the real-GPU measurement stage as an explicit
metric, since this is expected to be the project's actual bottleneck.
"""

import time
import json
import threading
from dataclasses import dataclass, asdict, field
from typing import Optional
from collections import deque
from pathlib import Path
from contextlib import contextmanager


@dataclass
class ThroughputMetrics:
    """Throughput metrics for a measurement phase."""
    phase: str
    start_time: float
    end_time: float
    duration_sec: float
    examples_completed: int
    examples_failed: int
    examples_per_sec: float
    examples_per_hour: float
    avg_time_per_example_sec: float
    peak_memory_mb: float
    gpu_utilization_pct: float


@dataclass
class ExperimentLogEntry:
    """Single experiment log entry."""
    timestamp: float
    run_id: str
    phase: str
    kernel_hash: str
    status: str  # 'started', 'completed', 'failed', 'timeout', 'skipped'
    duration_sec: float
    trials: list[float]
    median_ms: Optional[float]
    cv: Optional[float]
    throughput_examples_per_sec: float
    metadata: dict = field(default_factory=dict)


class ThroughputTracker:
    """Tracks and logs data-generation throughput."""
    
    def __init__(
        self,
        run_id: str,
        log_dir: str = "runs/logs",
        window_size: int = 100,
    ):
        self.run_id = run_id
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.window_size = window_size
        
        self._phase_start_times: dict[str, float] = {}
        self._phase_counts: dict[str, dict] = {}  # completed, failed
        self._recent_durations = deque(maxlen=window_size)
        self._lock = threading.Lock()
        
        self.log_file = self.log_dir / f"throughput_{run_id}.jsonl"
        self.summary_file = self.log_dir / f"summary_{run_id}.json"
    
    def start_phase(self, phase: str):
        """Mark the start of a measurement phase."""
        with self._lock:
            self._phase_start_times[phase] = time.time()
            if phase not in self._phase_counts:
                self._phase_counts[phase] = {'completed': 0, 'failed': 0}
    
    def end_phase(self, phase: str) -> ThroughputMetrics:
        """Mark the end of a phase and compute metrics."""
        with self._lock:
            if phase not in self._phase_start_times:
                return None
            
            start = self._phase_start_times.pop(phase)
            end = time.time()
            duration = end - start
            counts = self._phase_counts.get(phase, {'completed': 0, 'failed': 0})
            completed = counts['completed']
            failed = counts['failed']
            total = completed + failed
            
            metrics = ThroughputMetrics(
                phase=phase,
                start_time=start,
                end_time=end,
                duration_sec=duration,
                examples_completed=completed,
                examples_failed=failed,
                examples_per_sec=total / duration if duration > 0 else 0,
                examples_per_hour=(total / duration * 3600) if duration > 0 else 0,
                avg_time_per_example_sec=duration / total if total > 0 else 0,
                peak_memory_mb=0,  # Would need psutil
                gpu_utilization_pct=0,  # Would need nvidia-smi monitoring
            )
            
            return metrics
    
    def record_example(
        self,
        phase: str,
        kernel_hash: str,
        status: str,
        duration_sec: float,
        trials: list[float] = None,
        median_ms: float = None,
        cv: float = None,
        metadata: dict = None,
    ):
        """Record a single example result."""
        with self._lock:
            counts = self._phase_counts.get(phase, {'completed': 0, 'failed': 0})
            
            if status == 'completed':
                counts['completed'] += 1
                self._recent_durations.append(duration_sec)
            elif status in ('failed', 'timeout', 'skipped'):
                counts['failed'] += 1
            
            self._phase_counts[phase] = counts
        
        # Compute current throughput
        current_throughput = self.get_current_throughput(phase)
        
        entry = ExperimentLogEntry(
            timestamp=time.time(),
            run_id=self.run_id,
            phase=phase,
            kernel_hash=kernel_hash,
            status=status,
            duration_sec=duration_sec,
            trials=trials or [],
            median_ms=median_ms,
            cv=cv,
            throughput_examples_per_sec=current_throughput,
            metadata=metadata or {},
        )
        
        self._log_entry(entry)
    
    def get_current_throughput(self, phase: str) -> float:
        """Get current examples/sec for a phase (sliding window)."""
        with self._lock:
            if phase not in self._phase_start_times:
                return 0.0
            
            start = self._phase_start_times[phase]
            elapsed = time.time() - start
            if elapsed == 0:
                return 0.0
            
            completed = self._phase_counts.get(phase, {}).get('completed', 0)
            return completed / elapsed
    
    def get_sliding_throughput(self) -> float:
        """Get throughput based on recent examples (sliding window)."""
        with self._lock:
            if len(self._recent_durations) < 2:
                return 0.0
            
            # Use recent examples to compute rate
            window_sec = sum(self._recent_durations)
            if window_sec == 0:
                return 0.0
            
            return len(self._recent_durations) / window_sec
    
    def _log_entry(self, entry: ExperimentLogEntry):
        """Write log entry to file."""
        with open(self.log_file, 'a') as f:
            f.write(json.dumps(asdict(entry)) + '\n')
    
    def get_summary(self) -> dict:
        """Get overall experiment summary."""
        with self._lock:
            summary = {
                'run_id': self.run_id,
                'total_phases': len(self._phase_counts),
                'phases': {},
                'overall': {
                    'total_completed': sum(c['completed'] for c in self._phase_counts.values()),
                    'total_failed': sum(c['failed'] for c in self._phase_counts.values()),
                },
            }
            
            for phase, counts in self._phase_counts.items():
                start = self._phase_start_times.get(phase)
                if start:
                    elapsed = time.time() - start
                    total = counts['completed'] + counts['failed']
                    summary['phases'][phase] = {
                        'completed': counts['completed'],
                        'failed': counts['failed'],
                        'elapsed_sec': elapsed,
                        'examples_per_sec': total / elapsed if elapsed > 0 else 0,
                        'examples_per_hour': (total / elapsed * 3600) if elapsed > 0 else 0,
                    }
            
            # Write summary
            with open(self.summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            return summary
    
    def print_summary(self):
        """Print human-readable summary."""
        summary = self.get_summary()
        print(f"\n=== Throughput Summary for {self.run_id} ===")
        print(f"Overall: {summary['overall']['total_completed']} completed, "
              f"{summary['overall']['total_failed']} failed")
        
        for phase, metrics in summary['phases'].items():
            print(f"  {phase}: {metrics['completed']} completed, {metrics['failed']} failed, "
                  f"{metrics['examples_per_sec']:.2f} ex/s ({metrics['examples_per_hour']:.0f} ex/hr)")
        
        sliding = self.get_sliding_throughput()
        print(f"Sliding window throughput: {sliding:.2f} ex/s ({sliding * 3600:.0f} ex/hr)")


class PhaseTimer:
    """Context manager for timing a phase."""
    
    def __init__(self, tracker: ThroughputTracker, phase: str):
        self.tracker = tracker
        self.phase = phase
    
    def __enter__(self):
        self.tracker.start_phase(self.phase)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        metrics = self.tracker.end_phase(self.phase)
        if metrics:
            print(f"Phase '{self.phase}': {metrics.examples_completed} completed in "
                  f"{metrics.duration_sec:.1f}s = {metrics.examples_per_hour:.0f} ex/hr")


@contextmanager
def track_throughput(run_id: str, log_dir: str = "runs/logs"):
    """Context manager for a full throughput-tracked experiment."""
    tracker = ThroughputTracker(run_id, log_dir)
    try:
        yield tracker
    finally:
        tracker.get_summary()
        tracker.print_summary()


class ExperimentLogger:
    """High-level experiment logger with throughput tracking."""
    
    def __init__(self, run_id: str, log_dir: str = "runs/logs"):
        self.tracker = ThroughputTracker(run_id, log_dir)
        self.run_id = run_id
    
    def log_kernel_measurement(
        self,
        kernel_hash: str,
        launch_config: dict,
        trials_ms: list[float],
        stats: dict,
        status: str = 'completed',
    ):
        """Log a kernel measurement result."""
        self.tracker.record_example(
            phase='measurement',
            kernel_hash=kernel_hash,
            status=status,
            duration_sec=sum(trials_ms) / 1000 if trials_ms else 0,
            trials=trials_ms,
            median_ms=stats.get('median_ms'),
            cv=stats.get('cv'),
            metadata={
                'launch_config': launch_config,
                'log_mean': stats.get('log_mean'),
                'log_std': stats.get('log_std'),
                'outliers_removed': stats.get('outliers_removed', 0),
            },
        )
    
    def log_kernel_compilation(self, kernel_hash: str, success: bool, duration_sec: float):
        """Log kernel compilation result."""
        self.tracker.record_example(
            phase='compilation',
            kernel_hash=kernel_hash,
            status='completed' if success else 'failed',
            duration_sec=duration_sec,
            metadata={'stage': 'compilation'},
        )
    
    def log_kernel_generation(self, kernel_hash: str, duration_sec: float):
        """Log kernel generation result."""
        self.tracker.record_example(
            phase='generation',
            kernel_hash=kernel_hash,
            status='completed',
            duration_sec=duration_sec,
            metadata={'stage': 'generation'},
        )
    
    def log_preflight(self, kernel_hash: str, success: bool, duration_sec: float):
        """Log pre-flight validation result."""
        self.tracker.record_example(
            phase='preflight',
            kernel_hash=kernel_hash,
            status='completed' if success else 'failed',
            duration_sec=duration_sec,
            metadata={'stage': 'preflight'},
        )
    
    def print_summary(self):
        self.tracker.print_summary()
    
    def get_summary(self) -> dict:
        return self.tracker.get_summary()


if __name__ == '__main__':
    # Demo
    with track_throughput("demo_run") as tracker:
        # Simulate measurement phase
        with PhaseTimer(tracker, "measurement"):
            for i in range(10):
                time.sleep(0.01)  # Simulate work
                tracker.record_example(
                    phase="measurement",
                    kernel_hash=f"kernel_{i:04x}",
                    status="completed",
                    duration_sec=0.01,
                    trials=[5.0, 5.1, 4.9],
                    median_ms=5.0,
                    cv=0.02,
                )
        
        # Simulate compilation phase
        with PhaseTimer(tracker, "compilation"):
            for i in range(5):
                time.sleep(0.005)
                tracker.record_example(
                    phase="compilation",
                    kernel_hash=f"kernel_{i:04x}",
                    status="completed",
                    duration_sec=0.005,
                )