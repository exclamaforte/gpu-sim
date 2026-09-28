#!/usr/bin/env python3
"""Driver-hang and GPU-reset recovery procedure.

Defines and scripts the recovery path for when a kernel wedges the driver
badly enough that watchdog-kill isn't sufficient (Xid errors, GPU falls off bus).
"""

import subprocess
import time
import os
import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional
from pathlib import Path


logger = logging.getLogger(__name__)


class RecoveryAction(Enum):
    """Recovery actions in order of escalation."""
    NONE = "none"
    SOFT_RESET = "soft_reset"           # nvidia-smi -r
    HARD_RESET = "hard_reset"           # nvidia-smi --gpu-reset
    DRIVER_RELOAD = "driver_reload"     # modprobe -r nvidia && modprobe nvidia
    SYSTEM_REBOOT = "system_reboot"     # Full system reboot


class GPUHealth(Enum):
    """GPU health states."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"       # Throttling but functional
    HUNG = "hung"               # Driver responsive but GPU stuck
    XID_ERROR = "xid_error"     # Xid error in dmesg
    OFF_BUS = "off_bus"         # GPU fallen off PCIe bus
    UNKNOWN = "unknown"


@dataclass
class RecoveryResult:
    """Result of a recovery attempt."""
    success: bool
    action: RecoveryAction
    message: str
    duration_sec: float
    gpu_state_after: Optional[str] = None


class GPURecovery:
    """Handles GPU recovery procedures."""
    
    def __init__(self, gpu_id: int = 0, checkpoint_dir: str = "runs/checkpoints"):
        self.gpu_id = gpu_id
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.last_checkpoint: Optional[str] = None
    
    def check_gpu_health(self) -> GPUHealth:
        """Check GPU health via nvidia-smi and dmesg."""
        try:
            # Check nvidia-smi
            result = subprocess.run(
                ['nvidia-smi', '-i', str(self.gpu_id), '-q'],
                capture_output=True, text=True, timeout=10
            )
            
            if result.returncode != 0:
                # nvidia-smi failed - driver might be hung
                stderr = result.stderr.lower()
                if 'unable to determine' in stderr or 'not supported' in stderr:
                    return GPUHealth.OFF_BUS
                return GPUHealth.HUNG
            
            output = result.stdout
            
            # Check for Xid errors in dmesg (last 100 lines)
            xid_errors = self._check_xid_errors()
            if xid_errors:
                return GPUHealth.XID_ERROR
            
            # Check for throttling
            if 'Throttle' in output and 'Active' in output:
                return GPUHealth.DEGRADED
            
            return GPUHealth.HEALTHY
        
        except subprocess.TimeoutExpired:
            return GPUHealth.HUNG
        except Exception:
            return GPUHealth.UNKNOWN
    
    def _check_xid_errors(self) -> list[str]:
        """Check dmesg for recent Xid errors."""
        try:
            result = subprocess.run(
                ['dmesg', '-T', '-l', 'err,crit,alert,emerg'],
                capture_output=True, text=True, timeout=10
            )
            errors = []
            for line in result.stdout.split('\n'):
                if 'Xid' in line and ('GPU' in line or 'gpu' in line):
                    errors.append(line.strip())
            return errors[-10:]  # Last 10
        except Exception:
            return []
    
    def attempt_recovery(self, action: RecoveryAction) -> RecoveryResult:
        """Attempt a specific recovery action."""
        start = time.time()
        
        try:
            if action == RecoveryAction.SOFT_RESET:
                return self._soft_reset()
            elif action == RecoveryAction.HARD_RESET:
                return self._hard_reset()
            elif action == RecoveryAction.DRIVER_RELOAD:
                return self._driver_reload()
            elif action == RecoveryAction.SYSTEM_REBOOT:
                return self._system_reboot()
            else:
                return RecoveryResult(
                    success=False,
                    action=action,
                    message="Unknown recovery action",
                    duration_sec=time.time() - start,
                )
        except Exception as e:
            return RecoveryResult(
                success=False,
                action=action,
                message=f"Recovery failed with exception: {e}",
                duration_sec=time.time() - start,
            )
    
    def _soft_reset(self) -> RecoveryResult:
        """Soft reset: nvidia-smi -r (reset GPU)."""
        start = time.time()
        try:
            result = subprocess.run(
                ['nvidia-smi', '-i', str(self.gpu_id), '-r'],
                capture_output=True, text=True, timeout=30
            )
            time.sleep(3)  # Wait for reset
            
            if result.returncode == 0:
                return RecoveryResult(
                    success=True,
                    action=RecoveryAction.SOFT_RESET,
                    message="Soft reset successful",
                    duration_sec=time.time() - start,
                    gpu_state_after=self.check_gpu_health().value,
                )
            else:
                return RecoveryResult(
                    success=False,
                    action=RecoveryAction.SOFT_RESET,
                    message=f"Soft reset failed: {result.stderr}",
                    duration_sec=time.time() - start,
                )
        except Exception as e:
            return RecoveryResult(
                success=False,
                action=RecoveryAction.SOFT_RESET,
                message=f"Soft reset exception: {e}",
                duration_sec=time.time() - start,
            )
    
    def _hard_reset(self) -> RecoveryResult:
        """Hard reset: nvidia-smi --gpu-reset."""
        start = time.time()
        try:
            result = subprocess.run(
                ['nvidia-smi', '-i', str(self.gpu_id), '--gpu-reset'],
                capture_output=True, text=True, timeout=60
            )
            time.sleep(5)  # Wait for reset
            
            if result.returncode == 0:
                return RecoveryResult(
                    success=True,
                    action=RecoveryAction.HARD_RESET,
                    message="Hard reset successful",
                    duration_sec=time.time() - start,
                    gpu_state_after=self.check_gpu_health().value,
                )
            else:
                return RecoveryResult(
                    success=False,
                    action=RecoveryAction.HARD_RESET,
                    message=f"Hard reset failed: {result.stderr}",
                    duration_sec=time.time() - start,
                )
        except Exception as e:
            return RecoveryResult(
                success=False,
                action=RecoveryAction.HARD_RESET,
                message=f"Hard reset exception: {e}",
                duration_sec=time.time() - start,
            )
    
    def _driver_reload(self) -> RecoveryResult:
        """Reload NVIDIA driver: modprobe -r nvidia && modprobe nvidia."""
        start = time.time()
        try:
            # This requires root/sudo
            result = subprocess.run(
                ['sudo', 'modprobe', '-r', 'nvidia_uvm', 'nvidia_drm', 'nvidia_modeset', 'nvidia'],
                capture_output=True, text=True, timeout=30
            )
            time.sleep(2)
            
            result = subprocess.run(
                ['sudo', 'modprobe', 'nvidia'],
                capture_output=True, text=True, timeout=30
            )
            time.sleep(3)
            
            if result.returncode == 0:
                return RecoveryResult(
                    success=True,
                    action=RecoveryAction.DRIVER_RELOAD,
                    message="Driver reload successful",
                    duration_sec=time.time() - start,
                    gpu_state_after=self.check_gpu_health().value,
                )
            else:
                return RecoveryResult(
                    success=False,
                    action=RecoveryAction.DRIVER_RELOAD,
                    message=f"Driver reload failed: {result.stderr}",
                    duration_sec=time.time() - start,
                )
        except Exception as e:
            return RecoveryResult(
                success=False,
                action=RecoveryAction.DRIVER_RELOAD,
                message=f"Driver reload exception: {e}",
                duration_sec=time.time() - start,
            )
    
    def _system_reboot(self) -> RecoveryResult:
        """Request system reboot (last resort)."""
        start = time.time()
        try:
            # Log the need for reboot
            logger.critical("GPU recovery exhausted - requesting system reboot")
            
            # Write reboot flag for orchestrator
            reboot_flag = self.checkpoint_dir / "REBOOT_REQUIRED"
            reboot_flag.write_text(f"GPU {self.gpu_id} requires reboot at {time.time()}")
            
            # Actually reboot (requires sudo)
            # subprocess.run(['sudo', 'reboot'], check=True)
            
            return RecoveryResult(
                success=False,  # Reboot means process ends
                action=RecoveryAction.SYSTEM_REBOOT,
                message="System reboot required - flag written",
                duration_sec=time.time() - start,
            )
        except Exception as e:
            return RecoveryResult(
                success=False,
                action=RecoveryAction.SYSTEM_REBOOT,
                message=f"Reboot request failed: {e}",
                duration_sec=time.time() - start,
            )
    
    def auto_recover(self) -> RecoveryResult:
        """Automatically attempt recovery in escalating order."""
        health = self.check_gpu_health()
        logger.warning(f"GPU {self.gpu_id} health: {health.value}")
        
        if health == GPUHealth.HEALTHY:
            return RecoveryResult(
                success=True,
                action=RecoveryAction.NONE,
                message="GPU healthy, no recovery needed",
                duration_sec=0,
                gpu_state_after=health.value,
            )
        
        # Escalation order based on health state
        if health == GPUHealth.DEGRADED:
            actions = [RecoveryAction.SOFT_RESET, RecoveryAction.HARD_RESET, 
                       RecoveryAction.DRIVER_RELOAD, RecoveryAction.SYSTEM_REBOOT]
        elif health == GPUHealth.HUNG:
            actions = [RecoveryAction.SOFT_RESET, RecoveryAction.HARD_RESET,
                       RecoveryAction.DRIVER_RELOAD, RecoveryAction.SYSTEM_REBOOT]
        elif health == GPUHealth.XID_ERROR:
            actions = [RecoveryAction.HARD_RESET, RecoveryAction.DRIVER_RELOAD,
                       RecoveryAction.SYSTEM_REBOOT]
        elif health == GPUHealth.OFF_BUS:
            actions = [RecoveryAction.DRIVER_RELOAD, RecoveryAction.SYSTEM_REBOOT]
        else:
            actions = [RecoveryAction.SOFT_RESET, RecoveryAction.HARD_RESET,
                       RecoveryAction.DRIVER_RELOAD, RecoveryAction.SYSTEM_REBOOT]
        
        for action in actions:
            logger.info(f"Attempting recovery: {action.value}")
            result = self.attempt_recovery(action)
            
            if result.success:
                logger.info(f"Recovery successful: {action.value}")
                return result
            else:
                logger.warning(f"Recovery failed: {action.value} - {result.message}")
        
        # All failed
        return RecoveryResult(
            success=False,
            action=RecoveryAction.SYSTEM_REBOOT,
            message="All recovery attempts failed",
            duration_sec=0,
        )
    
    def save_checkpoint(self, example_id: str, state: dict):
        """Save pipeline checkpoint."""
        import json
        checkpoint = {
            'example_id': example_id,
            'timestamp': time.time(),
            'state': state,
        }
        checkpoint_file = self.checkpoint_dir / f"checkpoint_{example_id}.json"
        checkpoint_file.write_text(json.dumps(checkpoint))
        self.last_checkpoint = example_id
        logger.info(f"Checkpoint saved: {example_id}")
    
    def load_latest_checkpoint(self) -> Optional[dict]:
        """Load the most recent checkpoint."""
        import json
        checkpoints = sorted(self.checkpoint_dir.glob("checkpoint_*.json"))
        if not checkpoints:
            return None
        
        latest = checkpoints[-1]
        checkpoint = json.loads(latest.read_text())
        self.last_checkpoint = checkpoint['example_id']
        logger.info(f"Loaded checkpoint: {self.last_checkpoint}")
        return checkpoint


def create_recovery_daemon(gpu_id: int = 0, check_interval: int = 30) -> GPURecovery:
    """Create and configure a recovery daemon."""
    recovery = GPURecovery(gpu_id)
    return recovery


if __name__ == '__main__':
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        recovery = GPURecovery()
        
        if cmd == 'health':
            health = recovery.check_gpu_health()
            print(f"GPU health: {health.value}")
            if health == GPUHealth.XID_ERROR:
                print(f"Xid errors: {recovery._check_xid_errors()}")
        
        elif cmd == 'recover':
            result = recovery.auto_recover()
            print(f"Recovery result: {result.success}, action: {result.action.value}, msg: {result.message}")
        
        elif cmd == 'soft':
            result = recovery.attempt_recovery(RecoveryAction.SOFT_RESET)
            print(f"Soft reset: {result.success}, msg: {result.message}")
        
        elif cmd == 'hard':
            result = recovery.attempt_recovery(RecoveryAction.HARD_RESET)
            print(f"Hard reset: {result.success}, msg: {result.message}")
        
        else:
            print("Usage: python gpu_recovery.py [health|recover|soft|hard]")
    else:
        print("Usage: python gpu_recovery.py [health|recover|soft|hard]")