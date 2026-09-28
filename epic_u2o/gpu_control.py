#!/usr/bin/env python3
"""GPU clock-locking and thermal-stability controls for reproducible measurements.

Implements nvidia-smi clock locking, power limit management, persistence mode,
and thermal monitoring with automatic cooldown.
"""

import subprocess
import time
import threading
import os
from dataclasses import dataclass
from typing import Optional
from contextlib import contextmanager


@dataclass
class GPUState:
    """Current GPU state snapshot."""
    gpu_id: int
    timestamp: float
    temperature_c: int
    power_draw_w: float
    power_limit_w: float
    clock_graphics_mhz: int
    clock_memory_mhz: int
    clock_sm_mhz: int
    throttle_reasons: list[str]
    persistence_mode: bool


class GPUController:
    """Controls GPU clocks, power limits, and thermal state."""
    
    def __init__(self, gpu_id: int = 0):
        self.gpu_id = gpu_id
        self._original_state: Optional[GPUState] = None
        self._locked = False
    
    def get_state(self) -> GPUState:
        """Query current GPU state via nvidia-smi."""
        try:
            # Query all relevant metrics
            queries = [
                'temperature.gpu',
                'power.draw',
                'power.limit',
                'clocks.gr',
                'clocks.mem',
                'clocks.sm',
                'clocks.throttle_reasons',
                'persistence_mode',
            ]
            
            cmd = [
                'nvidia-smi', '-i', str(self.gpu_id),
                '--query-gpu=' + ','.join(queries),
                '--format=csv,noheader,nounits'
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                raise RuntimeError(f"nvidia-smi failed: {result.stderr}")
            
            values = [v.strip() for v in result.stdout.strip().split(',')]
            
            return GPUState(
                gpu_id=self.gpu_id,
                timestamp=time.time(),
                temperature_c=int(values[0]) if values[0] != '[Not Supported]' else 0,
                power_draw_w=float(values[1]) if values[1] != '[Not Supported]' else 0.0,
                power_limit_w=float(values[2]) if values[2] != '[Not Supported]' else 0.0,
                clock_graphics_mhz=int(values[3]) if values[3] != '[Not Supported]' else 0,
                clock_memory_mhz=int(values[4]) if values[4] != '[Not Supported]' else 0,
                clock_sm_mhz=int(values[5]) if values[5] != '[Not Supported]' else 0,
                throttle_reasons=values[6].split(';') if values[6] != '[Not Supported]' else [],
                persistence_mode=values[7].strip().lower() == 'enabled',
            )
        except Exception as e:
            # Return default state on error
            return GPUState(
                gpu_id=self.gpu_id,
                timestamp=time.time(),
                temperature_c=0,
                power_draw_w=0.0,
                power_limit_w=0.0,
                clock_graphics_mhz=0,
                clock_memory_mhz=0,
                clock_sm_mhz=0,
                throttle_reasons=[],
                persistence_mode=False,
            )
    
    def enable_persistence_mode(self) -> bool:
        """Enable persistence mode to keep driver loaded."""
        try:
            subprocess.run(
                ['nvidia-smi', '-i', str(self.gpu_id), '-pm', '1'],
                capture_output=True, timeout=10, check=True
            )
            return True
        except subprocess.CalledProcessError:
            return False
    
    def disable_persistence_mode(self) -> bool:
        """Disable persistence mode."""
        try:
            subprocess.run(
                ['nvidia-smi', '-i', str(self.gpu_id), '-pm', '0'],
                capture_output=True, timeout=10, check=True
            )
            return True
        except subprocess.CalledProcessError:
            return False
    
    def set_power_limit(self, watts: int) -> bool:
        """Set GPU power limit in watts."""
        try:
            subprocess.run(
                ['nvidia-smi', '-i', str(self.gpu_id), '-pl', str(watts)],
                capture_output=True, timeout=10, check=True
            )
            return True
        except subprocess.CalledProcessError:
            return False
    
    def get_max_power_limit(self) -> int:
        """Get maximum supported power limit."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '-i', str(self.gpu_id), '-q', '-d', 'POWER'],
                capture_output=True, text=True, timeout=10
            )
            for line in result.stdout.split('\n'):
                if 'Max Power Limit' in line:
                    return int(line.split(':')[1].strip().split()[0])
        except Exception:
            pass
        return 600  # Default for RTX 5090
    
    def lock_clocks(self, graphics_mhz: Optional[int] = None, memory_mhz: Optional[int] = None) -> bool:
        """Lock GPU clocks to specific frequencies.
        
        If frequencies not provided, locks to maximum stable clocks.
        Use 0 to unlock.
        """
        try:
            if graphics_mhz is None or memory_mhz is None:
                # Get max clocks
                result = subprocess.run(
                    ['nvidia-smi', '-i', str(self.gpu_id), '-q', '-d', 'CLOCK'],
                    capture_output=True, text=True, timeout=10
                )
                for line in result.stdout.split('\n'):
                    if 'Graphics' in line and 'Max' in line and graphics_mhz is None:
                        graphics_mhz = int(line.split(':')[1].strip().split()[0])
                    if 'Memory' in line and 'Max' in line and memory_mhz is None:
                        memory_mhz = int(line.split(':')[1].strip().split()[0])
                
                if graphics_mhz is None:
                    graphics_mhz = 0
                if memory_mhz is None:
                    memory_mhz = 0
            
            # Lock clocks (0 means unlock)
            if graphics_mhz == 0 and memory_mhz == 0:
                cmd = ['nvidia-smi', '-i', str(self.gpu_id), '-rgc']
            elif graphics_mhz == 0:
                cmd = ['nvidia-smi', '-i', str(self.gpu_id), '-lgc', f'{memory_mhz},{memory_mhz}']
            elif memory_mhz == 0:
                cmd = ['nvidia-smi', '-i', str(self.gpu_id), '-lgc', f'{graphics_mhz},{graphics_mhz}']
            else:
                cmd = ['nvidia-smi', '-i', str(self.gpu_id), '-lgc', f'{graphics_mhz},{memory_mhz}']
            
            subprocess.run(cmd, capture_output=True, timeout=10, check=True)
            self._locked = True
            return True
        except subprocess.CalledProcessError as e:
            print(f"Failed to lock clocks: {e}")
            return False
    
    def unlock_clocks(self) -> bool:
        """Unlock GPU clocks (restore dynamic boosting)."""
        try:
            subprocess.run(
                ['nvidia-smi', '-i', str(self.gpu_id), '-rgc'],
                capture_output=True, timeout=10, check=True
            )
            self._locked = False
            return True
        except subprocess.CalledProcessError:
            return False
    
    def reset_clocks(self) -> bool:
        """Reset clocks to default."""
        return self.unlock_clocks()
    
    def save_state(self):
        """Save current GPU state for later restoration."""
        self._original_state = self.get_state()
    
    def restore_state(self) -> bool:
        """Restore GPU to saved state."""
        if not self._original_state:
            return self.reset_clocks()
        
        ok = True
        if self._original_state.persistence_mode:
            ok &= self.enable_persistence_mode()
        else:
            ok &= self.disable_persistence_mode()
        
        ok &= self.set_power_limit(int(self._original_state.power_limit_w))
        ok &= self.unlock_clocks()
        
        return ok
    
    def is_throttling(self, state: Optional[GPUState] = None) -> bool:
        """Check if GPU is throttling."""
        if state is None:
            state = self.get_state()
        
        throttle_indicators = [
            'HW Thermal Slowdown',
            'HW Power Brake Slowdown',
            'SW Thermal Slowdown',
            'SW Power Cap',
            'Sync Boost',
        ]
        
        for reason in state.throttle_reasons:
            if any(indicator in reason for indicator in throttle_indicators):
                return True
        return False
    
    def get_throttle_reasons(self, state: Optional[GPUState] = None) -> list[str]:
        """Get active throttle reasons."""
        if state is None:
            state = self.get_state()
        return state.throttle_reasons


class ThermalMonitor:
    """Monitors GPU temperature and triggers cooldown if needed."""
    
    def __init__(
        self,
        controller: GPUController,
        temp_limit_c: int = 83,  # RTX 5090 throttle temp
        cooldown_temp_c: int = 70,
        check_interval_sec: float = 2.0,
    ):
        self.controller = controller
        self.temp_limit_c = temp_limit_c
        self.cooldown_temp_c = cooldown_temp_c
        self.check_interval = check_interval_sec
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self.throttle_count = 0
        self.max_temp_seen = 0
        self.callback = None
    
    def start(self):
        """Start monitoring in background thread."""
        self._running = True
        self._thread = threading.Thread(target=self._monitor, daemon=True)
        self._thread.start()
    
    def stop(self):
        """Stop monitoring."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
    
    def _monitor(self):
        while self._running:
            state = self.controller.get_state()
            self.max_temp_seen = max(self.max_temp_seen, state.temperature_c)
            
            if self.controller.is_throttling(state):
                self.throttle_count += 1
                if self.callback:
                    self.callback('throttle', state)
            
            if state.temperature_c >= self.temp_limit_c:
                if self.callback:
                    self.callback('temp_limit', state)
                # Trigger cooldown
                self._cooldown()
            
            time.sleep(self.check_interval)
    
    def _cooldown(self):
        """Pause and wait for GPU to cool down."""
        if self.callback:
            self.callback('cooldown_start', self.controller.get_state())
        
        # Unlock clocks to reduce power
        self.controller.unlock_clocks()
        
        while self._running:
            state = self.controller.get_state()
            if state.temperature_c <= self.cooldown_temp_c:
                break
            time.sleep(5.0)  # Check every 5 seconds during cooldown
        
        # Re-lock clocks after cooldown
        self.controller.lock_clocks()
        
        if self.callback:
            self.callback('cooldown_end', self.controller.get_state())
    
    def get_stats(self) -> dict:
        """Get monitoring statistics."""
        return {
            'throttle_count': self.throttle_count,
            'max_temp_c': self.max_temp_seen,
            'current_temp_c': self.controller.get_state().temperature_c,
        }


@contextmanager
def measurement_session(
    gpu_id: int = 0,
    power_limit_w: Optional[int] = None,
    lock_clocks: bool = True,
    persistence_mode: bool = True,
    temp_limit_c: int = 83,
):
    """Context manager for a full measurement session with GPU controls.
    
    Usage:
        with measurement_session(gpu_id=0, power_limit_w=600) as ctrl:
            # Run measurements
            state = ctrl.get_state()
            ...
    """
    controller = GPUController(gpu_id)
    controller.save_state()
    
    try:
        if persistence_mode:
            controller.enable_persistence_mode()
        
        if power_limit_w:
            controller.set_power_limit(power_limit_w)
        else:
            # Set to max
            controller.set_power_limit(controller.get_max_power_limit())
        
        if lock_clocks:
            controller.lock_clocks()
        
        monitor = ThermalMonitor(controller, temp_limit_c=temp_limit_c)
        monitor.start()
        
        yield controller
        
    finally:
        monitor.stop()
        controller.restore_state()


def prepare_measurement_environment(gpu_id: int = 0) -> GPUController:
    """Prepare GPU for measurement run - lock clocks, set power limit, enable persistence."""
    controller = GPUController(gpu_id)
    controller.save_state()
    
    # Enable persistence mode
    controller.enable_persistence_mode()
    
    # Set power limit to maximum
    max_power = controller.get_max_power_limit()
    controller.set_power_limit(max_power)
    
    # Lock clocks to maximum stable
    controller.lock_clocks()
    
    return controller


def cleanup_measurement_environment(controller: GPUController):
    """Restore GPU to original state after measurement run."""
    controller.restore_state()


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'status':
        ctrl = GPUController()
        state = ctrl.get_state()
        print(f"GPU {state.gpu_id} State:")
        print(f"  Temperature: {state.temperature_c}°C")
        print(f"  Power: {state.power_draw_w:.1f}W / {state.power_limit_w:.1f}W")
        print(f"  Clocks: Graphics={state.clock_graphics_mhz}MHz, Memory={state.clock_memory_mhz}MHz, SM={state.clock_sm_mhz}MHz")
        print(f"  Throttle reasons: {state.throttle_reasons}")
        print(f"  Persistence mode: {state.persistence_mode}")
    
    elif len(sys.argv) > 1 and sys.argv[1] == 'lock':
        ctrl = GPUController()
        ctrl.lock_clocks()
        print("Clocks locked to maximum")
    
    elif len(sys.argv) > 1 and sys.argv[1] == 'unlock':
        ctrl = GPUController()
        ctrl.unlock_clocks()
        print("Clocks unlocked")
    
    elif len(sys.argv) > 1 and sys.argv[1] == 'prepare':
        ctrl = prepare_measurement_environment()
        print("Measurement environment prepared")
        state = ctrl.get_state()
        print(f"  Power limit: {state.power_limit_w}W")
        print(f"  Clocks locked: Graphics={state.clock_graphics_mhz}MHz, Memory={state.clock_memory_mhz}MHz")
    
    elif len(sys.argv) > 1 and sys.argv[1] == 'cleanup':
        ctrl = GPUController()
        cleanup_measurement_environment(ctrl)
        print("Measurement environment cleaned up")
    
    else:
        print("Usage: python gpu_control.py [status|lock|unlock|prepare|cleanup]")