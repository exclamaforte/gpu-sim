# gpu-sim-w4v: bytes-keyed dtype generality
op_latency supports dtype_bytes (default 4) and indexes by bytes. sim_trace_173.py supports --dtype-bytes. Replay of 84z fp16: median 0.0199, max 0.0675 (all pass). See replay_metrics.json.
