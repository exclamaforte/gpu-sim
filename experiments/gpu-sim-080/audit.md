# gpu-sim-080: propagation audit of canonical_byte_table.md citations

## Summary

`canonical_byte_table.md` was ruled REFERENCE-UNVERIFIED (gpu-sim-4yw, PI ruling 7bb3764).
Its byte-factor column contradicts its own cited traces on every row. eut re-measured
and found FACTOR-1.0-ABOVE-512KiB (per-rank byte factor 1.00 at >=512KiB, 2 runs each,
both ranks agreeing; below 512KiB UNMEASURED).

This audit enumerates every committed artifact that cites canonical_byte_table.md,
states what number it borrows, and eut's ruling (CONFIRMED / CHANGED / UNDECIDED).

## Citation inventory

### 1. experiments/gpu-sim-m3d.10/gate_artifact.md (line 186 pointer + duplicated table lines 104-112)

**Numbers borrowed:** byte factor column: 2.0 (2S) at 4KiB/64KiB/256KiB, 1.0 (S) at 512KiB/1MiB/2MiB.
Also: "byte factor 1.0 per direction; 2.0 total across both directions" (line 114).

**eut ruling:**
- >=512KiB: CONFIRMED (1.00 at 512KiB/1MiB/2MiB, 2 runs each, both ranks agreeing)
- <512KiB: UNDECIDED (4KiB ranks disagree 2.00/1.00; 64-384KiB invalid under exact-multiple gate)

**Action taken:** gate_artifact.md table annotated. Byte factor at >=512KiB marked CONFIRMED.
Byte factor at <512KiB marked UNDECIDED (not 2.0). "2.0 total across both directions"
qualified as model assumption, not measured (model now uses 1.0 at n=2 per ggh).

### 2. experiments/gpu-sim-pkb/test_pkb.py (lines 98-99)

**Numbers borrowed:** per-phase pieces (P) column: 1, 4, 4, 8, 8, 8.

**eut ruling:** P is CONFIRMED by phase_payload_table.md (measured, distinct from
canonical_byte_table). eut did not re-measure P; phase_payload_table.md (m3d.8 net
plugin) is the authoritative source. The P values in canonical_byte_table.md happen
to match phase_payload_table.md.

**Action taken:** test_pkb.py re-grounded to parse P from phase_payload_table.md
instead of canonical_byte_table.md. The reference is now the measured source.

### 3. experiments/gpu-sim-pkb/pre_registration.md (line 7)

**Numbers borrowed:** none specific; cites canonical_byte_table as "agrees with its own cited traces."

**eut ruling:** CHANGED. canonical_byte_table does NOT agree with its own traces (4yw).

**Action taken:** none. Pre-registration is a historical record of what was planned;
editing it would falsify the record. The 4yw ruling is cited in the bead's results.

### 4. experiments/gpu-sim-4yw/PI_check_reference_table.md, leg1.py, pre_registration.md, results.md

**Numbers borrowed:** various (these artifacts DOCUMENT the reference failure).

**eut ruling:** N/A (these are the investigation artifacts, not consumers of the table).

**Action taken:** none. Historical records.

### 5. REPORT.md (experiments/gpu-sim-m3d/REPORT.md)

**Numbers borrowed:** none from canonical_byte_table directly. REPORT.md Limit 1
discusses timing (SMT sibling contention), not byte factor. Limit 3 mentions
per-isend accounting (8nt, deferred).

**eut ruling:** N/A (no byte-table numbers in REPORT.md).

**Action taken:** none. Checked: REPORT.md sections 4-5 do not rest on byte-table numbers.

## Conclusions

- P values (1,4,4,8,8,8): CONFIRMED by phase_payload_table.md (measured). Not affected by
  canonical_byte_table failure. test_pkb.py re-grounded to cite phase_payload_table.md.
- Byte factor >=512KiB (1.0): CONFIRMED by eut.
- Byte factor <512KiB (old 2.0): UNDECIDED. Model uses 1.0 at n=2 (ggh) for all sizes,
  but eut only confirms 1.0 at >=512KiB. Below 512KiB is UNMEASURED.
- gate_artifact.md duplicated table: annotated with eut rulings.
