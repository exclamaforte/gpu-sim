#!/usr/bin/env bash
# fake_sysfs.sh <rank> <outdir> -- build the /sys/class/infiniband tree NCCL's
# IB path needs, with that rank's HCA sitting behind that rank's GPU bridge.
#
# Read sysfs_spoof.c first: NCCL reads exactly
#   /sys/class/infiniband/<dev>/device   (realpath -> a PCI device directory)
# and truncates the last character of the result to merge the ports of one PCI
# device.  So `device` must resolve to a REAL path, and it should be a bridge
# the rank's GPU also lives under, or the topology says the NIC is far from the
# GPU and NCCL reports no usable path.
#
# The GPU is found from sysfs alone (class 0x030000 = VGA, 0x030200 = 3D), NOT
# from CUDA: this script has to run on the CPU-only side of the lab, and a
# CUDA call here would need the GPU queue to answer a question sysfs can.
#
# Ranks are NOT given the same bridge: two ranks sharing a PCI path are merged
# into one node by NCCL's cross-rank topology merge, which is the opposite of
# what we are building.
set -euo pipefail

rank=${1:?usage: fake_sysfs.sh <rank> <outdir>}
out=${2:?usage: fake_sysfs.sh <rank> <outdir>}
dev=${3:-mlx5_0}

mapfile -t gpus < <(for d in /sys/bus/pci/devices/*; do
    c=$(cat "$d/class" 2>/dev/null || true)
    case "$c" in 0x030000|0x030200) echo "$(basename "$d")";; esac
done | sort)

if (( ${#gpus[@]} == 0 )); then
    # No display/3D device: keep the tree buildable so the CPU stages can run,
    # and say so instead of silently pointing at something odd.
    gpu=""
    bridge=$(realpath -e /sys/bus/pci/devices/"$(ls /sys/bus/pci/devices | head -1)"/.. )
    note="no PCI display device on this host; NIC pinned to $bridge"
else
    gpu=${gpus[$(( rank % ${#gpus[@]} ))]}
    bridge=$(realpath -e "/sys/bus/pci/devices/$gpu/..")
    note="gpu $gpu shares bridge $bridge with $dev"
fi

mkdir -p "$out/infiniband/$dev/ports/1/gids" "$out/infiniband/$dev/ports/1/gid_attrs/ndevs"
ln -sfn "$bridge" "$out/infiniband/$dev/device"
# Not read by the IB path (only by the RoCE gid-type lookup), but a tree with
# only one leaf invites the question "is the spoof incomplete or is NCCL?".
printf '0x%04x' $(( 0x248a + rank )) > "$out/infiniband/$dev/node_guid"
printf '0x%04x' $(( 0x248a + rank )) > "$out/infiniband/$dev/node_type"
printf '4: ACTIVE\n' > "$out/infiniband/$dev/ports/1/state"
printf '400\n' > "$out/infiniband/$dev/ports/1/rate"
printf 'fe80:0000:0000:0000:0000:0000:0000:%04x\n' $(( rank + 1 )) \
    > "$out/infiniband/$dev/ports/1/gids/0"
printf 'eth%d\n' "$rank" > "$out/infiniband/$dev/ports/1/gid_attrs/ndevs/0"

cat > "$out/manifest.json" <<JSON
{"rank": $rank, "device": "$dev", "device_symlink": "$bridge",
 "gpu_bdf": "${gpu:-null}", "note": "$note"}
JSON
echo "fake sysfs rank $rank: $out/infiniband/$dev/device -> $bridge ($note)"
