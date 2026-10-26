"""Crop floater gaussians from an INRIA 3DGS .ply, keeping the dense scene core.

3DGS reconstructions accumulate a sparse halo of far 'floater' gaussians (here ~1%
out to +/-240 units while the real scene is ~+/-4). Cropping them cleans the splat,
shrinks the file, and de-risks the Isaac Sim 5.1 NuRec far-from-origin artifact bug.

Keeps every vertex whose |x|,|y|,|z| are all <= half_extent (a box crop), preserves
all 62 columns, and rewrites a valid binary_little_endian .ply with the updated count.

Usage: python ply_crop.py IN.ply OUT.ply [half_extent=20]
"""
import re
import sys

import numpy as np


def main(inp, outp, half):
    with open(inp, "rb") as f:
        raw = f.read()
    end = raw.index(b"end_header\n") + len(b"end_header\n")
    header = raw[:end].decode("ascii", "replace")
    nverts = int(re.search(r"element vertex (\d+)", header).group(1))
    props = re.findall(r"property\s+\w+\s+(\w+)", header)
    nprops = len(props)

    data = np.frombuffer(raw, dtype="<f4", offset=end, count=nverts * nprops).reshape(nverts, nprops)
    xyz = data[:, :3]
    keep = (np.abs(xyz) <= half).all(axis=1)
    kept = data[keep]
    print("kept {} / {} ({:.2f}%); dropped {} floaters beyond +/-{} units".format(
        kept.shape[0], nverts, 100.0 * kept.shape[0] / nverts, nverts - kept.shape[0], half))

    kxyz = kept[:, :3].astype(np.float64)
    print("cropped bbox min:", np.round(kxyz.min(0), 3).tolist(), "max:", np.round(kxyz.max(0), 3).tolist())

    new_header = re.sub(r"element vertex \d+", "element vertex {}".format(kept.shape[0]), header)
    with open(outp, "wb") as f:
        f.write(new_header.encode("ascii"))
        f.write(np.ascontiguousarray(kept, dtype="<f4").tobytes())
    print("wrote", outp)


if __name__ == "__main__":
    inp, outp = sys.argv[1], sys.argv[2]
    half = float(sys.argv[3]) if len(sys.argv) > 3 else 20.0
    main(inp, outp, half)
