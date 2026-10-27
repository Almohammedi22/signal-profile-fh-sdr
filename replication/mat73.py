"""Minimal read-only reader for MATLAB v7.3 (HDF5, superblock v0) numeric variables.

Supports what MATLAB writes for plain numeric arrays: symbol-table groups (B-tree v1 + local heap),
object header v1 with continuation blocks, dataspace v1/v2, fixed-point and floating-point types,
data layout v3 (contiguous, compact, chunked) and deflate/shuffle filters. Arrays are returned in
MATLAB orientation (HDF5 dimensions reversed). Anything else raises NotImplementedError.
"""
import struct, zlib
import numpy as np


class Mat73:
    def __init__(self, path):
        self.f = open(path, "rb")
        self.base = self._find_base()
        sb = self._read(self.base, 96)
        assert sb[8] == 0, "only superblock version 0 is supported"
        ent = sb[56:96]
        _, self.root_oh = struct.unpack("<QQ", ent[:16])

    # ---------------------------------------------------------------- basics
    def _find_base(self):
        for off in (0, 512, 1024, 2048):
            self.f.seek(off)
            if self.f.read(8) == b"\x89HDF\r\n\x1a\n":
                return off
        raise ValueError("no HDF5 signature")

    def _read(self, addr, n):
        self.f.seek(addr); return self.f.read(n)

    def _abs(self, rel):
        return self.base + rel

    # ---------------------------------------------------------------- object headers
    def _messages(self, oh_rel):
        hdr = self._read(self._abs(oh_rel), 16)
        ver, _, nmsg, _, hsize = struct.unpack("<BBHII", hdr[:12])
        assert ver == 1, f"object header version {ver} not supported"
        blocks, msgs = [(self._abs(oh_rel) + 16, hsize)], []
        while blocks:
            start, size = blocks.pop(0)
            data = self._read(start, size); pos = 0
            while pos + 8 <= len(data) and len(msgs) < nmsg:
                mtype, msize, flags = struct.unpack("<HHB", data[pos:pos + 5])
                body = data[pos + 8:pos + 8 + msize]
                if mtype == 0x10:                                   # continuation
                    caddr, clen = struct.unpack("<QQ", body[:16]); blocks.append((self._abs(caddr), clen))
                msgs.append((mtype, body)); pos += 8 + msize
        return msgs

    # ---------------------------------------------------------------- groups
    def _heap_name(self, heap_rel, off):
        h = self._read(self._abs(heap_rel), 32)
        assert h[:4] == b"HEAP"
        dseg = struct.unpack("<Q", h[24:32])[0]
        raw = self._read(self._abs(dseg) + off, 256)
        return raw.split(b"\x00", 1)[0].decode("latin-1")

    def _walk_btree(self, node_rel, heap_rel, out):
        n = self._read(self._abs(node_rel), 24)
        assert n[:4] == b"TREE" and n[4] == 0
        level, used = n[5], struct.unpack("<H", n[6:8])[0]
        body = self._read(self._abs(node_rel) + 24, (used * 2 + 1) * 8)
        children = [struct.unpack("<Q", body[8 + i * 16: 16 + i * 16])[0] for i in range(used)]
        for c in children:
            if level > 0:
                self._walk_btree(c, heap_rel, out)
            else:
                s = self._read(self._abs(c), 8)
                assert s[:4] == b"SNOD"
                nsym = struct.unpack("<H", s[6:8])[0]
                ents = self._read(self._abs(c) + 8, nsym * 40)
                for i in range(nsym):
                    lno, oh = struct.unpack("<QQ", ents[i * 40:i * 40 + 16])
                    out[self._heap_name(heap_rel, lno)] = oh

    def members(self, oh_rel=None):
        oh_rel = self.root_oh if oh_rel is None else oh_rel
        for mtype, body in self._messages(oh_rel):
            if mtype == 0x11:
                bt, heap = struct.unpack("<QQ", body[:16]); out = {}
                self._walk_btree(bt, heap, out); return out
        raise ValueError("not a group")

    # ---------------------------------------------------------------- datasets
    def info(self, name):
        oh = self.members()[name]
        d = {"attrs": {}}
        for mtype, b in self._messages(oh):
            if mtype == 0x01:                                       # dataspace
                ver, rank, flags = b[0], b[1], b[2]
                off = 8 if ver == 1 else 4
                d["shape"] = tuple(struct.unpack("<%dQ" % rank, b[off:off + 8 * rank]))
            elif mtype == 0x03:                                     # datatype
                cls, size = b[0] & 0x0F, struct.unpack("<I", b[4:8])[0]
                if cls == 1:
                    d["dtype"] = np.dtype("<f%d" % size)
                elif cls == 0:
                    signed = bool(b[1] & 0x08); d["dtype"] = np.dtype(("<i%d" if signed else "<u%d") % size)
                else:
                    d["dtype"] = f"class{cls}"
            elif mtype == 0x08:                                     # layout
                assert b[0] == 3, f"layout version {b[0]}"
                d["layout"] = b[1]
                if b[1] == 1:
                    d["addr"], d["size"] = struct.unpack("<QQ", b[2:18])
                elif b[1] == 2:
                    nd = b[2]; d["btree"] = struct.unpack("<Q", b[3:11])[0]
                    d["chunk"] = struct.unpack("<%dI" % nd, b[11:11 + 4 * nd])
                elif b[1] == 0:
                    sz = struct.unpack("<H", b[2:4])[0]; d["compact"] = b[4:4 + sz]
            elif mtype == 0x0B:                                     # filters
                ver, nf = b[0], b[1]; pos = 8 if ver == 1 else 2; fl = []
                for _ in range(nf):
                    fid, nlen, flg, ncv = struct.unpack("<HHHH", b[pos:pos + 8]); pos += 8
                    if ver == 1 or fid >= 256:
                        pos += (nlen + 7) // 8 * 8 if ver == 1 else nlen
                    pos += 4 * ncv + (4 if ver == 1 and ncv % 2 else 0); fl.append(fid)
                d["filters"] = fl
            elif mtype == 0x0C:                                     # attribute (MATLAB_class)
                try:
                    nlen = struct.unpack("<H", b[2:4])[0]; name_ = b[8:8 + nlen].split(b"\x00")[0].decode()
                    d["attrs"][name_] = b
                except Exception:
                    pass
        return d

    def read(self, name):
        d = self.info(name)
        if not isinstance(d.get("dtype"), np.dtype):
            raise NotImplementedError(f"{name}: unsupported type {d.get('dtype')}")
        dt, shape = d["dtype"], d["shape"]
        if d["layout"] == 1:
            arr = np.frombuffer(self._read(self._abs(d["addr"]), d["size"]), dtype=dt).reshape(shape)
        elif d["layout"] == 0:
            arr = np.frombuffer(d["compact"], dtype=dt).reshape(shape)
        else:
            arr = np.zeros(shape, dtype=dt)
            chunk = d["chunk"][:-1]
            self._read_chunks(d["btree"], len(shape), chunk, dt, d.get("filters", []), arr)
        return arr.T                                                # MATLAB orientation

    def _read_chunks(self, node_rel, rank, chunk, dt, filters, arr):
        n = self._read(self._abs(node_rel), 24)
        assert n[:4] == b"TREE" and n[4] == 1
        level, used = n[5], struct.unpack("<H", n[6:8])[0]
        ksz = 8 + 8 * (rank + 1)
        body = self._read(self._abs(node_rel) + 24, used * (ksz + 8) + ksz)
        for i in range(used):
            k = body[i * (ksz + 8): i * (ksz + 8) + ksz]
            csize, fmask = struct.unpack("<II", k[:8])
            offs = struct.unpack("<%dQ" % (rank + 1), k[8:8 + 8 * (rank + 1)])[:rank]
            child = struct.unpack("<Q", body[i * (ksz + 8) + ksz: i * (ksz + 8) + ksz + 8])[0]
            if level > 0:
                self._read_chunks(child, rank, chunk, dt, filters, arr); continue
            raw = self._read(self._abs(child), csize)
            for j, fid in reversed(list(enumerate(filters))):
                if fmask & (1 << j):
                    continue
                if fid == 1:
                    raw = zlib.decompress(raw)
                elif fid == 2:
                    raw = np.frombuffer(raw, np.uint8).reshape(dt.itemsize, -1).T.tobytes()
                else:
                    raise NotImplementedError(f"filter {fid}")
            block = np.frombuffer(raw, dtype=dt)[:int(np.prod(chunk))].reshape(chunk)
            sl = tuple(slice(o, min(o + c, s)) for o, c, s in zip(offs, chunk, arr.shape))
            arr[sl] = block[tuple(slice(0, x.stop - x.start) for x in sl)]


def _info_oh(self, oh):
    """info() for an object header address (used for MATLAB string references)."""
    members_backup = self.members
    self.members = lambda oh_rel=None: {"__x__": oh}
    try:
        return Mat73.info(self, "__x__")
    finally:
        self.members = members_backup


def strings(self, name):
    """Decode a MATLAB cell array of strings stored as object references."""
    d = self.info(name)
    raw = d["compact"] if d.get("layout") == 0 else None
    if raw is None:
        raise NotImplementedError("only compact reference arrays")
    refs = struct.unpack("<%dQ" % (len(raw) // 8), raw)
    out = []
    for r in refs:
        i = _info_oh(self, r)
        if i["layout"] == 1:
            a = np.frombuffer(self._read(self._abs(i["addr"]), i["size"]), dtype=i["dtype"])
        elif i["layout"] == 0:
            a = np.frombuffer(i["compact"], dtype=i["dtype"])
        else:
            a = np.zeros(i["shape"], dtype=i["dtype"]); self._read_chunks(i["btree"], len(i["shape"]), i["chunk"][:-1], i["dtype"], i.get("filters", []), a); a = a.ravel()
        out.append("".join(chr(c) for c in a.ravel()))
    return out


Mat73.strings = strings
