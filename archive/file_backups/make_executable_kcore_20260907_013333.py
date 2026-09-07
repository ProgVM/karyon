# make_executable_kcore.py
import os
import sys

LOADER_SCRIPT = r"""#!/usr/bin/env bash
_SELF="$0"
python3 - "$_SELF" "$@" << "KCORE_RUNNER"
import sys, os, struct, json, zlib, tempfile

container_path = sys.argv[1]
user_args = sys.argv[2:]

if not os.path.exists(container_path):
    if os.path.exists(os.path.basename(container_path)):
        container_path = os.path.basename(container_path)

with open(container_path, "rb") as f:
    data = f.read()

# Dynamic search for binary payload (split signature to avoid matching script text)
sig_v5 = b"KC" + b"ORE" + bytes([5, 0, 0])
sig_v1 = b"KC" + b"ORE" + bytes([1, 0, 0])
magic_offset = data.rfind(sig_v5)
if magic_offset == -1:
    magic_offset = data.rfind(sig_v1)

if magic_offset == -1:
    print(f"[Error] KCORE binary payload signature not found in {container_path}!")
    sys.exit(1)

f = open(container_path, "rb")
f.seek(magic_offset)
magic = f.read(8)

header_raw = f.read(24)
header_size, num_sections, total_file_size, flags = struct.unpack("<IIQQ", header_raw)

sections = []
for _ in range(num_sections):
    sec_raw = f.read(64)
    s_type, s_flags, offset, size, align = struct.unpack("<IIQQQ", sec_raw[:32])
    s_name = sec_raw[32:].rstrip(b"\x00").decode("utf-8", errors="replace")
    sections.append({"type": s_type, "flags": s_flags, "offset": magic_offset + offset, "size": size, "name": s_name})

sec_logic = next(s for s in sections if s["type"] == 2)
f.seek(sec_logic["offset"])
logic_raw = f.read(sec_logic["size"])
if sec_logic["flags"] & 1:
    logic_raw = zlib.decompress(logic_raw)
logic = json.loads(logic_raw.decode("utf-8"))

with tempfile.TemporaryDirectory() as tmpdir:
    sys.path.insert(0, tmpdir)
    for filename, code in logic.items():
        filepath = os.path.join(tmpdir, filename)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as sf:
            sf.write(code)
            
    from karyon_entity import KaryonEntity
    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    entity = KaryonEntity.load(container_path, device=device)
    
    if len(user_args) > 0 and user_args[0] in ("sleep", "--sleep"):
        print("🌙 [Self-Executing KCORE] Entering deep allostatic sleep & morphogenesis...")
        pruned = entity.sleep()
        print(f"☀️ [Self-Executing KCORE] Sleep complete. Pruned {pruned} synapses.")
        entity.save(container_path)
    elif len(user_args) > 0 and user_args[0] in ("info", "--info"):
        print("=== KARYON-CORE SELF-EXECUTABLE CONTAINER ===")
        print(f"  Container Path : {container_path}")
        print(f"  Genome Params  : {len(list(entity.brain.parameters()))}")
        print(f"  Device Engine  : {device.upper()}")
        print(f"  Homeostatic u_t: {entity.hu.state[0].tolist()}")
    else:
        prompt = " ".join(user_args) if len(user_args) > 0 else "Hello Karyon, are you executing directly from your own container?"
        print(f"Human: {prompt}")
        print("Karyon: ", end="", flush=True)
        for event in entity.interact(prompt, max_tokens=60):
            if event["status"] == "token":
                print(event["text"], end="", flush=True)
        print()
        entity.save(container_path)
KCORE_RUNNER
exit $?
"""

def make_executable(filepath="karyon_soul.kcore"):
    with open(filepath, "rb") as f:
        data = f.read()

    # Search for real binary payload with rfind
    sig_v5 = b"KC" + b"ORE" + bytes([5, 0, 0])
    sig_v1 = b"KC" + b"ORE" + bytes([1, 0, 0])
    idx = data.rfind(sig_v5)
    if idx == -1:
        idx = data.rfind(sig_v1)
    if idx == -1:
        raise ValueError(f"No KCORE binary payload in {filepath}")

    payload = data[idx:]
    with open(filepath, "wb") as f:
        f.write(LOADER_SCRIPT.encode("utf-8"))
        f.write(payload)

    os.chmod(filepath, 0o755)
    print(f"Successfully converted '{filepath}' into a self-executing container!")

if __name__ == "__main__":
    make_executable()
