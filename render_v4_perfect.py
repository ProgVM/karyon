import gzip, base64, re
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# Load & Decompress Bloodbath
with open('/kaggle/input/datasets/mcpeorakul/geometry-dash/bloodbath.gdlvl', 'rb') as f:
    raw = f.read().strip()
raw += b'=' * ((4 - len(raw) % 4) % 4)
plist_text = gzip.decompress(base64.urlsafe_b64decode(raw)).decode('utf-8', errors='replace')
lvl_data_raw = re.search(r'<k>k4</k><s>(.*?)</s>', plist_text).group(1).encode('utf-8')
lvl_data_raw += b'=' * ((4 - len(lvl_data_raw) % 4) % 4)
lvl_decompressed = gzip.decompress(base64.urlsafe_b64decode(lvl_data_raw)).decode('utf-8', errors='replace')

raw_objs = lvl_decompressed.split(';')

# Categorization
SOLID_BLOCKS = {1, 2, 3, 4, 6, 7, 193, 211, 268, 368}
SPIKES_NORMAL = {8, 39, 103, 286, 392} # OID 8: big, 9/103: small, 39: medium
SPIKES_SMALL = {9, 103, 468}
ORBS_PORTALS = {36, 141, 1332, 10, 11, 12, 13, 47, 111, 660} # Orbs & portals

fig, ax = plt.subplots(figsize=(20, 6), dpi=300)
ax.set_facecolor('#0a0405') # Deep dark blood theme

parsed_cnt = 0

for o_str in raw_objs[1:]:
    if not o_str.strip(): continue
    kv = dict(re.findall(r'(\d+),([^,]+)', o_str))
    try:
        oid = int(kv.get('1', 0))
        x = float(kv.get('2', 0))
        y = float(kv.get('3', 0))
        rot_raw = float(kv.get('6', 0))
        flip_x = int(kv.get('4', 0)) == 1
        flip_y = int(kv.get('5', 0)) == 1
        scale = float(kv.get('32', 1.0))
        
        # Filter range X in [-50, 1150]
        if not (-50 <= x <= 1150 and -50 <= y <= 400):
            continue
            
        parsed_cnt += 1
        
        # Effective rotation formula for GD
        # Standard unrotated orientation: Spikes point UP (+Y)
        # Rot is clockwise rotation in degrees
        # FlipY mirrors across horizontal axis
        eff_rot = rot_raw
        if flip_y:
            eff_rot += 180.0
        eff_rot = eff_rot % 360.0
        
        # Radians for trigonometry
        rad = np.radians(-eff_rot) # Matplotlib rotates counter-clockwise, so -eff_rot
        
        # Render Solid Blocks
        if oid in SOLID_BLOCKS:
            w, h = 30.0 * scale, 30.0 * scale
            rect = patches.Rectangle((-w/2, -h/2), w, h, linewidth=1.2, edgecolor='#ff2233', facecolor='#330508', alpha=0.9)
            t = matplotlib.transforms.Affine2D().rotate_deg(-eff_rot).translate(x, y) + ax.transData
            rect.set_transform(t)
            ax.add_patch(rect)
            
        # Render Spikes
        elif oid in SPIKES_NORMAL or oid in SPIKES_SMALL:
            is_small = oid in SPIKES_SMALL
            w = 30.0 * scale if not is_small else 18.0 * scale
            h = 30.0 * scale if not is_small else 18.0 * scale
            
            # Base triangle pointing UP (+Y): base at -h/2, tip at +h/2
            pts = np.array([
                [-w/2, -h/2],
                [ w/2, -h/2],
                [ 0.0,  h/2]
            ])
            
            # Apply counter-clockwise rotation matching GD clockwise
            cos_a, sin_a = np.cos(rad), np.sin(rad)
            rot_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
            rotated_pts = pts @ rot_mat.T
            translated_pts = rotated_pts + np.array([x, y])
            
            poly = patches.Polygon(translated_pts, closed=True, 
                                   linewidth=1.0, 
                                   edgecolor='#ffaa00' if is_small else '#ffffff', 
                                   facecolor='#ff3300', 
                                   alpha=0.95)
            ax.add_patch(poly)
            
        # Render Orbs & Portals
        elif oid in ORBS_PORTALS:
            circle = patches.Circle((x, y), 12.0 * scale, linewidth=1.5, edgecolor='#00ffff', facecolor='#0088ff', alpha=0.7)
            ax.add_patch(circle)
            
        # Render Other Decor
        else:
            rect = patches.Rectangle((x-5, y-5), 10, 10, linewidth=0.5, edgecolor='#661122', facecolor='none', alpha=0.3)
            ax.add_patch(rect)
            
    except ValueError:
        continue

ax.set_xlim(-50, 1150)
ax.set_ylim(-10, 320)
ax.set_aspect('equal')
ax.grid(True, color='#220a10', linestyle='--', linewidth=0.5)
ax.set_title("Bloodbath — Fixed Spike Rotation & FlipY Matrix (GD True Geometry)", fontsize=14, color='white', fontweight='bold')
ax.set_xlabel("X Coordinate (pixels)", color='white')
ax.set_ylabel("Y Coordinate (pixels)", color='white')
ax.tick_params(colors='white')

plt.tight_layout()
plt.savefig('experiments/renders/bloodbath_precise_v4.png', dpi=300, bbox_inches='tight', facecolor='#0a0405')
print(f"Saved v4 render to experiments/renders/bloodbath_precise_v4.png (Objects: {parsed_cnt})")
