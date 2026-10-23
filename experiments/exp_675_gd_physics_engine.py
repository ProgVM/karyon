import math, base64, gzip, re

# === GD PHYSICAL CONSTANTS (240 Hz) ===
UNIT = 30.0               # px per block
PHYS_DT = 1.0 / 240.0     # 240Hz physics step
GROUND_Y = 0.0

SPEEDS = [8.372, 10.386, 12.914, 15.6, 19.2]
GRAV_CUBE = [91.387, 93.137, 93.040, 93.429, 93.429]
JUMP_CUBE = [19.116, 20.124, 20.556, 20.214, 20.214]

PHYS = {
    'cube': {'maxFall': 27.0, 'rotSpeed': 415.4},
    'ship': {
        'holdFall': 46.568, 'holdRise': 37.255,
        'relRise': 44.706, 'relFall': 29.804,
        'maxUp': 14.4, 'maxDown': 11.52, 'miniCapMult': 1.0 / 0.85
    },
    'ball': {'gravity': 55.882, 'maxFall': 27.0, 'clickPop': 6.037},
    'ufo': {'gravFall': 37.255, 'gravRise': 55.882, 'tap': 12.6, 'maxUp': 14.4, 'maxDown': 11.52},
    'wave': {'slope': 1.0, 'miniSlope': 2.0},
    'robot': {'initVel': 10.062, 'maxHold': 0.278, 'gravity': 83.823, 'maxFall': 27.0},
    'spider': {'gravity': 55.882, 'maxFall': 27.0},
    'mini': {'size': 0.6, 'jump': 0.8}
}

ORB_VEL = {
    'yellow': {'v': 20.124}, 'pink': {'v': 14.489}, 'red': {'v': 27.771},
    'green': {'v': 20.124, 'flip': True}, 'blue': {'v': -8.05, 'flip': True}, 'black': {'v': -27.0}
}
PAD_VEL = {
    'yellow': {'v': 28.8}, 'pink': {'v': 18.72}, 'red': {'v': 36.0}, 'blue': {'v': -11.52, 'flip': True}
}

PLAYER_SIZE = 1.0
PLAYER_INNER = 0.3
LAND_SNAP = 0.333
LAND_SNAP_SHIP = 0.2
FLIP_VEL_MULT = 0.5

CEILING_MODES = {'ship': True, 'wave': True, 'ball': True, 'spider': True, 'ufo': True}

# === OBJECT MAPPING DATABASE ===
# Mapping raw GD object IDs to physical types and hitboxes
OBJ_CATALOG = {
    # Blocks
    '1': {'type': 'block', 'solid': True, 'w': 1.0, 'h': 1.0},
    '2': {'type': 'block', 'solid': True, 'w': 1.0, 'h': 1.0},
    '3': {'type': 'block', 'solid': True, 'w': 1.0, 'h': 1.0},
    '4': {'type': 'block', 'solid': True, 'w': 1.0, 'h': 1.0},
    '5': {'type': 'block', 'solid': True, 'w': 1.0, 'h': 1.0},
    '6': {'type': 'block', 'solid': True, 'w': 1.0, 'h': 1.0},
    '7': {'type': 'block', 'solid': True, 'w': 1.0, 'h': 1.0},
    '237': {'type': 'block', 'solid': True, 'w': 1.0, 'h': 1.0},
    '503': {'type': 'block', 'solid': True, 'w': 1.0, 'h': 1.0},
    '504': {'type': 'block', 'solid': True, 'w': 1.0, 'h': 1.0},

    # Spikes & Hazards
    '8': {'type': 'spike', 'hazard': True, 'w': 0.2, 'h': 0.4},
    '39': {'type': 'spike_small', 'hazard': True, 'w': 0.2, 'h': 0.19, 'oy': -0.28},
    '103': {'type': 'spike_tiny', 'hazard': True, 'w': 0.2, 'h': 0.1},
    '94': {'type': 'sawblade', 'hazard': True, 'r': 0.44},
    '135': {'type': 'sawblade_big', 'hazard': True, 'r': 0.79},
    '193': {'type': 'spikeball', 'hazard': True, 'r': 0.4},

    # Portals
    '10': {'type': 'portal', 'act': 'gravUp', 'w': 0.83, 'h': 2.5},
    '11': {'type': 'portal', 'act': 'gravDown', 'w': 0.83, 'h': 2.5},
    '12': {'type': 'portal', 'act': 'cube', 'w': 1.13, 'h': 2.87},
    '13': {'type': 'portal', 'act': 'ship', 'w': 1.13, 'h': 2.87},
    '47': {'type': 'portal', 'act': 'ball', 'w': 1.13, 'h': 2.87},
    '111': {'type': 'portal', 'act': 'ufo', 'w': 0.9, 'h': 2.7},
    '660': {'type': 'portal', 'act': 'wave', 'w': 1.13, 'h': 2.87},
    '744': {'type': 'portal', 'act': 'robot', 'w': 1.13, 'h': 2.87},
    '1331': {'type': 'portal', 'act': 'spider', 'w': 1.13, 'h': 2.87},
    '99': {'type': 'portal', 'act': 'mini', 'w': 1.13, 'h': 2.87},
    '101': {'type': 'portal', 'act': 'big', 'w': 0.9, 'h': 2.7},

    # Speeds
    '200': {'type': 'speed', 'speedIdx': 0, 'w': 1.3, 'h': 1.0},
    '201': {'type': 'speed', 'speedIdx': 1, 'w': 1.3, 'h': 1.0},
    '202': {'type': 'speed', 'speedIdx': 2, 'w': 1.3, 'h': 1.0},
    '203': {'type': 'speed', 'speedIdx': 3, 'w': 1.3, 'h': 1.0},
    '1334': {'type': 'speed', 'speedIdx': 4, 'w': 1.3, 'h': 1.0},

    # Pads & Orbs
    '35': {'type': 'pad', 'pad': 'yellow', 'w': 0.83, 'h': 0.14, 'oy': -0.43},
    '140': {'type': 'pad', 'pad': 'pink', 'w': 0.83, 'h': 0.17, 'oy': -0.42},
    '1332': {'type': 'pad', 'pad': 'red', 'w': 0.97, 'h': 0.23, 'oy': -0.38},
    '67': {'type': 'pad', 'pad': 'blue', 'w': 0.83, 'h': 0.14, 'oy': -0.43},

    '36': {'type': 'orb', 'orb': 'yellow', 'r': 0.6},
    '141': {'type': 'orb', 'orb': 'pink', 'r': 0.6},
    '1333': {'type': 'orb', 'orb': 'red', 'r': 0.6},
    '84': {'type': 'orb', 'orb': 'blue', 'r': 0.6},
    '1022': {'type': 'orb', 'orb': 'green', 'r': 0.6},
    '1705': {'type': 'orb', 'orb': 'black', 'r': 0.6},
}

class GDPlayer:
    def __init__(self):
        self.x = 0.0
        self.y = GROUND_Y + 0.5
        self.vy = 0.0
        self.mode = 'cube'
        self.speedIdx = 1
        self.gravDir = 1  # 1 = down, -1 = up
        self.mini = False
        self.onGround = True
        self.dead = False
        self.won = False
        self.thrusting = False
        self.holdTime = 0.0

    def half(self):
        s = PHYS['mini']['size'] if self.mini else 1.0
        return (PLAYER_SIZE * s) / 2.0

    def inner_half(self):
        s = PHYS['mini']['size'] if self.mini else 1.0
        return (PLAYER_INNER * s) / 2.0

class GDEngine:
    def __init__(self, level_objects, ceilH=12.0):
        self.objects = sorted(level_objects, key=lambda o: o['x'])
        self.player = GDPlayer()
        self.time = 0.0
        self.ceilH = ceilH
        self.used_orbs = set()

    def step(self, hold=False, press_buffered=False):
        p = self.player
        if p.dead or p.won:
            return

        dt = PHYS_DT
        speed = SPEEDS[p.speedIdx]
        half = p.half()
        prev_y = p.y
        miniJ = PHYS['mini']['jump'] if p.mini else 1.0
        flyCap = PHYS['ship']['miniCapMult'] if p.mini else 1.0

        # Horizontal
        p.x += speed * dt
        g = p.gravDir

        # Vertical Mode Updates
        if p.mode == 'cube':
            grav = GRAV_CUBE[p.speedIdx]
            jv = JUMP_CUBE[p.speedIdx]
            if (hold or press_buffered) and p.onGround:
                p.vy = jv * miniJ * g
                p.onGround = False
            p.vy -= grav * dt * g
            p.vy = max(-PHYS['cube']['maxFall'], min(PHYS['cube']['maxFall'], p.vy))

        elif p.mode == 'ship':
            rising = (p.vy * g) > 0
            if hold:
                a = PHYS['ship']['holdRise'] if rising else PHYS['ship']['holdFall']
            else:
                a = -(PHYS['ship']['relRise'] if rising else PHYS['ship']['relFall'])
            p.vy += a * dt * g
            up = PHYS['ship']['maxUp'] * flyCap
            dn = PHYS['ship']['maxDown'] * flyCap
            if g == 1:
                p.vy = max(-dn, min(up, p.vy))
            else:
                p.vy = max(-up, min(dn, p.vy))

        elif p.mode == 'ball':
            if press_buffered and p.onGround:
                p.gravDir *= -1
                p.vy = -PHYS['ball']['clickPop'] * p.gravDir
                p.onGround = False
            p.vy -= PHYS['ball']['gravity'] * dt * p.gravDir
            p.vy = max(-PHYS['ball']['maxFall'], min(PHYS['ball']['maxFall'], p.vy))

        elif p.mode == 'ufo':
            if press_buffered:
                tap = PHYS['ufo']['tap'] * miniJ
                if p.vy * g < tap:
                    p.vy = tap * g
                p.onGround = False
            grav = PHYS['ufo']['gravRise'] if (p.vy * g > 0) else PHYS['ufo']['gravFall']
            p.vy -= grav * dt * g
            up = PHYS['ufo']['maxUp'] * flyCap
            dn = PHYS['ufo']['maxDown'] * flyCap
            if g == 1:
                p.vy = max(-dn, min(up, p.vy))
            else:
                p.vy = max(-up, min(dn, p.vy))

        elif p.mode == 'wave':
            slope = (PHYS['wave']['miniSlope'] if p.mini else PHYS['wave']['slope']) * speed
            p.vy = (slope if hold else -slope) * g

        p.y += p.vy * dt
        p.onGround = False

        # Floor / Ceiling Collision
        if p.y - half <= GROUND_Y:
            p.y = GROUND_Y + half
            if p.gravDir == 1:
                p.onGround = True
                p.vy = 0
            elif p.vy < 0:
                p.vy = 0

        if p.y + half >= self.ceilH and (p.mode in CEILING_MODES or p.gravDir == -1):
            p.y = self.ceilH - half
            if p.gravDir == -1:
                p.onGround = True
                p.vy = 0
            elif p.vy > 0:
                p.vy = 0

        # Object Collisions
        inner_half = p.inner_half()
        for idx, o in enumerate(self.objects):
            if o['x'] < p.x - 5.0: continue
            if o['x'] > p.x + 5.0: break

            spec = OBJ_CATALOG.get(o['id'])
            if not spec: continue

            # Circle Hazard
            if 'r' in spec and spec.get('hazard'):
                r = spec['r'] * o.get('scale', 1.0)
                dx = max(abs(p.x - o['x']) - half, 0.0)
                dy = max(abs(p.y - o['y']) - half, 0.0)
                if dx*dx + dy*dy < r*r:
                    p.dead = True
                    return

            # AABB Hazard / Solid / Portal / Pad / Orb
            w = spec.get('w', 1.0) * o.get('scale', 1.0)
            h = spec.get('h', 1.0) * o.get('scale', 1.0)
            ox = spec.get('ox', 0.0)
            oy = spec.get('oy', 0.0)
            hb_x = o['x'] + ox
            hb_y = o['y'] + oy

            # AABB Overlap check
            if abs(p.x - hb_x) < half + w/2.0 and abs(p.y - hb_y) < half + h/2.0:
                if spec.get('hazard'):
                    p.dead = True
                    return
                elif spec.get('type') == 'portal':
                    act = spec['act']
                    if act in ['cube', 'ship', 'ball', 'ufo', 'wave', 'robot', 'spider']:
                        p.mode = act
                    elif act == 'gravUp': p.gravDir = -1
                    elif act == 'gravDown': p.gravDir = 1
                    elif act == 'mini': p.mini = True
                    elif act == 'big': p.mini = False
                elif spec.get('type') == 'speed':
                    p.speedIdx = spec['speedIdx']
                elif spec.get('type') == 'pad':
                    pad = spec['pad']
                    p.vy = PAD_VEL[pad]['v'] * p.gravDir
                    p.onGround = False
                    if PAD_VEL[pad].get('flip'):
                        p.gravDir *= -1

                elif spec.get('solid'):
                    # Inner box check for wall crash death
                    if abs(p.x - hb_x) < inner_half + w/2.0 and abs(p.y - hb_y) < inner_half + h/2.0:
                        top = hb_y + h/2.0
                        bot = hb_y - h/2.0
                        if prev_y - half >= top - LAND_SNAP and p.vy <= 0:
                            p.y = top + half
                            if p.gravDir == 1:
                                p.onGround = True
                                p.vy = 0
                        elif prev_y + half <= bot + LAND_SNAP and p.vy >= 0:
                            p.y = bot - half
                            if p.gravDir == -1:
                                p.onGround = True
                                p.vy = 0
                        else:
                            p.dead = True
                            return

        self.time += dt


def load_bloodbath_level():
    with open('/kaggle/input/datasets/mcpeorakul/geometry-dash/bloodbath.gdlvl', 'rb') as f:
        raw = f.read().strip()
    raw += b'=' * ((4 - len(raw) % 4) % 4)
    plist_text = gzip.decompress(base64.urlsafe_b64decode(raw)).decode('utf-8', errors='replace')
    lvl_data_raw = re.search(r'<k>k4</k><s>(.*?)</s>', plist_text).group(1).encode('utf-8')
    lvl_data_raw += b'=' * ((4 - len(lvl_data_raw) % 4) % 4)
    lvl_decompressed = gzip.decompress(base64.urlsafe_b64decode(lvl_data_raw)).decode('utf-8', errors='replace')

    objects = lvl_decompressed.split(';')
    parsed_objs = []
    for obj_str in objects[1:]:
        if not obj_str.strip(): continue
        parts = obj_str.split(',')
        kv = {parts[i]: parts[i+1] for i in range(0, len(parts)-1, 2)}
        parsed_objs.append({
            'id': kv.get('1'),
            'x': float(kv.get('2', 0)) / 30.0,
            'y': float(kv.get('3', 0)) / 30.0,
            'rot': float(kv.get('6', 0)),
            'flip_x': kv.get('4') == '1',
            'flip_y': kv.get('5') == '1',
            'scale': float(kv.get('32', 1.0))
        })
    return parsed_objs


if __name__ == '__main__':
    print("Loading Bloodbath objects...")
    objs = load_bloodbath_level()
    print(f"Loaded {len(objs)} objects from bloodbath.gdlvl.")
    engine = GDEngine(objs)
    print("Simulating 500 physics substeps (2.08 seconds of game time)...")
    for step_i in range(500):
        engine.step(hold=(step_i > 100 and step_i < 200))
        if engine.player.dead:
            print(f"Player died at step {step_i}, X={engine.player.x:.2f}, Y={engine.player.y:.2f}, Mode={engine.player.mode}")
            break
    if not engine.player.dead:
        print(f"Player survived 500 steps! X={engine.player.x:.2f}, Y={engine.player.y:.2f}, Mode={engine.player.mode}")
