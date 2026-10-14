"""Data-grounded mission viz v1: S2/THRML mission with energy path, LiDAR fan, repulsion vectors."""
import bpy, numpy as np, math, os, csv

# telemetry: new full CSV preferred, old flight CSV as fallback for path
TELE = "C:/Users/whtet/Documents/muse/s2_mission_telemetry.csv"
FALLBACK = "C:/Users/whtet/Documents/muse/s2_flight_bodyaware.csv"
OCC = "C:/Users/whtet/Documents/muse/room_004_occ.npy"
V4SCENE = "C:/Users/whtet/Documents/muse/techdemo_hero_v4.py"
OUT = "C:/Users/whtet/Documents/muse/mission_viz_v1.png"
VS = 0.25

# ---------- load trajectory ----------
traj, energies, lidar_hits, clearances = [], [], [], []
src = TELE if os.path.exists(TELE) else FALLBACK
print("telemetry source:", src)
with open(src) as f:
    r = csv.DictReader(f)
    cols = r.fieldnames
    for row in r:
        traj.append((float(row['x']), float(row['y']), float(row['z'])))
        if 'energy_chosen' in cols and row.get('energy_chosen'):
            energies.append(float(row['energy_chosen']))
        if 'clearance_m' in cols and row.get('clearance_m'):
            clearances.append(float(row['clearance_m']))
        if 'lidar_hits' in cols and row.get('lidar_hits'):
            hits = []
            for h in row['lidar_hits'].split('|'):
                if h.strip():
                    hits.append(tuple(float(v) for v in h.split(';')))
            lidar_hits.append(hits)

# recenter to room (same as mission video)
xs = [p[0] for p in traj]; ys = [p[1] for p in traj]
mx, my = (min(xs)+max(xs))/2, (min(ys)+max(ys))/2
traj = [(x-mx, y-my, z) for x, y, z in traj]
if lidar_hits:
    lidar_hits = [[(hx-mx, hy-my, hz) for hx, hy, hz in hits] for hits in lidar_hits]
print(f"{len(traj)} points, energies: {len(energies)}, lidar frames: {len(lidar_hits)}")

# viz moment: closest approach (min clearance), else midpoint
if clearances:
    vi = int(np.argmin(clearances))
else:
    vi = len(traj)//2
print("viz moment index:", vi, "pos:", traj[vi])


# ============ V4 SCENE (inlined) ============
"""Tech-demo hero still v3: cinematic drone, Track-To camera, volumetric fog. Blender 5.2."""
import bpy, numpy as np, math, os

OUT = "C:/Users/whtet/Documents/muse/techdemo_hero_v4.png"
OCC = "C:/Users/whtet/Documents/muse/room_004_occ.npy"
VS = 0.25

# ---------- scene reset ----------
bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete()
for c in list(bpy.data.collections):
    if c.name != "Collection": bpy.data.collections.remove(c)

# ---------- room (merged per z-layer) ----------
occ = np.load(OCC)
mat = bpy.data.materials.new("WallMat")
mat.use_nodes = True
bsdf = mat.node_tree.nodes["Principled BSDF"]
bsdf.inputs["Base Color"].default_value = (0.07, 0.08, 0.10, 1)
bsdf.inputs["Metallic"].default_value = 1.0
bsdf.inputs["Roughness"].default_value = 0.32

for iz in range(occ.shape[2]):
    layer = occ[:, :, iz]
    if not layer.any(): continue
    boxes = []
    for ix in range(occ.shape[0]):
        for iy in range(occ.shape[1]):
            if layer[ix, iy]:
                bpy.ops.mesh.primitive_cube_add(size=VS,
                    location=((ix-16)*VS, (iy-16)*VS, (iz-8)*VS))
                boxes.append(bpy.context.active_object)
    if boxes:
        bpy.ops.object.select_all(action='DESELECT')
        for b in boxes: b.select_set(True)
        bpy.context.view_layer.objects.active = boxes[0]
        bpy.ops.object.join()
        bpy.context.active_object.data.materials.append(mat)
print("room built")

# ---------- floor ----------
bpy.ops.mesh.primitive_plane_add(size=14, location=(0, 0, -2.01))
fl = bpy.context.active_object
fmat = bpy.data.materials.new("FloorMat"); fmat.use_nodes = True
fbsdf = fmat.node_tree.nodes["Principled BSDF"]
fbsdf.inputs["Base Color"].default_value = (0.02, 0.022, 0.028, 1)
fbsdf.inputs["Metallic"].default_value = 1.0
fbsdf.inputs["Roughness"].default_value = 0.15
fl.data.materials.append(fmat)

# ---------- volumetric fog cube ----------
bpy.ops.mesh.primitive_cube_add(size=16, location=(0, 0, 1))
fog = bpy.context.active_object
fogmat = bpy.data.materials.new("FogMat"); fogmat.use_nodes = True
nt = fogmat.node_tree
nt.nodes.clear()
vol = nt.nodes.new("ShaderNodeVolumePrincipled")
vol.inputs["Density"].default_value = 0.008
vol.inputs["Color"].default_value = (0.35, 0.5, 0.7, 1)
vol.inputs["Anisotropy"].default_value = 0.4
out = nt.nodes.new("ShaderNodeOutputMaterial")
nt.links.new(vol.outputs[0], out.inputs["Volume"])
fog.data.materials.append(fogmat)
print("fog added")

# ---------- drone ----------
def emat(name, color, strength=18.0):
    m = bpy.data.materials.new(name); m.use_nodes = True
    e = m.node_tree.nodes.new("ShaderNodeEmission")
    e.inputs["Color"].default_value = (*color, 1)
    e.inputs["Strength"].default_value = strength
    m.node_tree.links.new(e.outputs[0], m.node_tree.nodes["Material Output"].inputs[0])
    return m

carbon = bpy.data.materials.new("Carbon"); carbon.use_nodes = True
cbsdf = carbon.node_tree.nodes["Principled BSDF"]
cbsdf.inputs["Base Color"].default_value = (0.012, 0.012, 0.018, 1)
cbsdf.inputs["Metallic"].default_value = 0.9
cbsdf.inputs["Roughness"].default_value = 0.25

cyan = emat("CyanGlow", (0.12, 0.95, 1.0), 22.0)
orange = emat("OrangeGlow", (1.0, 0.40, 0.06), 14.0)
white = emat("WhiteStrobe", (1.0, 1.0, 1.0), 28.0)

bpy.ops.mesh.primitive_cube_add(size=0.36, location=(0, 0, 1.6))
body = bpy.context.active_object
body.name = "DroneBody"
body.scale = (1.5, 1.1, 0.38)
body.data.materials.append(carbon)

bpy.ops.mesh.primitive_cube_add(size=0.22, location=(0, 0, 1.74))
tp = bpy.context.active_object
tp.scale = (1.2, 0.9, 0.14)
tp.data.materials.append(carbon)

for sx, sy in [(1,1),(1,-1),(-1,1),(-1,-1)]:
    ax, ay = 0.32*sx, 0.24*sy
    bpy.ops.mesh.primitive_cylinder_add(radius=0.028, depth=0.40,
        location=(ax/2, ay/2, 1.6), rotation=(0, math.pi/2, math.atan2(ay,ax)))
    bpy.context.active_object.data.materials.append(carbon)
    bpy.ops.mesh.primitive_cylinder_add(radius=0.05, depth=0.09, location=(ax, ay, 1.64))
    bpy.context.active_object.data.materials.append(cyan)
    bpy.ops.mesh.primitive_torus_add(major_radius=0.062, minor_radius=0.009, location=(ax, ay, 1.64))
    bpy.context.active_object.data.materials.append(cyan)
    bpy.ops.mesh.primitive_cube_add(size=0.028, location=(ax, ay, 1.70))
    bpy.context.active_object.data.materials.append(carbon)
    for ang in [0.5, -0.5]:
        bpy.ops.mesh.primitive_cube_add(size=0.028, location=(ax, ay, 1.70))
        bl = bpy.context.active_object
        bl.scale = (7.5, 0.9, 0.18); bl.rotation_euler[2] = ang
        bl.data.materials.append(carbon)

bpy.ops.mesh.primitive_plane_add(size=0.30, location=(0, 0, 1.45))
bpy.context.active_object.data.materials.append(orange)
bpy.ops.mesh.primitive_cube_add(size=0.035, location=(0.28, 0, 1.62))
bpy.context.active_object.data.materials.append(white)
print("drone built")

# ---------- THREE-POINT CINEMATIC LIGHTING ----------
# key: warm area from upper front-right
bpy.ops.object.light_add(type='AREA', location=(4, -3, 5))
key = bpy.context.active_object
key.data.energy = 2500; key.data.color = (1.0, 0.88, 0.72); key.data.size = 4
# fill: cool from front-left, weaker
bpy.ops.object.light_add(type='AREA', location=(-3, -4, 2))
fill = bpy.context.active_object
fill.data.energy = 700; fill.data.color = (0.45, 0.65, 1.0); fill.data.size = 5
# rim: cyan from behind, strongest for edge separation
bpy.ops.object.light_add(type='AREA', location=(-2, 5, 3.5))
rim = bpy.context.active_object
rim.data.energy = 3200; rim.data.color = (0.12, 0.72, 1.0); rim.data.size = 3
# orange kick from below
bpy.ops.object.light_add(type='POINT', location=(0, 0, -1.0))
und = bpy.context.active_object
und.data.energy = 180; und.data.color = (1.0, 0.42, 0.10)
print("lights set")

# ---------- CAMERA with Track To (bulletproof aiming) ----------
bpy.ops.object.empty_add(location=(0, 0, 1.65))
# lift drone above the central wall cluster into open air

target = bpy.context.active_object
target.name = "DroneTarget"

bpy.ops.object.camera_add(location=(1.6, -1.5, 1.1))
cam = bpy.context.active_object
bpy.context.scene.camera = cam
cam.data.lens = 40

# Track To constraint: camera -Z aims at target, Y up
tt = cam.constraints.new(type='TRACK_TO')
tt.target = target
tt.track_axis = 'TRACK_NEGATIVE_Z'
tt.up_axis = 'UP_Y'
print("camera tracking drone")


# ============ VIZ BODY ============
OUT = "C:/Users/whtet/Documents/muse/mission_viz_v1.png"  # re-assert after v4 inline

# ---------- energy-colored breadcrumb path ----------
if energies:
    emin, emax = min(energies), max(energies)
    erange = max(emax-emin, 1e-9)
else:
    emin, erange = 0, 1

def energy_color(e):
    t = (e-emin)/erange  # 0=low energy (good) -> 1=high energy (near obstacle)
    # cyan -> yellow -> orange-red
    if t < 0.5:
        u = t*2
        return (0.1+0.8*u, 0.9-0.1*u, 1.0-0.7*u)
    else:
        u = (t-0.5)*2
        return (0.9+0.1*u, 0.8-0.6*u, 0.3-0.25*u)

for i, p in enumerate(traj):
    e = energies[i] if energies else 0
    bpy.ops.mesh.primitive_ico_sphere_add(radius=0.035, subdivisions=2, location=p)
    s = bpy.context.active_object
    m = bpy.data.materials.new(f"PathMat{i}")
    m.use_nodes = True
    em = m.node_tree.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = (*energy_color(e), 1)
    em.inputs["Strength"].default_value = 6.0
    m.node_tree.links.new(em.outputs[0], m.node_tree.nodes["Material Output"].inputs[0])
    s.data.materials.append(m)
print("path breadcrumbs placed")

# ---------- drone at viz moment ----------
bpy.ops.object.empty_add(location=traj[vi])
rig = bpy.context.active_object
rig.name = "VizRig"
parts = [o for o in bpy.data.objects
         if o.type == 'MESH' and abs(o.location.z - 1.6) < 0.6
         and abs(o.location.x) < 1.2 and abs(o.location.y) < 1.2]
bpy.ops.object.select_all(action='DESELECT')
for o in parts: o.select_set(True)
rig.select_set(True)
bpy.context.view_layer.objects.active = rig
bpy.ops.object.parent_set(type='OBJECT', keep_transform=True)
bpy.ops.object.select_all(action='DESELECT')
for o in parts:  # recenter on rig
    o.location.x -= (0.0 - traj[vi][0])
    o.location.y -= (0.0 - traj[vi][1])
    o.location.z -= (1.6 - traj[vi][2])
print("drone placed at viz moment,", len(parts), "parts")

# ---------- LiDAR fan at viz moment ----------
if lidar_hits and vi < len(lidar_hits):
    hits = lidar_hits[vi]
    dp = traj[vi]
    fanmat = bpy.data.materials.new("FanMat"); fanmat.use_nodes = True
    fem = fanmat.node_tree.nodes.new("ShaderNodeEmission")
    fem.inputs["Color"].default_value = (0.2, 0.8, 1.0, 1)
    fem.inputs["Strength"].default_value = 1.2
    fanmat.node_tree.links.new(fem.outputs[0], fanmat.node_tree.nodes["Material Output"].inputs[0])
    fanmat.blend_method = 'BLEND'
    hitmat = bpy.data.materials.new("HitMat"); hitmat.use_nodes = True
    hem = hitmat.node_tree.nodes.new("ShaderNodeEmission")
    hem.inputs["Color"].default_value = (0.4, 1.0, 1.0, 1)
    hem.inputs["Strength"].default_value = 12.0
    hitmat.node_tree.links.new(hem.outputs[0], hitmat.node_tree.nodes["Material Output"].inputs[0])
    for hx, hy, hz in hits:
        # ray line
        dx, dy, dz = hx-dp[0], hy-dp[1], hz-dp[2]
        dist = math.sqrt(dx*dx+dy*dy+dz*dz)
        if dist < 1e-6: continue
        bpy.ops.mesh.primitive_cylinder_add(radius=0.006, depth=dist,
            location=((dp[0]+hx)/2, (dp[1]+hy)/2, (dp[2]+hz)/2))
        ray = bpy.context.active_object
        # orient along direction
        ray.rotation_euler = (math.acos(dz/dist),
                              0,
                              math.atan2(dy, dx) - math.pi/2)
        ray.data.materials.append(fanmat)
        # hit point
        bpy.ops.mesh.primitive_ico_sphere_add(radius=0.025, subdivisions=2, location=(hx, hy, hz))
        bpy.context.active_object.data.materials.append(hitmat)
    print(f"lidar fan: {len(hits)} rays")

# ---------- repulsion vectors: path points pushed from nearest wall ----------
occ = np.load(OCC)
repmat = bpy.data.materials.new("RepMat"); repmat.use_nodes = True
rem = repmat.node_tree.nodes.new("ShaderNodeEmission")
rem.inputs["Color"].default_value = (1.0, 0.45, 0.1, 1)
rem.inputs["Strength"].default_value = 8.0
repmat.node_tree.links.new(rem.outputs[0], repmat.node_tree.nodes["Material Output"].inputs[0])

occ_idx = np.argwhere(occ)
shown = 0
for i in range(0, len(traj), max(1, len(traj)//12)):  # ~12 vectors max
    px, py, pz = traj[i]
    # nearest occupied voxel center
    best, bd = None, 1e9
    for ix, iy, iz in occ_idx:
        wx, wy, wz = (ix-16)*VS, (iy-16)*VS, (iz-8)*VS
        d = (wx-px)**2 + (wy-py)**2 + (wz-pz)**2
        if d < bd: bd, best = d, (wx, wy, wz)
    if best and bd < 1.0:  # within 1m
        # vector from wall to path point (repulsion direction), scaled
        vx, vy, vz = px-best[0], py-best[1], pz-best[2]
        vl = math.sqrt(vx*vx+vy*vy+vz*vz) + 1e-9
        L = 0.35
        ex, ey, ez = px+vx/vl*L, py+vy/vl*L, pz+vz/vl*L
        bpy.ops.mesh.primitive_cylinder_add(radius=0.012, depth=L,
            location=((px+ex)/2, (py+ey)/2, (pz+ez)/2))
        vec = bpy.context.active_object
        vec.rotation_euler = (math.acos((ez-pz)/L), 0, math.atan2(ey-py, ex-px) - math.pi/2)
        vec.data.materials.append(repmat)
        shown += 1
print(f"repulsion vectors: {shown}")

# ---------- camera: frame drone + path ----------
tgt = bpy.data.objects.get("DroneTarget")
if tgt:
    tgt.location = traj[vi]
cam = bpy.data.objects.get("Camera")
if cam:
    for c in cam.constraints:
        if c.type == 'TRACK_TO':
            c.target = tgt
    cam.location = (traj[vi][0]+2.2, traj[vi][1]-2.4, traj[vi][2]+1.6)

# ---------- render still ----------
scene = bpy.context.scene
scene.render.engine = 'CYCLES'
scene.render.resolution_x, scene.render.resolution_y = 1920, 1080
scene.cycles.samples = 320
scene.cycles.device = 'GPU'
prefs = bpy.context.preferences.addons['cycles'].preferences
prefs.compute_device_type = 'HIP'
prefs.get_devices()
for d in prefs.devices:
    d.use = ('RX 9070 XT' in d.name)
scene.render.film_transparent = False
scene.render.image_settings.file_format = 'PNG'
scene.render.filepath = OUT
bpy.ops.render.render(write_still=True)
print("MISSION_VIZ_DONE", os.path.exists(OUT))
