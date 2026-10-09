"""Convert the hand / screwdriver URDFs to USD with the Isaac Sim 6.0 URDF importer.

Isaac Lab's ``UrdfFileCfg`` relies on the legacy ``_urdf`` interface that is not shipped with
Isaac Sim 6.0, so we convert once offline and spawn the resulting USD files.

Usage:
    ~/env_isaaclab6/bin/python scripts/convert_urdf_to_usd.py
"""

import argparse
import os
import re
import shutil
import tempfile

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--force", action="store_true", help="re-convert assets that already exist")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import omni.kit.app  # noqa: E402

manager = omni.kit.app.get_app().get_extension_manager()
manager.set_extension_enabled_immediate("isaacsim.asset.importer.urdf", True)
from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig  # noqa: E402
from pxr import PhysxSchema, Usd, UsdPhysics  # noqa: E402

# contact / rest offset of the original IsaacGym config (sim.physx.contact_offset / rest_offset)
CONTACT_OFFSET = 0.002
REST_OFFSET = 0.0


def set_collision_offsets(usd_dir):
    """Author contact/rest offsets on every collider (they live in instanceable prototypes)."""
    for root, _, files in os.walk(usd_dir):
        for name in files:
            if not name.endswith((".usda", ".usd")):
                continue
            stage = Usd.Stage.Open(os.path.join(root, name))
            changed = False
            for prim in stage.Traverse(Usd.TraverseInstanceProxies()):
                if prim.IsInstanceProxy() or not prim.HasAPI(UsdPhysics.CollisionAPI):
                    continue
                api = PhysxSchema.PhysxCollisionAPI.Apply(prim)
                api.CreateContactOffsetAttr().Set(CONTACT_OFFSET)
                api.CreateRestOffsetAttr().Set(REST_OFFSET)
                changed = True
            if changed:
                stage.GetRootLayer().Save()

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "assets")
URDFS = [
    "xhand_left/urdf/xhand_left.urdf",
    "screw/driver/0000_stripe.urdf",
    "screw/driver/0001_stripe.urdf",
    "screw/trinut/0003_stripe.urdf",
    "screw/boxnut/0001_stripe.urdf",
    "screw/crossnut/0004_stripe.urdf",
]

MESH_DIR = os.path.normpath(os.path.join(ROOT, "meshes"))
tmp_dir = tempfile.mkdtemp()

for rel in URDFS:
    urdf_path = os.path.normpath(os.path.join(ROOT, rel))
    name = os.path.splitext(os.path.basename(rel))[0]
    # screw assets share file names across subsets (e.g. driver/0001_stripe, boxnut/0001_stripe)
    out_dir = os.path.join(ROOT, "usd", rel.split("/")[1]) if rel.startswith("screw/") else os.path.join(ROOT, "usd")
    out_dir = os.path.normpath(out_dir)
    if os.path.exists(os.path.join(out_dir, name, f"{name}.usda")) and not args.force:
        print(f"[convert] {rel} exists, skipping (use --force)")
        continue
    if rel.startswith("screw/"):
        # the screw URDFs reference ``../meshes/*.stl`` but the meshes live in ``assets/meshes``
        with open(urdf_path) as f:
            text = f.read()
        text = re.sub(r'filename="\.\./meshes/([^"]+)"', lambda m: f'filename="{MESH_DIR}/{m.group(1)}"', text)
        urdf_path = os.path.join(tmp_dir, os.path.basename(urdf_path))
        with open(urdf_path, "w") as f:
            f.write(text)
    cfg = URDFImporterConfig(urdf_path=urdf_path, usd_path=out_dir, collision_type="Convex Hull",
                             allow_self_collision=False)
    path = URDFImporter(cfg).import_urdf()
    set_collision_offsets(os.path.dirname(path))
    # The importer fixes the base with a maximal "root_joint" to the world, which is not part of the
    # articulation, so root poses written from Isaac Lab are ignored. Deactivate it; the env sets
    # `fix_root_link=True` instead, which makes a real fixed-base articulation.
    stage = Usd.Stage.Open(path)
    for prim in stage.Traverse():
        if prim.GetName() == "root_joint" and prim.IsA(UsdPhysics.FixedJoint):
            prim.SetActive(False)
    stage.GetRootLayer().Save()
    print(f"[convert] {rel} -> {path}")

shutil.rmtree(tmp_dir)
app.close()
