from nuscenes.nuscenes import NuScenes

nusc = NuScenes(
    version="v1.0-mini",
    dataroot="data/nuscenes",
    verbose=True
)

print("Number of scenes:", len(nusc.scene))