from nuscenes.nuscenes import NuScenes

nusc = NuScenes(
    version="v1.0-mini",
    dataroot="data/nuscenes",
    verbose=False
)

scene = nusc.scene[0]

sample_token = scene["first_sample_token"]

sample = nusc.get("sample", sample_token)

# Pick first object
ann_token = sample["anns"][0]

print("Tracking one object through time\n")

while ann_token != "":

    ann = nusc.get("sample_annotation", ann_token)

    print(
        ann["category_name"],
        "Position:",
        ann["translation"][:2]
    )

    ann_token = ann["next"]
    