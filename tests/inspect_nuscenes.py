from nuscenes.nuscenes import NuScenes

nusc = NuScenes(version="v1.0-mini", dataroot="data/nuscenes", verbose=False)

scene = nusc.scene[0]
print("Scene:", scene["name"])
print("First sample token:", scene["first_sample_token"])

sample = nusc.get("sample", scene["first_sample_token"])
print("Sample token:", sample["token"])
print("Number of annotations:", len(sample["anns"]))

ann_token = sample["anns"][0]
ann = nusc.get("sample_annotation", ann_token)
print("First annotation category:", ann["category_name"])
print("Translation:", ann["translation"])
print("Size:", ann["size"])