import musdb

mus = musdb.DB(root="data/", download=True)

train_tracks = mus.load_mus_tracks(subsets="train")
test_tracks  = mus.load_mus_tracks(subsets="test")

print(f"Train: {len(train_tracks)} tracks")
print(f"Test:  {len(test_tracks)} tracks")
