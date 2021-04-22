from datasets.lending_club import LendingClub

root = '/Users/iv7/data'
a = LendingClub(root=root, accepted=True, partition="train")[:]

print("Accpted applications:")
print(f"X_numeric.shape: {a['X_numeric'].shape}")
print(f"X_categorical.shape: {a['X_categorical'].shape}")
print(f"y.shape: {a['target'].shape}")

print("Rejected applications:")
r = LendingClub(root=root, accepted=False, partition="train")[:]
print(f"X_numeric.shape: {r['X_numeric'].shape}")
print(f"X_categorical.shape: {r['X_categorical'].shape}")
