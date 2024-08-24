# convert filenames to dictionary and save as piclle file
import pickle

index_to_filename = {}
i = 0
with open("filenames.txt", "r") as f:
    for line in f.readlines():
        index_to_filename[i] = line.strip()
        i += 1
with open('index_to_filenames.pkl', 'wb') as f:
    pickle.dump(index_to_filename, f, pickle.HIGHEST_PROTOCOL)