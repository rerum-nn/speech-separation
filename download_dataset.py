import kagglehub

# Download latest version
path = kagglehub.dataset_download("rerumnn/dla-dataset")

print("Path to dataset files:", path)