import os

synthetic_dir = "data/atc_dataset/synthetic"

for file in os.listdir(synthetic_dir):
    file_path = os.path.join(synthetic_dir, file)
    try:
        os.remove(file_path)
        print(f"🗑️ Deleted {file_path}")
    except Exception as e:
        print(f"❌ Error deleting {file_path}: {e}")

print("✅ All synthetic files have been deleted.")
