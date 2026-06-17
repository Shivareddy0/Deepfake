import os
import shutil
import sys

def download_and_setup():
    try:
        import kagglehub
    except ImportError:
        print("[!] kagglehub library not found. Please install it in your virtual environment:")
        print("    .\\backend\\venv\\Scripts\\pip install kagglehub")
        return

    print("[*] Downloading CIFAKE dataset using kagglehub...")
    # Download latest version
    download_path = kagglehub.dataset_download("birdy654/cifake-real-and-ai-generated-synthetic-images")
    print(f"[+] Dataset downloaded to: {download_path}")

    # Determine destination root
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    dest_root = os.path.join(project_root, "data", "images", "cifake")
    dest_real = os.path.join(dest_root, "REAL")
    dest_fake = os.path.join(dest_root, "FAKE")

    os.makedirs(dest_real, exist_ok=True)
    os.makedirs(dest_fake, exist_ok=True)

    print(f"[*] Organizing and copying files to {dest_root}...")

    copied_real = 0
    copied_fake = 0

    for root, dirs, files in os.walk(download_path):
        parent_dir = os.path.basename(root).upper()
        if parent_dir == "REAL":
            target_dir = dest_real
        elif parent_dir == "FAKE":
            target_dir = dest_fake
        else:
            continue

        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp')):
                src_file = os.path.join(root, file)
                # Prepend the parent subdirectory name (e.g. 'train' or 'test') to prevent filename collisions
                subfolder_name = os.path.basename(os.path.dirname(root))
                new_filename = f"{subfolder_name}_{file}"
                dst_file = os.path.join(target_dir, new_filename)
                
                shutil.copy2(src_file, dst_file)
                if parent_dir == "REAL":
                    copied_real += 1
                else:
                    copied_fake += 1

    print(f"[+] Setup completed successfully!")
    print(f"    - Destination: {dest_root}")
    print(f"    - REAL Images Copied: {copied_real}")
    print(f"    - FAKE Images Copied: {copied_fake}")

if __name__ == "__main__":
    download_and_setup()
