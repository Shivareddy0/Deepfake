import os
import sys
import urllib.request

# Define weights directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")

# Model definitions (URLs and local filenames)
MODELS = {
    "standard": {
        "spatial_efficientnet.pth": "https://download.pytorch.org/models/efficientnet_b0_rwightman-3dd3cd2a.pth",
        "audio_resnet.pth": "https://download.pytorch.org/models/resnet18-f37072fd.pth"
    },
    "ultra": {
        "spatial_convnext_large.pth": "https://download.pytorch.org/models/convnext_large-ea09fe54.pth",
        "audio_wav2vec2_large.bin": "https://huggingface.co/facebook/wav2vec2-xls-r-300m/resolve/main/pytorch_model.bin"
    }
}

def download_progress(block_num, block_size, total_size):
    """
    Console progress reporter for urllib.
    """
    read_so_far = block_num * block_size
    if total_size > 0:
        percent = min(100.0, read_so_far * 100 / total_size)
        sys.stdout.write(f"\rDownloading... {percent:.1f}% ({read_so_far / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB)")
    else:
        sys.stdout.write(f"\rDownloading... {read_so_far / (1024*1024):.1f} MB")
    sys.stdout.flush()

def main():
    print("==========================================================")
    print("       ANTIGRAVITY DEEPFAKE SHIELD WEIGHTS DOWNLOADER     ")
    print("==========================================================")
    print(f"Target Directory: {WEIGHTS_DIR}")
    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    print("\nSelect Download Tier:")
    print(" [1] Standard Models (~66 MB) - Light and fast for initial verification")
    print(" [2] Ultra-Forensics Models (~2.5 GB) - Deep inspection (ConvNeXt-L & Wav2Vec2)")
    print(" [3] Download ALL models (~2.6 GB)")
    
    try:
        choice = input("\nEnter choice (1-3, default 1): ").strip()
    except (KeyboardInterrupt, SystemExit):
        print("\nAborted.")
        return
    
    if choice == "2":
        selected_tiers = ["ultra"]
    elif choice == "3":
        selected_tiers = ["standard", "ultra"]
    else:
        selected_tiers = ["standard"]

    print(f"\nQueueing downloads for: {selected_tiers} tiers...")
    
    for tier in selected_tiers:
        print(f"\n--- Processing {tier.upper()} Tier Models ---")
        for filename, url in MODELS[tier].items():
            dest_path = os.path.join(WEIGHTS_DIR, filename)
            if os.path.exists(dest_path):
                print(f"[-] {filename} already exists. Skipping.")
                continue
            
            print(f"[+] Downloading {filename}...")
            print(f"    Source: {url}")
            try:
                req = urllib.request.Request(
                    url,
                    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
                )
                with urllib.request.urlopen(req) as response:
                    total_size = int(response.info().get('Content-Length', 0))
                    block_size = 1024 * 8
                    read_so_far = 0
                    with open(dest_path, 'wb') as out_file:
                        while True:
                            buffer = response.read(block_size)
                            if not buffer:
                                break
                            read_so_far += len(buffer)
                            out_file.write(buffer)
                            # Update progress
                            if total_size > 0:
                                percent = min(100.0, read_so_far * 100 / total_size)
                                sys.stdout.write(f"\rDownloading... {percent:.1f}% ({read_so_far / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB)")
                            else:
                                sys.stdout.write(f"\rDownloading... {read_so_far / (1024*1024):.1f} MB")
                            sys.stdout.flush()
                print(f"\n[✓] Successfully downloaded {filename}!")
            except Exception as e:
                print(f"\n[✗] Failed to download {filename}: {e}")
                # Clean up partial download
                if os.path.exists(dest_path):
                    try:
                        os.remove(dest_path)
                    except Exception:
                        pass

    print("\n==========================================================")
    print("Weights download complete. Place weights directory inside ")
    print("the backend folder prior to running the FastAPI server.   ")
    print("==========================================================")

if __name__ == "__main__":
    main()
