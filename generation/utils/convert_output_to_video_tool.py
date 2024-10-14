import os
import argparse
from utils import video_utils
from concurrent.futures import ThreadPoolExecutor

def render_video(dir_path, output_path, video_utils_instance):
    """Render video for a specific directory."""
    video_utils_instance.render_gaussian_splatting_video(dir_path, output_path, 24)

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Render videos from obj files.")
    parser.add_argument(
        "--dir", default="logs", type=str, help="Directory where obj files are stored"
    )
    parser.add_argument(
        "--out", default="videos", type=str, help="Directory where videos will be saved"
    )
    return parser.parse_args()

def main():
    """Main entry point of the script."""
    args = parse_arguments()

    # Create output directory only if it doesn't exist
    os.makedirs(args.out, exist_ok=True)

    # Initialize VideoUtils instance once
    video_utils_instance = video_utils.VideoUtils(512, 512, 4, 5, 10, -30, 10)

    # Use ThreadPoolExecutor for parallel processing if there are multiple directories to render
    with ThreadPoolExecutor() as executor:
        executor.submit(render_video, args.dir, args.out, video_utils_instance)

if __name__ == "__main__":
    main()
