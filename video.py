import cv2
import os
import subprocess
import shutil


INPUT_VIDEO = "input.mp4"
OUTPUT_VIDEO = "output.mp4"
TEMP_DIR = "temp"
NEURAL_DREAM_SCRIPT = "neural_dream.py"


def process_frame(input_frame, output_frame):
    cmd = [
        "python", NEURAL_DREAM_SCRIPT,
        "-content_image", input_frame,
        "-output_image", output_frame,
        "-gpu", "mps",
        "-save_iter", "0"
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"\nError processing frame:")
        print(result.stderr)
        raise RuntimeError(f"Failed to process frame: {input_frame}")


def process_video():
    input_frames_dir = os.path.join(TEMP_DIR, "input")
    output_frames_dir = os.path.join(TEMP_DIR, "output")

    os.makedirs(input_frames_dir, exist_ok=True)
    os.makedirs(output_frames_dir, exist_ok=True)

    try:
        cap = cv2.VideoCapture(INPUT_VIDEO)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {INPUT_VIDEO}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        print(f"Video: {total_frames} frames, {fps} FPS, {width}x{height}")

        print("Extracting frames...")
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_path = os.path.join(
                input_frames_dir, f"frame_{frame_count:06d}.jpg"
            )
            cv2.imwrite(frame_path, frame)
            frame_count += 1

            if frame_count % 10 == 0:
                print(f"Extracted {frame_count}/{total_frames} frames", end="\r")

        cap.release()
        print(f"\nExtracted all {frame_count} frames")

        # Process each frame
        print("Processing frames...")
        for i in range(180):  # TODO: change back to frame_count
            input_frame = os.path.join(
                input_frames_dir, f"frame_{i:06d}.jpg"
            )
            output_frame = os.path.join(
                output_frames_dir, f"frame_{i:06d}.jpg"
            )

            print(f"Processing frame {i}/{frame_count}")
            process_frame(input_frame, output_frame)

        print(f"Processed all {frame_count} frames")

        # Reassemble video (FIXED)
        print("Reassembling video...")

        fourcc = cv2.VideoWriter_fourcc(*"avc1")  # H.264
        out = cv2.VideoWriter(
            OUTPUT_VIDEO,
            fourcc,
            fps,
            (width, height)
        )

        if not out.isOpened():
            raise RuntimeError("Failed to open VideoWriter with H.264")

        for i in range(180):  # TODO: change back to frame_count
            output_frame = os.path.join(
                output_frames_dir, f"frame_{i:06d}.jpg"
            )
            frame = cv2.imread(output_frame)

            if frame is None:
                print(f"\nWarning: Could not read frame {i}")
                continue

            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))

            out.write(frame)

            if (i + 1) % 10 == 0:
                print(f"Writing frame {i + 1}/{frame_count}", end="\r")

        out.release()
        print(f"Video saved to {OUTPUT_VIDEO}")

    finally:
        print("Cleaning up...")
        if os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)
        print("Done!")


if __name__ == "__main__":
    process_video()
