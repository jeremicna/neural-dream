import cv2
import os
import shutil
from dreamer import DeepDreamer # Import the class we created

INPUT_VIDEO = "input.mp4"
OUTPUT_VIDEO = "output.mp4"
TEMP_DIR = "temp"

def process_video():
    input_frames_dir = os.path.join(TEMP_DIR, "input")
    output_frames_dir = os.path.join(TEMP_DIR, "output")

    os.makedirs(input_frames_dir, exist_ok=True)
    os.makedirs(output_frames_dir, exist_ok=True)

    try:        
        dreamer_args = [
            "-gpu", "mps",
            "-save_iter", "0",
            "-print_iter", "0",
            "-num_iterations", "10",
        ]
        
        dreamer = DeepDreamer(dreamer_args)
        print("Model loaded successfully.")

        cap = cv2.VideoCapture(INPUT_VIDEO)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {INPUT_VIDEO}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video: {total_frames} frames, {fps} FPS, {width}x{height}")

        frame_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            input_frame_path = os.path.join(input_frames_dir, f"frame_{frame_count:06d}.jpg")
            output_frame_path = os.path.join(output_frames_dir, f"frame_{frame_count:06d}.jpg")
            
            cv2.imwrite(input_frame_path, frame)

            print(f"Processing frame {frame_count}/{total_frames}")

            dreamer.dream(input_frame_path, output_frame_path)
            
            frame_count += 1

            if frame_count > 100:  # Limit for testing
                print("\nFrame limit reached for testing.")
                break

        cap.release()
        print(f"\nProcessed {frame_count} frames.")

        print("Reassembling video...")
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

        for i in range(frame_count):
            output_frame = os.path.join(output_frames_dir, f"frame_{i:06d}.jpg")
            if not os.path.exists(output_frame):
                output_frame = os.path.join(output_frames_dir, f"frame_{i:06d}.png")
            
            frame = cv2.imread(output_frame)
            if frame is None:
                print(f"Warning: Frame {i} missing/corrupt.")
                continue
                
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
                
            out.write(frame)

        out.release()
        print(f"Video saved to {OUTPUT_VIDEO}")

    finally:
        print("Cleaning up...")
        if os.path.exists(TEMP_DIR):
            shutil.rmtree(TEMP_DIR)
        print("Done!")

if __name__ == "__main__":
    process_video()