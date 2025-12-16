import cv2
import os
import shutil
import numpy as np
from dreamer import DeepDreamer
import optical_flow as flow_est

INPUT_VIDEO = "input.mp4"
OUTPUT_VIDEO = "output.mp4"
TEMP_DIR = "temp"

# 0.0 = Pure Warp (blurry), 1.0 = No Warp (flickering).
BLEND_WEIGHT = 0.3

def process_video():
    cwd = os.getcwd()
    abs_temp_dir = os.path.join(cwd, TEMP_DIR)
    
    try:
        input_frames_dir = os.path.join(abs_temp_dir, "input")
        output_frames_dir = os.path.join(abs_temp_dir, "output")
        flow_frames_dir = os.path.join(abs_temp_dir, "flow")

        print(f"Creating temporary directories at: {abs_temp_dir}")
        os.makedirs(input_frames_dir, exist_ok=True)
        os.makedirs(output_frames_dir, exist_ok=True)
        os.makedirs(flow_frames_dir, exist_ok=True)

        raft_model, raft_transforms, device = flow_est.init_raft()
        
        dreamer_args = [
            "-gpu", "mps",
            "-image_size", "1024",
            "-save_iter", "0",
            "-print_iter", "0",
            "-num_iterations", "1",
        ]
        dreamer = DeepDreamer(dreamer_args)
        print("Dreamer model loaded.")

        if not os.path.exists(INPUT_VIDEO):
            raise FileNotFoundError(f"Input video not found at: {INPUT_VIDEO}")

        cap = cv2.VideoCapture(INPUT_VIDEO)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {INPUT_VIDEO}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video: {total_frames} frames, {fps} FPS, {width}x{height}")

        frame_count = 0
        prev_frame = None
        prev_dream = None

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            print(f"Processing frame {frame_count}/{total_frames}")

            input_frame_path = os.path.join(
                input_frames_dir, f"frame_{frame_count:06d}.jpg"
            )
            output_frame_path = os.path.join(
                output_frames_dir, f"frame_{frame_count:06d}.jpg"
            )
            flow_path = os.path.join(
                flow_frames_dir, f"flow_{frame_count:06d}.jpg"
            )

            img_to_dream = frame.copy()

            if prev_frame is not None:
                flow_data, flow_vis = flow_est.estimate_flow(
                    prev_frame,
                    frame,
                    raft_model,
                    raft_transforms,
                    device
                )

                cv2.imwrite(flow_path, flow_vis)

                if prev_dream is not None:
                    if prev_dream.shape[:2] != frame.shape[:2]:
                        prev_dream = cv2.resize(
                            prev_dream,
                            (frame.shape[1], frame.shape[0]),
                            interpolation=cv2.INTER_LINEAR
                        )

                    warped_prev_dream = flow_est.warp_image(
                        prev_dream, flow_data
                    )

                    img_to_dream = cv2.addWeighted(
                        frame,
                        BLEND_WEIGHT,
                        warped_prev_dream,
                        (1 - BLEND_WEIGHT),
                        0
                    )
            else:
                cv2.imwrite(flow_path, np.zeros_like(frame))

            prev_frame = frame.copy()

            cv2.imwrite(input_frame_path, img_to_dream)
            dreamer.dream(input_frame_path, output_frame_path)

            if os.path.exists(output_frame_path):
                prev_dream = cv2.imread(output_frame_path)

                if prev_dream is None:
                    print(
                        f"Warning: Could not read dream output "
                        f"{output_frame_path}"
                    )
                else:
                    if prev_dream.shape[:2] != (height, width):
                        prev_dream = cv2.resize(
                            prev_dream,
                            (width, height),
                            interpolation=cv2.INTER_LINEAR
                        )

                        cv2.imwrite(output_frame_path, prev_dream)
            else:
                print(
                    f"Warning: Dreamer did not produce output at "
                    f"{output_frame_path}"
                )

            frame_count += 1

        cap.release()
        print(f"Processed {frame_count} frames.")

        print("Reassembling video...")
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        out = cv2.VideoWriter(
            OUTPUT_VIDEO, fourcc, fps, (width, height)
        )

        frames_written = 0
        for i in range(frame_count):
            output_frame = os.path.join(
                output_frames_dir, f"frame_{i:06d}.jpg"
            )

            if os.path.exists(output_frame):
                frame_read = cv2.imread(output_frame)
                if frame_read is not None:
                    if frame_read.shape[:2] != (height, width):
                        frame_read = cv2.resize(
                            frame_read, (width, height)
                        )
                    out.write(frame_read)
                    frames_written += 1

        out.release()
        print(
            f"Video saved to {OUTPUT_VIDEO} "
            f"with {frames_written} frames."
        )

    finally:
        if os.path.exists(abs_temp_dir):
            print(f"Cleaning up temp directory: {abs_temp_dir}")
            shutil.rmtree(abs_temp_dir)

if __name__ == "__main__":
    process_video()