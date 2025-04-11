import cv2
import time
import numpy as np
import os
from datetime import datetime

# Configuration
IR_THRESHOLD = 230
sampling_interval = 0.0667  # 15Hz (~66.7ms between bits)
num_bits_to_collect = 6

# Known IR patterns
known_patterns = {
    "Player 1": [1, 1, 0, 0, 1, 0],
    "Player 2": [1, 1, 0, 1, 0, 0],
}

# Initialize camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 60)  # 60 FPS should suffice for 15Hz sampling

# FPS variables
prev_time = time.time()
frame_count = 0
fps = 0

# Session folder
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
session_dir = f"ir_session_{timestamp}"
os.makedirs(session_dir, exist_ok=True)

# Collection variables
collecting = False
collected_bits = []
bit_positions = []
bit_timestamps = []
next_sample_time = 0
pattern_counter = 1
current_pattern_dir = None
bit_dirs = []

print(f"Monitoring for double IR pulse... Press 'q' to quit.\nSaving frames to: {session_dir}")

def detect_bit_and_position(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    max_val = np.max(gray)
    bit = 1 if max_val > IR_THRESHOLD else 0
    position = None
    if bit == 1:
        max_pos = np.unravel_index(np.argmax(gray), gray.shape)
        position = (max_pos[1], max_pos[0])  # (x, y)
    return bit, position, gray

def create_pattern_folder(pattern_number):
    pattern_path = os.path.join(session_dir, f"pattern_{pattern_number}")
    os.makedirs(pattern_path, exist_ok=True)
    dirs = []
    for i in range(1, num_bits_to_collect + 1):
        bit_path = os.path.join(pattern_path, f"bit_{i}")
        os.makedirs(bit_path, exist_ok=True)
        dirs.append(bit_path)
    return pattern_path, dirs

def save_gray_frame(gray, folder, bit_index):
    img_name = f"bit{bit_index}_{datetime.now().strftime('%H%M%S_%f')}.png"
    path = os.path.join(folder, img_name)
    cv2.imwrite(path, gray)

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # FPS calculation
        frame_count += 1
        now = time.time()
        elapsed = now - prev_time
        if elapsed >= 1.0:
            fps = frame_count / elapsed
            frame_count = 0
            prev_time = now

        bit, position, gray = detect_bit_and_position(frame)

        # Show FPS
        cv2.putText(gray, f"FPS: {fps:.2f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
        cv2.imshow("IR Detection View", gray)

        if not collecting:
            if bit == 1:
                time.sleep(sampling_interval)  # Wait 66.7ms for second pulse
                ret, frame = cap.read()
                if not ret:
                    break
                second_bit, second_position, second_gray = detect_bit_and_position(frame)
                if second_bit == 1:
                    collecting = True
                    collected_bits = [1, 1]
                    bit_positions = [position, second_position]

                    # Capture timestamps
                    t1 = time.time()
                    t2 = time.time()
                    bit_timestamps = [t1, t2]

                    # Create folder for this pattern
                    current_pattern_dir, bit_dirs = create_pattern_folder(pattern_counter)
                    print(f"\n📂 Started pattern_{pattern_counter} in {current_pattern_dir}")
                    print("Started collecting IR bits at 15Hz (66.7ms intervals)...")

                    # Save first two bits
                    save_gray_frame(gray, bit_dirs[0], 1)
                    save_gray_frame(second_gray, bit_dirs[1], 2)

                    # Print Δt for bit 1 and 2
                    print(f"🕒 Bit 1: 1, Δt = 0.0000 sec")
                    print(f"🕒 Bit 2: 1, Δt = {(t2 - t1):.4f} sec")

                    # Setup sampling
                    next_sample_time = t2 + sampling_interval

        else:
            now = time.time()
            if now >= next_sample_time:
                ret, frame = cap.read()
                if not ret:
                    break
                sample_time = time.time()
                bit, position, gray = detect_bit_and_position(frame)
                collected_bits.append(bit)
                bit_positions.append(position)
                bit_timestamps.append(sample_time)

                index = len(collected_bits)
                save_gray_frame(gray, bit_dirs[index - 1], index)

                delta_t = sample_time - bit_timestamps[index - 2]
                print(f"🕒 Bit {index}: {bit}, Δt = {delta_t:.4f} sec")

                # Schedule next sample
                next_sample_time += sampling_interval

                # Adjust for drift or processing delay
                sleep_time = next_sample_time - time.time()
                if sleep_time > 0:
                    time.sleep(sleep_time)

                if len(collected_bits) == num_bits_to_collect:
                    print(f"\n✅ Detected pattern: {collected_bits}")
                    matched = False
                    for player, pattern in known_patterns.items():
                        if collected_bits == pattern:
                            for i in reversed(range(len(collected_bits))):
                                if collected_bits[i] == 1 and bit_positions[i]:
                                    px, py = bit_positions[i]
                                    print(f"👤 {player} triggered at position: x={px}, y={py}")
                                    break
                            matched = True
                            break
                    if not matched:
                        print("❌ No matching pattern found.")

                    collecting = False
                    pattern_counter += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Stopped by user.")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n🗂️ Saved all IR pattern folders inside: {session_dir}")