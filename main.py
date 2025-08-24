import cv2
import mediapipe as mp
import numpy as np
import time
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import threading
import os

DEBUG = True

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
coords = []


def calculate_angle(a, b, c):
    """Calculates the angle between three points (e.g., a joint)."""
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle


def analyze_squat(landmarks, width, height):
    """Analyze squat form by choosing the more visible side of the body."""
    score = 100
    rep_counts = False
    stage = ""

    # --- Visibility Check ---
    left_hip_lm = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    left_knee_lm = landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value]
    left_ankle_lm = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    right_hip_lm = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    right_knee_lm = landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value]
    right_ankle_lm = landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]

    left_visibility = left_hip_lm.visibility + left_knee_lm.visibility + left_ankle_lm.visibility
    right_visibility = right_hip_lm.visibility + right_knee_lm.visibility + right_ankle_lm.visibility

    side_used = 'LEFT' if left_visibility > right_visibility else 'RIGHT'

    if side_used == 'LEFT':
        shoulder_lm = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        hip_lm = left_hip_lm
        knee_lm = left_knee_lm
        ankle_lm = left_ankle_lm
    else:
        shoulder_lm = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        hip_lm = right_hip_lm
        knee_lm = right_knee_lm
        ankle_lm = right_ankle_lm

    # Get coordinates from the chosen side's landmarks
    shoulder = [shoulder_lm.x, shoulder_lm.y]
    hip = [hip_lm.x, hip_lm.y]
    knee = [knee_lm.x, knee_lm.y]
    ankle = [ankle_lm.x, ankle_lm.y]

    # Calculate angles
    knee_angle = calculate_angle(hip, knee, ankle)
    hip_angle = calculate_angle(shoulder, hip, knee)

    # Original logic for rep counting and stage
    if knee_angle < 100:
        rep_counts = True

    return {
        'knee_angle': knee_angle,
        'hip_angle': hip_angle,
        'knee_pos': knee,
        'hip_pos': hip,
        'primary_y': hip[1],
        'primary_angle': knee_angle,
        'rep_counts': rep_counts,
        'stage': stage
    }


def analyze_pushup(landmarks, width, height):
    """Analyze push-up form using the more visible arm for elbow angle."""
    score = 100
    rep_counts = False

    # --- Visibility Check for arms ---
    left_shoulder_lm = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    left_elbow_lm = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    left_wrist_lm = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    right_shoulder_lm = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    right_elbow_lm = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
    right_wrist_lm = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

    left_visibility = left_shoulder_lm.visibility + left_elbow_lm.visibility + left_wrist_lm.visibility
    right_visibility = right_shoulder_lm.visibility + right_elbow_lm.visibility + right_wrist_lm.visibility

    side_used = 'LEFT' if left_visibility > right_visibility else 'RIGHT'

    if side_used == 'LEFT':
        shoulder = [left_shoulder_lm.x, left_shoulder_lm.y]
        elbow = [left_elbow_lm.x, left_elbow_lm.y]
        wrist = [left_wrist_lm.x, left_wrist_lm.y]
        hip = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x, landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y]
        knee = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y]
    else:
        shoulder = [right_shoulder_lm.x, right_shoulder_lm.y]
        elbow = [right_elbow_lm.x, right_elbow_lm.y]
        wrist = [right_wrist_lm.x, right_wrist_lm.y]
        hip = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y]
        knee = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x, landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y]

    # Calculate angles
    elbow_angle = calculate_angle(shoulder, elbow, wrist)
    body_angle = calculate_angle(shoulder, hip, knee)

    if elbow_angle < 90:
        rep_counts = True

    # Original return dictionary (no rep counting)
    return {
        'elbow_angle': elbow_angle,
        'body_angle': body_angle,
        'elbow_pos': elbow,
        'hip_pos': hip,
        'primary_angle': elbow_angle,
        'primary_y': shoulder[1],
        'rep_counts': rep_counts,
    }


def analyze_pullup(landmarks, width, height):
    """Analyze pull-up form using the more visible arm."""

    score = 100
    rep_counts = False

    # --- Visibility Check for arms ---
    left_shoulder_lm = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    left_elbow_lm = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value]
    left_wrist_lm = landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    right_shoulder_lm = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    right_elbow_lm = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value]
    right_wrist_lm = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]

    left_visibility = left_shoulder_lm.visibility + left_elbow_lm.visibility + left_wrist_lm.visibility
    right_visibility = right_shoulder_lm.visibility + right_elbow_lm.visibility + right_wrist_lm.visibility

    side_used = 'LEFT' if left_visibility > right_visibility else 'RIGHT'

    if side_used == 'LEFT':
        shoulder = [left_shoulder_lm.x, left_shoulder_lm.y]
        elbow = [left_elbow_lm.x, left_elbow_lm.y]
        wrist = [left_wrist_lm.x, left_wrist_lm.y]
    else:
        shoulder = [right_shoulder_lm.x, right_shoulder_lm.y]
        elbow = [right_elbow_lm.x, right_elbow_lm.y]
        wrist = [right_wrist_lm.x, right_wrist_lm.y]

    nose_y = landmarks[mp_pose.PoseLandmark.NOSE.value].y
    left_hand_y = left_wrist_lm.y
    right_hand_y = right_wrist_lm.y
    hands_y = (left_hand_y + right_hand_y) / 2

    elbow_angle = calculate_angle(shoulder, elbow, wrist)

    if nose_y < hands_y and elbow_angle < 45:
        rep_counts = True

    return {
        'elbow_angle': elbow_angle,
        'elbow_pos': elbow,
        'shoulder_pos': shoulder,
        'primary_angle': elbow_angle,
        'rep_counts': rep_counts,
        'primary_y': nose_y
    }


def calculate_scaled_dimensions(original_width, original_height, target_width):
    """Calculate scaled dimensions maintaining aspect ratio."""
    if target_width is None: return original_width, original_height
    aspect_ratio = original_height / original_width
    return target_width, int(target_width * aspect_ratio)


def calculate_score(min_angle, perfect_angle, current_eccentric_time, current_concentric_time):
    # score penalty ratio
    epsilon = 1.4

    # perfect eccentric / concentric ratio
    perfect_ratio = 1.5

    max_score = 100
    penalty = 0

    if perfect_angle < min_angle:
        penalty += (perfect_angle - min_angle) * epsilon

    ratio = current_eccentric_time / current_concentric_time
    if ratio < perfect_ratio:
        penalty += (perfect_ratio - ratio) * epsilon

    return max_score - penalty


def analyze_video(exercise_type, video_source, target_width=1500):
    """Main analysis function that processes the video."""
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: Could not open video source")
        messagebox.showerror("Error", f"Could not open video file: {video_source}")
        return

    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    display_width, display_height = calculate_scaled_dimensions(original_width, original_height, target_width)
    width, height = original_width, original_height

    # --- State and Counter Variables ---
    counter = 0
    stage = "concentric"
    prev_rep_counts = False
    y_avg = 0
    frames_per_stage = 0
    min_angle = 180

    perfect_angles = {
        "pullup": 40,
        "squat": 45,
        "pushup": 30
    }
    perfect_angle = perfect_angles[exercise_type]

    phase_start_time = None  # Timestamp when the current phase started
    current_concentric_time = 0.0
    current_eccentric_time = 0.0
    rep_timings = []  # List to store the final timings for each completed rep

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Reached the end of the video.")
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark

                if phase_start_time is None:
                    phase_start_time = time.time()

                # Analyze based on exercise type (Original structure)
                if exercise_type == "squat":
                    analysis = analyze_squat(landmarks, width, height)
                    cv2.putText(image, f"Knee: {int(analysis['knee_angle'])}",
                                tuple(np.multiply(analysis['knee_pos'], [width, height]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(image, f"Hip: {int(analysis['hip_angle'])}",
                                tuple(np.multiply(analysis['hip_pos'], [width, height]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                elif exercise_type == "pushup":
                    analysis = analyze_pushup(landmarks, width, height)
                    cv2.putText(image, f"Elbow: {int(analysis['elbow_angle'])}",
                                tuple(np.multiply(analysis['elbow_pos'], [width, height]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.putText(image, f"Body: {int(analysis['body_angle'])}",
                                tuple(np.multiply(analysis['hip_pos'], [width, height]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                elif exercise_type == "pullup":
                    analysis = analyze_pullup(landmarks, width, height)
                    cv2.putText(image, f"Elbow: {int(analysis['elbow_angle'])}",
                                tuple(np.multiply(analysis['elbow_pos'], [width, height]).astype(int)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

                # --- Repetition Counting and Scoring Logic ---
                cur_primary_y = analysis['primary_y']
                rep_counts = analysis['rep_counts']
                cur_angle = analysis['primary_angle']

                min_angle = min(min_angle, cur_angle)

                # --- Rep Completion Logic ---
                if rep_counts and not prev_rep_counts and (
                        time.time() - phase_start_time > 0.2 or counter == 0):
                    y_avg = cur_primary_y
                    counter += 1

                # --- Phase Change Logic ---
                phase_changed = False
                # going UP - type excercies (e.g., pullup)
                if exercise_type in ["pullup"]:
                    if stage == "concentric" and cur_primary_y > y_avg:
                        stage = "eccentric"
                        phase_changed = True
                    elif stage == "eccentric" and cur_primary_y < y_avg:
                        stage = "concentric"
                        phase_changed = True
                # going DOWN - type excercises (e.g., squat, pushup)
                elif exercise_type in ["squat", "pushup"]:
                    if stage == "concentric" and cur_primary_y < y_avg:
                        stage = "eccentric"
                        phase_changed = True
                    elif stage == "eccentric" and cur_primary_y > y_avg:
                        stage = "concentric"
                        phase_changed = True

                if phase_changed:
                    duration = time.time() - phase_start_time
                    if stage == "eccentric":  # was concentric
                        current_concentric_time += duration
                    else:  # was eccantric, so if counter > 0 it was end of rep
                        if counter and duration > 1.5:
                            current_eccentric_time += duration

                            score = calculate_score(min_angle, perfect_angle, current_eccentric_time,
                                                    current_concentric_time)

                            rep_timings.append({
                                'rep': counter,
                                'concentric': round(current_concentric_time, 2),
                                'eccentric': round(current_eccentric_time, 2)
                            })
                            print(
                                f"Rep {counter} Timings -> Concentric: {current_concentric_time:.2f}s, Eccentric: {current_eccentric_time:.2f}s")
                            print(f"Score: {score:.2f}")

                            current_concentric_time = 0.0
                            current_eccentric_time = 0.0
                            min_angle = 180

                    phase_start_time = time.time()
                    frames_per_stage = 0

                y_avg = ((y_avg * frames_per_stage) + cur_primary_y) / (frames_per_stage + 1)
                frames_per_stage += 1
                prev_rep_counts = rep_counts

            except Exception as e:
                pass

            # --- Render the status box ---
            cv2.rectangle(image, (0, 0), (550, 120), (245, 117, 16), -1)

            cv2.putText(image, 'REPS', (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(counter), (10, 95), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.putText(image, 'PHASE', (160, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, stage, (150, 95), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2, cv2.LINE_AA)

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

            if target_width:
                image = cv2.resize(image, (display_width, display_height))

            cv2.imshow(f'{exercise_type.upper()} Analysis - Press Q to quit', image)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    print(f"\nSession completed! Total {exercise_type}s performed: {counter}")


class ExerciseAnalyzerGUI:
    def __init__(self, root):
        self.width_var = None
        self.file_label = None
        self.root = root
        self.root.title("Exercise Analysis Tool")
        self.root.geometry("500x700")
        self.root.configure(bg='#f0f0f0')

        # Variables
        self.selected_exercise = tk.StringVar(value="pullup")
        self.selected_file = tk.StringVar()

        self.create_widgets()

    def create_widgets(self):
        # Main title
        title_label = tk.Label(self.root, text="Exercise Analysis Tool",
                               font=("Arial", 20, "bold"),
                               bg='#f0f0f0', fg='#333')
        title_label.pack(pady=20)

        # Exercise selection frame
        exercise_frame = tk.LabelFrame(self.root, text="Select Exercise",
                                       font=("Arial", 12, "bold"),
                                       bg='#f0f0f0', fg='#333', padx=20, pady=10)
        exercise_frame.pack(pady=10, padx=20, fill='x')

        # Exercise radio buttons
        exercises = [("Pull-up", "pullup"), ("Push-up", "pushup"), ("Squat", "squat")]

        for text, value in exercises:
            rb = tk.Radiobutton(exercise_frame, text=text, variable=self.selected_exercise,
                                value=value, font=("Arial", 11), bg='#f0f0f0', fg='#333')
            rb.pack(anchor='w', pady=5)

        # File selection frame
        file_frame = tk.LabelFrame(self.root, text="Select Video File",
                                   font=("Arial", 12, "bold"),
                                   bg='#f0f0f0', fg='#333', padx=20, pady=10)
        file_frame.pack(pady=10, padx=20, fill='x')

        # File selection button
        file_button = tk.Button(file_frame, text="Browse Video File",
                                command=self.browse_file,
                                font=("Arial", 11), bg='#4CAF50', fg='white',
                                relief='flat', padx=20, pady=8)
        file_button.pack(pady=10)

        # Selected file display
        self.file_label = tk.Label(file_frame, text="No file selected",
                                   font=("Arial", 10), bg='#f0f0f0', fg='#666',
                                   wraplength=400)
        self.file_label.pack(pady=5)

        # Options frame
        options_frame = tk.LabelFrame(self.root, text="Options",
                                      font=("Arial", 12, "bold"),
                                      bg='#f0f0f0', fg='#333', padx=20, pady=10)
        options_frame.pack(pady=10, padx=20, fill='x')

        # Target width option
        tk.Label(options_frame, text="Display Width:",
                 font=("Arial", 10), bg='#f0f0f0', fg='#333').pack(anchor='w')

        self.width_var = tk.StringVar(value="1500")
        width_entry = tk.Entry(options_frame, textvariable=self.width_var,
                               font=("Arial", 10), width=10)
        width_entry.pack(anchor='w', pady=5)

        # Start analysis button
        start_button = tk.Button(self.root, text="Start Analysis",
                                 command=self.start_analysis,
                                 font=("Arial", 14, "bold"), bg='#2196F3', fg='white',
                                 relief='flat', padx=30, pady=12)
        start_button.pack(pady=30)

    def browse_file(self):
        """Open file dialog to select video file."""
        filetypes = [
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.webm"),
            ("All files", "*.*")
        ]

        filename = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=filetypes
        )

        if filename:
            self.selected_file.set(filename)
            # Display just the filename, not the full path
            display_name = os.path.basename(filename)
            self.file_label.config(text=f"Selected: {display_name}", fg='#2196F3')
        else:
            self.file_label.config(text="No file selected", fg='#666')

    def start_analysis(self):
        """Start the video analysis."""
        if not self.selected_file.get():
            messagebox.showerror("Error", "Please select a video file first!")
            return

        if not os.path.exists(self.selected_file.get()):
            messagebox.showerror("Error", "Selected file does not exist!")
            return

        try:
            target_width = int(self.width_var.get()) if self.width_var.get().strip() else 1500
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid width value!")
            return

        # Start analysis in a separate thread to prevent GUI freezing
        analysis_thread = threading.Thread(
            target=analyze_video,
            args=(self.selected_exercise.get(), self.selected_file.get(), target_width)
        )
        analysis_thread.daemon = True
        analysis_thread.start()

        # Minimize the main window during analysis
        self.root.iconify()


def main():
    """Main function to run the GUI application."""
    root = tk.Tk()
    app = ExerciseAnalyzerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()