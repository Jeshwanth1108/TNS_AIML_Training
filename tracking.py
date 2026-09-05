from ultralytics import YOLO
import cv2
import os


def process_video(input_video, output_video):

    # Load YOLO model
    model = YOLO("yolov8n.pt")

    # Open input video
    cap = cv2.VideoCapture(input_video)

    if not cap.isOpened():
        raise Exception("Could not open input video")

    # Video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Output video
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(
        output_video,
        fourcc,
        fps,
        (width, height)
    )

    # Counting line
    line_x = width // 2

    margin = 25

    left_line = line_x - margin
    right_line = line_x + margin

    # Tracking information
    last_side = {}
    counted_ids = set()

    # Counters
    left_to_right = 0
    right_to_left = 0

    frame_number = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_number += 1

        # YOLO + ByteTrack
        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            classes=[0],
            conf=0.35,
            verbose=False
        )

        if results[0].boxes.id is not None:

            boxes = results[0].boxes.xyxy.cpu().numpy()
            ids = results[0].boxes.id.cpu().numpy().astype(int)

            for box, track_id in zip(boxes, ids):

                x1, y1, x2, y2 = map(int, box)

                # Bottom-center point
                center_x = (x1 + x2) // 2
                bottom_y = y2

                # Determine side
                if center_x < left_line:
                    current_side = "LEFT"

                elif center_x > right_line:
                    current_side = "RIGHT"

                else:
                    current_side = "CENTER"

                # Initialize track
                if track_id not in last_side:

                    if current_side in ["LEFT", "RIGHT"]:
                        last_side[track_id] = current_side

                else:

                    previous_side = last_side[track_id]

                    # LEFT → RIGHT
                    if (
                        previous_side == "LEFT"
                        and current_side == "RIGHT"
                        and track_id not in counted_ids
                    ):

                        left_to_right += 1
                        counted_ids.add(track_id)

                    # RIGHT → LEFT
                    elif (
                        previous_side == "RIGHT"
                        and current_side == "LEFT"
                        and track_id not in counted_ids
                    ):

                        right_to_left += 1
                        counted_ids.add(track_id)

                    # Update side only outside counting zone
                    if current_side in ["LEFT", "RIGHT"]:
                        last_side[track_id] = current_side

                # Draw bounding box
                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # Draw ID
                cv2.putText(
                    frame,
                    f"ID: {track_id}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

                # Draw tracking point
                cv2.circle(
                    frame,
                    (center_x, bottom_y),
                    5,
                    (0, 0, 255),
                    -1
                )

        # Draw counting line
        cv2.line(
            frame,
            (line_x, 0),
            (line_x, height),
            (255, 0, 0),
            3
        )

        # Draw counting zone
        cv2.line(
            frame,
            (left_line, 0),
            (left_line, height),
            (0, 255, 255),
            1
        )

        cv2.line(
            frame,
            (right_line, 0),
            (right_line, height),
            (0, 255, 255),
            1
        )

        # Total count
        total_count = left_to_right + right_to_left

        # Display counters
        cv2.putText(
            frame,
            f"Total: {total_count}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Left -> Right: {left_to_right}",
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Right -> Left: {right_to_left}",
            (20, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # Write frame
        out.write(frame)

    cap.release()
    out.release()

    return {
        "total": left_to_right + right_to_left,
        "left_to_right": left_to_right,
        "right_to_left": right_to_left,
        "frames": frame_number,
        "output_video": output_video
    }