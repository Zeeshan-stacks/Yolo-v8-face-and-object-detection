import cv2
import argparse
from ultralytics import YOLO
import supervision as sv
import numpy as np

# Define the detection zone (left half of the screen)
ZONE_POLYGON = np.array([
    [0, 0],
    [0.5, 0],
    [0.5, 1],
    [0, 1]
])

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 live")
    parser.add_argument(
        "--webcam-resolution",
        default=[1280, 720],
        nargs=2,
        type=int
    )
    return parser.parse_args()

def main():
    args = parse_arguments()
    frame_width, frame_height = args.webcam_resolution

    # Open webcam and set resolution
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

    # Load YOLOv8 model
    model = YOLO("yolov8l.pt")

    # Initialize box annotator (text_thickness removed to avoid error)
    box_annotator = sv.BoxAnnotator(
        thickness=2,
        color=sv.Color.RED
    )

    # Scale and set up the detection zone
    zone_polygon = (ZONE_POLYGON * np.array(args.webcam_resolution)).astype(int)
    zone = sv.PolygonZone(polygon=zone_polygon)
    # , frame_resolution_wh = tuple(args.webcam_resolution)
    zone_annotator = sv.PolygonZoneAnnotator(
        zone=zone,
        color=sv.Color.RED,
        thickness=2,
        text_scale=2
    )

    # Main loop
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run inference
        result = model(frame, agnostic_nms=True)[0]
        detections = sv.Detections.from_ultralytics(result)

        # Create labels
        labels = [
            f"{model.model.names[class_id]} {confidence:.2f}"
            for class_id, confidence in zip(detections.class_id, detections.confidence)
        ]

        # Annotate with boxes and labels
        frame = box_annotator.annotate(
            scene=frame,
            detections=detections
        )

        # Draw labels manually
        for xyxy, label in zip(detections.xyxy, labels):
            x1, y1, x2, y2 = map(int, xyxy)
            cv2.putText(
                frame,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
        # Check if detection is inside the zone
        zone.trigger(detections=detections)

        # Draw zone overlay
        frame = zone_annotator.annotate(scene=frame)

        # Display result
        cv2.imshow("YOLOv8", frame)

        # Exit on ESC key
        if cv2.waitKey(30) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
