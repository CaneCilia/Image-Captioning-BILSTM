import cv2
from ultralytics import YOLO

def run_live_yolo():
    print("Loading YOLOv5 model...")
    # Load the YOLO model (yolov5su is the updated YOLOv5 architecture in ultralytics)
    model = YOLO('yolov5su.pt') 
    
    print("Starting webcam...")
    # Open the default camera (index 0)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
        
    print("Press 'q' to quit.")
    
    while True:
        # Read a frame from the webcam
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture image.")
            break
            
        # Run YOLO inference on the frame
        results = model(frame, verbose=False)
        
        # Visualize the results on the frame
        annotated_frame = results[0].plot()
        
        # Display the annotated frame
        cv2.imshow("YOLO Live Object Detection", annotated_frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    run_live_yolo()
