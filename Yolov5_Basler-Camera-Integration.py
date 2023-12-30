efrom pypylon import pylon
import cv2
import os
import numpy as np
import torch
import cv2
from pathlib import Path


path1 = "/home/s186/Downloads/Data_colection/weights/best.pt"
# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom',path1)
# Set the model to evaluation mode
model.eval()

# Define the classes of objects that the model can detect
classes = model.module.names if hasattr(model, 'module') else model.names
camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly) 
converter = pylon.ImageFormatConverter()
converter.OutputPixelFormat = pylon.PixelType_BGR8packed
converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

# Specify the custom path and filename for the output video
custom_path = '/media/kshitizkumar/3D50BB136F37ADE5/Live Demo/'
custom_path2 = '/media/kshitizkumar/3D50BB136F37ADE5/Live Demo/'
video_filename1 = 'lingampally-2.avi'
video_filename2 = 'Ling_Results-2.avi'
output_path = os.path.join(custom_path, video_filename1)
output_path2 = os.path.join(custom_path2,video_filename2) 
# Define the video writer
fps = 20  # Adjust the frame rate as needed
video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (1920, 1080))  # Update resolution if needed
video_writer2 = cv2.VideoWriter(output_path2, cv2.VideoWriter_fourcc(*'XVID'), fps, (1920, 1080))
confidence_score = 0.73
while camera.IsGrabbing():
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
    if grabResult.GrabSucceeded():
        # Access the image data
        image = converter.Convert(grabResult)
        img = image.GetArray()
        frame = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_AREA)
        # Convert the frame from BGR to RGB format
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        video_writer.write(frame)

        # Run the model on the input image
        results = model(img)

        # Parse the output and draw bounding boxes on the image
        for result in results.xyxy[0]:
            xmin, ymin, xmax, ymax, score, class_id = result.tolist()
            if score >= confidence_score:
                xmin = int(xmin)
                ymin = int(ymin)
                xmax = int(xmax)
                ymax = int(ymax)
                if classes[int(class_id)]:
                    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
                    # s = str(classes[int(class_id)])+str(score: .2f)
                    s = f"{classes[int(class_id)]} Score: {score:.2f}"
                    cv2.putText(frame,s, (xmin, ymin - 5),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Write the frame to the output video
        video_writer2.write(frame)

        # Display the image
        cv2.imshow('Object Detection', frame)

        # Exit the loop if the 'q' key is pressed
        if cv2.waitKey(1) == ord('q'):
            break

    grabResult.Release()

# Releasing resources
camera.StopGrabbing()
video_writer.release()
cv2.destroyAllWindows()

