import math
import cv2
import numpy as np
import time
import mediapipe as mp
import matplotlib.pyplot as plt
from numpy import average

# Initializing mediapipe pose class.
mp_pose = mp.solutions.pose

# Setting up the Pose function.
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.3, model_complexity=2)

# Initializing mediapipe drawing class, useful for annotation.
mp_drawing = mp.solutions.drawing_utils



def detectPose(image, pose, display=True):
    
    # Create a copy of the input image.
    output_image = image.copy()
    
    # Convert the image from BGR into RGB format.
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Perform the Pose Detection.
    results = pose.process(imageRGB)

    # Retrieve the height and width of the input image.
    h, w, _ = image.shape
    
    # Initialize a list to store the detected landmarks.
    landmarks = []


        # Iterate over the detected landmarks.
    for landmark in results.pose_landmarks.landmark:
            
            #Append the landmark into the list.
        landmarks.append((int(landmark.x * w), int(landmark.y * h),
                                  (landmark.z * w)))
    #mid = np.average(landmarks[11][0:2],landmarks[12][0:2])
    mid_x = int(np.round((landmarks[11][0]+landmarks[12][0])/2))
    mid_y = int(np.round((landmarks[11][1]+landmarks[12][1])/2))
    print(landmarks[0][0:2])
    
    cv2.circle(output_image,(mid_x,mid_y),10,(255, 0, 0),-10)
   
    #Neck
    nose_2_x = int(np.round(landmarks[0][0]))
    nose_2_y = int(np.round(landmarks[0][1] + 100 ))
    cv2.line(output_image,(mid_x,mid_y),(nose_2_x,nose_2_y),(0, 51, 204),3)

    #Chin 
    cv2.circle(output_image,(nose_2_x,nose_2_y),10,(255, 0, 0),-10)
    #Left arm
    cv2.circle(output_image,landmarks[12][0:2],10,(255, 0, 0),-10)
    cv2.circle(output_image,landmarks[14][0:2],10,(255, 0, 0),-10)
    cv2.circle(output_image,landmarks[16][0:2],10,(255, 0, 0),-10)
    
    cv2.line(output_image,landmarks[12][0:2],landmarks[14][0:2],(204, 0, 204),3)
    cv2.line(output_image,landmarks[14][0:2],landmarks[16][0:2],(204, 0, 204),3)
   
    #Right arm
    cv2.circle(output_image,landmarks[11][0:2],10,(255, 0, 0),-10)
    cv2.circle(output_image,landmarks[13][0:2],10,(255, 0, 0),-10)
    cv2.circle(output_image,landmarks[15][0:2],10,(255, 0, 0),-10)

    cv2.line(output_image,landmarks[11][0:2],landmarks[13][0:2],(34,255,34),3)
    cv2.line(output_image,landmarks[13][0:2],landmarks[15][0:2],(34,255,34),3)
    
    #Shoulder
    cv2.line(output_image,landmarks[11][0:2],landmarks[12][0:2],(255, 255, 102),3)
   
    
    # Check if the original input image and the resultant image are specified to be displayed.
      
        # Return the output image and the found landmarks.
    return output_image, landmarks

def calculateAngle(landmark1, landmark2, landmark3):
 
    # Get the required landmarks coordinates.
    x1, y1,_ = landmark1
    x2, y2,_ = landmark2
    x3, y3,_ = landmark3
 
    # Calculate the angle between the three points
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    
    # Check if the angle is less than zero.
    if angle < 0:
 
        # Add 360 to the found angle.
        angle += 360
    
    # Return the calculated angle.
    return angle

# Setup Pose function for video.
#count = 0
#stage = None
pose_video = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, model_complexity=1)
pTime = 0


# Initialize the VideoCapture object to read from a video stored in the disk.
video = cv2.VideoCapture(0)

# Iterate until the video is accessed successfully.
while True:
    # Read a frame.
    ok, frame = video.read()
    
    # Check if frame is not read properly.
    if not ok:
        
        # Break the loop.
        break
    
    # Flip the frame horizontally for natural (selfie-view) visualization.
    frame = cv2.flip(frame, 1)
    
    # Get the width and height of the frame
    frame_height, frame_width, _ =  frame.shape
    
    # Resize the frame while keeping the aspect ratio.
    frame = cv2.resize(frame, (int(frame_width * (640 / frame_height)), 640))
    
    # Perform Pose landmark detection.
    frame, landmarks = detectPose(frame, pose_video, display=False)
    
    # Set the time for this frame to the current time.
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
        
        # Write the calculated number of frames per second on the frame. 
    
    # Update the previous frame time to this frame time.
    # As this frame will become previous frame in next iteration.
    
    # Display the frame.
    #cv2.putText(frame, str(int(fps)), (50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0), 3)

    angle = calculateAngle(landmarks[11][0:3], landmarks[13][0:3], landmarks[15][0:3])
    angle= int(np.round(angle))
    #Counter 
    if angle < 75 :
        cv2.line(frame,landmarks[11][0:2],landmarks[13][0:2],(0,0,255),3)
        cv2.line(frame,landmarks[13][0:2],landmarks[15][0:2],(0,0,255),3) 
        
        

    else : 
    #if angle < 50 and stage == "Down" :  
    #    stage = "Up"
        cv2.line(frame,landmarks[11][0:2],landmarks[13][0:2],(0,255,0),3)
        cv2.line(frame,landmarks[13][0:2],landmarks[15][0:2],(0,255,0),3)
    #    count=+1
    #    print(count)
    

    # Display the calculated angle.
    org = (200, 50)
    cv2.putText(frame, "Angle",(10,50), cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,0), 3)

    cv2.putText(frame, str(int(np.round(angle))),org, cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,0), 3)
    #cv2.putText(frame, str(int(count)), (30,30), cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0), 3)

    cv2.imshow('Pose Detection', frame)
    print(f'The calculated right arm angle is {angle}')
    # Wait until a key is pressed.

    
    # Check if 'ESC' is pressed.
    if 0xFF & cv2.waitKey(5) == 27:
        
        # Break the loop.
        break

# Release the VideoCapture object.
video.release()

# Close the windows.
cv2.destroyAllWindows()