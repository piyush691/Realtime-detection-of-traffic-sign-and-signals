# Detecting Objects in Real Time with OpenCV deep learning library
# Importing needed libraries

import numpy as np
import cv2
import time

# Reading stream video from camera

# reading stream video from camera
camera = cv2.VideoCapture(0)

# Preparing variables for the frames
h, w = None, None

# Loading YOLO v3 network

with open('classes.names') as f:
    # Getting labels reading every line and putting them into the list
    labels = [line.strip() for line in f]


# Loading trained YOLO v3 Objects Detector
# with the help of 'dnn' library from OpenCV
network = cv2.dnn.readNetFromDarknet('yolov3.cfg','yolov3_final.weights')

# Getting list with names of all layers from YOLO v3 network
layers_names_all = network.getLayerNames()

# Getting only output layers' names that we need from YOLO v3 algorithm
layers_names_output = \
    [layers_names_all[i[0] - 1] for i in network.getUnconnectedOutLayers()]

# Setting minimum probability to eliminate weak predictions
probability_minimum = 0.5

# Setting threshold for filtering weak bounding boxes with non-maximum suppression
threshold = 0.3

# Generating colours for representing every detected object
# with function randint(low, high=None, size=None, dtype='l')
colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')


# Reading frames in the loop

# Defining loop for catching frames
while True:
    # Capturing frame-by-frame from camera
    _, frame = camera.read()

    # Getting spatial dimensions of the frame and all other frames have the same dimension
    if w is None or h is None:
        # Slicing from tuple only first two elements
        h, w = frame.shape[:2]


    # Getting blob from current frame
    # Resulted shape has number of frames, number of channels, width and height
    # blob = cv2.dnn.blobFromImage(image, scalefactor=1.0, size, mean, swapRB=True)
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                 swapRB=True, crop=False)

    #Implementing Forward pass

    network.setInput(blob)  # setting blob as input to the network
    start = time.time()
    output_from_network = network.forward(layers_names_output)
    end = time.time()

    # Showing spent time for single current frame
    print('Current frame took {:.5f} seconds'.format(end - start))


    # Getting bounding boxes

    # Preparing lists for detected bounding boxes,
    # obtained confidences and class's number
    bounding_boxes = []
    confidences = []
    class_numbers = []

    for result in output_from_network:
        # Going through all detections from current output layer
        for detected_objects in result:
            # Getting 3 classes' probabilities for current detected object
            scores = detected_objects[5:]
            # Getting index of the class with the maximum value of probability
            class_current = np.argmax(scores)
            # Getting value of probability for defined class
            confidence_current = scores[class_current]


            # Eliminating weak predictions with minimum probability
            if confidence_current > probability_minimum:

                box_current = detected_objects[0:4] * np.array([w, h, w, h])

                x_center, y_center, box_width, box_height = box_current
                x_min = int(x_center - (box_width / 2))
                y_min = int(y_center - (box_height / 2))

                # Adding results into prepared lists
                bounding_boxes.append([x_min, y_min,
                                       int(box_width), int(box_height)])
                confidences.append(float(confidence_current))
                class_numbers.append(class_current)



    # Non-maximum suppression

    results = cv2.dnn.NMSBoxes(bounding_boxes, confidences,
                               probability_minimum, threshold)


   # Drawing bounding boxes and labels


    # Checking if there is at least one detected object
    # after non-maximum suppression
    if len(results) > 0:
        # Going through indexes of results
        for i in results.flatten():
            # Getting current bounding box coordinates,
            # its width and height
            x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
            box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]

            # Preparing colour for current bounding box
            colour_box_current = colours[class_numbers[i]].tolist()


            # Drawing bounding box on the original current frame
            cv2.rectangle(frame, (x_min, y_min),
                          (x_min + box_width, y_min + box_height),
                          colour_box_current, 2)

            # Preparing text with label and confidence for current bounding box
            text_box_current = '{}: {:.4f}'.format(labels[int(class_numbers[i])],
                                                   confidences[i])

            # Putting text with label and confidence on the original image
            cv2.putText(frame, text_box_current, (x_min, y_min - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour_box_current, 2)


   # Showing processed frames in OpenCV Window


    cv2.namedWindow('YOLO v3 Real Time Detections', cv2.WINDOW_NORMAL)
    cv2.imshow('YOLO v3 Real Time Detections', frame)

    # Breaking the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


# Releasing camera
camera.release()
# Destroying all opened OpenCV windows
cv2.destroyAllWindows()

