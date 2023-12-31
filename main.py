import cv2
from cvzone.HandTrackingModule import HandDetector  # Make sure cvzone is installed

cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=2)

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img, draw=True)  # Enable draw parameter

    for hand in hands:
        # Draw Bounding Box
        bbox = hand["bbox"]
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 255), 2)

        # Draw Landmarks
        lmList = hand["lmList"]
        for landmark in lmList:
            # Extracting x and y coordinates
            x, y, z = landmark
            cv2.circle(img, (int(x), int(y)), 5, (0, 255, 0), cv2.FILLED)

        # Hand Type and Index
        handType = hand["type"]
        index = hands.index(hand) + 1
        cv2.putText(img, f"{handType} Hand - Index: {index}", (10, 30 * index), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2)

        # Finger Information
        fingers = detector.fingersUp(hand)

        if fingers[0] == 1 and fingers[1] == 1 and all(fingers[2:]):
            cv2.putText(img, f" five!!! - Hand {index}", (int(hand["center"][0]), int(hand["center"][1]) - 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            cv2.putText(img, f"Fingers: {fingers.count(1)}", (int(hand["center"][0]), int(hand["center"][1]) - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
