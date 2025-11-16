import cv2
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

cap=cv2.VideoCapture(0)
hands=mp_hands.Hands()
while True:
    data,image=cap.read()     #flip the image
    image=cv2.cvtColor(cv2.flip(image,1),cv2.COLOR_BGR2RGB)  #storing the image
    result=hands.process(image)  #processing the image
    image=cv2.cvtColor(image,cv2.COLOR_RGB2BGR)
    if result.multi_hand_landmarks:   #if hands are detected
        for hand_landmarks in result.multi_hand_landmarks:  #for each hand
            mp_drawing.draw_landmarks( #drawing the landmarks
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS)
    cv2.imshow("handtracking",image)  #displaying the image
    cv2.waitKey(1)