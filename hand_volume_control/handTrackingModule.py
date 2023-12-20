import cv2 as cv
import mediapipe as mp

class handDetector():
    def __init__(self, mode = False, maxHands = 2, complexity = 1, detectCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.complexity = complexity
        self.detectCon = detectCon

        self.trackCon = trackCon
        
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.complexity, self.detectCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils
    
    def findHand(self,frame,draw = True):
        frameRGB = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
        self.result = self.hands.process(frameRGB)
        #print(result.multi_hand_landmarks)
    
        if self.result.multi_hand_landmarks:
            for handLM in self.result.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame,handLM,self.mpHands.HAND_CONNECTIONS)
        return frame
    
    def findPosition(self,frame,handNo=0):
        LMlist = []
        if self.result.multi_hand_landmarks:
            handLM = self.result.multi_hand_landmarks[handNo]
            for ID,LM in enumerate(handLM.landmark):
                #print(id,LM)
                h,w,c = frame.shape
                cx , cy = int(LM.x*w), int(LM.y*h)
                LMlist.append([ID,cx,cy])
                #print(id,cx,cy)
        return LMlist


#dummy code

def main():
    cap = cv.VideoCapture(0)
    detector = handDetector()
    while True:
        ret, frame = cap.read()
        frame = detector.findHand(frame)
        LMlist = detector.findPosition(frame)
        if LMlist:
            print(LMlist[4])
        cv.imshow("Video",frame)
        if cv.waitKey(20) & 0xFF == ord('q'):
            break
    cap.release
    cv.destroyAllWindows()
    cv.waitKey(0)


if __name__ == "__main__": 
    main()
    