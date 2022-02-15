import cv2
import imutils
from time import sleep

avg = None
video = cv2.VideoCapture('people-capture.mp4')
yvalues = list()
motion = list()
count1 = 0
count2 = 0


def find_majority(motion):
    mymap = {}
    maximum = ("", 0)
    for n in motion:
        if n in mymap:
            mymap[n] += 1
        else:
            mymap[n] = 1

        if mymap[n] > maximum[1]:
            maximum = (n, mymap[n])

    return maximum


while(video.isOpened()):
    ret, frame = video.read()

    flag = True
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    tempo = float(1 / 100)
    sleep(tempo)
    frame = imutils.resize(frame, width=500)
    # cv2.imshow("Original_video", frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gaussian = cv2.GaussianBlur(gray, (21, 21), 1)
    # cv2.imshow("Gaussian_Blur", gaussian)

    if avg is None:
        avg = gaussian.copy().astype('float')
        continue

    cv2.accumulateWeighted(gaussian, avg, 0.5)
    accumulate = cv2.convertScaleAbs(avg)
    cv2.imshow("accumulate", accumulate)
    diff = cv2.absdiff(gray, accumulate)
    thresh = cv2.threshold(diff, 9, 255, cv2.THRESH_BINARY)[1]
    cv2.imshow("thresh", thresh)
    dilate = cv2.dilate(thresh, None, iterations=2)
    (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for c in cnts:
        if cv2.contourArea(c) < 5000:
            continue

        (x, y, w, h) = cv2.boundingRect(c)
        yvalues.append(y)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        flag = False
    # cv2.imshow("Original_video", frame)

    no_y = len(yvalues)

    if(no_y > 2):
        difference = yvalues[no_y - 1] - yvalues[no_y - 2]
        if (difference > 0):
            motion.append(1)
        else:
            motion.append(0)

    if flag is True:
        if (no_y > 5):
            val, times = find_majority(motion)
            if val == 1 and times >= 10:
                count1 += 1
            else:
                count2 += 1
        yvalues = []
        motion = []

    height, width = frame.shape[:2]
    x1 = width // 2
    y1 = height // 2

    cv2.line(frame, (0, y1 + 100), (500, y1 + 100), (0, 255, 0), 2)
    cv2.line(frame, (0, y1 + 200), (500, y1+ 200), (0, 255, 0), 2)
    cv2.putText(frame, "In: {}".format(count1), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, "Out: {}".format(count2), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.imshow("Original_video", frame)


    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
