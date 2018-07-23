import numpy as np
import argparse
import os
import imutils
import glob
import cv2

srcPath = "D:\\python\\workspaces\\rs\\"

ap = argparse.ArgumentParser()
ap.add_argument("-t", "--template", required=True, help=srcPath+"template.png")
ap.add_argument("-i", "--images", required=True, help=srcPath+"test2.png")
ap.add_argument("-v", "--visualize", help="Flag indicating whether or not to visualize each iteration")
args = vars(ap.parse_args())

template = cv2.imread(args["template"])
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
template = cv2.Canny(template, 50, 200)
(tH, tW) = template.shape[:2]

for imagePath in glob.glob(args["images"]+".png"):

    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    found = None
    cv2.namedWindow("Display window", cv2.WINDOW_AUTOSIZE)

    for scale in np.linspace(0.2, 1.0, 20)[::-1]:
        resized = imutils.resize(gray, width=int(gray.shape[1] * scale))
        r = gray.shape[1] / float(resized.shape[1])

        if(resized.shape[0] < tH or resized.shape[1] < tW):
            break
        edged = cv2.Canny(resized, 50, 200)
        result = cv2.matchTemplate(edged, template, cv2.TM_CCOEFF)

        (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(result)
        minThresh = (minVal + 1e-6) * 1.5
        matchLocations = np.where(result <= minThresh)

        if args.get("visualize", False):
            clone = np.dstack([edged, edged, edged])
            for(x, y) in zip(matchLocations[1], matchLocations[0]):
                cv2.rectangle(clone, (x, y), (x+ tW, y+ tH), (0, 0, 255), 2)
            cv2.imshow("Display window", clone)
            cv2.waitKey(0)

        if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, r)

    (_, maxLoc, r) = found
    (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
    (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))

    cv2.rectangle(image, (startX, startY), (endX, endY), (0,0,255), 2)
    cv2.imwrite(os.path.join(srcPath, "rs.png"), image)
    cv2.waitKey(0)
