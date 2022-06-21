import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


def featureHomography(frame):
    """ Applies feature matching and homography to match badminton court template (based on the following tutorial https://docs.opencv.org/3.4/d1/de0/tutorial_py_feature_homography.html)

    Args: 
        frame: the frame to be matched
    
    Returns: 
        [np.int32(dst)]: list of points describing the court
        img2: the original frame
        img3: the processed frame
        

    """
    MIN_MATCH_COUNT = 10
    img1 = cv.imread("template.png",0) # queryImage
    img2 = cv.imread(frame, 0) # trainImage
    # Initiate SIFT detector
    sift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1,None)
    kp2, des2 = sift.detectAndCompute(img2,None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    search_params = dict(checks = 50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    # store all the good matches as per Lowe's ratio test.
    good = []
    for m,n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    if len(good)>MIN_MATCH_COUNT:
        src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
        dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC,5.0)
        matchesMask = mask.ravel().tolist()
        h,w = img1.shape
        pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
        dst = cv.perspectiveTransform(pts,M)
        img2 = cv.polylines(img2,[np.int32(dst)],True,0,3, cv.LINE_AA)
    else:
        print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
        matchesMask = None

    draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                    singlePointColor = None,
                    matchesMask = matchesMask, # draw only inliers
                    flags = 2)
    img3 = cv.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
    return np.int32(dst), img2, img3

def cropImage(pts, img):
    """ Crops image with the court's bounding box (based on https://stackoverflow.com/a/48301735)

    Args: 
        pts: a list of points that describe the court
        img: the frame to be processed
    
    Returns: an image
    """
    ## (1) Crop the bounding rects
    rect = cv.boundingRect(pts)
    x,y,w,h = rect
    cropped = img[y:y+h, x:x+w].copy()

    # TODO: perspective transform image to further crop

    scale = 1 / 4
    (HEIGHT, WIDTH) = img.shape[:2]
    cv.imshow("dst2", cv.resize(cropped, (int(WIDTH * scale), int(HEIGHT * scale))))

dst, img2, img3 = featureHomography("test.png")
cropImage(dst, img2)
plt.imshow(img3, 'gray'),plt.show()