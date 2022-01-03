import math

import cv2


def readFrame():
    """ Loads frame into program as an RGB image and rescales it

    :return
         A cv2 image corresponding to the read frame
    """
    # reading frame and rescaling
    scale = 1 / 4

    frame = cv2.imread("test.png", cv2.IMREAD_COLOR)
    (HEIGHT, WIDTH) = frame.shape[:2]
    frameRGB = cv2.resize(frame, (int(WIDTH * scale), int(HEIGHT * scale)))

    return frameRGB


def processContours():
    """ Finds all contours in frame

    :return
            A list of cv2 contours
    """
    img = readFrame()

    # converting frames to grayscale
    frameGrey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # contour detection
    ret, thresh = cv2.threshold(frameGrey, 127, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return contours


def extractQuadrilaterals():
    """ determines if a contour is a quadrilateral. If a quadrilateral is found, adds it to list "quadrilaterals"

    :return
         A list of contours, and a cv2 image with all quadrilateral contours in the frame as an overlay
    """
    i = 0
    img = readFrame()
    quadrilaterals = []

    for contour in processContours():
        # here we are ignoring first counter because findContours function detects whole image as shape
        if i == 0:
            i = 1
            continue

        # cv2.approxPloyDP() function to approximate the shape
        approx = cv2.approxPolyDP(
            contour, 0.01 * cv2.arcLength(contour, True), True)

        # detect quadrilaterals (to find the 4 quadrilaterals that are used to define the court's bounds) and draw
        if len(approx) == 4:
            cv2.drawContours(img, [contour], -1, (0, 255, 0), 1, lineType=cv2.LINE_AA)
            quadrilaterals.append(contour)
    return quadrilaterals, img


def courtBoundingBox():
    """ Using quadrilaterals in frame, determine bounds of the court by searching for at least 2 matching squares which
    define corners of the court. Other bounds can be determined by propagating lines according to official badminton
    court guidelines

    :return
         court bounding box and a cv2 image with bounding quadrilateral contours in the frame as an overlay
    """
    quadrilaterals, img = extractQuadrilaterals()

    for i in quadrilaterals:
        similar = similarContours(i, quadrilaterals)
        if len(similar) == 2:
            _, _, w, h = cv2.boundingRect(i)
            print(verifyDistance(similar[0], similar[1], w, h))
            cv2.drawContours(img, similar, -1, (255, 0, 0), 5, lineType=cv2.LINE_AA)
    return img


def similarContours(c, contours):
    """ Using the height and width of a contour, find all contours in the given list of contours that are similar to
    the given contour

    :param c: A contour to find
    :param contours: List of contours

    :return
        List of similar contours
    """
    threshhold = 0.50  # margin of error
    matching = []
    _, _, wi, hi = cv2.boundingRect(c)
    areai = cv2.contourArea(c)
    try:
        # TODO: fix similarity determination
        extenti = wi * hi / areai
        for j in contours:
            _, _, wj, hj = cv2.boundingRect(j)
            print("break")
            if wj - (wj * threshhold) <= wi <= wj + (wj * threshhold):
                matching.append(j)
    except ZeroDivisionError:
        pass
    return matching


def verifyDistance(c1, c2, w, h):
    """ Verifies that the distance between two contours follow the appropriate ratios for a badminton court

    :param c1: A contour
    :param c2: A contour
    :param w: contour width
    :param h: contour height

    :return
         A boolean statement (true: appropriate distancing, false: inappropriate distancing)
    """
    # extract extreme points from both contours

    l1, r1, t1, b1 = findExtremePoints(c1)
    l2, r2, t2, b2 = findExtremePoints(c2)

    # find points with smallest distance to one another
    l, r = findSmallerSet(l1, r2, l2, r1)  # l: leftmost point, r: rightmost point
    t, b = findSmallerSet(t1, b2, t2, b1)  # t: topmost point, b: bottommost point

    t1, t2 = findSmallerSet(l, r, t, b)

    if t1 == l:
        # this means camera is pointing perpendicular to the net
        return checkDoubleServiceLine(t1, t2, w) and not checkSinglesSideline(t, b, h)
    else:
        # this means camera is pointing parallel to the net
        return checkDoubleServiceLine(t1, t2, h) and not checkSinglesSideline(l, r, h)


def findSmallerSet(a1, a2, b1, b2):
    """ finds the smaller pair of point pairs (a1, b2 or a2, b1)

    :param a1: point in contour 1
    :param a2: point in contour 2
    :param b1: point in contour 1
    :param b2: point in contour 2
    :return: two integers
    """
    if abs(a1[0] - a2[0]) < abs(b1[0] - b2[0]):
        return a1, a2
    else:
        return b1, b2


def findExtremePoints(c):
    """ find all extreme points in a contour

    :param c: a contour
    :return:
        l: leftmost point
        r: rightmost point
        t: topmost point
        b: bottommost point
    """
    l = tuple(c[c[:, :, 0].argmin()][0])
    r = tuple(c[c[:, :, 0].argmax()][0])
    t = tuple(c[c[:, :, 1].argmin()][0])
    b = tuple(c[c[:, :, 1].argmax()][0])
    return l, r, t, b


def checkDoubleServiceLine(t1, t2, length):
    """ Verifies if given tuples have distances matching a "Double Service Line"

    :param
        t1:
            a tuple representing a contour's extreme point
        t2:
            a tuple representing a contour's extreme point
        length:
            a float representing the width or height of a contour

    :return
        A boolean statement (true: appropriate distancing, false: inappropriate distancing)
    """
    # standardize l to official size then multiply by distance between service lines to find expected distance between
    # bounding quadrilaterals
    expectedDistance = (length / 0.76) * 11.88

    # find distance between t1 and t2
    realDistance = math.sqrt(((t1[0] - t2[0]) ^ 2) + ((t1[1] - t2[1]) ^ 2))
    return round(expectedDistance) == round(realDistance)


def checkSinglesSideline(t1, t2, length):
    """ Verifies if given tuples have distances matching a "Singles Sideline"

    :param
        t1:
            a tuple representing a contour's extreme point
        t2:
            a tuple representing a contour's extreme point
        length:
            an integer representing the width or height of a contour

    :return
        A boolean statement (true: appropriate distancing, false: inappropriate distancing)
    """
    # standardize l to official size then multiply by distance between sidelines to find expected distance between
    # bounding quadrilaterals
    expectedDistance = (length / 0.46) * 5.18

    # find distance between t1 and t2
    realDistance = math.sqrt((t1[0] - t2[0]) ^ 2 + (t1[1] - t2[1]) ^ 2)

    return round(expectedDistance) == round(realDistance)


def transformFrame():
    """ Using the bounding quadrilaterals, shift perspective into "bird's eye view" by transforming bounding quadrilaterals
    into rectangles and stretch image into standard badminton court dimensions.

    :return
        cv2 image of the court.
    """


cv2.imshow("display", courtBoundingBox())
cv2.waitKey(0)
cv2.destroyAllWindows()
