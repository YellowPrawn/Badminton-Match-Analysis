import cv2


def readFrame():
    """ Read a frame from given video.

    Loads frame into program as an RGB image and rescales it

    Returns:
         A cv2 image corresponding to the read frame
    """
    # reading frame and rescaling
    scale = 1 / 4

    frame = cv2.imread("test.png", cv2.IMREAD_COLOR)
    (HEIGHT, WIDTH) = frame.shape[:2]
    frameRGB = cv2.resize(frame, (int(WIDTH * scale), int(HEIGHT * scale)))

    return frameRGB


def processContours():
    """ Processes contours in given frame

    Finds all contours in frame

    Returns:
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
    """ Finds all quadrilateral contours

    determines if a contour is a quadrilateral. If a quadrilateral is found, adds it to list "quadrilaterals"

    Returns:
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
    """ Finds bounds of badminton court

    Using quadrilaterals in frame, determine bounds of the court by searching for at least 2 matching squares which
    define corners of the court. Other bounds can be determined by propagating lines according to official badminton
    court guidelines

    Returns:
         court bounding box and a cv2 image with bounding quadrilateral contours in the frame as an overlay
    """
    quadrilaterals, img = extractQuadrilaterals()

    # for contour in quadrilaterals:

    return img


def transformFrame():
    """ standardizes court perspective

    Using the bounding quadrilaterals, shift perspective into "bird's eye view" by transforming bounding quadrilaterals
    into rectangles and stretch image into standard badminton court dimensions.

    Returns:
        cv2 image of the court.
    """


cv2.imshow("display", courtBoundingBox())
cv2.waitKey(0)
cv2.destroyAllWindows()
