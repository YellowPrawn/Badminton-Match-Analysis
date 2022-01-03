# Badminton Match Analysis

The objective of this project is to detect the most common form of winning a badminton match in the 2020 Olympics.

The ways a player can lose a round is through the following:

- In-court loss (IN)
- Out-of-court loss (OUT)
- Other; all of which will be considered as fouls for simplicity (FOUL) 

The former can be determined by:
1. Obtaining the video frames where the scoreboard changes
2. Finding the 4 rectangles in the corners of the badminton court (named bounding quadrilaterals in this project)
3. Transforming the image such that the badminton court is in a standardized, flat plane
4. Detecting the location of the shuttlecock (with respect to the classifications listed in the above)

