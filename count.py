import numpy as np
import cv2

image = cv2.imread('pipe1.png')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (11, 11), 0)
canny = cv2.Canny(blur, 20, 120, 3)
dilated = cv2.dilate(canny, (1, 1), iterations=0)

(cnt, hierarchy) = cv2.findContours(
	dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cv2.drawContours(rgb, cnt, -1, (0, 255, 0), 2)

# Calculate the number of pipes
number_of_pipes = len(cnt)

# Add text to the image indicating the pipe count
cv2.putText(rgb, "Number of pipes: " + str(number_of_pipes),
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

# Display the image
cv2.imshow('Number of pipes', rgb)
cv2.waitKey(0)

# Save the image with the count
cv2.imwrite('pipe_count1.jpg', rgb)
