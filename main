import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # installing tesseract ocr directory

# Capturing number plate of vehicle. Use quality camera for more accurate results
vid = cv2.VideoCapture(0)
while True:
    ret, image = vid.read()
    cv2.imshow('image', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):  # press 'q' to capture photo
        break
cv2.imwrite('CarPictures/car.jpg', image)  # it is saved in this location
vid.release()
cv2.destroyAllWindows()

# Now to read image file
image = cv2.imread('CarPictures/car.jpg')
# We will resize and standardize our image to 500
image = cv2.resize(image, (500, int(image.shape[0] * (500.0 / image.shape[1]))))
# We will display original image when it will start finding
cv2.imshow("Original Image", image)  # here original image is the name of window can give your suitable name

# Now we will convert image to gray scale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Gray Scale Image", gray)

# Now we will reduce noise from our image and make it smooth
gray = cv2.bilateralFilter(gray, 11, 17, 17)
cv2.imshow("Smoother Image", gray)

# Now we will find the edges of images
edged = cv2.Canny(gray, 170, 200)
cv2.imshow("Canny edge", edged)

# Now we will find the contours based on the images
cnts, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# We will create a copy of our original image to draw all the contours
image1 = image.copy()
cv2.drawContours(image1, cnts, -1, (0, 255, 0), 3)
cv2.imshow("Canny After Contouring", image1)


# We will sort them on the basis of their areas and select top 30 areas
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]
NumberPlateCount = None

# To draw top 30 contours we will make a copy of the original image and use it
image2 = image.copy()
cv2.drawContours(image2, cnts, -1, (0, 255, 0), 3)
cv2.imshow("Top 30 Contours", image2)

# Now we will run a for loop on our contours to find the best possible contour of our expected number plate
count = 0
name = 1  # name of our cropped image

for i in cnts:
    perimeter = cv2.arcLength(i, True)
    approx = cv2.approxPolyDP(i, 0.02 * perimeter, True)
    if len(approx) == 4:  # 4 means it has 4 corners which will be most probably our number plate as it also has 4 corners
        NumberPlateCount = approx
        # Now we will crop that rectangle part
        x, y, w, h = cv2.boundingRect(i)
        crp_img = image[y:y + h, x:x + w]
        cv2.imwrite(f'CarPictures/{name}.png', crp_img)
        name += 1
        break

# Now we will draw contour in our main image that we have identified as a number plate
cv2.drawContours(image, [NumberPlateCount], -1, (0, 255, 0), 3)
cv2.imshow("Final Image", image)

# We will crop only the part of the number plate
crop_img_loc = 'CarPictures/1.png'
cv2.imshow("Cropped Image", cv2.imread(crop_img_loc))

# Now what we do is by using pytesseract module we will convert our image into text
text = pytesseract.image_to_string(crop_img_loc, lang='eng')
print("Number is:", text)
text = ''.join(e for e in text if e.isalnum())  # modify our text to have no spaces

# Create function to read our database
def check_if_string_in_file(file_name, string_to_search):
    with open(file_name, 'r') as read_obj:
        for line in read_obj:
            if string_to_search in line:
                return True
    return False

# Print whether the vehicle is registered or not
if check_if_string_in_file('./Database/Database.txt', text) and text != "":
    print('Registered')
else:
    print("Not Registered")

cv2.waitKey(0)
