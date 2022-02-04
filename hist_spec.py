import cv2
import numpy as np

def near_values(array_img, final):
    
    # here we are finding the difference 
    difference = array_img - final
    
    # We do not want less values so we dont consider the pixels having negative difference 
    mask_neg_diff= np.ma.less_equal(difference, -1)
    
    if np.all(mask_neg_diff):
        
        # it finds the absolute difference 
        min_index = np.abs(difference).argmin()
        
        # returns minimum index of  nearest ( if target is greater than any value)
        return min_index 
    
    masked_diff = np.ma.masked_array(difference, mask_neg_diff)
    
    # returns the min value that is nearest
    return masked_diff.argmin()

def match_histogram(image1, image2):

    #find the shape of image1
    shape1 = image1.shape
    
    #it converts image into an array
    image1 = image1.ravel()
    image2 = image2.ravel()

    # it  get the set of unique values and their indices and counts
    img_values, img_index, img_counts = np.unique(image1, return_inverse=True,return_counts=True)
    img2_values, img2_counts = np.unique(image2, return_counts=True)

    # Calculate sk for first image
    img_quantiles = np.cumsum(img_counts).astype(np.float64)
    img_quantiles /= img_quantiles[-1]
    
    # Calculate sk for second image
    img2_quantiles = np.cumsum(img2_counts).astype(np.float64)
    img2_quantiles /= img2_quantiles[-1]

    # Rounding off values
    img_round = np.around(img_quantiles*255)
    img2_round = np.around(img2_quantiles*255)
    
    # Map the rounded values
    nearest=[]
    for data_values in img_round[:]:
        nearest.append(near_values(img2_round,data_values))
    nearest = np.array(nearest,dtype='uint8')

    return nearest[img_index].reshape(shape1)




# reading the image and store it in image object
image1 = cv2.imread('c:\\Users\\User\\Downloads\\imag2.jpg')
image2 = cv2.imread('c:\\Users\\User\\Downloads\\imag3.jpg')


# It Do The Histogram Matching By Using User Define Function
finalimage = match_histogram(image1, image2)

# It Shows The Output Image
cv2.imshow('finalimage',np.array(finalimage,dtype='uint8'))
cv2.imshow('IMAGE 1',image1)
cv2.imshow('IMAGE 2',image2)

cv2.waitKey(0)
cv2.destroyAllWindows()