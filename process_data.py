# preprocessing the data

# helper scale function
def scale(x, feature_range=(-1, 1)):
    # this function takes in an image x and returns that image, scaled with a feature_range of pixel values from -1 to 1.
    # This function assumes that the input x is already scaled from 0-1.

    x = (x * 2) - 1
    return x

def process_data(images):
    # current range
    img = images[0]
    print ('Min: ', img.min())
    print ('Max: ', img.max())

    scaled_img = scale(img)

    print('Scaled min: ', scaled_img.min())
    print('Scaled max: ', scaled_img.max())

    return scaled_img
