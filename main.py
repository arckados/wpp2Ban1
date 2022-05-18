import sys

import cv2 as cv
import cv2
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import os
import pathlib

def remove_small_objects(img, min_size=150):
    # find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img)
    # connectedComponentswithStats yields every seperated component with information on each of them, such as size
    # the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
    sizes = stats[1:, -1]
    nb_components = nb_components - 1

    # your answer image
    img2 = img
    # for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components-1):
        if sizes[i] < min_size:
            img2[output == i + 1] = 0

    return img2

class mrtimage:

    def __init__(self, path, filename):
        self.filename = filename
        self.imgpath = path+"/"+filename
        self.img = pydicom.read_file(self.imgpath)

        self.labelpath = path+"/Save/Autosave/"+filename.replace(".IMA",".txt")
        self.labelfile = None
        self.setlable()

    def setlable(self):
        #self.label = open(self.labelpath, "r").readline()

        try:
            self.labelfile = open(self.labelpath, "r").readlines()
            #print(self.labelpath)
        except:

            pass

class mrtfolder:

    def __init__(self, path):
        self.path = path
        self.images = []
        self.addimagesfrompath()

    def addimagesfrompath(self):

        imafiles = os.listdir(self.path)

        for i in imafiles:
            if i[-1] == "A" or i[-1] == "a":
                self.images.append(mrtimage(self.path,i))

    def showimage(self):

        for i in range(9):
            plt.subplot(3, 3, i + 1)
            plt.imshow(self.images[i].img.pixel_array)
            plt.axis("off")

        plt.show()




def main(args):

    base_dir, Gruppedir = os.path.join(args[0]),os.path.join(args[1])
    dirlist = os.listdir(base_dir)
    mrtdata = []
    for i in dirlist:
        mrtdata.append(mrtfolder("./OrthosisMRT/"+i))

    imagespath = Gruppedir+"/images"
    labelspath = Gruppedir+"/lables"
    coloredpath = Gruppedir+"/colored"

    if not os.path.exists(Gruppedir):
        os.mkdir(Gruppedir)
    if not os.path.exists(imagespath):
        os.mkdir(imagespath)
    if not os.path.exists(labelspath):
        os.mkdir(labelspath)
    if not os.path.exists(coloredpath):
        os.mkdir(coloredpath)

    for i in mrtdata:
        for j in i.images:
            im = j.img.pixel_array.astype(float)  # get image array
            rescaled_image = (np.maximum(im, 0) / im.max()) * 255  # float pixels
            final_image = np.uint8(rescaled_image)  # integers pixels

            #cv2.imwrite(imagespath+"/"+j.filename.replace(".IMA",".png"), final_image)

    #mrtdata[0].showimage()
    #print(mrtdata[0].images[1].labelfile)

# Press the green button in the gutter to run the script.
def test(args):
    base_dir, Gruppedir = os.path.join(args[0]), os.path.join(args[1])
    folder = "B_BL_TSE_DIXON_CONTROL_LEG_45MIN_OPP_0041"
    imagename = "NUTRIHEP_BASELINE_B.MR.NUTRIHEP_23NA_1H.0041.0002.2014.04.02.11.40.34.515625.11064117"


    img = pydicom.read_file(base_dir+"/"+folder+"/"+imagename+".IMA")
    im = img.pixel_array.astype(float)  # get image array
    rescaled_image = (np.maximum(im, 0) / im.max()) * 255  # float pixels
    image = np.uint8(rescaled_image)
    plt.subplot(3, 3, 1)
    plt.imshow(image)
    ######
    ksize = (5, 5)
    blured = cv2.blur(image, ksize)
    plt.subplot(3, 3, 3)
    plt.imshow(blured)
    ########
    ret, thresh = cv.threshold(blured, 36, 255, cv.THRESH_BINARY)

    plt.subplot(3, 3, 4)
    plt.imshow(thresh)
    # cv2.imshow("thresh", thresh)
    # cv2.waitKey()

    # red color boundaries [B, G, R]
    lower = [0, 0, 0]
    upper = [0, 0, 0]

    # create NumPy arrays from the boundaries
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")

    # find the colors within the specified boundaries and apply
    # the mask
    #mask = cv2.inRange(image, lower, upper)
    print(image.shape)
    output = cv2.bitwise_and(image, image)

    blank_image = np.zeros((256, 256, 3), np.uint8)



    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print(len(contours))

    if len(contours) != 0:
        c = max(contours, key=cv2.contourArea)
        # draw in blue the contours that were founded
        cv2.drawContours(blank_image, c, -1, (0,255,0), 3)

        # # find the biggest countour (c) by the area
        # c = max(contours, key=cv2.contourArea)
        # x, y, w, h = cv2.boundingRect(c)
        #
        # # draw the biggest contour (c) in green
        # cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # show the images
    re=alpha_blend(np.zeros((256, 256, 3), np.uint8),image,blank_image)
    cv2.imshow("a",re)
    #cv2.imshow("Result", np.hstack([image, blank_image]))
    cv2.waitKey()

    drawed = cv.drawContours(thresh, contours, 2, (0, 255, 0))
    # cv2.imshow("drawed", drawed)
    # cv2.waitKey()


    plt.subplot(3, 3, 5)
    plt.imshow(drawed)
    # for i in range(9):
    #
    #     plt.imshow(self.images[i].img.pixel_array)
    #     plt.axis("off")

    #plt.show()

    #plt.show()

def alpha_blend(background_, foreground_, mask_):
    background = background_.copy()
    foreground = foreground_.copy()
    mask = mask_.copy()

    background = background.astype(float)
    foreground = foreground.astype(float)
    mask = mask.astype(float) / 255
    foreground = cv2.multiply(mask, foreground)
    background = cv2.multiply(1.0 - mask, background)
    image = cv2.add(foreground, background)

    return image
if __name__ == '__main__':
    args = sys.argv[1:]
    test(args)
    #main(args)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
