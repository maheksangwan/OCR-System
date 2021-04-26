"""
Character Detection

The goal of this task is to implement an optical character recognition system consisting of Enrollment, Detection and Recognition sub tasks

Please complete all the functions that are labelled with '# TODO'. When implementing the functions,
comment the lines 'raise NotImplementedError' instead of deleting them.

Do NOT modify the code provcounted.
Please follow the gucountelines mentioned in the project1.pdf
Do NOT import any library (function, module, etc.).
"""


import argparse
import json
import os
import glob
import cv2
import numpy as np


def read_image(img_path, show=False):
    """Reads an image into memory as a grayscale array.
    """
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    if show:
        show_image(img)

    return img

def show_image(img, delay=1000):
    """Shows an image.
    """
    cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('image', img)
    cv2.waitKey(delay)
    cv2.destroyAllWindows()

def parse_args():
    parser = argparse.ArgumentParser(description="cse 473/573 project 1.")
    parser.add_argument(
        "--test_img", type=str, default="./data/test_img.jpg",
        help="path to the image used for character detection (do not change this arg)")
    parser.add_argument(
        "--character_folder_path", type=str, default="./data/characters",
        help="path to the characters folder")
    parser.add_argument(
        "--result_saving_directory", dest="rs_directory", type=str, default="./",
        help="directory to which results are saved (do not change this arg)")
    args = parser.parse_args()
    return args

def ocr(test_img, characters):
    """Step 1 : Enroll a set of characters. Also, you may store features in an intermediate file.
       Step 2 : Use connected component labeling to detect various characters in an test_img.
       Step 3 : Taking each of the character detected from previous step,
         and your features for each of the enrolled characters, you are required to a recognition or matching.

    Args:
        test_img : image that contains character to be detected.
        characters_list: list of characters along with name for each character.

    Returns:
    a nested list, where each element is a dictionary with {"bbox" : (x(int), y (int), w (int), h (int)), "name" : (string)},
        x: row that the character appears (starts from 0).
        y: column that the character appears (starts from 0).
        w: wcountth of the detected character.
        h: height of the detected character.
        name: name of character provcounted or "UNKNOWN".
        Note : the order of detected characters should follow english text reading pattern, i.e.,
            list should start from top left, then move from left to right. After finishing the first line, go to the next line and continue.
        
    """
    # TODO Add your code here. Do not modify the return and input arguments


    #features = enrollment(characters)

    #print(features[1])

    #for i in features:
    #    detection(i)

    enrollment(characters)
    p = detection(test_img)
    recognition(characters,p)

    raise NotImplementedError

def enrollment(characters):
    """ Args:
        You are free to deccounte the input arguments.
    Returns:
    You are free to deccounte the return.
    """
    # TODO: Step 1 : Your Enrollment code should go here.

    labels = []
    features = []

    for c in characters:
        labels.append(c[0])
        features.append(c[1])
        # print(c[0], c[1].shape)
        
    #return labels, features
    
    for i in range(len(characters)):
        hog = cv2.HOGDescriptor()
        h = hog.compute(characters[i][1])
        array = np.array(h) 
    
    return array

    cv2.waitKey(0)
    cv2.destroyAllWindows


    


    # raise NotImplementedError


#     for r in img_section




# left = label[i-1,j]
# above = label[i,j-1]
#     neighbor_array = [left,above]
#     return neighbor_array




def detection(feature):
    """ 
    Use connected component labeling to detect various characters in an test_img.
    Args:
        You are free to deccounte the input arguments.
    Returns:
    You are free to deccounte the return.
    """
    # TODO: Step 2 : Your Detection code should go here.

    m = feature.shape[0] 
    n = feature.shape[1] 

 

    new = np.zeros(feature.shape)
    label = np.zeros(feature.shape)
    new_label = 0
    dict_label = []
    count = 0


    for row in range(m):
        for column in range(n):
            if feature[row][column] < 127 :
                new[row][column] = 1
            else:
                new[row][column] = 0


   


    # first pass 
    for row in range(m):
        for column in range(n):
            if new[row][column] == 0:
                label[row][column] = 0
            else:
                try:
                    left = label[row-1][column]
                except:
                    left = 0
                try:
                    above = label[row][column-1]
                except:
                    above = 0
                current_neighbor = [left, above]
                if current_neighbor == [0,0] :
                    new_label= new_label + 1
                    label[row][column] = new_label
                else :
                    if np.min(current_neighbor) == 0 or current_neighbor[0] == current_neighbor[1] :
                        label[row][column] = np.max(current_neighbor)
                    else:
                        label[row][column] = np.min(current_neighbor)
                        if count == 0:
                            dict_label.append(current_neighbor)
                            count += 1
                        else:
                            test = 0
                            for c in range(count) :
                                temp = set(dict_label[c]).intersection(set(current_neighbor))
                                if len(temp) != 0 :
                                    dict_label[c] = set(dict_label[c]).union(current_neighbor)
                                    np.array(dict_label, dtype=object)
                                    test += 1
                            if test == 0:
                                count += 1
                                np.array(dict_label, dtype=object)
                                dict_label.append(current_neighbor)
                
    
    val = label[row][column]

    # second pass
    for row in range(m):
        for column in range(n):
            for c in range(count):
                if (val in dict_label[c]) and val !=0 :
                    val = min(dict_label[c]) 
                if (val == min(dict_label[c])):
                    val = c+1
            
            # if  val> 0 :
                # while val in dict_labels.keys():
                    # root = dict_labels[val]
                    # val = root
                    # label[row][column] = root



    #BoundingBox
    box = {}
    xmin = 1000
    ymin = 1000
    xmax = 0
    ymax = 0

    for y in range(len(label)):
        for x,val in np.ndenumerate(label[y]):
            x=x[0]
            if val !=0 :
                if val not in box :
                    box[val] = [xmin,ymin,xmax,ymax]
                thisLabel = box[val]
                if thisLabel[0]>x:
                    box[val][0] = x
                if thisLabel[2]<x:
                    box[val][2] = x
                if thisLabel[1]>y:
                    box[val][1] = y
                if thisLabel[3]<y:
                    box[val][3] = y

    return box

    


    #raise NotImplementedError

def recognition(characters, dict):
    """ 
    Args:
        You are free to deccounte the input arguments.
    Returns:
    You are free to deccounte the return.
    """
    # TODO: Step 3 : Your Recognition code should go here.
    for key in dict:
        #hog = cv2.HOGDescriptor()
        labels = enrollment(dict[key])
        #labels= cv2.HOGDescriptor(dict[key],100,200) # replace with your matching function 
        temp = characters[0][1] 
        for x in range (5):
            label_width = len(dict[key][1])
            label_height = len(dict[key])
            temp = cv2.resize(characters[x][1],(label_width,label_height))
            diff = labels - temp 
            error = np.sum(diff)/(label_width*label_height)
            
            

    
 
    raise NotImplementedError


def save_results(coordinates, rs_directory):
    """
    Donot modify this code
    """
    results = []
    with open(os.path.join(rs_directory, 'results.json'), "w") as file:
        json.dump(results, file)


def main():
    args = parse_args()
    
    characters = []

    all_character_imgs = glob.glob(args.character_folder_path+ "/*")
    
    for each_character in all_character_imgs :
        character_name = "{}".format(os.path.split(each_character)[-1].split('.')[0])
        characters.append([character_name, read_image(each_character, show=False)])

    test_img = read_image(args.test_img)

    results = ocr(test_img, characters)

    save_results(results, args.rs_directory)


if __name__ == "__main__":
    main()
