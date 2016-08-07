import numpy as np
import cv2
import re
import sys
import os
import copy


def gui_start(impath):        
    pathbase = os.path.splitext(impath)[0]
    vedit_name = pathbase
    vedit_name += ".vtx"
    vertices = None
    vertices = parse_normal(vedit_name)
    image = cv2.imread(impath)
    editloop(image, vertices, pathbase)

def parse_normal(filename):
    if not os.path.isfile(filename):
        return []
    p = re.compile("^[0-9 \.]{2,}?$", re.M)
    results = p.findall(open(filename).read())
    for i in range(0, len(results)):
        tokens = results[i].split(' ')
        x_rel = float(tokens[0])
        y_rel = float(tokens[1])
        results[i] = (x_rel, y_rel)
    return results
    
def save_vertex_file(vertices, filename):
    print("saving file to " + filename)
    result = ""
    for entry in vertices:
        result += str(entry[0]) + " " + str(entry[1]) + "\n"
    file = open(filename, 'w')
    file.write(result)
    file.close()

def save_vertex_image(image, filename):
    print("saving image to " + filename)
    cv2.imwrite(filename, image)
    
def add_vertex(image, x, y, index, puttext = True):
    sizes = [8, 1.5, 3]
    cv2.circle(image,(x,y),sizes[0],(255,0,0),-1)
    if puttext:
        cv2.putText(image, str(index), (x,y), cv2.FONT_HERSHEY_SIMPLEX, sizes[1], (255, 255, 150), thickness = sizes[2])
    return image
    
def editloop(image, v_list, pathbase):    
    original_image = image.copy()
    for i in range(0, len(v_list)):
        add_vertex(image, int(v_list[i][0] * image.shape[1]), int(v_list[i][1] * image.shape[0]), i)
    image_copy = image.copy()
    closure = {'image':image, 'image_copy':image_copy, 'undo': False}
    def mouse_callback(event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            closure["image_copy"] = closure["image"].copy()
            closure["image"] = add_vertex(closure["image"], x, y, len(v_list))
            x_dim = x / (closure["image"].shape[1] * 1.0)
            y_dim = y / (closure["image"].shape[0] * 1.0)
            v_list.append((x_dim, y_dim))
            closure['undo'] = True
        elif event == cv2.EVENT_RBUTTONDOWN:
            xypoint = np.asarray([ (x / (closure["image"].shape[1] * 1.0), y / (closure["image"].shape[0] * 1.0)) ])
    cv2.namedWindow("Edit Vertices", cv2.WINDOW_NORMAL)    
    cv2.imshow("Edit Vertices", image)    
    cv2.setMouseCallback("Edit Vertices", mouse_callback)
    while True: 
        cv2.imshow("Edit Vertices", closure["image"])
        key = cv2.waitKey(1)
        if key == ord("q"): #quit
            print("exiting gui")
            break
        elif key == ord("z"): #undo
            if closure['undo']:
                closure["image"] = closure["image_copy"].copy()
                if len(v_list) > 0:
                    v_list.pop()
                closure['undo'] = False
        elif key == ord("s"): #save
            save_vertex_file(v_list, pathbase + ".vtx")
            save_vertex_image(closure["image"], pathbase + "_v.jpg")
        elif key == ord("c"): #clear
            v_list = []
            closure["image"] = original_image.copy()