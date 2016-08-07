import argparse
import cv2
import os
from datetime import datetime
import numpy as np
import gui
import engine

np.set_printoptions(precision = 10, suppress=True)
parser = argparse.ArgumentParser()
parser.add_argument("-path", action="store", help="path to image to tesselate")
parser.add_argument("-gui", action="store_true", help="open gui to specify target patch region")
parser.add_argument("-L", action="store_true", help="Perform lighting adjustments") #lighting adjustment on or off
parser.add_argument("-aspect", type=float, help="aspect ratio for the image patch")
parser.add_argument("-debug", action="store_true", help="write debug images") #whether or not to output debug images
parser.add_argument("-v", "--verbose", action="store_true", help="display progress, timing info, etc")
args = vars(parser.parse_args())
""" utility functions """
global_stamp = None
def stamp():
    global global_stamp
    old = global_stamp
    global_stamp = datetime.now()
    if old is None:
        return -1
    else:
        delta = global_stamp - old
        micro = delta.seconds * 1000000
        micro += delta.microseconds
        return micro/1000.0
def ps(desc = ""):
    if not args["verbose"]:
        return
    print( str(stamp()) + " " + desc)
def get_simple_base(path):
    return os.path.splitext(path)[0]
def get_base(path):
    return os.path.splitext(os.path.basename(path))[0]
def get_dir(path):
    return os.path.dirname(path)
def ifnotmakedir(dir):
    if(not os.path.isdir(dir)):
        os.mkdir(dir)
def make_output_name(original_image_path, suffix):
    return get_simple_base(original_image_path) + suffix + ".png"
def valid_write(path, image):
    cv2.imwrite(path, engine.validate(image))
"""primary functions"""
def pipeline(image, path):
    aspect_ratio = args["aspect"]
    if not aspect_ratio:
        aspect_ratio = 1
    vertex_path = get_simple_base(path) + ".vtx"
    #pipeline step 1: extract region
    ps("\nbeginning texture extraction:")
    if args["gui"]:
        gui.gui_start(path)
    ps("\nisolating region:")
    image = engine.extract_region(image, engine.parse_swap(vertex_path, image.shape), aspect_ratio)
    if args["debug"]:
        valid_write(make_output_name(path, "_1_region"), image)
    #pipeline step 2 : lighting adjustment
    image = image.astype(float)
    if args["L"]:
        ps("\nremoving lighting effects:")
        image = engine.light_normalize(image)
        if args["debug"]:
            valid_write(make_output_name(path, "_2_lighting"), image)
    image = np.clip(image, 0, 255)
    #pipeline step 4A : attempt to tesselate via quilting
    ps("\nTesselating Patch via lowest difference seam:")
    image_seam = engine.texture_quilt(image)
    #pipeline step 4B : attempt to tesselate via blending
    ps("\nTesselating Patch via blending:")
    image_blend = engine.texture_blend(image)
    #pipeline step 4C : attempt to tesselate via reflection
    ps("\nTesselating Patch Via Reflection:")
    image_reflect = engine.texture_reflect(image)
    #pipeline step 4D : attempt to tesselate via synthesis    
    #pipeline end : write results
    ps("\nwriting results:")
    if args["debug"]:
        valid_write(make_output_name(path, "_sample_tess_reflect"), engine.example_tesselation(image_reflect))
        valid_write(make_output_name(path, "_sample_tess_seam"), engine.example_tesselation(image_seam))
        valid_write(make_output_name(path, "_sample_tess_blend"), engine.example_tesselation(image_blend))
    valid_write(make_output_name(path, "_result_reflect"), image_reflect)
    valid_write(make_output_name(path, "_result_seam"), image_seam)
    valid_write(make_output_name(path, "_result_blend"), image_blend)
    ps("\nall complete")
        
pipeline(cv2.imread(args["path"]), args["path"])