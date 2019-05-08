import numpy as np
import sys, os
import argparse
import caffe_pb2 as cq

f = open('inceptionv3.caffemodel', 'r')
cq2 = cq.NetParameter()
cq2.ParseFromString(f.read())
f.close()
print ("name 1st layer: " + cq2.layers[0].name) 
