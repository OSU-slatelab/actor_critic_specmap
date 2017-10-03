#!/usr/bin/python

import re
import sys
import os


#011c02015
#011c02010 
#currentfile = open(sys.argv[1])
#outputfile = open(sys.argv[1] + ".out","w");
#for line in currentfile:
#   wrds = line.split()
#   outputfile.write(wrds[0][:8] + "  " + wrds[1]+ "\n")

currentfile = open(sys.argv[1])
outputfile = open(sys.argv[1]+".out","w");
for line in currentfile:
    wrds = line.split()
    s = list(wrds[0])
    s[-1] = '0'
    s = "".join(s)
    wrds[0] = s
    outputfile.write(wrds[0] + "  " + wrds[1]+ "\n")

