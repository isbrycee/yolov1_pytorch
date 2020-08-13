# !/usr/bin/python
# -*- coding:utf-8 -*-
# author: Bryce

import os


reqs = []
with open("requirements.txt", mode='r') as f_old:
    lines = f_old.readlines()
    lines = lines[2:]
    for line in lines:
        line = line.split()
        temp = line[0] + "==" + line[1]
        reqs.append(temp)

with open("requirements.txt", mode='w') as f:
    for line in reqs:
        f.write(line + "\n")
