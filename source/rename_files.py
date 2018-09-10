#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 09:30:41 2018

@author: liuy08
"""

import os, glob, shutil

def rename_files(path):
    """
    """
    
    home = os.path.abspath(path)
    file_path = os.path.join(home, "*.png")
    img_files = glob.glob(file_path)
    all_files = []
    try:
        all_files = os.listdir(home)
    except NotADirectoryError:
        return
        
    no_files = False
    if len(all_files) == 0:
        no_files = True
    if len(all_files) == 1 and all_files[0] == ".DS_Store":
        no_files = True
    
    if no_files:
        return
    
    if len(img_files) != 0:
        for file_name in img_files:
            old_file_path = os.path.join(home, file_name)
            prefix = os.path.split(path)[1] + "_"
            tail = os.path.split(file_name)[1]
            new_file_path = os.path.join(home, prefix + tail)
            os.rename(old_file_path, new_file_path)
    
    if not no_files and len(img_files) == 0:
        for sub_path in all_files:
            sub_path = os.path.join(home, sub_path)
            rename_files(sub_path)
            

def pool_files(path, to_path):
    """
    """
    
    home = os.path.abspath(path)
    new_path = os.path.abspath(to_path)
    try:
        os.mkdir(new_path)
    except os.error:
        pass
    
    if home == new_path:
        return
    
    file_path = os.path.join(home, "*.png")
    img_files = glob.glob(file_path)
    all_files = []
    try:
        all_files = os.listdir(home)
    except NotADirectoryError:
        return
        
    no_files = False
    if len(all_files) == 0:
        no_files = True
    if len(all_files) == 1 and all_files[0] == ".DS_Store":
        no_files = True
    
    if no_files:
        return
    
    
    if len(img_files) != 0:
        for file_name in img_files:
            old_file_path = os.path.join(home, file_name)
            new_file_name = os.path.split(file_name)[1]
            new_file_path = os.path.join(new_path, new_file_name)
            shutil.copy(old_file_path, new_file_path)
    
    if not no_files and len(img_files) == 0:
        for sub_path in all_files:
            sub_path = os.path.join(home, sub_path)
            pool_files(sub_path, to_path)
    
        
def main():
    home = os.path.abspath("..")
    data_path = os.path.join(home, "data")
    pool_path = os.path.join(data_path, "pool")
    try:
        os.mkdir(pool_path)
    except os.error:
        pass
    rename_files(data_path)
    pool_files(data_path, pool_path)
    
if __name__ == "__main__":
    main()
    