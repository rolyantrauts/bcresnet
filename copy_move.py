#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import glob
import os
import shutil
import random

def split(source_dir='./in', dest_dir='./out', qty=0):
  
  if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)
      
  if not os.path.exists(source_dir):
    print("Source_dir = " + source_dir + " does not exist!")
    exit()

  source = glob.glob(os.path.join(source_dir, '*.wav'))
  if len(source) < qty:
    print("Not enough files found *.wav")
    exit()
  count = 0
  
  source_qty = len(source)
  for count in range(0, qty):
      rand_sample = int(random.random() * source_qty)
      shutil.copy(source[rand_sample], dest_dir + "/" + os.path.basename(source[rand_sample]))
      source.pop(rand_sample)
      source_qty = len(source)


def main_body():
  parser = argparse.ArgumentParser()
  parser.add_argument('--source_dir', default='./in', help='source dir location')
  parser.add_argument('--dest_dir', type=str, default='./out', help='dest dir location')
  parser.add_argument('--qty', type=int, help='Qty to copy')
  args = parser.parse_args()

  if args.dest_dir == None:
    args.dest_dir = "./out"
  
  split(args.source_dir, args.dest_dir, args.qty)
    
if __name__ == '__main__':
  main_body()



