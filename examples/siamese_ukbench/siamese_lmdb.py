import numpy as np
import lmdb
import random
import sys
from PIL import Image

sys.path.append('/home/xuzhang/caffe/python')
import caffe
#import pdb
#pdb.set_trace()

test_flag = 1;
if test_flag:
    N = 500
    begin_number = 2001
    end_number = 2549
else:
    N = 10000
    begin_number = 0
    end_number = 2000
image_dir = '../../data/ukbench/images/'

# We need to prepare the database for the size. We'll set it 10 times
# greater than what we theoretically need. There is little drawback to
# setting this too big. If you still run into problem after raising
# this, you might want to try saving fewer entries in a single
# transaction.
imsize = 227
map_size = N*227*227*3*10

if test_flag:
    env = lmdb.open('ukbench_siamese_test', map_size=map_size)
else: 
    env = lmdb.open('ukbench_siamese_train', map_size=map_size)

with env.begin(write=True) as txn:
    # txn is a Transaction object
    for i in range(N):
        if i%100==0:
            print(i)
        index = random.randint(begin_number,end_number)
        im1 = Image.open(image_dir+'ukbench{:05}.jpg'.format(index*4))
        flag = random.randint(1,4);
        if flag==4:
            im2 = Image.open(image_dir+'ukbench{:05}_0.jpg'.format(index*4))
            flag = 0
        else:
            im2 = Image.open(image_dir+'ukbench{:05}.jpg'.format(index*4+flag))
            flag = 1
        d = im1.height-im1.width
        dy = np.floor(np.max(d,0)/2)
        dx = np.floor(np.max(-d,0)/2)
        im1 = im1.crop((dx,dy,im1.width-dx,im1.height-dy))
        im1 = im1.resize((imsize,imsize),Image.BILINEAR)
        d = im2.height-im2.width
        dy = np.floor(np.max(d,0)/2)
        dx = np.floor(np.max(-d,0)/2)
        im2 = im2.crop((dx,dy,im2.width-dx,im2.height-dy))
        im2 = im2.resize((imsize,imsize),Image.BILINEAR)
        new_im = Image.new('RGB', (imsize,imsize*2))
        new_im.paste(im1, (0,0))
        new_im.paste(im2, (0,imsize))
        
        datum = caffe.proto.caffe_pb2.Datum()
        datum.channels = 3*2
        datum.height = im1.height
        datum.width = im1.width
        datum.data = im1.tobytes()+im2.tobytes()  # or .tostring() if numpy < 1.9
        datum.label = int(flag)
        str_id = '{:08}'.format(i)

        # The encode is only essential in Python 3
        txn.put(str_id.encode('ascii'), datum.SerializeToString())
