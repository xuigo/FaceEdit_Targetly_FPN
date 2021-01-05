import tensorflow as tf
import numpy as np
import cv2
import os
import tqdm 
import sys

from FPN  import FPN
from FPN.resnet import get_latent
from Decoder.ops import *
slim = tf.contrib.slim

MODEL_DIR='./checkpoints/fpn_id_loss_4'   # which checkpoint to use
FILE_DIR='./test/images'                  # image dir

META_PATH=os.path.join(MODEL_DIR,'model.meta')
MODEL_PATH=os.path.join(MODEL_DIR,'model')
SAVE_DIR=os.path.join('./test/results',os.path.split(MODEL_DIR)[-1])  # reconstruct dir

class Specific_Face_FPN:

    def __init__(self):
        self._init_check()
        self.x=tf.placeholder(tf.float32,[None,256,256,3],name='x')
        self.feature=FPN.build_whole_detection_network(self.x)
        self.latent=get_latent(self.feature)
        self.alpha = tf.constant(0.0, dtype=tf.float32, shape=[])
        self.fake_images = generator(self.latent, alpha=self.alpha, target_img_size=256, is_training=False)
        self.restore()

    def _init_check(self):
        #check file path
        for dir in [MODEL_DIR,FILE_DIR]:
            if not os.path.isdir(dir):
                raise ValueError
        self.files=[os.path.join(FILE_DIR,file) for file in os.listdir(FILE_DIR)]
        if len(self.files)==0:raise ValueError 
        os.makedirs(SAVE_DIR,exist_ok=True)
    
    def get_files(self):
        for file in self.files:
            #print(file)
            src=cv2.resize(cv2.imread(file),(256,256))
            src=src/127.5-1.
            src=np.array(src)[np.newaxis,:]
            yield np.array(src)

    def back_process(self,img):
        drange_min, drange_max = -1.0, 1.0
        scale = 255.0 / (drange_max - drange_min)
        scaled_image = img * scale + (0.5 - drange_min * scale)
        scaled_image = np.clip(scaled_image, 0, 255)
        #img = cv2.cvtColor(scaled_image.astype('uint8'), cv2.COLOR_RGB2BGR)
        img=scaled_image.astype('uint8')
        return img  
    
    def restore(self):
        self.sess=tf.Session()
        variables_to_restore=tf.trainable_variables()
        tf.global_variables_initializer().run(session=self.sess)
        saver = tf.train.Saver()
        restorer,restore_ckpt=FPN.get_restorer()
        restorer.restore(self.sess, restore_ckpt)
        load_fn = slim.assign_from_checkpoint_fn(MODEL_PATH, variables_to_restore, ignore_missing_vars=True)   
        load_fn(self.sess)
        
    def run(self):
        index=0
        for src in self.get_files():
            index+=1
            latent_code,reconstruct_image=self.sess.run([self.latent,self.fake_images],feed_dict={self.x:src})
            samples=np.concatenate(src,axis=0)
            results=np.concatenate(reconstruct_image,axis=0)
            out=np.concatenate((samples,results),axis=1)
            cv2.imwrite(os.path.join(SAVE_DIR,'res_{}.jpg'.format(index)),self.back_process(out))
            sys.stdout.write('Processing: {} / {} \r'.format(index,len(self.files)))
            sys.stdout.flush()
            
    def load_single_img(self,file):
        src=cv2.resize(cv2.imread(file),(256,256))
        src=src/127.5-1.
        src=np.array(src)[np.newaxis,:]
        return src
    
    def interPolate(self,src1,src2):
        img1=self.load_single_img(src1)
        img2=self.load_single_img(src2)
        latent_code_1,reconstruct_img_1=self.sess.run([self.latent,self.fake_images],feed_dict={self.x:img1})
        latent_code_2,reconstruct_img_2=self.sess.run([self.latent,self.fake_images],feed_dict={self.x:img2})
        interpolate_imgs=[]
        for alpha in [i/10. for i in range(0,10)]:
            latent_code= latent_code_1*alpha + latent_code_2*(1-alpha)
            interpolate_img=self.sess.run(self.fake_images,feed_dict={self.latent:latent_code})
            interpolate_imgs.append(interpolate_img)
        interpolate_imgs_np=np.array(interpolate_imgs)
        interpolate_res=np.concatenate([img2,*tuple(interpolate_imgs),img1],axis=0)
        interpolate_res=np.concatenate(interpolate_res,axis=1)
        cv2.imwrite('interpolate_res.jpg',self.back_process(interpolate_res))
        
if __name__=='__main__':
    sff=Specific_Face_FPN()
    #sff.run()  # image reconstructe function
    sff.interPolate('./test/images/1.jpg','./test/images/7.jpg')  # image interpolate fucntion


