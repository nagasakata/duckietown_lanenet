

import time

import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pygame

from lanenet_model import lanenet
from lanenet_model import lanenet_postprocess
from local_utils.config_utils import parse_config_utils
from local_utils.log_util import init_logger

from pprint import pprint

CFG = parse_config_utils.lanenet_cfg
LOG = init_logger.get_logger(log_file_name_prefix='lanenet_test')

tf.compat.v1.disable_eager_execution()

class Try_lanenet():

    def __init__(self, weights_path = "./weights/tusimple_lanenet.ckpt"):
        
        self.input_tensor = tf.compat.v1.placeholder(dtype=tf.float32, shape=[1, 256, 512, 3], name='input_tensor')

        net = lanenet.LaneNet(phase='test', cfg=CFG)

        self.binary_seg_ret, self.instance_seg_ret = net.inference(input_tensor=self.input_tensor, name='LaneNet', reuse=tf.compat.v1.AUTO_REUSE)
        

        self.postprocessor = lanenet_postprocess.LaneNetPostProcessor(cfg=CFG)

        # Set sess configuration
        self.sess_config = tf.compat.v1.ConfigProto()
        self.sess_config.gpu_options.per_process_gpu_memory_fraction = CFG.GPU.GPU_MEMORY_FRACTION
        self.sess_config.gpu_options.allow_growth = CFG.GPU.TF_ALLOW_GROWTH
        self.sess_config.gpu_options.allocator_type = 'BFC'

        self.sess = tf.compat.v1.Session(config=self.sess_config)

        with tf.compat.v1.variable_scope(name_or_scope='moving_avg'):
            #tf.compat.v1.get_variable_scope().reuse_variables()
            variable_averages = tf.train.ExponentialMovingAverage(
                CFG.SOLVER.MOVING_AVE_DECAY)
            variables_to_restore = variable_averages.variables_to_restore()

        self.saver = tf.compat.v1.train.Saver(variables_to_restore)

        with self.sess.as_default():
            self.saver.restore(sess=self.sess, save_path=weights_path)

        pygame.init()
        width, height = 512, 256
        self.screen = pygame.display.set_mode((width, height))
        

    def minmax_scale(self, input_arr):
        """

        :param input_arr:
        :return:
        """
        min_val = np.min(input_arr)
        max_val = np.max(input_arr)

        output_arr = (input_arr - min_val) * 255.0 / (max_val - min_val)

        return output_arr


    def execute_lanenet(self, 
                        raw_image, 
                        with_lane_fit = True):
            
            raw_image = raw_image[50:470, :]
            
            #self.change_color(raw_image)

            with self.sess.as_default():

                sub_image = cv2.resize(raw_image, (512, 256), interpolation=cv2.INTER_LINEAR)
                source_image = sub_image
                image = cv2.resize(sub_image, (512, 256), interpolation=cv2.INTER_LINEAR)
                image = image / 127.5 - 1.0

                binary_seg_image, instance_seg_image = self.sess.run(
                    [self.binary_seg_ret, self.instance_seg_ret],
                    feed_dict={self.input_tensor: [image]}
                    )

                postprocess_result = self.postprocessor.postprocess(
                    binary_seg_result=binary_seg_image[0],
                    instance_seg_result=instance_seg_image[0],
                    source_image=source_image,
                    with_lane_fit=with_lane_fit,
                    data_source='tusimple'
                )

                mask_image = postprocess_result['mask_image']

                if not postprocess_result['mask_image'] is None:

                    if with_lane_fit:
                        lane_params = postprocess_result['fit_params']
                        LOG.info('Model have fitted {:d} lanes'.format(len(lane_params)))
                        for i in range(len(lane_params)):
                            LOG.info('Fitted 2-order lane {:d} curve param: {}'.format(i + 1, lane_params[i]))

                    self.image_with_pygame(mask_image)
                    self.image_with_plt(mask_image, source_image, binary_seg_image)


    def image_with_pygame(self, opencv_image):

        opencv_image = opencv_image[:,:,::-1]
        shape = opencv_image.shape[1::-1]
        pygame_image = pygame.image.frombuffer(opencv_image.tostring(), shape, 'RGB')

        self.screen.blit(pygame_image, (0, 0))
        pygame.display.update()

        return


    def image_with_plt(self, 
                    mask_image,
                    source_image,
                    binary_seg_image):
        
        #plt.figure('mask_image')
        #plt.imshow(mask_image[:, :, (2, 1, 0)])
        plt.figure('src_image')
        plt.imshow(source_image[:, :, (2, 1, 0)])
        #plt.figure('binary_image')
        #plt.imshow(binary_seg_image[0] * 255, cmap='gray')
        plt.show(block=False)
        plt.close

        return


    def change_color(self, image, r=50, g=50, b=50):
        for i in range(len(image)):
            for j in image[i]:
                if j[2] <= 50:
                    j[0] = 0
                    j[1] = 0
                    j[2] = 0
