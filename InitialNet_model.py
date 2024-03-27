

import tensorflow as tf
import numpy as np
from collections import namedtuple


LEAKY_ALPHA = 0.1
MAX_DISP = 40
MEAN_VALUE = 100.
INPUT_SIZE = (384, 768, 3)



monodepth_parameters = namedtuple('parameters',
                        'batch_size, '
                        'num_threads, '
                        'max_disp, '
                        'loss_weights, '
                        'full_summary')

class DispNetModel(object):
    """monodepth model"""
    def __init__(self,mode, left_batch, right_batch, dis_batch, weight_decay,max_disp, loss_weights):
        self.mode=mode
        self.left_batch = left_batch
        self.right_batch = right_batch
        self.dis_batch = dis_batch
        self.weight_decay = weight_decay
        self.max_disp=max_disp
        self.loss_weights=loss_weights
        self.initializer = tf.contrib.layers.variance_scaling_initializer()

        # build the model
        self.build_model_dilated()

        # build the output
        self.build_outputs()

        if self.mode == 'test':
            return
        # calculate the losses
        if self.mode == "fine_tune":
            self.build_losses_finetune_kitti()
            # build the summaries
            self.build_summaries_finetune()
        else:
            self.build_losses()
            # build the summaries
            self.build_summaries()



    def conv2d(self, x, kernel_shape, strides=1, relu=True, padding='SAME', scope="conv"):

        W = tf.get_variable("weights", kernel_shape, initializer=self.initializer)
        b = tf.get_variable("biases", kernel_shape[3], initializer=tf.constant_initializer(0.0))

        with tf.name_scope(scope):
            x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding=padding)
            x = tf.nn.bias_add(x, b)

            tf.summary.histogram("W", W)
            tf.summary.histogram("b", b)

            if relu:
                x = tf.maximum(LEAKY_ALPHA * x, x)
        return x

    def conv2d_dilated(self, x, kernel_shape, rate=1,  relu=True, padding='SAME', scope="dilated_conv"):

        W = tf.get_variable("weights", kernel_shape, initializer=self.initializer)
        b = tf.get_variable("biases", kernel_shape[3], initializer=tf.constant_initializer(0.0))

        with tf.name_scope(scope):
            x = tf.nn.atrous_conv2d(x, W, rate, padding=padding)
            x = tf.nn.bias_add(x, b)

            tf.summary.histogram("W", W)
            tf.summary.histogram("b", b)

            if relu:
                x = tf.nn.relu(x, name = None)
        return x


    def conv2d_transpose(self, x, kernel_shape, strides=1, relu=True):

        W = tf.get_variable("weights", kernel_shape, initializer=self.initializer)
        output_shape = [tf.shape(x)[0],
                        tf.shape(x)[1] * strides, tf.shape(x)[2] * strides, kernel_shape[2]]

        with tf.name_scope("deconv"):
            x = tf.nn.conv2d_transpose(x, W, output_shape, strides=[1, strides, strides, 1],
                                       padding='SAME')
            tf.summary.histogram("W", W)
            if relu:
                x = tf.maximum(LEAKY_ALPHA * x, x)
        return x

    def upsampling_block(self, bottom, skip_connection, input_channels, output_channels, skip_input_channels):
        with tf.variable_scope("deconv"):
            # deconvolutional layer
            deconv = self.conv2d_transpose(bottom, [4, 4, output_channels, input_channels], strides=2)
        with tf.variable_scope("predict"):
            # predict layer
            predict = self.conv2d(bottom, [3, 3, input_channels, 1], strides=1, relu=False)
            tf.summary.histogram("predict", predict)
        with tf.variable_scope("up_predict"):
            # upascale the predict layer
            upsampled_predict = self.conv2d_transpose(predict, [4, 4, 1, 1], strides=2, relu=False)
        with tf.variable_scope("concat"):
            concat = self.conv2d(tf.concat([skip_connection, deconv, upsampled_predict], axis=3),
                            [3, 3, output_channels + skip_input_channels + 1, output_channels],
                            strides=1, relu=False)
        return concat, predict



    def atrous_spatial_pyramid_pooling_new(self, net, input_channels, depth=256):


        # Employ 3x3 convolutions with different atrous rates.
        with tf.variable_scope("conv_3x3_1"):
            at_pool3x3_1 = self.conv2d_dilated(net,[3, 3, input_channels, depth],rate=2)
        with tf.variable_scope("conv_3x3_1_1"):
            at_pool1x1_1 = self.conv2d(at_pool3x3_1, [1, 1, depth, 32], strides=1)

        with tf.variable_scope("conv_3x3_2"):
            at_pool3x3_2 = self.conv2d_dilated(net, [3, 3, input_channels, depth], rate=3)
        with tf.variable_scope("conv_3x3_2_1"):
            at_pool1x1_2 = self.conv2d(at_pool3x3_2, [1, 1, depth, 32], strides=1)

        with tf.variable_scope("conv_3x3_3"):
            at_pool3x3_3 = self.conv2d_dilated(net, [3, 3, input_channels, depth], rate=6)
        with tf.variable_scope("conv_3x3_3_1"):
            at_pool1x1_3 = self.conv2d(at_pool3x3_3, [1, 1, depth, 32], strides=1)

        with tf.variable_scope("concat"):
            net = tf.concat((at_pool1x1_1, at_pool1x1_2, at_pool1x1_3), axis=3,
                        name="concat")
        with tf.variable_scope("conv_1x1_output"):
            net = self.conv2d(net,[1, 1, 32*3, depth])

        return net

    def correlation_map(self, x, y, max_disp):
        corr_tensors = []
        w=tf.shape(y)[2]
        for i in range(-max_disp, 0, 1):
            shifted = tf.pad(tf.slice(y, [0] * 4, [-1, -1, w + i, -1]),
                             [[0, 0], [0, 0], [-i, 0], [0, 0]], "CONSTANT")
            corr = tf.reduce_mean(tf.multiply(shifted, x), axis=3)  # 在通道上取平均值
            corr_tensors.append(corr)
        for i in range(0,max_disp + 1,1):
            shifted = tf.pad(tf.slice(y, [0, 0, i, 0], [-1] * 4),
                             [[0, 0], [0, 0], [0, i], [0, 0]], "CONSTANT")
            corr = tf.reduce_mean(tf.multiply(shifted, x), axis=3)
            corr_tensors.append(corr)
        return tf.transpose(tf.stack(corr_tensors),
                            perm=[1, 2, 3, 0])

    def correlation_map_max(self, x, y, max_disp):
        corr_tensors = []
        w=tf.shape(y)[2]
        for i in range(-max_disp, 0, 2):
            shifted = tf.pad(tf.slice(y, [0] * 4, [-1, -1, w + i, -1]),
                             [[0, 0], [0, 0], [-i, 0], [0, 0]], "CONSTANT")
            corr = tf.reduce_mean(tf.multiply(shifted, x), axis=3)  # 在通道上取平均值
            corr_tensors.append(corr)
        for i in range(0,max_disp + 1,2):
            shifted = tf.pad(tf.slice(y, [0, 0, i, 0], [-1] * 4),
                             [[0, 0], [0, 0], [0, i], [0, 0]], "CONSTANT")
            corr = tf.reduce_mean(tf.multiply(shifted, x), axis=3)
            corr_tensors.append(corr)
        return tf.transpose(tf.stack(corr_tensors),
                            perm=[1, 2, 3, 0])

    def correlation_map_left(self, x, y, max_disp):
        corr_tensors = []
        w = tf.shape(y)[2]
        for i in range(-max_disp, 1, 1):
            shifted = tf.pad(tf.slice(y, [0] * 4, [-1, -1, w + i, -1]),
                             [[0, 0], [0, 0], [-i, 0], [0, 0]], "CONSTANT")
            corr = tf.reduce_mean(tf.multiply(shifted, x), axis=3)
            corr_tensors.append(corr)
        return tf.transpose(tf.stack(corr_tensors),
                            perm=[1, 2, 3, 0])

    def gradient_x(self, img):
        gx = img[:, :, :-1, :] - img[:, :, 1:, :]
        return gx

    def gradient_y(self, img):
        gy = img[:, :-1, :, :] - img[:, 1:, :, :]
        return gy

    def gradient_gaussion(self,img,sigma):

        halfwid = 2 * sigma
        num= tf.to_int32(2*halfwid+1)

        [xx, yy] = tf.meshgrid(tf.linspace(-halfwid,halfwid,num),
                               tf.linspace(-halfwid,halfwid,num))

        tmp=-(tf.square(xx)+tf.square(yy))/(2*np.square(sigma))
        tmp_exp=tf.exp(tmp)
        dx= tf.multiply(xx,tmp_exp)
        dy= tf.multiply(yy,tmp_exp)

        gaussion_x_filter = tf.reshape(dx, [num, num, 1, 1])
        gaussion_y_filter = tf.reshape(dy, [num, num, 1, 1])

        gradient_x= tf.nn.conv2d(img, gaussion_x_filter,
                          strides=[1, 1, 1, 1], padding='SAME')
        gradient_y = tf.nn.conv2d(img, gaussion_y_filter,
                                  strides=[1, 1, 1, 1], padding='SAME')

        return  gradient_x, gradient_y

    def get_disparity_smoothness_gaussion(self, disp_est, disp_label):

        disp_est_gradients_x,disp_est_gradients_y= self.gradient_gaussion(disp_est,1.0)
        disp_label_gradients_x, disp_label_gradients_y = self.gradient_gaussion(disp_label, 1.0)

        # smoothness_x = tf.reduce_mean(tf.square(tf.abs(disp_est_gradients_x - disp_label_gradients_x)))
        # smoothness_y = tf.reduce_mean(tf.square(tf.abs(disp_est_gradients_y - disp_label_gradients_y)))
        smoothness_x = tf.reduce_mean(tf.abs(disp_est_gradients_x - disp_label_gradients_x))
        smoothness_y = tf.reduce_mean(tf.abs(disp_est_gradients_y - disp_label_gradients_y))

        return smoothness_x + smoothness_y

    def get_disparity_smoothness(self, disp_est, disp_label):

        disp_est_gradients_x = self.gradient_x(disp_est)
        disp_est_gradients_y = self.gradient_y(disp_est)

        disp_label_gradients_x = self.gradient_x(disp_label)
        disp_label_gradients_y = self.gradient_y(disp_label)

        smoothness_x = tf.reduce_mean(tf.abs(disp_est_gradients_x - disp_label_gradients_x))
        smoothness_y = tf.reduce_mean(tf.abs(disp_est_gradients_y - disp_label_gradients_y))
        return smoothness_x + smoothness_y

    def get_disparity_smoothness_finetune(self, disp_est, disp_label):

        # creat a constant tensor with labes
        min_dis_constant = tf.zeros_like(disp_label, dtype=tf.float32)

        # test is the labes is smaller than 0
        mask_min = tf.greater(disp_label, min_dis_constant)
        mask_min = tf.cast(mask_min, dtype=tf.float32)

        disp_est_new =tf.multiply(disp_est,mask_min)

        disp_est_gradients_x = self.gradient_x(disp_est_new)
        disp_est_gradients_y = self.gradient_y(disp_est_new)

        disp_label_gradients_x = self.gradient_x(disp_label)
        disp_label_gradients_y = self.gradient_y(disp_label)

        smoothness_x = tf.reduce_mean(tf.abs(disp_est_gradients_x - disp_label_gradients_x))
        smoothness_y = tf.reduce_mean(tf.abs(disp_est_gradients_y - disp_label_gradients_y))
        return smoothness_x + smoothness_y

    def get_disparity_smoothness_gaussion_finetune(self, disp_est, disp_label, disp_label_fill):

        disp_est_gradients_x, disp_est_gradients_y = self.gradient_gaussion(disp_est, 1.0)
        disp_label_gradients_x, disp_label_gradients_y = self.gradient_gaussion(disp_label_fill, 1.0)

        dif_gd_x = tf.abs(disp_est_gradients_x - disp_label_gradients_x)
        dif_gd_y=tf.abs(disp_est_gradients_y - disp_label_gradients_y)


        # creat a constant tensor with labes
        min_dis_constant = tf.zeros_like(disp_label, dtype=tf.float32)

        # test is the label is smaller than 0
        mask_min = tf.greater(disp_label, min_dis_constant)

        mask_min = tf.cast(mask_min,dtype=tf.int32)


        dif_gd_x_new = tf.dynamic_partition(dif_gd_x,mask_min,2)
        dif_gd_y_new= tf.dynamic_partition(dif_gd_y, mask_min,2)

        smoothness_x = tf.reduce_mean(dif_gd_x_new[1])
        smoothness_y = tf.reduce_mean(dif_gd_y_new[1])
        return smoothness_x + smoothness_y


    def L1_loss(self, disp_est, disp_label):

        # L1 loss for traning
        loss = tf.reduce_mean(tf.abs(disp_est - disp_label))
        return loss

    def Disp_Regre_Loss_L1_L2(self,dis_est, disp_label):

        # L1 and L2 loss based on the value of EPE error (pixel_wise)

        # first calculate the substraction of dis_est and disp_label
        diff=tf.abs(dis_est-disp_label)

        #creat a constant tensor with values are 1.0
        min_differ= tf.ones_like(disp_label, dtype=tf.float32)

        # compare the difference with 1.0
        mask_max1 = tf.greater(diff,min_differ)
        mask_max1 = tf.cast(mask_max1, dtype=tf.int32)

        new_differs = tf.dynamic_partition(diff,mask_max1,2)
        min_1_diff=new_differs[0]
        max_1_diff =new_differs[1]

        #get the L1 loss for difference value larger than 1
        loss_l1= tf.reduce_sum(max_1_diff)
        loss_l2 = tf.reduce_sum(tf.square(min_1_diff))

        loss= (loss_l1+loss_l2)/tf.to_float(tf.size(max_1_diff)+tf.size(min_1_diff))

        return  loss

    def L1_loss_finetune(self,disp_est, disp_label):

        # creat a constant tensor with labes
        min_dis_constant = tf.zeros_like(disp_label, dtype=tf.float32)

        # test is the labes is smaller than 0
        mask_min = tf.greater(disp_label, min_dis_constant)
        mask_min = tf.cast(mask_min,dtype=tf.int32)

        new_labels = tf.dynamic_partition(disp_label,mask_min,2)
        new_est= tf.dynamic_partition(disp_est, mask_min,2)

        epe=tf.abs(new_est[1]-new_labels[1])
        nonlin_epe=tf.pow(tf.maximum(epe, 1e-3),0.4)

        loss = tf.reduce_mean(epe)

        return loss

    def EPE_finetune(self,disp_est, disp_label):

        # creat a constant tensor with labes
        min_dis_constant = tf.zeros_like(disp_label, dtype=tf.float32)

        # test is the labes is smaller than 0
        mask_min = tf.greater(disp_label, min_dis_constant)
        mask_min = tf.cast(mask_min,dtype=tf.int32)

        new_labels = tf.dynamic_partition(disp_label,mask_min,2)
        new_est= tf.dynamic_partition(disp_est, mask_min,2)

        epe=tf.reduce_mean(tf.abs(new_est[1]-new_labels[1]))


        return epe


    def build_model_dilated(self):
        # set convenience functions
        conv2d = self.conv2d
        upsampling_block = self.upsampling_block

        # creat the the dispnet correlation model
        regularizer = tf.contrib.layers.l2_regularizer(self.weight_decay)

        # -------------------------encoder part--------------------------#
        with tf.variable_scope('network_dilated'):
            with tf.variable_scope('1-encoder'):
                with tf.variable_scope("conv1", regularizer=regularizer) as scope:
                    conv1a = conv2d(self.left_batch, [7, 7, 3, 64], strides=2)
                    scope.reuse_variables()
                    conv1b = conv2d(self.right_batch, [7, 7, 3, 64], strides=2)

                with tf.variable_scope("ASPP_layer", regularizer=regularizer) as scope:
                    aspp_feature_a = self.atrous_spatial_pyramid_pooling_new(conv1a, 64, 64)
                    scope.reuse_variables()
                    aspp_feature_b = self.atrous_spatial_pyramid_pooling_new(conv1b, 64, 64)

                with tf.variable_scope("conv2", regularizer=regularizer) as scope:
                    conv2a = conv2d(aspp_feature_a, [5, 5, 64, 128], strides=2)
                    scope.reuse_variables()
                    conv2b = conv2d(aspp_feature_b, [5, 5, 64, 128], strides=2)

                with tf.variable_scope("conv_redir", regularizer=regularizer):
                    conv_redir = conv2d(conv2a, [1, 1, 128, 64], strides=1)

                with tf.name_scope("correlation"):
                    corr = self.correlation_map_left(conv2a, conv2b, max_disp=self.max_disp)
                    corr_dir= tf.concat([corr, conv_redir], axis=3)

                with tf.variable_scope("conv3", regularizer=regularizer):
                    conv3 = conv2d(corr_dir,
                                   [5, 5, self.max_disp + 1 + 64, 256], strides=2)
                    with tf.variable_scope("1"):
                        conv3_1 = conv2d(conv3, [3, 3, 256, 256], strides=1)

                with tf.variable_scope("conv4", regularizer=regularizer):
                    conv4 = conv2d(conv3_1, [3, 3, 256, 512], strides=2)
                    with tf.variable_scope("1"):
                        conv4_1 = conv2d(conv4, [3, 3, 512, 512], strides=1)
                with tf.variable_scope("conv5", regularizer=regularizer):
                    conv5 = conv2d(conv4_1, [3, 3, 512, 1024], strides=2)
                    with tf.variable_scope("1"):
                        conv5_1 = conv2d(conv5, [3, 3, 1024, 1024], strides=1)


            # -------------------------decoder part--------------------------#
            with tf.variable_scope('2-decoder'):
                with tf.variable_scope("up4"):
                    concat4, self.predict5 = upsampling_block(conv5_1, conv4_1, 1024, 512, 512)
                with tf.variable_scope("up3"):
                    concat3, self.predict4 = upsampling_block(concat4, conv3_1, 512, 256, 256)
                with tf.variable_scope("up2"):
                    concat2, self.predict3 = upsampling_block(concat3, conv2a, 256, 128, 128)
                with tf.variable_scope("up1"):
                    concat1, self.predict2 = upsampling_block(concat2, conv1a, 128, 64, 64)
                with tf.variable_scope("up0"):
                    concat0, self.predict1 = upsampling_block(concat1, self.left_batch, 64, 32, 3)
                with tf.variable_scope("prediction"):
                    self.predict0 = conv2d(concat0, [3, 3, 32, 1], strides=1, relu=False)


    def build_outputs(self):
        # store the disparities
        with tf.variable_scope('disparities'):
            self.disp_est  = [self.predict0, self.predict1, self.predict2, self.predict3, self.predict4, self.predict5]



    def build_losses(self):

        with tf.variable_scope('loss'):
            # Down sample the ground truth of disparities
            self.downsample_dis()

            # calculate the loss according to the weight schedule
            self.disp_losses = [self.Disp_Regre_Loss_L1_L2(self.disp_est[i], self.disp_downsample[i]) for i in range(6)]

            # apply the loss weight schedule to calculate the all losses from differetn layears
            self.disp_loss= tf.add_n([self.disp_losses[i]*self.loss_weights[i] for i in range(6)])

            # ---------------------disp gradient losses-----------------------------------------
            # calculate the disp gradient losses according to the weight schedule
            self.disp_grad_losses = [self.get_disparity_smoothness_gaussion(self.disp_est[i], self.disp_downsample[i]) for i in
                                     range(6)]

            # apply the loss weight schedule to calculate the smoothness loss from different layers
            self.disp_smoothness = tf.add_n([self.disp_grad_losses[i] * self.loss_weights[i] for i in range(2)])


            # apply the L2 regularization loss
            self.reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

            # the total loss
            self.total_loss = self.disp_loss + self.reg_loss + self.disp_smoothness
            #self.total_loss = self.disp_loss + self.reg_loss+ self.disp_smoothness

            # the EPE error of disparity: the highest layer loss
            self.EPE_error=tf.reduce_mean(tf.abs(self.disp_est[0]-self.disp_downsample[0]))
            #self.EPE_error = self.disp_losses[0]

    def build_losses_finetune(self):

        with tf.variable_scope('loss'):
            # Down sample the ground truth of disparities
            self.downsample_dis()

            # calculate the loss according to the weight schedule
            self.disp_losses= [self.L1_loss_finetune(self.disp_est[i], self.disp_downsample[i]) for i in range (6)]


            # apply the loss weight schedule to calculate the all losses from differetn layears
            self.disp_loss= tf.add_n([self.disp_losses[i]*self.loss_weights[i] for i in range(6)])


            #---------------------disp gradient losses-----------------------------------------
            # calculate the disp gradient losses according to the weight schedule
            #get the filled disparity labels
            self.disp_grad_losses=[self.get_disparity_smoothness_finetune(self.disp_est[i],self.disp_downsample[i]) for i in range(6)]
            # apply the loss weight schedule to calculate the smoothness loss from different layers
            self.disp_smoothness=tf.add_n([self.disp_grad_losses[i]*self.loss_weights[i] for i in range(6)])


            # apply the L2 regularization loss
            self.reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

            # the total loss
            self.total_loss = self.disp_loss + self.reg_loss + 0.1*self.disp_smoothness
            #self.total_loss = self.disp_loss + self.reg_loss

            # the EPE error of disparity: the highest layer loss
            self.EPE_error = self.EPE_finetune(self.disp_est[0],self.disp_downsample[0])
            #self.EPE_error = self.disp_losses[0]

    def build_losses_finetune_kitti(self):

        with tf.variable_scope('loss'):
            # Down sample the ground truth of disparities
            self.downsample_dis()

            # calculate the loss according to the weight schedule
            self.disp_losses = [self.L1_loss_finetune(self.disp_est[i], self.disp_downsample[i][:,:,:,0:1]) for i in range(6)]

            # apply the loss weight schedule to calculate the all losses from differetn layears
            self.disp_loss = tf.add_n([self.disp_losses[i] * self.loss_weights[i] for i in range(6)])

            # ---------------------disp gradient losses-----------------------------------------
            # calculate the disp gradient losses according to the weight schedule
            # get the filled disparity labels
            self.disp_grad_losses = [self.get_disparity_smoothness_gaussion_finetune(self.disp_est[i], self.disp_downsample[i][:,:,:,0:1],self.disp_downsample[i][:,:,:,1:2])
                                     for i in range(2)]
            # apply the loss weight schedule to calculate the smoothness loss from different layers
            self.disp_smoothness = tf.add_n([self.disp_grad_losses[i] * self.loss_weights[i] for i in range(2)])

            # apply the L2 regularization loss
            self.reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

            # the total loss
            #self.total_loss = self.disp_loss + self.reg_loss + self.disp_smoothness
            self.total_loss = self.disp_loss + self.reg_loss
            #self.total_loss = self.disp_loss

            # the EPE error of disparity: the highest layer loss
            self.EPE_error = self.EPE_finetune(self.disp_est[0], self.disp_downsample[0][:,:,:,0:1])
            #self.EPE_error = self.disp_losses[0]

    def downsample_dis(self):
        self.disp_downsample = {}
        # get the height and width of ground truth disparity image
        height, width = tf.shape(self.dis_batch)[1], tf.shape(self.dis_batch)[2]

        # down sample the disparity and change the disparity value according the down scale
        for n in range(0, 6):
            self.disp_downsample[n] = tf.image.resize_nearest_neighbor(self.dis_batch, [tf.to_int32(height / np.power(2, n)),
                                                                   tf.to_int32(width / np.power(2, n))])
            self.disp_downsample[n] = self.disp_downsample[n] * (1. / np.power(2, n))

    def build_summaries(self):

        # summary

        # summary the loss
        with tf.name_scope("losses"):
            tf.summary.scalar('disp_loss', self.disp_loss)
            tf.summary.scalar('disp_smoothness', self.disp_smoothness)
            tf.summary.scalar('reg_loss', self.reg_loss)
            tf.summary.scalar('all_loss', self.total_loss)
            for i in range(6):
                tf.summary.scalar('disp_loss_' + str(i), self.disp_losses[i])
                tf.summary.scalar('loss_weight' + str(i), self.loss_weights[i])

        # summary the images
        with tf.name_scope("images"):
            tf.summary.image("left", self.left_batch,max_outputs=1)
            tf.summary.image("right", self.right_batch,max_outputs=1)

        # summary the disparities
        with tf.name_scope("disparites"):
            # the ground truth of disparity
            tf.summary.image("disp0_gt",self.dis_batch,max_outputs=1)

            for i in range(6):
                tf.summary.image("disp" + str(i),self.disp_est[i],max_outputs=1)

    def build_summaries_finetune(self):

        # summary

        # summary the loss
        with tf.name_scope("losses"):
            tf.summary.scalar('disp_loss', self.disp_loss)
            tf.summary.scalar('disp_smoothness', self.disp_smoothness)
            tf.summary.scalar('reg_loss', self.reg_loss)
            tf.summary.scalar('all_loss', self.total_loss)
            for i in range(6):
                tf.summary.scalar('disp_loss_' + str(i), self.disp_losses[i])
                tf.summary.scalar('loss_weight' + str(i), self.loss_weights[i])

        # summary the images
        with tf.name_scope("images"):
            tf.summary.image("left", self.left_batch,max_outputs=1)
            tf.summary.image("right", self.right_batch,max_outputs=1)

        # summary the disparities
        with tf.name_scope("disparites"):
            # the ground truth of disparity
            tf.summary.image("disp0_gt",self.dis_batch[:,:,:,0:1],max_outputs=1)
            tf.summary.image("disp0_gt_fill", self.dis_batch[:, :, :, 1:2], max_outputs=1)
            for i in range(6):
                tf.summary.image("disp" + str(i),self.disp_est[i],max_outputs=1)


