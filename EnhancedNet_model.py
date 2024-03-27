

import tensorflow as tf
import numpy as np
from  bilinear_sampler import  bilinear_sampler_1d_h


LEAKY_ALPHA = 0.1


class DispNetModel(object):
    def __init__(self, left_batch, right_batch,  weight_decay, max_disp):

        self.left_batch = left_batch
        self.right_batch = right_batch
        self.weight_decay = weight_decay
        self.max_disp = max_disp
        self.initializer = tf.contrib.layers.variance_scaling_initializer()

        # build the model
        self.build_model_dilated()
        # self.build_model()

        # build the output
        self.build_outputs()

    def build_model_dilated(self):
        # set convenience functions
        conv2d = self.conv2d
        upsampling_block = self.upsampling_block

        # creat the the dispnet correlation model
        regularizer = tf.contrib.layers.l2_regularizer(self.weight_decay)

        # -------------------------encoder part--------------------------#
        with tf.variable_scope('network_dilated', reuse=tf.AUTO_REUSE):
            with tf.variable_scope('1-encoder'):
                with tf.variable_scope("conv1", regularizer=regularizer) as scope:
                    self.conv1a = conv2d(self.left_batch, [7, 7, 3, 64], strides=2)
                    scope.reuse_variables()
                    self.conv1b = conv2d(self.right_batch, [7, 7, 3, 64], strides=2)

                with tf.variable_scope("ASPP_layer", regularizer=regularizer) as scope:
                    aspp_feature_a = self.atrous_spatial_pyramid_pooling_new(self.conv1a, 64, 64)
                    scope.reuse_variables()
                    aspp_feature_b = self.atrous_spatial_pyramid_pooling_new(self.conv1b, 64, 64)

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
                    concat1, self.predict2 = upsampling_block(concat2, self.conv1a, 128, 64, 64)
                with tf.variable_scope("up0"):
                    concat0, self.predict1 = upsampling_block(concat1, self.left_batch, 64, 32, 3)
                with tf.variable_scope("prediction"):
                    self.predict0 = conv2d(concat0, [3, 3, 32, 1], strides=1, relu=False)

    def build_outputs(self):
        # store the disparities
        with tf.variable_scope('disparities'):
            self.disp_est  = [self.predict0, self.predict1, self.predict2, self.predict3, self.predict4, self.predict5]

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

    def conv2d_dilated(self, x, kernel_shape, rate=1, relu=True, padding='SAME', scope="dilated_conv"):

        W = tf.get_variable("weights", kernel_shape, initializer=self.initializer)
        b = tf.get_variable("biases", kernel_shape[3], initializer=tf.constant_initializer(0.0))

        with tf.name_scope(scope):
            x = tf.nn.atrous_conv2d(x, W, rate, padding=padding)
            x = tf.nn.bias_add(x, b)

            tf.summary.histogram("W", W)
            tf.summary.histogram("b", b)

            if relu:
                x = tf.nn.relu(x, name=None)
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


class DispRefineModel(object):
    def __init__(self,mode, left_batch, right_batch, dis_batch, left_conv, right_conv, left_initial_dis,weight_decay,dis_offset, is_train, loss_weights):
        self.mode=mode
        self.left_batch = left_batch
        self.right_batch = right_batch
        self.dis_batch = dis_batch
        self.left_conv=left_conv
        self.right_conv=right_conv
        # self.conv2a=conv2a
        # self.conv2b=conv2b
        self.left_initial_dis=left_initial_dis
        self.weight_decay = weight_decay
        self.dis_offset=dis_offset
        self.is_train=is_train
        self.loss_weights=loss_weights
        self.initializer = tf.contrib.layers.variance_scaling_initializer()

        # build the model
        self.build_refine_model_EncodeDecode()

        # build the output

        self.build_outputs()


        if self.mode == 'test':
            return

        if self.mode =="fine_tune":

            self.build_losses_finetune()
            self.build_summaries_EncodeDecode()

        else:
            self.build_losses_EncodeDecode()
            # build the summaries
            self.build_summaries_EncodeDecode()

    def conv2d(self, x, kernel_shape, strides=1, relu=True, bn= False,  regularizer=None, padding='SAME', scope="conv"):

        W = tf.get_variable("weights", kernel_shape, initializer=self.initializer, regularizer=regularizer)
        b = tf.get_variable("biases", kernel_shape[3], initializer=tf.constant_initializer(0.0))

        with tf.name_scope(scope):
            x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding=padding)
            x = tf.nn.bias_add(x, b)

            tf.summary.histogram("W", W)
            tf.summary.histogram("b", b)

            if bn:
                x = tf.layers.batch_normalization(x, training=self.is_train, momentum=0.99)

            if relu:
                x = tf.maximum(LEAKY_ALPHA * x, x)
        return x

    def conv2d_dilated(self, x, kernel_shape, rate=1,  leaky_relu=True, bn= False, regularizer=None,padding='SAME', scope="dilated_conv"):

        W = tf.get_variable("weights", kernel_shape, initializer=self.initializer, regularizer=regularizer)
        b = tf.get_variable("biases", kernel_shape[3], initializer=tf.constant_initializer(0.0))

        with tf.name_scope(scope):
            x = tf.nn.atrous_conv2d(x, W, rate, padding=padding)
            x = tf.nn.bias_add(x, b)

            tf.summary.histogram("W", W)
            tf.summary.histogram("b", b)
            if bn:
                x = tf.layers.batch_normalization(x, training=self.is_train, momentum=0.99)
            if leaky_relu:
                x = tf.maximum(0.2 * x, x)
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

    def upsampling_block_res(self, bottom, pr_dis, input_channels, output_channels):
        with tf.variable_scope("deconv"):
            # deconvolutional layer
            deconv = self.conv2d_transpose(bottom, [4, 4, output_channels, input_channels], strides=2)
        with tf.variable_scope("predict"):
            # predict layer
            predict = self.conv2d(bottom, [3, 3, input_channels, 1], strides=1, relu=False)
            #predict=res+pr_dis
            tf.summary.histogram("predict", predict)
        with tf.variable_scope("up_predict"):
            # upascale the predict layer
            upsampled_predict = self.conv2d_transpose(predict, [4, 4, 1, 1], strides=2, relu=False)
        with tf.variable_scope("concat"):
            concat = self.conv2d(tf.concat([ deconv, upsampled_predict], axis=3),
                                 [3, 3, output_channels + 1, output_channels],
                                 strides=1, relu=False)
        return concat, predict

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


    def Recon_error(self, left_conv, right_conv, left_dis, dis_offset):

        # using the bilinear sample to generate left conv
        error_tensors = []
        # creat a constant tensor with labes
        ones = tf.ones_like(left_dis, dtype=tf.float32)

        for i in np.arange(-dis_offset, dis_offset+0.25, 0.25):

            # redefine the left_dis
            offset=ones*i
            left_dis_i=left_dis+ offset
            recon_leftconv=bilinear_sampler_1d_h(right_conv,-left_dis_i)

            error=tf.reduce_mean(tf.abs(left_conv-recon_leftconv), axis=3)
            error_tensors.append(error)
        return tf.transpose(tf.stack(error_tensors),
                            perm=[1, 2, 3, 0])

    def Fined_Correlation_error(self, left_conv, right_conv, left_dis, dis_offset):

        # using the bilinear sample to generate left conv
        corr_tensors = []
        error_tensors =[]
        # creat a constant tensor with labes
        ones = tf.ones_like(left_dis, dtype=tf.float32)

        for i in np.arange(-dis_offset, dis_offset+0.25, 0.25):

            # redefine the left_dis
            offset=ones*i
            left_dis_i=left_dis+ offset
            recon_leftconv=bilinear_sampler_1d_h(right_conv,-left_dis_i)

            corr=tf.reduce_mean(tf.multiply(left_conv,recon_leftconv), axis=3)
            error = tf.reduce_mean(tf.abs(left_conv- recon_leftconv), axis=3)
            corr_tensors.append(corr)
            error_tensors.append(error)

        fined_corr=tf.transpose(tf.stack(corr_tensors),perm=[1, 2, 3, 0])
        recon_error=tf.transpose(tf.stack(error_tensors),perm=[1, 2, 3, 0])
        return fined_corr,recon_error



    def Squeeze_excitation_layer(self, input_x, out_dim, ratio, layer_name):
        with tf.name_scope(layer_name):
            with tf.name_scope(layer_name +'_global_pool'):
                squeeze = tf.reduce_mean(input_x, [1, 2], keep_dims=True)

            with tf.name_scope(layer_name + '_fully_connected1'):
                excitation = tf.layers.dense(inputs=squeeze, units=out_dim / ratio, use_bias=True)
                excitation = tf.nn.relu(excitation)
            with tf.name_scope(layer_name + '_fully_connected2'):
                excitation = tf.layers.dense(inputs=excitation, units=out_dim, use_bias=True)
                excitation = tf.nn.sigmoid(excitation)

            excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])

            scale = input_x * excitation
            return scale

    def ResiBlolck(self,x,in_channel, out_channel, dilation=1, regularizer=None):

        # conv1
        residual=x
        conv2d_dilated=self.conv2d_dilated
        with tf.variable_scope("conv1"):
            conv1=conv2d_dilated(x,[3,3,in_channel,out_channel],rate=dilation,leaky_relu=True, regularizer=regularizer)
        # conv2
        with tf.variable_scope("conv2"):
            conv2=conv2d_dilated(conv1,[3,3,in_channel,out_channel], rate=dilation,leaky_relu=False, regularizer=regularizer)
            conv2 +=residual
            # leacy relu
            out = tf.maximum(0.2 * conv2, conv2)

        return out

    def build_refine_model_ResBlocks(self):

        conv2d = self.conv2d
        resiblock=self.ResiBlolck
        # creat the the dispnet correlation model
        regularizer = tf.contrib.layers.l2_regularizer(self.weight_decay)

        with tf.variable_scope('refined_net'):

            with tf.name_scope("conv1"):

                height, width = tf.shape(self.left_initial_dis)[1], tf.shape(self.left_initial_dis)[2]
                conv1a= tf.image.resize_bilinear(self.left_conv,[height,width],align_corners=True)
                conv1b = tf.image.resize_bilinear(self.right_conv, [height, width], align_corners=True)

            with tf.variable_scope("conv1_press") as scope:
                conv1a_press= conv2d(conv1a,[3,3,64,16], strides=1,regularizer=regularizer)
                scope.reuse_variables()
                conv1b_press = conv2d(conv1b, [3, 3, 64, 16], strides=1,regularizer=regularizer)

            with tf.name_scope("correlation"):
                fined_corr= self.Fined_Correlation(conv1a_press,conv1b_press,self.left_initial_dis,self.dis_offset)
                corr_dir = tf.concat([fined_corr, conv1a_press], axis=3)

            with tf.name_scope("photometirc_error"):
                recon_conv1a=bilinear_sampler_1d_h(conv1b_press,-self.left_initial_dis)
                ph_error=tf.abs(recon_conv1a-conv1a_press)
                ph_concat=tf.concat([ph_error,conv1a_press],axis=3)
                with tf.variable_scope("ph_conv"):
                    ph_conv = conv2d(ph_concat, [3, 3, 32, 16],strides=1, regularizer=regularizer)

            with tf.name_scope("concat_all"):
                 concat_all = tf.concat([corr_dir,ph_conv], axis=3)
                 num_channels=8*self.dis_offset+1+16+16
                 with tf.variable_scope("conv_corr"):
                     all_conv=conv2d(concat_all,[3,3,num_channels,32],bn=True,strides=1,regularizer=regularizer)

            with tf.variable_scope("ResBlock"):
                with tf.variable_scope("1"):
                    res1=resiblock(all_conv,32,32,1,regularizer=regularizer)
                with tf.variable_scope("2"):
                    res2=resiblock(res1,32,32,1,regularizer=regularizer)
                with tf.variable_scope("3"):
                    res3=resiblock(res2,32,32,1,regularizer=regularizer)
                with tf.variable_scope("4"):
                    res4=resiblock(res3,32,32,1,regularizer=regularizer)
                with tf.variable_scope("5"):
                    res5=resiblock(res4,32,32,1,regularizer=regularizer)
                with tf.variable_scope("6"):
                    res6=resiblock(res5,32,32,1,regularizer=regularizer)
            with tf.variable_scope("conv_res1"):
                self.conv_res1=conv2d(res6,[3,3,32,1],strides=1, relu=False,regularizer=regularizer)
            with tf.name_scope("RefineDis"):
                refined_dis=self.left_initial_dis+self.conv_res1
                self.refined_dis= tf.nn.relu(refined_dis, name = None)


    def build_refine_model_EncodeDecode(self):

        conv2d = self.conv2d
        upsampling_block_res=self.upsampling_block_res
        # creat the the dispnet correlation model
        regularizer = tf.contrib.layers.l2_regularizer(self.weight_decay)

        with tf.variable_scope('refined_net',reuse=tf.AUTO_REUSE):

            with tf.name_scope("dis_downsample"):

                height, width = tf.shape(self.left_initial_dis)[1], tf.shape(self.left_initial_dis)[2]

                self.left_dis_2 = tf.image.resize_bilinear(self.left_initial_dis,
                                                       [tf.to_int32(height / 2),
                                                        tf.to_int32(width / 2)])
                self.left_dis_2 = self.left_dis_2 * (1. /2.0)


            with tf.variable_scope("conv1a_press") as scope:
                conv1a_press = conv2d(self.left_conv, [1, 1, 64, 16], strides=1, regularizer=regularizer)
                scope.reuse_variables()
                conv1b_press = conv2d(self.right_conv, [1, 1, 64, 16], strides=1, regularizer=regularizer)


            with tf.name_scope("correlation"):
                fined_corr, recon_error = self.Fined_Correlation_error(conv1a_press, conv1b_press, self.left_dis_2,
                                                                       self.dis_offset)
                recon_leftconv = bilinear_sampler_1d_h(conv1b_press, -self.left_dis_2)

            with tf.name_scope("concate_all"):

                r_concates_all=tf.concat([fined_corr,recon_error, conv1a_press], axis=3)
                #r_concates_all = tf.concat([fined_corr, conv1a_press], axis=3)
                #r_concates_all = tf.concat([fined_corr, conv1a_press, recon_leftconv], axis=3)
                channels= (2*4*self.dis_offset+1)*2+16
                #channels = (2 * 4 * self.dis_offset + 1) + 16
                #channels = (2 * 4 * self.dis_offset + 1) + 16+16
                with tf.variable_scope("r_conv1_1"):
                    r_conv1_1 = conv2d(r_concates_all, [3, 3, channels, 64],strides=1, regularizer=regularizer)

            with tf.variable_scope("r_conv2"):
                r_conv2 = conv2d(r_conv1_1, [3, 3, 64, 128], strides=2, regularizer=regularizer)
            with tf.variable_scope("r_conv2_1"):
                r_conv2_1 = conv2d(r_conv2, [3, 3, 128, 128], strides=1,regularizer=regularizer)

            with tf.variable_scope("r_conv3"):
                r_conv3 = conv2d(r_conv2_1, [3, 3, 128, 256], strides=2, regularizer=regularizer)
            with tf.variable_scope("r_conv3_1"):
                r_conv3_1 = conv2d(r_conv3, [3, 3, 256, 256], strides=1, regularizer=regularizer)

                # -------------------------decoder part--------------------------#
            with tf.variable_scope("up2"):
                self.left_dis_8 = tf.image.resize_bilinear(self.left_initial_dis,
                                                           [tf.to_int32(height / 8),
                                                            tf.to_int32(width / 8)])
                self.left_dis_8 = self.left_dis_8 * (1. / 8.0)
                concat2, self.res_3 = upsampling_block_res(r_conv3_1, self.left_dis_8, 256, 128)
                self.predict3 = self.res_3 + self.left_dis_8
            with tf.variable_scope("up1"):
                self.left_dis_4 = tf.image.resize_bilinear(self.left_initial_dis,
                                                           [tf.to_int32(height / 4),
                                                            tf.to_int32(width / 4)])
                self.left_dis_4 = self.left_dis_4 * (1. / 4.0)
                concat1, self.res_2 = upsampling_block_res(concat2, self.left_dis_4, 128, 64)
                self.predict2 = self.res_2 + self.left_dis_4
            with tf.variable_scope("up0"):
                concat0, self.res_1 = upsampling_block_res(concat1, self.left_dis_2, 64, 32)
                self.predict1 = self.res_1 + self.left_dis_2
            with tf.variable_scope("prediction"):
                self.res_0 = conv2d(concat0, [3, 3, 32, 1], strides=1, relu=False)
                self.predict0 = self.res_0 + self.left_initial_dis

    def build_outputs(self):
        # store the disparities
        with tf.variable_scope('disparities'):
            self.disp_refine = [self.predict0, self.predict1, self.predict2, self.predict3]

    def downsample_dis(self):
        self.disp_downsample = {}
        # get the height and width of ground truth disparity image
        height, width = tf.shape(self.dis_batch)[1], tf.shape(self.dis_batch)[2]

        # down sample the disparity and change the disparity value according the down scale
        for n in range(0, 4):
            self.disp_downsample[n] = tf.image.resize_nearest_neighbor(self.dis_batch, [tf.to_int32(height / np.power(2, n)),
                                                                   tf.to_int32(width / np.power(2, n))])
            self.disp_downsample[n] = self.disp_downsample[n] * (1. / np.power(2, n))


    def pixel_wise_softmax(self, logits):
        max_axis = tf.reduce_max(logits, axis=3, keepdims=True)
        exponential_map = tf.exp(logits - max_axis)
        normalize = tf.reduce_sum(exponential_map, axis=3, keepdims=True)
        return exponential_map / normalize

    def L1_loss(self, disp_est, disp_label):

        # L1 loss for traning
        loss = tf.reduce_mean(tf.abs(disp_est - disp_label))
        return loss

    def Smooth_L1loss(self, disp_est, disp_label):

        # smooth l1 loss for reducing the oscillations problem
        loss=tf.losses.huber_loss(disp_label,disp_est)

        return loss

    def EPE_finetune(self,disp_est, disp_label):

        # creat the valid map
        valid_map = tf.where(tf.equal(disp_label, 0), tf.zeros_like(disp_label, dtype=tf.float32),
                             tf.ones_like(disp_label, dtype=tf.float32))

        epe=tf.reduce_sum(valid_map * tf.abs(disp_est - disp_label)) / tf.reduce_sum(valid_map)


        return epe

    def Occ_loss(self, logits, labels):

        # est_occmap: n,w,h,2
        flat_logits = tf.reshape(logits, [-1, 2])
        flat_labels = tf.reshape(labels, [-1, 2])

        #calculate the cross cropy loss
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits,
                                                                         labels=flat_labels))

        return  loss


    def build_losses_ResBlocks(self):

        # refined disarity regression loss
        self.dis_loss= self.L1_loss(self.refined_dis,self.dis_batch)

        # apply the L2 regularization loss

        self.reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        # total loss
        self.total_loss= self.dis_loss + self.reg_loss

        self.EPE = self.L1_loss(self.refined_dis, self.dis_batch)

        return self.total_loss

    def build_losses_EncodeDecode(self):

        # Down sample the ground truth of disparities
        self.downsample_dis()

        # calculate the loss according to the weight schedule
        self.disp_losses = [self.L1_loss(self.disp_refine[i], self.disp_downsample[i]) for i in range(4)]

        # apply the loss weight schedule to calculate the all losses from differetn layears
        self.disp_loss = tf.add_n([self.disp_losses[i] * self.loss_weights[i] for i in range(4)])

        # apply the L2 regularization loss
        self.reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        # total loss
        self.total_loss = self.disp_loss + self.reg_loss
        #self.total_loss = self.disp_loss

        self.EPE = self.L1_loss(self.disp_refine[0], self.dis_batch[0])

        return self.total_loss

    def L1_loss_finetune(self, disp_est, disp_label):

        # creat the valid map
        valid_map = tf.where(tf.equal(disp_label, 0), tf.zeros_like(disp_label, dtype=tf.float32),
                             tf.ones_like(disp_label, dtype=tf.float32))

        loss = tf.reduce_sum(valid_map * tf.abs(disp_est - disp_label)) / tf.reduce_sum(valid_map)

        return  loss

    def build_losses_finetune(self):

        # Down sample the ground truth of disparities
        self.downsample_dis()

        # calculate the loss according to the weight schedule
        self.disp_losses = [self.L1_loss_finetune(self.disp_refine[i], self.disp_downsample[i]) for i in range(4)]

        # apply the loss weight schedule to calculate the all losses from differetn layears
        self.disp_loss = tf.add_n([self.disp_losses[i] * self.loss_weights[i] for i in range(4)])

        # apply the L2 regularization loss

        self.reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        # total loss
        self.total_loss= 2*self.disp_loss + self.reg_loss

        self.EPE = self.EPE_finetune(self.disp_refine[0],self.dis_batch[0])

        return self.total_loss

    def build_summaries(self):

        # summary
        # summary the loss
        with tf.name_scope("losses"):
            tf.summary.scalar('disp_loss', self.dis_loss)
            tf.summary.scalar('reg_loss', self.reg_loss)
            tf.summary.scalar('total_loss', self.total_loss)
        # summary the images
        with tf.name_scope("images"):
            tf.summary.image("left", self.left_batch,max_outputs=1)
            tf.summary.image("right", self.right_batch,max_outputs=1)

        # summary the disparities
        with tf.name_scope("disparites"):
            # the ground truth of disparity
            tf.summary.image("disp0_gt",self.dis_batch,max_outputs=1)
            tf.summary.image("disp_left_initial", self.left_initial_dis, max_outputs=1)
            tf.summary.image("disp_left_refined", self.refined_dis, max_outputs=1)
            tf.summary.image("disp_left_error", self.conv_res1, max_outputs=1)

    def build_summaries_EncodeDecode(self):

        # summary
        # summary the loss
        with tf.name_scope("losses"):
            tf.summary.scalar('disp_loss', self.disp_loss)
            tf.summary.scalar('reg_loss', self.reg_loss)
            tf.summary.scalar('total_loss', self.total_loss)
            for i in range(4):
                tf.summary.scalar('disp_loss_' + str(i), self.disp_losses[i])
                tf.summary.scalar('loss_weight' + str(i), self.loss_weights[i])
        # summary the images
        with tf.name_scope("images"):
            tf.summary.image("left", self.left_batch, max_outputs=1)
            tf.summary.image("right", self.right_batch, max_outputs=1)

        # summary the disparities
        with tf.name_scope("disparites"):
            # the ground truth of disparity
            tf.summary.image("disp0_gt", self.dis_batch, max_outputs=1)
            tf.summary.image("disp_left_initial", self.left_initial_dis, max_outputs=1)

            for i in range(4):
                tf.summary.image("disp_refine" + str(i), self.disp_refine[i], max_outputs=1)


class DispRefineModel_occ(object):
    def __init__(self, mode, left_batch, right_batch, dis_batch, left_conv, right_conv, left_initial_dis, weight_decay,
                 dis_offset, is_train, loss_weights,occ_labels):
        self.mode = mode
        self.left_batch = left_batch
        self.right_batch = right_batch
        self.dis_batch = dis_batch
        self.left_conv = left_conv
        self.right_conv = right_conv
        self.occ_labels = occ_labels
        # self.conv2a=conv2a
        # self.conv2b=conv2b
        self.left_initial_dis = left_initial_dis
        self.weight_decay = weight_decay
        self.dis_offset = dis_offset
        self.is_train = is_train
        self.loss_weights = loss_weights
        self.initializer = tf.contrib.layers.variance_scaling_initializer()

        # build the model
        self.build_refine_model_EncodeDecode()

        # build the output

        self.build_outputs()

        if self.mode == 'test':
            return

        if self.mode == "fine_tune":

            self.build_losses_finetune()
            self.build_summaries_EncodeDecode()

        else:
            self.build_losses_EncodeDecode()
            # build the summaries
            self.build_summaries_EncodeDecode()

    def conv2d(self, x, kernel_shape, strides=1, relu=True, bn=False, regularizer=None, padding='SAME', scope="conv"):

        W = tf.get_variable("weights", kernel_shape, initializer=self.initializer, regularizer=regularizer)
        b = tf.get_variable("biases", kernel_shape[3], initializer=tf.constant_initializer(0.0))

        with tf.name_scope(scope):
            x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding=padding)
            x = tf.nn.bias_add(x, b)

            tf.summary.histogram("W", W)
            tf.summary.histogram("b", b)

            if bn:
                x = tf.layers.batch_normalization(x, training=self.is_train, momentum=0.99)

            if relu:
                x = tf.maximum(LEAKY_ALPHA * x, x)
        return x

    def conv2d_dilated(self, x, kernel_shape, rate=1, leaky_relu=True, bn=False, regularizer=None, padding='SAME',
                       scope="dilated_conv"):

        W = tf.get_variable("weights", kernel_shape, initializer=self.initializer, regularizer=regularizer)
        b = tf.get_variable("biases", kernel_shape[3], initializer=tf.constant_initializer(0.0))

        with tf.name_scope(scope):
            x = tf.nn.atrous_conv2d(x, W, rate, padding=padding)
            x = tf.nn.bias_add(x, b)

            tf.summary.histogram("W", W)
            tf.summary.histogram("b", b)
            if bn:
                x = tf.layers.batch_normalization(x, training=self.is_train, momentum=0.99)
            if leaky_relu:
                x = tf.maximum(0.2 * x, x)
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

    def upsampling_block_res(self, bottom, pr_dis, input_channels, output_channels):
        with tf.variable_scope("deconv"):
            # deconvolutional layer
            deconv = self.conv2d_transpose(bottom, [4, 4, output_channels, input_channels], strides=2)
        with tf.variable_scope("predict"):
            # predict layer
            predict = self.conv2d(bottom, [3, 3, input_channels, 1], strides=1, relu=False)
            # predict=res+pr_dis
            tf.summary.histogram("predict", predict)
        with tf.variable_scope("up_predict"):
            # upascale the predict layer
            upsampled_predict = self.conv2d_transpose(predict, [4, 4, 1, 1], strides=2, relu=False)
        with tf.variable_scope("concat"):
            concat = self.conv2d(tf.concat([deconv, upsampled_predict], axis=3),
                                 [3, 3, output_channels + 1, output_channels],
                                 strides=1, relu=False)
        return concat, predict

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

    def Recon_error(self, left_conv, right_conv, left_dis, dis_offset):

        # using the bilinear sample to generate left conv
        error_tensors = []
        # creat a constant tensor with labes
        ones = tf.ones_like(left_dis, dtype=tf.float32)

        for i in np.arange(-dis_offset, dis_offset + 0.25, 0.25):
            # redefine the left_dis
            offset = ones * i
            left_dis_i = left_dis + offset
            recon_leftconv = bilinear_sampler_1d_h(right_conv, -left_dis_i)

            error = tf.reduce_mean(tf.abs(left_conv - recon_leftconv), axis=3)
            error_tensors.append(error)
        return tf.transpose(tf.stack(error_tensors),
                            perm=[1, 2, 3, 0])

    def Fined_Correlation_error(self, left_conv, right_conv, left_dis, dis_offset):

        # using the bilinear sample to generate left conv
        corr_tensors = []
        error_tensors = []
        # creat a constant tensor with labes
        ones = tf.ones_like(left_dis, dtype=tf.float32)

        for i in np.arange(-dis_offset, dis_offset + 0.25, 0.25):
            # redefine the left_dis
            offset = ones * i
            left_dis_i = left_dis + offset
            recon_leftconv = bilinear_sampler_1d_h(right_conv, -left_dis_i)

            corr = tf.reduce_mean(tf.multiply(left_conv, recon_leftconv), axis=3)
            error = tf.reduce_mean(tf.abs(left_conv - recon_leftconv), axis=3)
            corr_tensors.append(corr)
            error_tensors.append(error)

        fined_corr = tf.transpose(tf.stack(corr_tensors), perm=[1, 2, 3, 0])
        recon_error = tf.transpose(tf.stack(error_tensors), perm=[1, 2, 3, 0])
        return fined_corr, recon_error

    def Squeeze_excitation_layer(self, input_x, out_dim, ratio, layer_name):
        with tf.name_scope(layer_name):
            with tf.name_scope(layer_name + '_global_pool'):
                squeeze = tf.reduce_mean(input_x, [1, 2], keep_dims=True)

            with tf.name_scope(layer_name + '_fully_connected1'):
                excitation = tf.layers.dense(inputs=squeeze, units=out_dim / ratio, use_bias=True)
                excitation = tf.nn.relu(excitation)
            with tf.name_scope(layer_name + '_fully_connected2'):
                excitation = tf.layers.dense(inputs=excitation, units=out_dim, use_bias=True)
                excitation = tf.nn.sigmoid(excitation)

            excitation = tf.reshape(excitation, [-1, 1, 1, out_dim])

            scale = input_x * excitation
            return scale

    def ResiBlolck(self, x, in_channel, out_channel, dilation=1, regularizer=None):

        # conv1
        residual = x
        conv2d_dilated = self.conv2d_dilated
        with tf.variable_scope("conv1"):
            conv1 = conv2d_dilated(x, [3, 3, in_channel, out_channel], rate=dilation, leaky_relu=True,
                                   regularizer=regularizer)
        # conv2
        with tf.variable_scope("conv2"):
            conv2 = conv2d_dilated(conv1, [3, 3, in_channel, out_channel], rate=dilation, leaky_relu=False,
                                   regularizer=regularizer)
            conv2 += residual
            # leacy relu
            out = tf.maximum(0.2 * conv2, conv2)

        return out

    def build_refine_model_ResBlocks(self):

        conv2d = self.conv2d
        resiblock = self.ResiBlolck
        # creat the the dispnet correlation model
        regularizer = tf.contrib.layers.l2_regularizer(self.weight_decay)

        with tf.variable_scope('refined_net'):
            with tf.name_scope("conv1"):
                height, width = tf.shape(self.left_initial_dis)[1], tf.shape(self.left_initial_dis)[2]
                conv1a = tf.image.resize_bilinear(self.left_conv, [height, width], align_corners=True)
                conv1b = tf.image.resize_bilinear(self.right_conv, [height, width], align_corners=True)

            with tf.variable_scope("conv1_press") as scope:
                conv1a_press = conv2d(conv1a, [3, 3, 64, 16], strides=1, regularizer=regularizer)
                scope.reuse_variables()
                conv1b_press = conv2d(conv1b, [3, 3, 64, 16], strides=1, regularizer=regularizer)

            with tf.name_scope("correlation"):
                fined_corr = self.Fined_Correlation(conv1a_press, conv1b_press, self.left_initial_dis, self.dis_offset)
                corr_dir = tf.concat([fined_corr, conv1a_press], axis=3)

            with tf.name_scope("photometirc_error"):
                recon_conv1a = bilinear_sampler_1d_h(conv1b_press, -self.left_initial_dis)
                ph_error = tf.abs(recon_conv1a - conv1a_press)
                ph_concat = tf.concat([ph_error, conv1a_press], axis=3)
                with tf.variable_scope("ph_conv"):
                    ph_conv = conv2d(ph_concat, [3, 3, 32, 16], strides=1, regularizer=regularizer)

            with tf.name_scope("concat_all"):
                concat_all = tf.concat([corr_dir, ph_conv], axis=3)
                num_channels = 8 * self.dis_offset + 1 + 16 + 16
                with tf.variable_scope("conv_corr"):
                    all_conv = conv2d(concat_all, [3, 3, num_channels, 32], bn=True, strides=1, regularizer=regularizer)

            with tf.variable_scope("ResBlock"):
                with tf.variable_scope("1"):
                    res1 = resiblock(all_conv, 32, 32, 1, regularizer=regularizer)
                with tf.variable_scope("2"):
                    res2 = resiblock(res1, 32, 32, 1, regularizer=regularizer)
                with tf.variable_scope("3"):
                    res3 = resiblock(res2, 32, 32, 1, regularizer=regularizer)
                with tf.variable_scope("4"):
                    res4 = resiblock(res3, 32, 32, 1, regularizer=regularizer)
                with tf.variable_scope("5"):
                    res5 = resiblock(res4, 32, 32, 1, regularizer=regularizer)
                with tf.variable_scope("6"):
                    res6 = resiblock(res5, 32, 32, 1, regularizer=regularizer)
            with tf.variable_scope("conv_res1"):
                self.conv_res1 = conv2d(res6, [3, 3, 32, 1], strides=1, relu=False, regularizer=regularizer)
            with tf.name_scope("RefineDis"):
                refined_dis = self.left_initial_dis + self.conv_res1
                self.refined_dis = tf.nn.relu(refined_dis, name=None)

    def build_refine_model_EncodeDecode(self):

        conv2d = self.conv2d
        upsampling_block_res = self.upsampling_block_res
        # creat the the dispnet correlation model
        regularizer = tf.contrib.layers.l2_regularizer(self.weight_decay)

        with tf.variable_scope('refined_net', reuse=tf.AUTO_REUSE):
            with tf.name_scope("dis_downsample"):
                height, width = tf.shape(self.left_initial_dis)[1], tf.shape(self.left_initial_dis)[2]

                self.left_dis_2 = tf.image.resize_bilinear(self.left_initial_dis,
                                                           [tf.to_int32(height / 2),
                                                            tf.to_int32(width / 2)])
                self.left_dis_2 = self.left_dis_2 * (1. / 2.0)

            with tf.variable_scope("conv1a_press") as scope:
                conv1a_press = conv2d(self.left_conv, [1, 1, 64, 16], strides=1, regularizer=regularizer)
                scope.reuse_variables()
                conv1b_press = conv2d(self.right_conv, [1, 1, 64, 16], strides=1, regularizer=regularizer)

            with tf.name_scope("correlation"):
                fined_corr, recon_error = self.Fined_Correlation_error(conv1a_press, conv1b_press, self.left_dis_2,
                                                                       self.dis_offset)
                recon_leftconv = bilinear_sampler_1d_h(conv1b_press, -self.left_dis_2)

            with tf.name_scope("concate_all"):
                r_concates_all = tf.concat([fined_corr, recon_error, conv1a_press], axis=3)
                #r_concates_all = tf.concat([fined_corr, conv1a_press], axis=3)
                # r_concates_all = tf.concat([fined_corr, conv1a_press, recon_leftconv], axis=3)
                channels = (2 * 4 * self.dis_offset + 1) * 2 + 16
                #channels = (2 * 4 * self.dis_offset + 1) + 16
                # channels = (2 * 4 * self.dis_offset + 1) + 16+16
                with tf.variable_scope("r_conv1_1"):
                    r_conv1_1 = conv2d(r_concates_all, [3, 3, channels, 64], strides=1, regularizer=regularizer)

            with tf.variable_scope("r_conv2"):
                r_conv2 = conv2d(r_conv1_1, [3, 3, 64, 128], strides=2, regularizer=regularizer)
            with tf.variable_scope("r_conv2_1"):
                r_conv2_1 = conv2d(r_conv2, [3, 3, 128, 128], strides=1, regularizer=regularizer)

            with tf.variable_scope("r_conv3"):
                r_conv3 = conv2d(r_conv2_1, [3, 3, 128, 256], strides=2, regularizer=regularizer)
            with tf.variable_scope("r_conv3_1"):
                r_conv3_1 = conv2d(r_conv3, [3, 3, 256, 256], strides=1, regularizer=regularizer)

                # -------------------------decoder part--------------------------#
            with tf.variable_scope("up2"):
                self.left_dis_8 = tf.image.resize_bilinear(self.left_initial_dis,
                                                           [tf.to_int32(height / 8),
                                                            tf.to_int32(width / 8)])
                self.left_dis_8 = self.left_dis_8 * (1. / 8.0)
                concat2, self.res_3 = upsampling_block_res(r_conv3_1, self.left_dis_8, 256, 128)
                self.predict3 = self.res_3 + self.left_dis_8
            with tf.variable_scope("up1"):
                self.left_dis_4 = tf.image.resize_bilinear(self.left_initial_dis,
                                                           [tf.to_int32(height / 4),
                                                            tf.to_int32(width / 4)])
                self.left_dis_4 = self.left_dis_4 * (1. / 4.0)
                concat1, self.res_2 = upsampling_block_res(concat2, self.left_dis_4, 128, 64)
                self.predict2 = self.res_2 + self.left_dis_4
            with tf.variable_scope("up0"):
                concat0, self.res_1 = upsampling_block_res(concat1, self.left_dis_2, 64, 32)
                self.predict1 = self.res_1 + self.left_dis_2
            with tf.variable_scope("prediction"):
                self.res_0 = conv2d(concat0, [3, 3, 32, 1], strides=1, relu=False)
                self.predict0 = self.res_0 + self.left_initial_dis

    def build_outputs(self):
        # store the disparities
        with tf.variable_scope('disparities'):
            self.disp_refine = [self.predict0, self.predict1, self.predict2, self.predict3]

    def downsample_dis_occlables(self):
        self.disp_downsample = {}
        self.occ_downsample={}
        # get the height and width of ground truth disparity image
        height, width = tf.shape(self.dis_batch)[1], tf.shape(self.dis_batch)[2]

        # down sample the disparity and change the disparity value according the down scale
        for n in range(0, 4):
            self.disp_downsample[n] = tf.image.resize_nearest_neighbor(self.dis_batch,
                                                                       [tf.to_int32(height / np.power(2, n)),
                                                                        tf.to_int32(width / np.power(2, n))])

            self.disp_downsample[n] = self.disp_downsample[n] * (1. / np.power(2, n))

            self.occ_downsample[n] = tf.image.resize_nearest_neighbor(self.occ_labels,
                                                                       [tf.to_int32(height / np.power(2, n)),
                                                                        tf.to_int32(width / np.power(2, n))])

    def pixel_wise_softmax(self, logits):
        max_axis = tf.reduce_max(logits, axis=3, keepdims=True)
        exponential_map = tf.exp(logits - max_axis)
        normalize = tf.reduce_sum(exponential_map, axis=3, keepdims=True)
        return exponential_map / normalize

    def L1_loss(self, disp_est, disp_label,occ_labels):

        # get the occlusion mask from the labels
        non_occ_gt_mask = tf.cast(tf.slice(occ_labels, [0, 0, 0, 0], [-1, -1, -1, 1]), tf.float32)
        # L1 loss for traning
        disp_diff=(tf.abs(disp_est - disp_label))
        l1_loss = tf.reduce_sum(disp_diff * non_occ_gt_mask) / tf.reduce_sum(non_occ_gt_mask)
        return l1_loss

    def Smooth_L1loss(self, disp_est, disp_label):

        # smooth l1 loss for reducing the oscillations problem
        loss = tf.losses.huber_loss(disp_label, disp_est)

        return loss

    def gradient_gaussion(self, img, sigma):

        halfwid = 2 * sigma
        num = tf.to_int32(2 * halfwid + 1)

        [xx, yy] = tf.meshgrid(tf.linspace(-halfwid, halfwid, num),
                               tf.linspace(-halfwid, halfwid, num))

        tmp = -(tf.square(xx) + tf.square(yy)) / (2 * np.square(sigma))
        tmp_exp = tf.exp(tmp)
        dx = tf.multiply(xx, tmp_exp)
        dy = tf.multiply(yy, tmp_exp)

        gaussion_x_filter = tf.reshape(dx, [num, num, 1, 1])
        gaussion_y_filter = tf.reshape(dy, [num, num, 1, 1])

        gradient_x = tf.nn.conv2d(img, gaussion_x_filter,
                                  strides=[1, 1, 1, 1], padding='SAME')
        gradient_y = tf.nn.conv2d(img, gaussion_y_filter,
                                  strides=[1, 1, 1, 1], padding='SAME')

        return gradient_x, gradient_y

    def gradient_loss_occ(self, disp_est, disp_label,occ_labels):

        disp_est_gradients_x, disp_est_gradients_y = self.gradient_gaussion(disp_est, 1.0)
        disp_label_gradients_x, disp_label_gradients_y = self.gradient_gaussion(disp_label, 1.0)

        grad_diff_x=tf.abs(disp_est_gradients_x - disp_label_gradients_x)
        grad_diff_y=tf.abs(disp_est_gradients_y - disp_label_gradients_y)

        # get the occlusion mask from the labels
        non_occ_gt_mask = tf.cast(tf.slice(occ_labels, [0, 0, 0, 0], [-1, -1, -1, 1]), tf.float32)
        # get the recon_error with occlusion mask
        smoothness_x = tf.reduce_sum(grad_diff_x * non_occ_gt_mask) / tf.reduce_sum(non_occ_gt_mask)
        smoothness_y = tf.reduce_sum(grad_diff_y * non_occ_gt_mask) / tf.reduce_sum(non_occ_gt_mask)

        return smoothness_x + smoothness_y

    def gradient_loss(self, disp_est, disp_label):

        disp_est_gradients_x, disp_est_gradients_y = self.gradient_gaussion(disp_est, 1.0)
        disp_label_gradients_x, disp_label_gradients_y = self.gradient_gaussion(disp_label, 1.0)

        smoothness_x = tf.reduce_mean(tf.abs(disp_est_gradients_x - disp_label_gradients_x))
        smoothness_y = tf.reduce_mean(tf.abs(disp_est_gradients_y - disp_label_gradients_y))

        return smoothness_x + smoothness_y

    def EPE_finetune(self, disp_est, disp_label):

        # creat the valid map
        valid_map = tf.where(tf.equal(disp_label, 0), tf.zeros_like(disp_label, dtype=tf.float32),
                             tf.ones_like(disp_label, dtype=tf.float32))

        epe = tf.reduce_sum(valid_map * tf.abs(disp_est - disp_label)) / tf.reduce_sum(valid_map)

        return epe

    def Occ_loss(self, logits, labels):

        # est_occmap: n,w,h,2
        flat_logits = tf.reshape(logits, [-1, 2])
        flat_labels = tf.reshape(labels, [-1, 2])

        # calculate the cross cropy loss
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=flat_logits,
                                                                      labels=flat_labels))

        return loss

    def Recon_error_loss(self,left_batch,right_batch,disp_est,occ_labels):

        # calculate the reconstrution error for the left and  right image loss with occlusion mask

        # get the reconstrcted left image
        recon_left = bilinear_sampler_1d_h(right_batch, -disp_est)

        # get the difference between left and recon_left
        diff_left=tf.abs(left_batch-recon_left)

        #get the occlusion mask from the labels
        non_occ_gt_mask = tf.cast(tf.slice(occ_labels, [0, 0, 0, 0], [-1,-1,-1,1]), tf.float32)
        # get the recon_error with occlusion mask
        recon_loss=tf.reduce_sum(diff_left*non_occ_gt_mask)/tf.reduce_sum(non_occ_gt_mask)

        return recon_loss

    def build_losses_ResBlocks(self):

        # refined disarity regression loss
        self.dis_loss = self.L1_loss(self.refined_dis, self.dis_batch)

        # apply the L2 regularization loss

        self.reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        # total loss
        self.total_loss = self.dis_loss + self.reg_loss

        self.EPE = self.L1_loss(self.refined_dis, self.dis_batch)

        return self.total_loss

    def build_losses_EncodeDecode(self):

        # Down sample the ground truth of disparities
        self.downsample_dis_occlables()

        # calculate the loss according to the weight schedule
        self.disp_losses = [self.L1_loss(self.disp_refine[i], self.disp_downsample[i],self.occ_downsample[i]) for i in range(4)]

        # apply the loss weight schedule to calculate the all losses from differetn layears
        self.disp_loss = tf.add_n([self.disp_losses[i] * self.loss_weights[i] for i in range(4)])

        # calculate the disp gradient losses according to the weight schedule
        self.disp_grad_losses = [self.gradient_loss(self.disp_refine[i], self.disp_downsample[i]) for i
                                 in
                                 range(4)]

        # apply the loss weight schedule to calculate the gradient loss from different layers
        self.grad_loss = tf.add_n([self.disp_grad_losses[i] * self.loss_weights[i] for i in range(4)])

        # apply the reconstruction error loss
        self.recon_loss=self.Recon_error_loss(self.left_batch,self.right_batch,self.disp_refine[0],self.occ_labels)

        # apply the L2 regularization loss
        self.reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        # total loss
        #self.total_loss = self.disp_loss + self.recon_loss+ 2*self.grad_loss+ self.reg_loss
        self.total_loss = self.disp_loss + self.recon_loss + 2*self.grad_loss+self.reg_loss
        #self.total_loss = self.disp_loss + 10*self.recon_loss  + self.reg_loss
        # self.total_loss = self.disp_loss

        self.EPE_occ = self.L1_loss(self.disp_refine[0], self.dis_batch[0],self.occ_labels)
        self.EPE_all = tf.reduce_mean(tf.abs(self.disp_refine[0] -self.dis_batch[0]))

        return self.total_loss

    def L1_loss_finetune(self, disp_est, disp_label):

        # creat the valid map
        valid_map = tf.where(tf.equal(disp_label, 0), tf.zeros_like(disp_label, dtype=tf.float32),
                             tf.ones_like(disp_label, dtype=tf.float32))

        loss = tf.reduce_sum(valid_map * tf.abs(disp_est - disp_label)) / tf.reduce_sum(valid_map)

        return loss

    def build_losses_finetune(self):

        # Down sample the ground truth of disparities
        self.downsample_dis()

        # calculate the loss according to the weight schedule
        self.disp_losses = [self.L1_loss_finetune(self.disp_refine[i], self.disp_downsample[i]) for i in range(4)]

        # apply the loss weight schedule to calculate the all losses from differetn layears
        self.disp_loss = tf.add_n([self.disp_losses[i] * self.loss_weights[i] for i in range(4)])

        # apply the L2 regularization loss

        self.reg_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

        # total loss
        self.total_loss = 2 * self.disp_loss + self.reg_loss

        self.EPE = self.EPE_finetune(self.disp_refine[0], self.dis_batch[0])

        return self.total_loss


    def build_summaries_EncodeDecode(self):

        # summary
        # summary the loss
        with tf.name_scope("losses"):
            tf.summary.scalar('disp_loss', self.disp_loss)
            tf.summary.scalar('reg_loss', self.reg_loss)
            tf.summary.scalar('recon_loss', self.recon_loss)
            tf.summary.scalar('gradient_loss', self.grad_loss)
            tf.summary.scalar('total_loss', self.total_loss)
            for i in range(4):
                tf.summary.scalar('disp_loss_' + str(i), self.disp_losses[i])
                tf.summary.scalar('loss_weight' + str(i), self.loss_weights[i])
        # summary the images
        with tf.name_scope("images"):
            tf.summary.image("left", self.left_batch, max_outputs=1)
            tf.summary.image("right", self.right_batch, max_outputs=1)

        # summary the disparities
        with tf.name_scope("disparites"):
            # the ground truth of disparity
            tf.summary.image("disp0_gt", self.dis_batch, max_outputs=1)
            tf.summary.image("disp_left_initial", self.left_initial_dis, max_outputs=1)

            for i in range(4):
                tf.summary.image("disp_refine" + str(i), self.disp_refine[i], max_outputs=1)


