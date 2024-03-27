import os
import sys
import time
import math
import datetime
import argparse
import tensorflow as tf
import glob
import matplotlib.pyplot as plt
import  imageio.v2 as imageio
from imageio import  imread, imsave


from math import ceil

CORR = True


from EnhancedNet_model import *
from  util import writePFM



parser = argparse.ArgumentParser(description='EnhancedNet TensorFlow implementation.')

parser.add_argument('--left_path',                  type=str,   help='path to the left image', required=True)
parser.add_argument('--right_path',                 type=str,   help='path to the right image', required=True)
parser.add_argument('--pretrain_model',             type=str,   help='path to the pretrained model', required=True)
parser.add_argument('--net_type',                        type=str,   help='Initial Net or Enhanced net', default='initial')
parser.add_argument('--save_dir',                   type=str,   help='directory to save results', required=True)

args = parser.parse_args()




def get_crop_width_height(org_width, org_height):
    divisor = 64.
    crop_width = int(math.floor(org_width / divisor) * divisor)
    crop_height = int(math.floor(org_height / divisor) * divisor)
    return  crop_width, crop_height


def Inference_InitialNet(left_path,right_path, pretrain_path,save_dir):

    # load the left and right image batch and placeholders
    left_image_batch = tf.placeholder(tf.uint8, name="left_img")
    right_image_batch = tf.placeholder(tf.uint8, name="right_img")
    org_width = tf.placeholder(tf.int32, name="org_width")
    org_height = tf.placeholder(tf.int32, name="org_height")
    crop_width = tf.placeholder(tf.int32, name="org_width")
    crop_height = tf.placeholder(tf.int32, name="org_height")
    target = tf.placeholder(name="disp_gt", dtype=tf.float32)

    # resize the images to the crop size

    left = tf.image.convert_image_dtype(left_image_batch, tf.float32)
    right = tf.image.convert_image_dtype(right_image_batch, tf.float32)
    left = tf.image.resize_bilinear(left, [crop_height, crop_width])
    right = tf.image.resize_bilinear(right, [crop_height, crop_width])


    mean_left, _std = tf.nn.moments(left, axes=[0, 1, 2], keep_dims=True)
    mean_right, _std = tf.nn.moments(right, axes=[0, 1, 2], keep_dims=True)


    left = (left - mean_left)
    right = (right - mean_right)

    # load the first initial graph
    DispModel_left = DispNetModel(left, right, 0.0004, 80)


    # upsample the estimated disparity to the original size
    disp_pred_tf = tf.image.resize_bilinear(DispModel_left.disp_est[0], [org_height, org_width])
    disp_pred_tf = disp_pred_tf * tf.to_float(org_width) / tf.to_float(crop_width)

    # Session
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.96)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    # SAVER
    train_saver = tf.train.Saver()

    # INIT
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # RESTORE
    restore_path = tf.train.latest_checkpoint(pretrain_path)
    print("Restoring from %s" % restore_path)
    train_saver.restore(sess, restore_path)

    # read the left and righ image
    input_left_img = imread(left_path)
    input_right_img = imread(right_path)

    left_size=input_left_img.shape
    right_size = input_right_img.shape

    input_left = np.zeros((left_size[0], left_size[1], 3))
    input_right = np.zeros((right_size[0], right_size[1], 3))
    if len(left_size)==2 and len(right_size)==2:

        input_left[:, :, 0] = input_left_img
        input_left[:, :, 1] = input_left_img
        input_left[:, :, 2] = input_left_img


        input_right[:, :, 0] = input_right_img
        input_right[:, :, 1] = input_right_img
        input_right[:, :, 2] = input_right_img

    elif left_size[2] == 4 and right_size[2] == 4:
        input_left = input_left_img[:, :, :3]
        input_right = input_right_img[:, :, :3]
    else:
        input_left = input_left_img
        input_right = input_right_img


    # get the crop size
    _org_height, _org_width, _num_channels = input_left.shape
    _crop_width, _crop_height = get_crop_width_height(_org_width, _org_height)

    print("crop_width: %d  crop_height: %d " %(_crop_width, _crop_height))

    # create the feed dict
    feed_dict = {}

    feed_dict[org_height] = _org_height
    feed_dict[org_width] = _org_width
    feed_dict[crop_height] = _crop_height
    feed_dict[crop_width] = _crop_width


    # reshape to the batch shape
    newshape1 = (1, np.shape(input_left)[0], np.shape(input_left)[1], 3)

    input_left = np.reshape(input_left, newshape1)
    input_right = np.reshape(input_right, newshape1)

    feed_dict[left_image_batch] = input_left
    feed_dict[right_image_batch] = input_right

    disp_pred= sess.run(disp_pred_tf, feed_dict=feed_dict)

    # reshape the predicted disparity
    newshape = (np.shape(disp_pred)[1], np.shape(disp_pred)[2])
    disp_pred_to_img = np.reshape(disp_pred, newshape)

    # save the disparity
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # get the file name
    dirStr = os.path.basename(left_path)
    imgname = dirStr.split('.')[0]

    dis_save_path = save_dir + imgname + ".png"
    plt.imsave(dis_save_path, disp_pred_to_img, cmap="gray")
    save_dis_path_pfm = dis_save_path.replace(".png", ".pfm")
    writePFM(save_dis_path_pfm, disp_pred_to_img)


def Inference_EnhancedNet(left_path,right_path, pretrain_path,save_dir):

    # load the left and right image batch and placeholders
    left_image_batch = tf.placeholder(tf.uint8, name="left_img")
    right_image_batch = tf.placeholder(tf.uint8, name="right_img")
    org_width = tf.placeholder(tf.int32, name="org_width")
    org_height = tf.placeholder(tf.int32, name="org_height")
    crop_width = tf.placeholder(tf.int32, name="org_width")
    crop_height = tf.placeholder(tf.int32, name="org_height")
    target = tf.placeholder(name="disp_gt", dtype=tf.float32)

    # resize the images to the crop size

    left = tf.image.convert_image_dtype(left_image_batch, tf.float32)
    right = tf.image.convert_image_dtype(right_image_batch, tf.float32)
    left = tf.image.resize_bilinear(left, [crop_height, crop_width])
    right = tf.image.resize_bilinear(right, [crop_height, crop_width])


    mean_left, _std = tf.nn.moments(left, axes=[0, 1, 2], keep_dims=True)
    mean_right, _std = tf.nn.moments(right, axes=[0, 1, 2], keep_dims=True)


    left = (left - mean_left)
    right = (right - mean_right)

    # load the first initial graph
    DispModel_left = DispNetModel(left, right, 0.0004, 80)

    output_left_initial = DispModel_left.disp_est[0]

    conv1a = DispModel_left.conv1a
    conv1b = DispModel_left.conv1b


    # Build model
    is_train = tf.placeholder_with_default(False, shape=(), name='training')
    loss_weights = tf.placeholder(tf.float32, shape=(4), name="loss_weights")
    initial_loss_weights = [1., .2, .0, .0]
    model = DispRefineModel("test", left, right, target, conv1a, conv1b, output_left_initial, 0.0004, 5,
                            is_train, loss_weights)

    # upsample the estimated disparity to the original size
    disp_pred_tf = tf.image.resize_bilinear(model.disp_refine[0], [org_height, org_width])
    disp_pred_tf = disp_pred_tf * tf.to_float(org_width) / tf.to_float(crop_width)

    # Session
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.96)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

    # SAVER
    train_saver = tf.train.Saver()

    # INIT
    sess.run(tf.global_variables_initializer())
    sess.run(tf.local_variables_initializer())

    # RESTORE
    restore_path = tf.train.latest_checkpoint(pretrain_path)
    print("Restoring from %s" % restore_path)
    train_saver.restore(sess, restore_path)

    # read the left and righ image
    input_left_img = imread(left_path)
    input_right_img = imread(right_path)

    left_size=input_left_img.shape
    right_size = input_right_img.shape

    input_left = np.zeros((left_size[0], left_size[1], 3))
    input_right = np.zeros((right_size[0], right_size[1], 3))
    if len(left_size)==2 and len(right_size)==2:

        input_left[:, :, 0] = input_left_img
        input_left[:, :, 1] = input_left_img
        input_left[:, :, 2] = input_left_img


        input_right[:, :, 0] = input_right_img
        input_right[:, :, 1] = input_right_img
        input_right[:, :, 2] = input_right_img

    elif left_size[2] == 4 and right_size[2] == 4:
        input_left = input_left_img[:, :, :3]
        input_right = input_right_img[:, :, :3]
    else:
        input_left = input_left_img
        input_right = input_right_img


    # get the crop size
    _org_height, _org_width, _num_channels = input_left.shape
    _crop_width, _crop_height = get_crop_width_height(_org_width, _org_height)

    print("crop_width: %d  crop_height: %d " %(_crop_width, _crop_height))

    # create the feed dict
    feed_dict = {}

    feed_dict[org_height] = _org_height
    feed_dict[org_width] = _org_width
    feed_dict[crop_height] = _crop_height
    feed_dict[crop_width] = _crop_width

    feed_dict[is_train] = False
    feed_dict[loss_weights] = initial_loss_weights

    # reshape to the batch shape
    newshape1 = (1, np.shape(input_left)[0], np.shape(input_left)[1], 3)
    newshape2 = (1, np.shape(input_left)[0], np.shape(input_left)[1], 1)

    input_left = np.reshape(input_left, newshape1)
    input_right = np.reshape(input_right, newshape1)

    feed_dict[left_image_batch] = input_left
    feed_dict[right_image_batch] = input_right

    disp_pred= sess.run(disp_pred_tf, feed_dict=feed_dict)

    # reshape the predicted disparity
    newshape = (np.shape(disp_pred)[1], np.shape(disp_pred)[2])
    disp_pred_to_img = np.reshape(disp_pred, newshape)

    # save the disparity
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    # get the file name
    dirStr = os.path.basename(left_path)
    imgname = dirStr.split('.')[0]

    dis_save_path = save_dir + imgname + ".png"
    plt.imsave(dis_save_path, disp_pred_to_img, cmap="gray")
    save_dis_path_pfm = dis_save_path.replace(".png", ".pfm")
    writePFM(save_dis_path_pfm, disp_pred_to_img)




if __name__ == '__main__':


    if args.net_type=="initial":
        Inference_InitialNet(args.left_path, args.right_path, args.pretrain_model, args.save_dir)
    if args.net_type=="enhanced":
        Inference_EnhancedNet(args.left_path, args.right_path, args.pretrain_model, args.save_dir)

