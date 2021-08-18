import tensorflow as tf
import numpy as np
import time
from datetime import timedelta
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
from keras import regularizers
from tensorflow.python.keras.utils.kernelized_utils import exact_gaussian_kernel

def maximum_mean_discrepancy(x, y, kernel=exact_gaussian_kernel):
    cost = tf.reduce_mean(kernel(x, x, stddev=2))
    cost += tf.reduce_mean(kernel(y, y, stddev=2))
    cost -= 2 * tf.reduce_mean(kernel(x, y, stddev=2))
    cost = tf.where(cost > 0, cost, 0, name='value')
    return cost


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def parse_exmp(serial_exmp):
    feats = tf.parse_single_example(serial_exmp, features={'user_profile_0': tf.FixedLenFeature([8], tf.int64),
                                                           'user_profile_1': tf.VarLenFeature(tf.int64),
                                                           'user_profile_2': tf.VarLenFeature(tf.int64),
                                                           "user_RecAnchor": tf.VarLenFeature(tf.int64),
                                                           "user_statIds": tf.FixedLenFeature([12], tf.int64),
                                                           "user_profile_3": tf.FixedLenFeature([6], tf.int64),
                                                           "redict_weights": tf.FixedLenFeature([8], tf.float32),
                                                           "user_songTagids": tf.VarLenFeature(tf.int64),
                                                           "user_tagWeights": tf.VarLenFeature(tf.float32),
                                                           
                                                           "Wanchorids_long": tf.VarLenFeature(tf.int64) ,
                                                           "Wanchorids_long_len": tf.FixedLenFeature([], tf.int64),
                                                           
                                                           "Wanchorids_short": tf.VarLenFeature(tf.int64),
                                                           "Wanchorids_short_len": tf.FixedLenFeature([], tf.int64),
                                                           
                                                           "Wanchorids_noclick": tf.VarLenFeature(tf.int64),
                                                           "Wanchorids_noclick_len": tf.FixedLenFeature([], tf.int64),
                                                           
                                                           "Wanchorids_effect": tf.VarLenFeature(tf.int64),
                                                           "Wanchorids_effect_len": tf.FixedLenFeature([], tf.int64),

                                                           'anchor_profile_0': tf.FixedLenFeature([7], tf.int64),
                                                           'anchor_profile_1': tf.VarLenFeature(tf.int64),
                                                           'anchor_profile_2': tf.VarLenFeature(tf.int64),
                                                           'anchor_live_profile_0': tf.FixedLenFeature([4], tf.int64),
                                                           'anchor_live_profile_1': tf.VarLenFeature(tf.int64),
                                                           'anchor_stats': tf.FixedLenFeature([15], tf.int64),
                                                           'anchor_stat_values': tf.FixedLenFeature([15], tf.float32),
                                                           "anchor_songTagids": tf.VarLenFeature(tf.int64),
                                                           "anchor_tagWeights": tf.VarLenFeature(tf.float32),
                                                           "day_ctr_seq": tf.VarLenFeature( tf.float32),
                                                           "day_cvr_seq": tf.VarLenFeature( tf.float32),
                                                           "day_ctrid_seq": tf.VarLenFeature( tf.int64),
                                                           "day_cvrid_seq": tf.VarLenFeature( tf.int64),
                                                           "ctcvr_seq_len": tf.FixedLenFeature([], tf.int64),
                                                           "anchor_live_profile_2": tf.FixedLenFeature([38], tf.int64),

                                                           'user_id': tf.FixedLenFeature([], tf.int64),
                                                           'item_id': tf.FixedLenFeature([], tf.int64),
                                                           'anchorId': tf.FixedLenFeature([1], tf.int64),
                                                           'label_ctr': tf.FixedLenFeature([], tf.int64),
                                                           'label_cvr': tf.FixedLenFeature([], tf.int64),
                                                           "realtime_values": tf.FixedLenFeature([22], tf.float32),
                                                           'realtime_ids': tf.FixedLenFeature([22], tf.int64),
                                                           'fea_sim': tf.FixedLenFeature([2], tf.float32),
                                                           "hourId": tf.FixedLenFeature([1], tf.int64),
                                                           "dayOfWeek": tf.FixedLenFeature([1], tf.int64),
                                                           "live_position": tf.FixedLenFeature([1],tf.int64),
                                                           "resource_position": tf.FixedLenFeature([1], tf.int64)
                                                           })

    user_profile_0 = feats['user_profile_0']  # 使用VarLenFeature读入的是一个sparse_tensor，用该函数进行转换
    user_profile_1 = feats["user_profile_1"]
    user_profile_2 = feats["user_profile_2"]
    user_RecAnchor = feats["user_RecAnchor"]
    user_statIds = feats["user_statIds"]
    user_profile_3 = feats["user_profile_3"]

    redict_weights=feats["redict_weights"]
    user_songTagids=feats["user_songTagids"]
    user_tagWeights = feats["user_tagWeights"]
    
    Wanchorids_long=feats["Wanchorids_long"]
    Wanchorids_long_len = feats["Wanchorids_long_len"]
    
    Wanchorids_short = feats["Wanchorids_short"]
    Wanchorids_short_len = feats["Wanchorids_short_len"]
    
    Wanchorids_noclick = feats["Wanchorids_noclick"]
    Wanchorids_noclick_len = feats["Wanchorids_noclick_len"]
    
    Wanchorids_effect = feats["Wanchorids_effect"]
    Wanchorids_effect_len = feats["Wanchorids_effect_len"]

    anchor_profile_0 = feats['anchor_profile_0']
    anchor_profile_1 = feats["anchor_profile_1"]
    anchor_profile_2 = feats["anchor_profile_2"]
    anchor_live_profile_0 = feats["anchor_live_profile_0"]
    anchor_live_profile_1 = feats["anchor_live_profile_1"]
    anchor_stats = feats["anchor_stats"]
    anchor_stat_values = feats["anchor_stat_values"]
    anchor_live_profile_2=feats["anchor_live_profile_2"]

    day_ctr_seq = feats["day_ctr_seq"]
    day_cvr_seq =  feats["day_cvr_seq"]
    day_ctrid_seq =  feats["day_ctrid_seq"]
    day_cvrid_seq = feats["day_cvrid_seq"]
    ctcvr_seq_len= feats["ctcvr_seq_len"]

    anchor_songTagids = feats["anchor_songTagids"]
    anchor_tagWeights = feats["anchor_tagWeights"]

    anchorId = feats["anchorId"]
    fea_sim = feats["fea_sim"]

    realtime_values = feats["realtime_values"]
    realtime_ids = feats["realtime_ids"]
    hourId = feats["hourId"]
    dayOfWeek= feats["dayOfWeek"]
    live_position= feats["live_position"]
    resource_position= feats["resource_position"]


    user_id = feats["user_id"]
    item_id = feats["item_id"]

    label_ctcvr = feats["label_cvr"]
    label_ctr = feats["label_ctr"]

    return label_ctr, label_ctcvr, user_profile_0, user_profile_1, user_profile_2, user_RecAnchor, user_statIds, user_profile_3, \
           Wanchorids_long ,Wanchorids_long_len, Wanchorids_short ,Wanchorids_short_len,Wanchorids_noclick, Wanchorids_noclick_len, Wanchorids_effect, Wanchorids_effect_len, \
           anchor_profile_0, anchor_profile_1, anchor_profile_2, anchor_live_profile_0, anchor_live_profile_1, anchor_stats, anchor_stat_values, \
           user_songTagids, user_tagWeights, anchor_songTagids, anchor_tagWeights, \
           day_ctr_seq, day_cvr_seq, day_ctrid_seq, day_cvrid_seq, ctcvr_seq_len, anchorId, fea_sim, realtime_values, realtime_ids,\
           hourId, dayOfWeek,  redict_weights, live_position, anchor_live_profile_2, user_id


def get_dataset(fname):
    #dataset = tf.data.TFRecordDataset(fname)
    files = tf.data.Dataset.list_files(fname)
    read_func = lambda x: tf.data.TFRecordDataset(x)
    dataset = files.apply(tf.contrib.data.parallel_interleave(read_func,
                                                              cycle_length=32,
                                                              sloppy=False,
                                                              block_length=256,
                                                              buffer_output_elements=256,
                                                              prefetch_input_elements=256))


    return dataset.map(parse_exmp, num_parallel_calls=32)  # use padded_batch method if padding needed


def cal_group_auc(labels, preds, user_id_list):
    """Calculate group auc"""

    print('*' * 50)
    if len(user_id_list) != len(labels):
        raise ValueError(
            "impression id num should equal to the sample num," \
            "impression id num is {0}".format(len(user_id_list)),
            "impression id num is {0}".format(len(labels))
        )
    group_score = defaultdict(lambda: [])
    group_truth = defaultdict(lambda: [])
    for idx, truth in enumerate(labels):
        user_id = user_id_list[idx]
        score = preds[idx]
        truth = labels[idx]
        group_score[user_id].append(score)
        group_truth[user_id].append(truth)

    group_flag = defaultdict(lambda: False)
    for user_id in set(user_id_list):
        truths = group_truth[user_id]
        flag = False
        for i in range(len(truths) - 1):
            if truths[i] != truths[i + 1]:
                flag = True
                break
        group_flag[user_id] = flag

    impression_total = 0
    total_auc = 0
    #
    for user_id in group_flag:
        if group_flag[user_id]:
            auc = roc_auc_score(np.asarray(group_truth[user_id]), np.asarray(group_score[user_id]))
            total_auc += auc * len(group_truth[user_id])
            impression_total += len(group_truth[user_id])
    group_auc = float(total_auc) / impression_total
    group_auc = round(group_auc, 4)
    return group_auc



def eval_test_data(model, config, sess, file):
    # data preprocessing
    batch_size = config["batch_size"]
    start_time = time.time()

    data_test = get_dataset(file)
    data_test = data_test.prefetch(batch_size * 10).batch(batch_size)
    iterator_test = data_test.make_one_shot_iterator()
    next_element_test = iterator_test.get_next()

    y_pctr = []
    y_pcvr = []
    y_pctcvr = []

    y_label_ctr = []
    y_label_ctcvr = []
    loss_test = []
    newUser_index = []

    user_id_list = []
    i=0
    try:
        while True:
            label_ctr, label_ctcvr, user_profile_0, user_profile_1, user_profile_2, user_RecAnchor, user_statIds, user_profile_3, \
            Wanchorids_long, Wanchorids_long_len, Wanchorids_short, Wanchorids_short_len, Wanchorids_noclick, Wanchorids_noclick_len, Wanchorids_effect, Wanchorids_effect_len, \
            anchor_profile_0, anchor_profile_1, anchor_profile_2, anchor_live_profile_0, anchor_live_profile_1, anchor_stats, anchor_stat_values, \
            user_songTagids, user_tagWeights, anchor_songTagids, anchor_tagWeights, \
            day_ctr_seq, day_cvr_seq, day_ctrid_seq, day_cvrid_seq, ctcvr_seq_len, anchorId, fea_sim, realtime_values, realtime_ids, \
            hourId, dayOfWeek, redict_weights, live_position, anchor_live_profile_2, user_id = sess.run(next_element_test)

            if (len(anchor_profile_basefea) < batch_size):
                break

            y_label_ctr.extend(label_ctr)
            y_label_ctcvr.extend(label_ctcvr)
            user_id_list.extend(user_id)

            feed_dict = {model.user_profile_0: user_profile_0,
                         model.user_profile_1: user_profile_1,
                         model.user_profile_2: user_profile_2,
                         model.user_RecAnchor: user_RecAnchor,
                         model.user_statIds: user_statIds,
                         model.user_profile_3: user_profile_3,
                         model.redict_weights_pre: redict_weights,
                         model.user_songTagids: user_songTagids,
                         model.user_tagWeights: user_tagWeights,
                         ## 用户长短序列
                         ## 长期90天
                         model.Wanchorids_long: Wanchorids_long,
                         model.Wanchorids_long_len_pre: Wanchorids_long_len,
                         ## 短期30天
                         model.Wanchorids_short: Wanchorids_short,
                         model.Wanchorids_short_len_pre: Wanchorids_short_len,
                         ## noclick
                         model.Wanchorids_noclick: Wanchorids_noclick,
                         model.Wanchorids_noclick_len_pre: Wanchorids_noclick_len,
                         ## effect
                         model.Wanchorids_effect: Wanchorids_effect,
                         model.Wanchorids_effect_len_pre: Wanchorids_effect_len,
                         ##主播
                         model.anchor_profile_0: anchor_profile_0,
                         model.anchor_profile_1: anchor_profile_1,
                         model.anchor_profile_2: anchor_profile_2,
                         model.anchor_live_profile_0: anchor_live_profile_0,
                         model.anchor_live_profile_1: anchor_live_profile_1,
                         model.anchor_stats: anchor_stats,
                         model.anchor_stat_values: anchor_stat_values,
                         model.day_ctr_seq: day_ctr_seq,
                         model.day_cvr_seq: day_cvr_seq,
                         model.day_ctrid_seq: day_ctrid_seq,
                         model.day_cvrid_seq: day_cvrid_seq,
                         model.anchor_songTagids: anchor_songTagids,
                         model.anchor_tagWeights: anchor_tagWeights,
                         model.ctcvr_seq_len: ctcvr_seq_len,
                         model.anchor_live_profile_2: anchor_live_profile_2,
                         ## 实时特征
                         model.realtime_ids: realtime_ids,
                         model.realtime_values: realtime_values,
                         model.fea_sim: fea_sim,
                         model.anchorId: anchorId,
                         model.hourId: hourId,
                         model.dayOfWeek: dayOfWeek,
                         model.live_position: live_position,
                         model.label_ctr: label_ctr,
                         model.label_ctcvr: label_ctcvr,

                         model.keep_prob: 1.0,
                         model.train_phase: False
                         }

            temp_pctr, temp_pcvr, temp_pctcvr, temp_newUser_index = sess.run((model.pctr, model.pcvr, model.pctcvr, model.newUser_index), feed_dict=feed_dict)
            y_pctr.extend(temp_pctr)
            y_pcvr.extend(temp_pcvr)
            y_pctcvr.extend(temp_pctcvr)
            newUser_index.extend(temp_newUser_index)
            # loss_test.append(temp_loss)
            # user_id_list.extend(user_id.reshape(len(user_id)))
    except  Exception as e:
        print("error: ", e)
        pass

    print("------------------------------------------------------------------------------------------")
    print("max ctr: ", max(y_pctr))
    print("max cvr: ", max(y_pcvr))
    print("max ctcvr: ", max(y_pctcvr))
    debug_ctr = 0
    debug_cvr = 0
    debug_ctcvr = 0
    for i in range(0, len(y_pctr)):
        if y_pctr[i] > 0.8:
            debug_ctr += 1
        if y_pcvr[i] > 0.8:
            debug_cvr += 1
        if y_pctcvr[i] > 0.8:
            debug_ctcvr += 1
    print("debug ctr >0.8 : ", debug_ctr)
    print("debug cvr >0.8 : ", debug_cvr)
    print("debug ctcvr >0.8 : ", debug_ctcvr)

    y_label_ctr = np.array(y_label_ctr)
    y_pctr = np.array(y_pctr)

    y_label_ctcvr = np.array(y_label_ctcvr)
    y_pctcvr = np.array(y_pctcvr)
    user_id_list = np.array(user_id_list)

    # 计算新用户AUC
    auc_ctr_new = roc_auc_score(y_label_ctr[newUser_index], y_pctr[newUser_index])
    auc_ctrcvr_new = roc_auc_score(y_label_ctcvr[newUser_index], y_pctcvr[newUser_index])
    gauc_ctcvr_new =cal_group_auc(y_label_ctcvr[newUser_index],y_pctcvr[newUser_index], user_id_list[newUser_index])
    gauc_ctr_new = cal_group_auc(y_label_ctr[newUser_index], y_pctr[newUser_index], user_id_list[newUser_index])
    print("new user, CTR AUC is {0}, CTCVR AUC is {1}, CTR GAUC is {2}, CTCVR GAUC is {3}".format(auc_ctr_new, auc_ctrcvr_new, gauc_ctr_new, gauc_ctcvr_new))


    # 计算老用户AUC
    oldUser_index = list(map(lambda x: not x, newUser_index))
    auc_ctr_old = roc_auc_score(y_label_ctr[oldUser_index], y_pctr[oldUser_index])
    auc_ctrcvr_old = roc_auc_score(y_label_ctcvr[oldUser_index], y_pctcvr[oldUser_index])
    gauc_ctcvr_old =cal_group_auc(y_label_ctcvr[oldUser_index],y_pctcvr[oldUser_index], user_id_list[oldUser_index])
    gauc_ctr_old = cal_group_auc(y_label_ctr[oldUser_index], y_pctr[oldUser_index], user_id_list[oldUser_index])
    print("old user, CTR AUC is {0}, CTCVR AUC is {1}, CTR GAUC is {2}, CTCVR GAUC is {3}".format(auc_ctr_old, auc_ctrcvr_old, gauc_ctr_old, gauc_ctcvr_old))


    ## 计算GAUC
    auc_ctr = roc_auc_score(y_label_ctr, y_pctr)
    auc_ctrcvr = roc_auc_score(y_label_ctcvr, y_pctcvr)
    gauc_ctcvr=cal_group_auc(y_label_ctcvr,y_pctcvr, user_id_list)
    gauc_ctr = cal_group_auc(y_label_ctr, y_pctr, user_id_list)
    # gauc_cvr=cal_group_auc(y_label_ctcvr,y_pcvr,user_id_list)
    return auc_ctr, auc_ctrcvr, len(y_label_ctr), y_label_ctr.tolist().count(1), y_label_ctcvr.tolist().count(1), gauc_ctr, gauc_ctcvr


def make_train_feed_dict(model, batch):
    """make train feed dict for training"""

    feed_dict = {model.user_profile_0: batch[2],
                 model.user_profile_1: batch[3],
                 model.user_profile_2: batch[4],
                 model.user_RecAnchor: batch[5],
                 model.user_statIds: batch[6],
                 model.user_profile_3: batch[7],
                 
                 model.Wanchorids_long: batch[8],
                 model.Wanchorids_long_len_pre: batch[9],
                 
                 model.Wanchorids_short: batch[10],
                 model.Wanchorids_short_len_pre: batch[11],
                 
                 model.Wanchorids_noclick: batch[12],
                 model.Wanchorids_noclick_len_pre: batch[13],
                
                 model.Wanchorids_effect: batch[14],
                 model.Wanchorids_effect_len_pre: batch[15],
                 
                 model.anchor_profile_0: batch[16],
                 model.anchor_profile_1: batch[17],
                 model.anchor_profile_2: batch[18],
                 model.anchor_live_profile_0: batch[19],
                 model.anchor_live_profile_1: batch[20],
                 model.anchor_stats: batch[21],
                 model.anchor_stat_values: batch[22],
                 model.user_songTagids: batch[23],
                 model.user_tagWeights: batch[24],
                 model.anchor_songTagids: batch[25],
                 model.anchor_tagWeights: batch[26],

                 model.day_ctr_seq: batch[27],
                 model.day_cvr_seq: batch[28],
                 model.day_ctrid_seq: batch[29],
                 model.day_cvrid_seq: batch[30],
                 model.ctcvr_seq_len: batch[31],

                 model.anchorId: batch[32],
                 model.fea_sim: batch[33],
                 model.realtime_values: batch[34],
                 model.realtime_ids: batch[35],
                 model.hourId: batch[36],
                 model.dayOfWeek: batch[37],
                 model.redict_weights_pre: batch[38],
                 model.live_position: batch[39],
                 model.anchor_live_profile_2: batch[40],
                 ## 实时特征
                 model.label_ctr: batch[0],
                 model.label_ctcvr: batch[1],

                 model.keep_prob: 0.5,
                 model.train_phase: True
                 }
    return feed_dict


def run_train_step(model, sess, batch):
    feed_dict = make_train_feed_dict(model, batch)
    to_return = {
        'train_op1': model.train_op1,
        'loss': model.loss,
        'global_step1': model.global_step1,
        'train_op2': model.train_op2,
        'recloss': model.rec_loss,
        'global_step2': model.global_step2,
        'vis_rec_loss': model.vis_rec_loss,
        'sem_rec_loss': model.sem_rec_loss,
        'align_loss' : model.align_loss,
        'newUser_index' :model.newUser_index,
        'log': model.log,
        'log_label': model.log_label,
        'pre_ctr': model.pctr,
        'pre_ctcvr':model.pctcvr
    }
    return sess.run(to_return, feed_dict)


def run_train_step_model(model, sess, batch):
    feed_dict = make_train_feed_dict(model, batch)
    to_return = {
        'train_op1': model.train_op1,
        'loss': model.loss,
        'global_step1': model.global_step1
    }
    return sess.run(to_return, feed_dict)

def run_train_step_ZSL(model, sess, batch):
    feed_dict = make_train_feed_dict(model, batch)
    to_return = {
        'train_op2': model.train_op2,
        'recloss': model.rec_loss,
        'global_step2': model.global_step2,
        'vis_rec_loss': model.vis_rec_loss,
        'sem_rec_loss': model.sem_rec_loss,
        'align_loss' : model.align_loss
    }
    return sess.run(to_return, feed_dict)


def get_attn_weight(model, sess, batch):
    feed_dict = make_train_feed_dict(model, batch)
    return sess.run(model.alpha, feed_dict)

def dice(_x, axis=-1, epsilon=0.000000001, name=''):
    with tf.variable_scope(name_or_scope='', reuse=tf.AUTO_REUSE):
        alphas = tf.get_variable('alpha' + name, _x.get_shape()[-1],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
        beta = tf.get_variable('beta' + name, _x.get_shape()[-1],
                               initializer=tf.constant_initializer(0.0),
                               dtype=tf.float32)
    input_shape = list(_x.get_shape())

    reduction_axes = list(range(len(input_shape)))
    del reduction_axes[axis]
    broadcast_shape = [1] * len(input_shape)
    broadcast_shape[axis] = input_shape[axis]

    # case: train mode (uses stats of the current batch)
    mean = tf.reduce_mean(_x, axis=reduction_axes)
    brodcast_mean = tf.reshape(mean, broadcast_shape)
    std = tf.reduce_mean(tf.square(_x - brodcast_mean) + epsilon, axis=reduction_axes)
    std = tf.sqrt(std)
    brodcast_std = tf.reshape(std, broadcast_shape)
    x_normed = tf.layers.batch_normalization(_x, center=False, scale=False, name=name, reuse=tf.AUTO_REUSE)
    # x_normed = (_x - brodcast_mean) / (brodcast_std + epsilon)
    x_p = tf.sigmoid(beta * x_normed)

    return alphas * (1.0 - x_p) * _x + x_p * _x


def parametric_relu(_x):
    with tf.variable_scope(name_or_scope='', reuse=tf.AUTO_REUSE):
        alphas = tf.get_variable('alpha', _x.get_shape()[-1],
                                 initializer=tf.constant_initializer(0.0),
                                 dtype=tf.float32)
    pos = tf.nn.relu(_x)
    neg = alphas * (_x - abs(_x)) * 0.5

    return pos + neg


def batch_norm_layer( x, train_phase, scope_bn, batch_norm_decay):
    ##  updates_collections=None  强制更新
    bn_train = batch_norm(x, decay=batch_norm_decay, center=True, scale=True, updates_collections=None,
                          is_training=True, reuse=None, trainable=True, scope=scope_bn)
    bn_inference = batch_norm(x, decay=batch_norm_decay, center=True, scale=True, updates_collections=None,
                              is_training=False, reuse=True, trainable=True, scope=scope_bn)
    z = tf.cond(train_phase, lambda: bn_train, lambda: bn_inference)
    return z

def prelu(_x, scope=''):
    """parametric ReLU activation"""
    with tf.variable_scope(name_or_scope=scope, default_name="prelu"):
        _alpha = tf.get_variable("prelu_"+scope, shape=_x.get_shape()[-1], dtype=_x.dtype, initializer=tf.constant_initializer(0.1))
        return tf.maximum(0.0, _x) + _alpha * tf.minimum(0.0, _x)







def _bulid_dnn_withDice(X_input, deep_layers,l2_reg_lambda, batch_norm, train_phase,keep_prob,batch_norm_decay, scope):
    dnn_out=X_input
    with tf.variable_scope(scope+"_dnn" ):
        for i in range(len(deep_layers)):
            dnn_out = tf.contrib.layers.fully_connected(inputs=dnn_out, num_outputs=deep_layers[i],activation_fn=None,  \
                                                        weights_regularizer=tf.contrib.layers.l2_regularizer(
                                                            l2_reg_lambda), scope='mlp%d' % i)

            dnn_out = dice(dnn_out, name=scope+'_dice_%d' %i)

    return dnn_out

def _bulid_dnn_withPrelu(X_input, deep_layers,l2_reg_lambda, batch_norm, train_phase,keep_prob,batch_norm_decay, scope):
    dnn_out=X_input
    with tf.variable_scope(scope+"_dnn" ):
        for i in range(len(deep_layers)):
            dnn_out = tf.contrib.layers.fully_connected(inputs=dnn_out, num_outputs=deep_layers[i],activation_fn=None, \
                                                        weights_regularizer=tf.contrib.layers.l2_regularizer(
                                                            l2_reg_lambda), scope='mlp%d' % i)
            dnn_out= prelu(dnn_out, "dnn_prelu")

            if batch_norm:
                dnn_out = batch_norm_layer(dnn_out, train_phase=train_phase,
                                           scope_bn='bn_%d' % i,
                                           batch_norm_decay=batch_norm_decay)  # 放在RELU之后 https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md#bn----before-or-after-relu

            dnn_out = tf.nn.dropout(dnn_out,keep_prob)  # Apply Dropout after all BN layers and set dropout=0.8(drop_ratio=0.2)



def DMR_DinAttention(item_eb, item_his_eb, keys_length, context_his_eb, scope_bn ,mode="SUM"):
  '''
    queries:     [B, H]
    keys:        [B, T, H]
    keys_length: [B]
  '''
  with tf.variable_scope(scope_bn):
      q_hidden_units = item_eb.get_shape().as_list()[-1]
      item_eb_tile = tf.tile(item_eb, [1, tf.shape(item_his_eb)[1]])
      item_eb_tile = tf.reshape(item_eb_tile, [-1, tf.shape(item_his_eb)[1], q_hidden_units])
      if context_his_eb is None:
          query = item_eb_tile
      else:
          query = tf.concat([item_eb_tile, context_his_eb], axis=-1)

      query = tf.layers.dense(query, item_his_eb.get_shape().as_list()[-1], activation=None, name='dmr_align')
      query = prelu(query, scope='dmr_prelu')

      din_all = tf.concat([query, item_his_eb, query - item_his_eb, query * item_his_eb], axis=-1)
      d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att')
      d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att')
      d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att')
      d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(item_his_eb)[1]])
      scores = d_layer_3_all
      # Mask
      key_masks = tf.sequence_mask(keys_length, tf.shape(item_his_eb)[1])  # [B, T]
      key_masks = tf.expand_dims(key_masks, 1)  # [B, 1, T]
      paddings = tf.ones_like(scores) * (-2 ** 32 + 1)
      paddings_no_softmax = tf.zeros_like(scores)
      scores = tf.where(key_masks, scores, paddings)  # [B, 1, T]
      scores_no_softmax = tf.where(key_masks, scores, paddings_no_softmax)

      # Scale
      # outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)

      # Activation
      scores = tf.nn.softmax(scores)  # [B, 1, T]

      if mode == 'SUM':
          output = tf.matmul(scores, item_his_eb)  # [B, 1, E]
          output = tf.reduce_sum(output, axis=1)  # B,E
      else:
          scores = tf.reshape(scores, [-1, tf.shape(item_his_eb)[1]])
          output = item_his_eb * tf.expand_dims(scores, -1)
          output = tf.reshape(output, tf.shape(item_his_eb))

      return output, scores, scores_no_softmax

def deep_match(item_his_eb, context_his_eb, keys_length, EMBEDDING_DIM, scope_bn):
    with tf.variable_scope(scope_bn):
        query = context_his_eb
        query = tf.layers.dense(query, item_his_eb.get_shape().as_list()[-1], activation=None, name='dm_align')
        query = prelu(query, scope='dm_prelu')
        inputs = tf.concat([query, item_his_eb, query - item_his_eb, query * item_his_eb], axis=-1)  # B,T,E
        att_layer1 = tf.layers.dense(inputs, 80, activation=tf.nn.sigmoid, name='dm_att_1')
        att_layer2 = tf.layers.dense(att_layer1, 40, activation=tf.nn.sigmoid, name='dm_att_2')
        att_layer3 = tf.layers.dense(att_layer2, 1, activation=None, name='dm_att_3')  # B,T,1
        scores = tf.transpose(att_layer3, [0, 2, 1])  # B,1,T

        # Mask
        key_masks = tf.sequence_mask(keys_length, tf.shape(item_his_eb)[1])  # [B, T]
        key_masks = tf.expand_dims(key_masks, 1)  # [B, 1, T]
        paddings = tf.ones_like(scores) * (-2 ** 32 + 1)
        scores = tf.where(key_masks, scores, paddings)  # [B, 1, T]

        scores = tf.nn.softmax(scores)  # B, 1, T
        att_dm_item_his_eb = tf.matmul(scores, item_his_eb)  # B, 1, E

        dnn_layer1 = tf.layers.dense(att_dm_item_his_eb, EMBEDDING_DIM, activation=None, name='dm_fcn_1')
        dnn_layer1 = prelu(dnn_layer1, 'dm_fcn_1')  # B, 1, E

        # target mask
        user_vector = tf.reduce_sum(dnn_layer1, axis=1)  # B, E


    return user_vector, scores

def deep_match_2(item_his_eb, context_his_eb, keys_length):
    query = context_his_eb
    query = tf.layers.dense(query, item_his_eb.get_shape().as_list()[-1], activation=None, name='adm_align')
    query = prelu(query, scope='adm_prelu')
    inputs = tf.concat([query, item_his_eb, query-item_his_eb, query*item_his_eb], axis=-1) # B,T,E
    att_layer1 = tf.layers.dense(inputs, 80, activation=tf.nn.sigmoid, name='adm_att_1')
    att_layer2 = tf.layers.dense(att_layer1, 40, activation=tf.nn.sigmoid, name='adm_att_2')
    att_layer3 = tf.layers.dense(att_layer2, 1, activation=None, name='adm_att_3')  # B,T,1
    scores = tf.transpose(att_layer3, [0, 2, 1]) # B,1,T

    # Mask
    key_masks = tf.sequence_mask(keys_length, tf.shape(item_his_eb)[1])  # [B, T]
    key_masks = tf.expand_dims(key_masks, 1)  # [B, 1, T]
    paddings = tf.ones_like(scores) * (-2 ** 32 + 1)
    scores = tf.where(key_masks, scores, paddings)  # [B, 1, T]

    scores = tf.nn.softmax(scores)  # [B, 1, T]
    outputs = tf.matmul(scores, item_his_eb)  # [B, 1, H]
    outputs = tf.reduce_sum(outputs, axis=1)  # B,H

    return outputs

def raw_deep_match(item_his_eb, context_his_eb, keys_length, mid_his_batch, EMBEDDING_DIM, item_vectors, item_biases, n_mid, scope_bn):
    with tf.variable_scope(scope_bn):
        ## 做reverse 是为了 方便获取T-1
        item_his_eb = tf.reverse(item_his_eb, [1])
        context_his_eb = tf.reverse(context_his_eb, [1])
        mid_his_batch = tf.reverse(mid_his_batch, [1])

        query = context_his_eb
        query = tf.layers.dense(query, item_his_eb.get_shape().as_list()[-1], activation=None, name='dm_align')
        query = prelu(query, scope='dm_prelu')
        inputs = tf.concat([query, item_his_eb, query - item_his_eb, query * item_his_eb], axis=-1)  # B,T,E
        att_layer1 = tf.layers.dense(inputs, 80, activation=tf.nn.sigmoid, name='dm_att_1')
        att_layer2 = tf.layers.dense(att_layer1, 40, activation=tf.nn.sigmoid, name='dm_att_2')
        att_layer3 = tf.layers.dense(att_layer2, 1, activation=None, name='dm_att_3')  # B,T,1
        scores = tf.transpose(att_layer3, [0, 2, 1])  # B,1,T

        # mask
        bool_mask = tf.sequence_mask(keys_length, tf.shape(item_his_eb)[1])  # [B, T]
        bool_mask = tf.reverse(bool_mask, [1])  ## 因为最近发生行为的主播在 最后一个
        key_masks = tf.expand_dims(bool_mask, 1)  # B,1,T
        paddings = tf.ones_like(scores) * (-2 ** 32 + 1)
        scores = tf.where(key_masks, scores, paddings)  # B,1,T

        # tril
        scores_tile = tf.tile(tf.reduce_sum(scores, axis=1), [1, tf.shape(scores)[-1]])  # B, T*T
        scores_tile = tf.reshape(scores_tile, [-1, tf.shape(scores)[-1], tf.shape(scores)[-1]])  # B, T, T
        diag_vals = tf.ones_like(scores_tile)  # B, T, T
        # tril = tf.contrib.linalg.LinearOperatorTriL(diag_vals).to_dense()
        tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()  ## 将对角矩阵上方置为0
        paddings = tf.ones_like(tril) * (-2 ** 32 + 1)
        scores_tile = tf.where(tf.equal(tril, 0), paddings, scores_tile)  # B, T, T
        scores_tile = tf.nn.softmax(scores_tile)  # B, T, T
        att_dm_item_his_eb = tf.matmul(scores_tile, item_his_eb)  # B, T, E

        dnn_layer1 = tf.layers.dense(att_dm_item_his_eb, EMBEDDING_DIM, activation=None, name='dm_fcn_1')
        dnn_layer1 = prelu(dnn_layer1, 'dm_fcn_1')  # B, T, E

        # target mask
        user_vector = dnn_layer1[:, -1, :]  ## 学习了整个T 的user embeding

        match_mask_raw = tf.expand_dims(keys_length, -1)  # [B，1]
        match_mask_bool = tf.greater(match_mask_raw, 2)  ## 获得true false 的矩阵；要求用户序列至少为3个,才去计算辅助loss 要不然没有意义
        match_mask = tf.where(match_mask_bool, tf.ones_like(match_mask_raw, dtype=tf.float32),
                              tf.zeros(shape=tf.shape(match_mask_raw), dtype=tf.float32))  # [B,1]

        ##错误隐患，如果节点应用了下面 user_vector2, 就会报错，应该可能部分用户序列长度为1 导致index -2 是错误的；目前线上不会报错，因为graph 中不利用loss 所以不会保存错误节点。
        user_vector2 = dnn_layer1[:, -2, :] * match_mask  ## Ut-1 在T-1 的user embedding
        ## tf.reshape(match_mask, [-1, tf.shape(match_mask)[1], 1])[:, -2, :]  这部分我认为是阿里同学的trick控制部分用户序列少的不进入计算，其他序列长的 match_mask 和 mask 是一致的
        num_sampled = 2000
        loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(weights=item_vectors,
                                                         biases=item_biases,
                                                         labels=tf.cast(tf.reshape(mid_his_batch[:, -1], [-1, 1]),tf.int64),  ## 最近一次行为
                                                         inputs=user_vector2,
                                                         num_sampled=num_sampled,
                                                         num_classes=n_mid,
                                                         sampled_values=tf.nn.learned_unigram_candidate_sampler(
                                                             tf.cast(tf.reshape(mid_his_batch[:, 0], [-1, 1]),
                                                                     tf.int64), 1, num_sampled, True, n_mid)
                                                         ))
    return loss, user_vector, scores



def _build_extreme_FM(nn_input, filed_size,embedding_size, cross_layer_sizes, res=False, direct=False, bias=False, reduce_D=False, f_dim=2):
    hidden_nn_layers = []
    field_nums = []
    final_len = 0
    field_num = filed_size
    nn_input = tf.reshape(nn_input, shape=[-1, int(field_num), embedding_size])
    field_nums.append(int(field_num))
    hidden_nn_layers.append(nn_input)
    final_result = []
    split_tensor0 = tf.split(hidden_nn_layers[0], embedding_size * [1], 2)  # D*B*F*1
    with tf.variable_scope("exfm_part", initializer=tf.truncated_normal_initializer(stddev=0.01)) as scope:
        for idx, layer_size in enumerate(cross_layer_sizes):
            split_tensor = tf.split(hidden_nn_layers[-1], embedding_size * [1], 2)
            dot_result_m = tf.matmul(split_tensor0, split_tensor,
                                     transpose_b=True)  # 如果idx=0  dot_result_m shape : D*B*F*F
            dot_result_o = tf.reshape(dot_result_m, shape=[embedding_size, -1,
                                                           field_nums[0] * field_nums[-1]])  ##  shape : D * B * (F*F)
            dot_result = tf.transpose(dot_result_o, perm=[1, 0, 2])  ##  shape : B*D* (F*F)

            if reduce_D:
                filters0 = tf.get_variable("f0_" + str(idx),
                                           shape=[1, layer_size, field_nums[0], f_dim],
                                           dtype=tf.float32)
                filters_ = tf.get_variable("f__" + str(idx),
                                           shape=[1, layer_size, f_dim, field_nums[-1]],
                                           dtype=tf.float32)
                filters_m = tf.matmul(filters0, filters_)
                filters_o = tf.reshape(filters_m, shape=[1, layer_size, field_nums[0] * field_nums[-1]])
                filters = tf.transpose(filters_o, perm=[0, 2, 1])
            else:
                filters = tf.get_variable(name="f_" + str(idx),
                                          shape=[1, field_nums[-1] * field_nums[0], layer_size],
                                          dtype=tf.float32)
            # dot_result = tf.transpose(dot_result, perm=[0, 2, 1])
            curr_out = tf.nn.conv1d(dot_result, filters=filters, stride=1, padding='VALID')  # B*D* (Layer_size)

            # BIAS ADD
            if bias:
                # hparams.logger.info("bias")
                b = tf.get_variable(name="f_b" + str(idx),
                                    shape=[layer_size],
                                    dtype=tf.float32,
                                    initializer=tf.zeros_initializer())
                curr_out = tf.nn.bias_add(curr_out, b)


            curr_out = tf.identity(curr_out)

            curr_out = tf.transpose(curr_out, perm=[0, 2, 1])  # B*(Layer_size) *D

            if direct:
                direct_connect = curr_out
                next_hidden = curr_out
                final_len += layer_size
                field_nums.append(int(layer_size))

            else:
                if idx != len(cross_layer_sizes) - 1:
                    next_hidden, direct_connect = tf.split(curr_out, 2 * [int(layer_size / 2)], 1)
                    final_len += int(layer_size / 2)
                else:
                    direct_connect = curr_out
                    next_hidden = 0
                    final_len += layer_size
                field_nums.append(int(layer_size / 2))

            final_result.append(direct_connect)  ##  (多少个cross layers ) *  B * (Layer_size)  * D
            hidden_nn_layers.append(next_hidden)

        result = tf.concat(final_result, axis=1)  ## B * (num_cross * Layer_size) *D
        result = tf.reduce_sum(result, -1)  ## B * (num_cross * Layer_size)
        if res:
            w_nn_output1 = tf.get_variable(name='w_nn_output1',
                                           shape=[final_len, 128],
                                           dtype=tf.float32)
            b_nn_output1 = tf.get_variable(name='b_nn_output1',
                                           shape=[128],
                                           dtype=tf.float32,
                                           initializer=tf.zeros_initializer())

            exFM_out0 = tf.nn.xw_plus_b(result, w_nn_output1, b_nn_output1)
            exFM_out1 = tf._active_layer(logit=exFM_out0,
                                           activation="relu",
                                           )
            w_nn_output2 = tf.get_variable(name='w_nn_output2',
                                           shape=[128 + final_len, 1],
                                           dtype=tf.float32)
            b_nn_output2 = tf.get_variable(name='b_nn_output2',
                                           shape=[1],
                                           dtype=tf.float32,
                                           initializer=tf.zeros_initializer())

            exFM_in = tf.concat([exFM_out1, result], axis=1, name="user_emb")
            exFM_out = tf.nn.xw_plus_b(exFM_in, w_nn_output2, b_nn_output2)

        else:
            w_nn_output = tf.get_variable(name='w_nn_output',
                                          shape=[final_len, 1],
                                          dtype=tf.float32)
            b_nn_output = tf.get_variable(name='b_nn_output',
                                          shape=[1],
                                          dtype=tf.float32,
                                          initializer=tf.zeros_initializer())

            exFM_out = tf.nn.xw_plus_b(result, w_nn_output, b_nn_output)
    return exFM_out

def din_attention(queries, keys, keys_length):
  '''
    queries:     [B, H]
    keys:        [B, T, H]
    keys_length: [B]
  '''
  queries_hidden_units = queries.get_shape().as_list()[-1]
  queries = tf.tile(queries, [1, tf.shape(keys)[1]])
  queries = tf.reshape(queries, [-1, tf.shape(keys)[1], queries_hidden_units])
  din_all = tf.concat([queries, keys, queries-keys, queries*keys], axis=-1)
  d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att', reuse=tf.AUTO_REUSE)
  d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att', reuse=tf.AUTO_REUSE)
  d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att', reuse=tf.AUTO_REUSE)
  d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(keys)[1]])
  outputs = d_layer_3_all
  # Mask
  key_masks = tf.sequence_mask(keys_length, tf.shape(keys)[1])   # [B, T]
  key_masks = tf.expand_dims(key_masks, 1) # [B, 1, T]
  paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
  outputs = tf.where(key_masks, outputs, paddings)  # [B, 1, T]

  # Scale
  outputs = outputs / (keys.get_shape().as_list()[-1] ** 0.5)

  # Activation
  outputs = tf.nn.softmax(outputs)  # [B, 1, T]

  # Weighted sum
  outputs = tf.matmul(outputs, keys)  # [B, 1, H]

  return outputs


def attention(inputs, attention_size, time_major=False, return_alphas=False):
    """
    Attention mechanism layer which reduces RNN/Bi-RNN outputs with Attention vector.
    The idea was proposed in the article by Z. Yang et al., "Hierarchical Attention Networks
     for Document Classification", 2016: http://www.aclweb.org/anthology/N16-1174.
    Variables notation is also inherited from the article

    Args:
        inputs: The Attention inputs.
            Matches outputs of RNN/Bi-RNN layer (not final state):
                In case of RNN, this must be RNN outputs `Tensor`:
                    If time_major == False (default), this must be a tensor of shape:
                        `[batch_size, max_time, cell.output_size]`.
                    If time_major == True, this must be a tensor of shape:
                        `[max_time, batch_size, cell.output_size]`.
                In case of Bidirectional RNN, this must be a tuple (outputs_fw, outputs_bw) containing the forward and
                the backward RNN outputs `Tensor`.
                    If time_major == False (default),
                        outputs_fw is a `Tensor` shaped:
                        `[batch_size, max_time, cell_fw.output_size]`
                        and outputs_bw is a `Tensor` shaped:
                        `[batch_size, max_time, cell_bw.output_size]`.
                    If time_major == True,
                        outputs_fw is a `Tensor` shaped:
                        `[max_time, batch_size, cell_fw.output_size]`
                        and outputs_bw is a `Tensor` shaped:
                        `[max_time, batch_size, cell_bw.output_size]`.
        attention_size: Linear size of the Attention weights.
        time_major: The shape format of the `inputs` Tensors.
            If true, these `Tensors` must be shaped `[max_time, batch_size, depth]`.
            If false, these `Tensors` must be shaped `[batch_size, max_time, depth]`.
            Using `time_major = True` is a bit more efficient because it avoids
            transposes at the beginning and end of the RNN calculation.  However,
            most TensorFlow data is batch-major, so by default this function
            accepts input and emits output in batch-major form.
        return_alphas: Whether to return attention coefficients variable along with layer's output.
            Used for visualization purpose.
    Returns:
        The Attention output `Tensor`.
        In case of RNN, this will be a `Tensor` shaped:
            `[batch_size, cell.output_size]`.
        In case of Bidirectional RNN, this will be a `Tensor` shaped:
            `[batch_size, cell_fw.output_size + cell_bw.output_size]`.
    """

    if isinstance(inputs, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        inputs = tf.concat(inputs, 2)

    if time_major:
        # (T,B,D) => (B,T,D)
        inputs = tf.array_ops.transpose(inputs, [1, 0, 2])

    hidden_size = inputs.shape[2].value  # D value - hidden size of the RNN layer

    # Trainable parameters
    w_omega = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    b_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    u_omega = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    with tf.name_scope('v'):
        # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
        #  the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
        v = tf.tanh(tf.tensordot(inputs, w_omega, axes=1) + b_omega)

    # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
    vu = tf.tensordot(v, u_omega, axes=1, name='vu')  # (B,T) shape
    alphas = tf.nn.softmax(vu, name='alphas')  # (B,T) shape

    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
    output = tf.reduce_sum(inputs * tf.expand_dims(alphas, -1), 1)

    if not return_alphas:
        return output
    else:
        return output, alphas
