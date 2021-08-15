import os
import sys
import tensorflow as tf
from model import *
from util import *
from util2 import *
import warnings
warnings.filterwarnings("ignore")
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

def train(model, config, train_file, test_file, test_file2, train_date):
    print("Configuring TensorBoard and Saver...")
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    sess = tf.Session(config=tf_config)
    sess.run(tf.global_variables_initializer())

    print('Training and evaluating...')
    model_dir = "model_save/"
    save_path = os.path.join(model_dir, 'flj_' + train_date)
    saver = tf.train.Saver(max_to_keep=3)

    total_batch = 0  # 总批次
    print_per_batch = 4000  # 每多个batch 输出eval

    for eppch_tmp in range(config["train_epoch"]):
        print("-----------------  Epoch {0} start ! --------------------".format(eppch_tmp))

        batch_size = config["batch_size"]
        data_train = get_dataset(train_file)
        data_train = data_train.shuffle(20000).prefetch(batch_size * 10).batch(batch_size)

        iterator_train = data_train.make_one_shot_iterator()
        next_element_train = iterator_train.get_next()

        while True:
            label_ctr, label_cvr, user_profile_basefea, user_appcate1, user_appcate2, user_RecAnchor, user_statIds, user_topHourIds, \
            Wanchorids_long, Wanchorids_long_len, Wanchorids_short, Wanchorids_short_len, Wanchorids_noclick, Wanchorids_noclick_len, Wanchorids_effect, Wanchorids_effect_len, \
            anchor_profile_basefea, anchor_appcate1, anchor_appcate2, anchor_live_basefea, anchor_tagids, anchor_stats, anchor_stat_values, \
            user_songTagids, user_tagWeights, anchor_songTagids, anchor_tagWeights, \
            day_ctr_seq, day_cvr_seq, day_ctrid_seq, day_cvrid_seq, ctcvr_seq_len, anchorId, fea_sim, realtime_values, realtime_ids, \
            hourId, dayOfWeek, redict_weights, live_position, anchor_tagidonehot, user_id = sess.run(next_element_train)


            if (label_ctr.shape[0] < batch_size):
                break
            return_dict = run_train_step(model, sess, (label_ctr, label_cvr, user_profile_basefea, user_appcate1, user_appcate2, user_RecAnchor, user_statIds, user_topHourIds, \
            Wanchorids_long, Wanchorids_long_len, Wanchorids_short, Wanchorids_short_len, Wanchorids_noclick, Wanchorids_noclick_len, Wanchorids_effect, Wanchorids_effect_len, \
            anchor_profile_basefea, anchor_appcate1, anchor_appcate2, anchor_live_basefea, anchor_tagids, anchor_stats, anchor_stat_values, \
            user_songTagids, user_tagWeights, anchor_songTagids, anchor_tagWeights, \
            day_ctr_seq, day_cvr_seq, day_ctrid_seq, day_cvrid_seq, ctcvr_seq_len, anchorId, fea_sim, realtime_values, realtime_ids, \
            hourId, dayOfWeek, redict_weights, live_position, anchor_tagidonehot,user_id))

            total_batch += 1

            print("now total_batch is {0}, train loss is {1}, zsl_loss is {2}, vis_rec_loss is {3}, sem_rec_loss is {4}, align_loss is {5}"
                  .format(total_batch,return_dict['loss'], return_dict['recloss'], return_dict['vis_rec_loss'], return_dict['sem_rec_loss'],return_dict['align_loss']))

            if ((total_batch+1) % print_per_batch) == 0:
                saver.save(sess=sess, save_path=save_path)
                print("the mode is saved is {0}, train batch is {1}".format(save_path, total_batch))

    saver.save(sess=sess, save_path=save_path)
    print("the mode is saved is {0}, train batch is {1}".format(save_path, total_batch))

if __name__ == '__main__':
    print('Configuring DSTN SnpShot model...')

    config = {
        "base_fea_size": 2000,
        "anchorId_size": 600000,
        "anchor_livefea_size": 500,
        "user_stat_fea_size": 120,
        "realtimeFea_size": 2760,
        "statsFea_size": 210,
        "Tophour_fea_size": 80,
        "hour_size": 12,
        "day_fea_size": 30,
        "songtag_size": 90,
        "LivePosition_size":15,

        "profile_size": 8,
        "user_statIds_size": 12,
        "realtime_values_size": 22,
        "anchor_stats_size": 15,
        "live_size": 4,
        "day_ctr_size": 7,
        "day_cvr_size": 7,
        "user_topHourIds_size": 6,
        "tagidOnehot_size":38,

        "fea_sim_size": 2,
        "ctr_task_wgt": 0.7,
        "embedding_size": 32,
        "learning_rate": 0.001,
        "l2_reg_lambda": 0.0001,
        "batch_size": 1024,
        "n_class": 2,

        "batch_norm": 1,
        "batch_norm_decay": 0.995,
        ###  category
        "deep_layers": [512, 256, 128],
        "deep_layers_2": [512, 128],
        "cross_layer_sizes": [100, 100, 50],
        "freve_layers": "null",
        # random setting, may need fine-tune
        "train_epoch": 2
    }


    train_date = sys.argv[1]

    train_file= ""
    test_file = ""
    test_file2 = ""

    config["is_aux"] = True
    print("now train date: ", train_date)
    classifier = Model_DSTN_SNPSHOT(config)
    classifier.bulid_graph()
    train(classifier, config, train_file, test_file, test_file2, train_date)

