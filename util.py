import tensorflow as tf
import numpy as np
import time
from datetime import timedelta
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm
from keras import regularizers
from util2 import *



class DSTNEmbeddingLayerSnpShot(object):
    def __init__(self, config):
        self.read_parameters(config)
        self.create_placeholder()

    def build(self):
        #特征潜入
        self.make_embedding()
        self.make_position_embedding()

        #零样本重构新用户特征
        rec_loss = self.User_ZSL()

        #  FM Input、DNN Input、DMR Input
        FM_Input, FM_size = self.make_FM_Input()
        DNN_Input, DNN_size = self.make_DNN_Input(FM_Input, FM_size)
        self.make_DMR_Input()
        return FM_Input , FM_size, DNN_Input, DNN_size, rec_loss

    def User_ZSL(self):
        with tf.variable_scope("user_ZSL"):
            semantic_input  = self.make_semantic_feature()
            visual_input = self.make_visual_feature()

            semantic_hidden = self.make_hidden(semantic_input, "semantic")
            visual_hidden = self.make_hidden(visual_input, "visual")
            self.align_loss = self.compute_align_loss(semantic_hidden, visual_hidden)

            self.generate_features(semantic_hidden, visual_hidden)

            self.vis_rec_loss = self.compute_vis_rec_loss()
            self.sem_rec_loss = self.compute_sem_rec_loss()
            self.update_features()

        return self.vis_rec_loss + self.align_loss + self.sem_rec_loss

    def make_semantic_feature(self):
        row = tf.shape(self.anchor_appcate1_embeddings)[0]
        user_profile_basefea_att = attention(self.user_profile_basefea_embeddings, 128, return_alphas=False)
        user_appcate1_att = tf.reshape(self.user_appcate1_embeddings, [row, -1])
        user_appcate2_att = tf.reshape(self.user_appcate2_embeddings, [row, -1])
        anchor_profile_basefea_att = attention(self.anchor_profile_basefea_embeddings, 128, return_alphas=False)
        anchor_appcate1_att = tf.reshape(self.anchor_appcate1_embeddings, [row, -1])
        anchor_appcate2_att = tf.reshape(self.anchor_appcate2_embeddings, [row, -1])
        anchor_stat_att = attention(self.anchor_stat_embeddings, 128, return_alphas=False)
        day_ctrid_att = attention(self.day_ctrid_embeddings, 128, return_alphas=False)
        day_cvrid_att = attention(self.day_cvrid_embeddings, 128, return_alphas=False)
        userSongTag_att = tf.reshape(self.userSongTag_embeddings, [row, -1])
        anchorSongTag_att = tf.reshape(self.anchorSongTag_embeddings, [row, -1])
        anchor_live_basefea_att = attention(self.anchor_livebasefea_embeddings, 128, return_alphas=False)
        realtime_ids_att = attention(self.realtime_ids_embeddings, 128, return_alphas=False)
        hour_att = tf.reshape(self.hour_embeddings, [row, -1])
        anchor_tagids_att = tf.reshape(self.anchor_tagids_embeddings, [row, -1])
        livePosition_att = tf.reshape(self.livePosition_embeddings, [row, -1])
        Day_att = tf.reshape(self.Day_embeddings, [row, -1])

        semantic_att_embedding = tf.concat([user_profile_basefea_att, user_appcate1_att, user_appcate2_att, anchor_profile_basefea_att,
                                                 anchor_appcate1_att, anchor_appcate2_att, anchor_stat_att, day_ctrid_att, day_cvrid_att, userSongTag_att, anchorSongTag_att,
                                                 anchor_live_basefea_att, realtime_ids_att, hour_att, anchor_tagids_att, livePosition_att, Day_att], axis=1)

        day_ctr_value = tf.reshape(self.day_ctr_value, shape=[row, self.day_seq_len])
        day_cvr_value = tf.reshape(self.day_cvr_value, shape=[row, self.day_seq_len])
        semantic_values = tf.concat([self.anchor_stat_values, day_ctr_value, day_cvr_value, self.fea_sim, self.realtime_values], axis=-1)
        semantic_input = tf.concat([semantic_att_embedding, semantic_values], axis=1)
        semantic_input = tf.reshape(semantic_input, [row, 597])
        return semantic_input

    def make_visual_feature(self):
        row = tf.shape(self.userRecanchor_embeddings_pre)[0]
        userRecanchor_embeddings_att = tf.reshape(self.userRecanchor_embeddings_pre, [row, -1])
        user_stat_att = attention(self.user_stat_embedding_pre, 128, return_alphas=False)
        user_topHour_att = attention(self.user_topHour_embedding_pre, 128, return_alphas=False)
        anchor_his_fea_long_att = attention(self.Wanchorids_long_embeddings_pre, 128, return_alphas=False)
        anchor_his_fea_short_att = attention(self.Wanchorids_short_embeddings_pre, 128, return_alphas=False)
        anchor_his_fea_noclick_att = attention(self.Wanchorids_noclick_embeddings_pre, 128, return_alphas=False)
        anchor_his_fea_effect_att = attention(self.Wanchorids_effect_embeddings_pre, 128, return_alphas=False)

        visual_att_embedding = tf.concat(
            [userRecanchor_embeddings_att, user_stat_att, user_topHour_att, anchor_his_fea_long_att,
             anchor_his_fea_short_att, anchor_his_fea_noclick_att, anchor_his_fea_effect_att], axis = 1)
        long_len = tf.cast(tf.reshape(self.Wanchorids_long_len_pre,[row,1]), dtype=tf.float32 )
        short_len = tf.cast(tf.reshape(self.Wanchorids_short_len_pre,[row,1]), dtype=tf.float32)
        noclick_len = tf.cast(tf.reshape(self.Wanchorids_noclick_len_pre,[row,1]), dtype=tf.float32)
        effect_len = tf.cast(tf.reshape(self.Wanchorids_effect_len_pre,[row,1]), dtype=tf.float32)
        visual_values = tf.concat([long_len, short_len, noclick_len, effect_len, self.redict_weights_pre], axis = 1)
        visual_input = tf.concat([visual_att_embedding, visual_values], axis=1)
        visual_input = tf.reshape(visual_input, [row, 236])
        return visual_input

    def make_hidden(self, input, name):
        input = tf.nn.dropout(input, keep_prob=self.keep_prob, name = name+str(0))
        hidden = tf.layers.dense(input, 512, activation=tf.nn.leaky_relu, name=name+str(1))
        hidden = tf.layers.dense(tf.concat([hidden, input], axis=1), 256, activation=tf.nn.leaky_relu, name=name+str(3))
        return hidden

    def compute_align_loss(self, semantic_hidden, visual_hidden):
        oldUser_index = tf.reshape(tf.equal(self.newUser_index, False),[-1])
        row = tf.shape(semantic_hidden)[0]
        semantic_hidden_flat = tf.reshape(semantic_hidden, [row,-1])
        visual_hidden_flat = tf.reshape(visual_hidden, [row, -1])
        align_loss = tf.reduce_mean(tf.square(semantic_hidden_flat[oldUser_index] - visual_hidden_flat[oldUser_index]))
        return align_loss

    def generate_features(self, semantic_hidden, visual_hidden):
        self.userRecanchor_embeddings_rec_bysem, self.user_stat_embedding_rec_bysem, self.user_topHour_embedding_rec_bysem, \
        self.Wanchorids_long_embeddings_rec_bysem, self.Wanchorids_long_len_rec_bysem, self.Wanchorids_short_embeddings_rec_bysem, \
        self.Wanchorids_short_len_rec_bysem, self.Wanchorids_noclick_embeddings_rec_bysem, self.Wanchorids_noclick_len_rec_bysem, \
        self.Wanchorids_effect_embeddings_rec_bysem, self.Wanchorids_effect_len_rec_bysem, self.redict_weights_rec_bysem = self.generate_visual_features(semantic_hidden)

        self.userRecanchor_embeddings_rec_byvis, self.user_stat_embedding_rec_byvis, self.user_topHour_embedding_rec_byvis, \
        self.Wanchorids_long_embeddings_rec_byvis, self.Wanchorids_long_len_rec_byvis, self.Wanchorids_short_embeddings_rec_byvis, \
        self.Wanchorids_short_len_rec_byvis, self.Wanchorids_noclick_embeddings_rec_byvis, self.Wanchorids_noclick_len_rec_byvis, \
        self.Wanchorids_effect_embeddings_rec_byvis, self.Wanchorids_effect_len_rec_byvis, self.redict_weights_rec_byvis = self.generate_visual_features(visual_hidden)

        self.user_profile_basefea_rec_bysem, self.user_appcate1_rec_bysem, self.user_appcate2_rec_bysem, self.anchor_profile_basefea_rec_bysem, \
        self.anchor_appcate1_rec_bysem, self.anchor_appcate2_rec_bysem, self.anchor_stat_rec_bysem, self.anchor_stat_values_rec_bysem, \
        self.day_ctr_value_rec_bysem, self.day_cvr_value_rec_bysem, self.day_ctrid_rec_bysem, self.day_cvrid_rec_bysem, self.userSongTag_rec_bysem, \
        self.fea_sim_rec_bysem, self.anchor_livebasefea_rec_bysem, self.realtime_values_rec_bysem, self.realtime_ids_rec_bysem, self.hourId_rec_bysem, \
        self.anchor_tagids_rec_bysem, self.Day_rec_bysem, self.live_position_rec_bysem = self.generate_semantic_features(semantic_hidden)

        self.user_profile_basefea_rec_byvis, self.user_appcate1_rec_byvis, self.user_appcate2_rec_byvis, self.anchor_profile_basefea_rec_byvis, \
        self.anchor_appcate1_rec_byvis, self.anchor_appcate2_rec_byvis, self.anchor_stat_rec_byvis, self.anchor_stat_values_rec_byvis, \
        self.day_ctr_value_rec_byvis, self.day_cvr_value_rec_byvis, self.day_ctrid_rec_byvis, self.day_cvrid_rec_byvis, self.userSongTag_rec_byvis, \
        self.fea_sim_rec_byvis, self.anchor_livebasefea_rec_byvis, self.realtime_values_rec_byvis, self.realtime_ids_rec_byvis, self.hourId_rec_byvis, \
        self.anchor_tagids_rec_byvis, self.Day_rec_byvis, self.live_position_rec_byvis = self.generate_semantic_features(visual_hidden)

    def generate_feature_type_embedding(self, hidden_features, num):
        hidden_features = tf.nn.dropout(hidden_features, keep_prob=self.keep_prob)
        temp = tf.layers.dense(hidden_features, 512, activation=tf.nn.leaky_relu)
        temp = tf.layers.dense(tf.concat([temp, hidden_features], axis = 1), num * self.embedding_size, activation=tf.nn.leaky_relu )
        res = tf.reshape(temp, [-1, num, self.embedding_size])
        return res

    def generate_feature_type_len(self, hidden_features):
        hidden_features = tf.nn.dropout(hidden_features, keep_prob=self.keep_prob)
        temp = tf.layers.dense(hidden_features, 512, activation=tf.nn.leaky_relu)
        res = tf.layers.dense(tf.concat([temp, hidden_features], axis=1), 1, activation=tf.nn.relu)
        return res

    def generate_feature_type_values(self, hidden_features, num):
        hidden_features = tf.nn.dropout(hidden_features, keep_prob=self.keep_prob)
        temp = tf.layers.dense(hidden_features, 512, activation=tf.nn.leaky_relu)
        res = tf.layers.dense(tf.concat([temp,hidden_features],axis=1), num, activation=tf.nn.leaky_relu)
        return res

    def generate_visual_features(self, hidden_features):
        userRecanchor_embeddings_rec = self.generate_feature_type_embedding(hidden_features, 1)
        user_stat_embedding_rec = self.generate_feature_type_embedding(hidden_features, 12)
        user_topHour_embedding_rec = self.generate_feature_type_embedding(hidden_features, 6)
        Wanchorids_long_embeddings_rec = self.generate_feature_type_embedding(hidden_features, 50)
        Wanchorids_long_len_rec = self.generate_feature_type_len(hidden_features)
        Wanchorids_short_embeddings_rec =  self.generate_feature_type_embedding(hidden_features, 50)
        Wanchorids_short_len_rec  = self.generate_feature_type_len(hidden_features)
        Wanchorids_noclick_embeddings_rec =  self.generate_feature_type_embedding(hidden_features, 50)
        Wanchorids_noclick_len_rec  = self.generate_feature_type_len(hidden_features)
        Wanchorids_effect_embeddings_rec =  self.generate_feature_type_embedding(hidden_features, 50)
        Wanchorids_effect_len_rec  = self.generate_feature_type_len(hidden_features)
        redict_weights_rec = self.generate_feature_type_values(hidden_features, 8)
        return userRecanchor_embeddings_rec, user_stat_embedding_rec, user_topHour_embedding_rec, Wanchorids_long_embeddings_rec, Wanchorids_long_len_rec, \
               Wanchorids_short_embeddings_rec, Wanchorids_short_len_rec, Wanchorids_noclick_embeddings_rec, Wanchorids_noclick_len_rec, Wanchorids_effect_embeddings_rec, \
               Wanchorids_effect_len_rec, redict_weights_rec

    def generate_semantic_features(self, hidden_features):
        user_profile_basefea_rec = self.generate_feature_type_embedding(hidden_features, 8)
        user_appcate1_rec = self.generate_feature_type_embedding(hidden_features, 1)
        user_appcate2_rec = self.generate_feature_type_embedding(hidden_features, 1)
        anchor_profile_basefea_rec = self.generate_feature_type_embedding(hidden_features, 7)
        anchor_appcate1_rec = self.generate_feature_type_embedding(hidden_features, 1)
        anchor_appcate2_rec = self.generate_feature_type_embedding(hidden_features, 1)
        anchor_stat_rec = self.generate_feature_type_embedding(hidden_features, 16)
        anchor_stat_values_rec = self.generate_feature_type_values(hidden_features, 15)
        day_ctr_value_rec = self.generate_feature_type_values(hidden_features, self.day_seq_len)
        day_ctr_value_rec = tf.reshape(day_ctr_value_rec, [-1, self.day_seq_len, 1])
        day_cvr_value_rec = self.generate_feature_type_values(hidden_features, self.day_seq_len)
        day_cvr_value_rec = tf.reshape(day_cvr_value_rec, [-1, self.day_seq_len, 1])
        day_ctrid_rec = self.generate_feature_type_embedding(hidden_features, self.day_ctr_size)
        day_cvrid_rec = self.generate_feature_type_embedding(hidden_features, self.day_cvr_size)
        userSongTag_rec =  self.generate_feature_type_embedding(hidden_features, 1)
        fea_sim_rec = self.generate_feature_type_values(hidden_features, 2)
        anchor_livebasefea_rec = self.generate_feature_type_embedding(hidden_features, 4)
        realtime_values_rec = self.generate_feature_type_values(hidden_features, 22)
        realtime_ids_rec = self.generate_feature_type_embedding(hidden_features, 22)
        hourId_rec = self.generate_feature_type_embedding(hidden_features, 1)
        anchor_tagids_rec = self.generate_feature_type_embedding(hidden_features, 1)
        Day_rec = self.generate_feature_type_embedding(hidden_features, 1)
        live_position_rec = self.generate_feature_type_embedding(hidden_features, 1)
        return user_profile_basefea_rec, user_appcate1_rec, user_appcate2_rec, anchor_profile_basefea_rec, anchor_appcate1_rec, anchor_appcate2_rec, \
               anchor_stat_rec, anchor_stat_values_rec, day_ctr_value_rec, day_cvr_value_rec, day_ctrid_rec, day_cvrid_rec, userSongTag_rec,\
               fea_sim_rec, anchor_livebasefea_rec, realtime_values_rec, realtime_ids_rec, hourId_rec, anchor_tagids_rec, Day_rec, live_position_rec

    def compute_sem_rec_loss(self):
        oldUser_index = tf.reshape(tf.equal(self.newUser_index, False),[-1])
        count = tf.reduce_sum(tf.cast(oldUser_index, tf.int32),axis=0)
        rec_stack = tf.concat(
            [self.user_profile_basefea_rec_bysem[oldUser_index], self.user_appcate1_rec_bysem[oldUser_index], self.user_appcate2_rec_bysem[oldUser_index], self.anchor_profile_basefea_rec_bysem[oldUser_index],
             self.anchor_appcate1_rec_bysem[oldUser_index], self.anchor_appcate2_rec_bysem[oldUser_index], self.anchor_stat_rec_bysem[oldUser_index],  self.userSongTag_rec_bysem[oldUser_index],
             self.anchor_livebasefea_rec_bysem[oldUser_index], self.realtime_ids_rec_bysem[oldUser_index], self.hourId_rec_bysem[oldUser_index], self.anchor_tagids_rec_bysem[oldUser_index], self.Day_rec_bysem[oldUser_index],
             self.live_position_rec_bysem[oldUser_index], self.user_profile_basefea_rec_byvis[oldUser_index], self.user_appcate1_rec_byvis[oldUser_index], self.user_appcate2_rec_byvis[oldUser_index],
             self.anchor_profile_basefea_rec_byvis[oldUser_index], self.anchor_appcate1_rec_byvis[oldUser_index], self.anchor_appcate2_rec_byvis[oldUser_index], self.anchor_stat_rec_byvis[oldUser_index],
             self.userSongTag_rec_byvis[oldUser_index], self.anchor_livebasefea_rec_byvis[oldUser_index], self.realtime_ids_rec_byvis[oldUser_index], self.hourId_rec_byvis[oldUser_index],
             self.anchor_tagids_rec_byvis[oldUser_index], self.Day_rec_byvis[oldUser_index], self.live_position_rec_byvis[oldUser_index]], axis=1)
        rec_stack = tf.reshape(rec_stack, [count, -1])
        ctr_value_rec_bysem = tf.reshape(self.day_ctr_value_rec_bysem, [-1, self.day_seq_len])
        cvr_value_rec_bysem = tf.reshape(self.day_cvr_value_rec_bysem, [-1, self.day_seq_len])
        ctr_value_rec_byvis = tf.reshape(self.day_ctr_value_rec_byvis, [-1, self.day_seq_len])
        cvr_value_rec_byvis = tf.reshape(self.day_cvr_value_rec_byvis, [-1, self.day_seq_len])
        rec_stack = tf.concat([rec_stack, self.anchor_stat_values_rec_bysem[oldUser_index], ctr_value_rec_bysem[oldUser_index], cvr_value_rec_bysem[oldUser_index], self.fea_sim_rec_bysem[oldUser_index],
                               self.realtime_values_rec_bysem[oldUser_index], self.anchor_stat_values_rec_byvis[oldUser_index], ctr_value_rec_byvis[oldUser_index],
                               cvr_value_rec_byvis[oldUser_index], self.fea_sim_rec_byvis[oldUser_index], self.realtime_values_rec_byvis[oldUser_index]], axis=-1)

        stack = tf.concat(
            [self.user_profile_basefea_embeddings[oldUser_index], self.user_appcate1_embeddings[oldUser_index], self.user_appcate2_embeddings[oldUser_index], self.anchor_profile_basefea_embeddings[oldUser_index],
             self.anchor_appcate1_embeddings[oldUser_index], self.anchor_appcate2_embeddings[oldUser_index], self.anchor_stat_rec_bysem[oldUser_index], self.userSongTag_embeddings[oldUser_index],
             self.anchor_livebasefea_embeddings[oldUser_index], self.realtime_ids_embeddings[oldUser_index], self.hour_embeddings[oldUser_index], self.anchor_tagids_embeddings[oldUser_index], self.Day_embeddings[oldUser_index],
             self.livePosition_embeddings[oldUser_index], self.user_profile_basefea_embeddings[oldUser_index], self.user_appcate1_embeddings[oldUser_index], self.user_appcate2_embeddings[oldUser_index], self.anchor_profile_basefea_embeddings[oldUser_index],
             self.anchor_appcate1_embeddings[oldUser_index], self.anchor_appcate2_embeddings[oldUser_index], self.anchor_stat_rec_bysem[oldUser_index], self.userSongTag_embeddings[oldUser_index],
             self.anchor_livebasefea_embeddings[oldUser_index], self.realtime_ids_embeddings[oldUser_index], self.hour_embeddings[oldUser_index], self.anchor_tagids_embeddings[oldUser_index], self.Day_embeddings[oldUser_index],
             self.livePosition_embeddings[oldUser_index]], axis=1)
        stack = tf.reshape(stack, [count, -1])
        ctr_value = tf.reshape(self.day_ctr_value, [-1, self.day_seq_len])
        cvr_value = tf.reshape(self.day_cvr_value, [-1, self.day_seq_len])
        stack = tf.concat([stack, self.anchor_stat_values[oldUser_index], ctr_value[oldUser_index], cvr_value[oldUser_index], self.fea_sim[oldUser_index], self.realtime_values[oldUser_index],
                           self.anchor_stat_values[oldUser_index], ctr_value[oldUser_index], cvr_value[oldUser_index], self.fea_sim[oldUser_index], self.realtime_values[oldUser_index]], axis=-1)
        rec_loss = tf.reduce_mean(tf.square(stack - rec_stack))
        return rec_loss

    def compute_vis_rec_loss(self):
        oldUser_index = tf.reshape(tf.equal(self.newUser_index, False),[-1])
        count = tf.reduce_sum(tf.cast(oldUser_index,tf.int32),axis=0)
        rec_stack = tf.concat(
            [self.userRecanchor_embeddings_rec_bysem[oldUser_index], self.user_stat_embedding_rec_bysem[oldUser_index], self.user_topHour_embedding_rec_bysem[oldUser_index], self.Wanchorids_long_embeddings_rec_bysem[oldUser_index],
             self.Wanchorids_short_embeddings_rec_bysem[oldUser_index], self.Wanchorids_noclick_embeddings_rec_bysem[oldUser_index], self.Wanchorids_effect_embeddings_rec_bysem[oldUser_index],
             self.userRecanchor_embeddings_rec_byvis[oldUser_index], self.user_stat_embedding_rec_byvis[oldUser_index], self.user_topHour_embedding_rec_byvis[oldUser_index], self.Wanchorids_long_embeddings_rec_byvis[oldUser_index],
             self.Wanchorids_short_embeddings_rec_byvis[oldUser_index], self.Wanchorids_noclick_embeddings_rec_byvis[oldUser_index], self.Wanchorids_effect_embeddings_rec_byvis[oldUser_index]], axis=1)
        rec_stack = tf.reshape(rec_stack, [count, -1])
        rec_stack = tf.concat([rec_stack, self.redict_weights_rec_bysem[oldUser_index], self.redict_weights_rec_byvis[oldUser_index], self.Wanchorids_long_len_rec_bysem[oldUser_index],
                               self.Wanchorids_short_len_rec_bysem[oldUser_index], self.Wanchorids_noclick_len_rec_bysem[oldUser_index], self.Wanchorids_effect_len_rec_bysem[oldUser_index],
                               self.Wanchorids_long_len_rec_byvis[oldUser_index], self.Wanchorids_short_len_rec_byvis[oldUser_index], self.Wanchorids_noclick_len_rec_byvis[oldUser_index], self.Wanchorids_effect_len_rec_byvis[oldUser_index]], axis=-1)

        self.log_label = self.Wanchorids_long_len_pre
        stack = tf.concat(
            [self.userRecanchor_embeddings_pre[oldUser_index], self.user_stat_embedding_pre[oldUser_index], self.user_topHour_embedding_pre[oldUser_index], self.Wanchorids_long_embeddings_pre[oldUser_index],
             self.Wanchorids_short_embeddings_pre[oldUser_index],self.Wanchorids_noclick_embeddings_pre[oldUser_index], self.Wanchorids_effect_embeddings_pre[oldUser_index],
             self.userRecanchor_embeddings_pre[oldUser_index], self.user_stat_embedding_pre[oldUser_index], self.user_topHour_embedding_pre[oldUser_index], self.Wanchorids_long_embeddings_pre[oldUser_index],
             self.Wanchorids_short_embeddings_pre[oldUser_index], self.Wanchorids_noclick_embeddings_pre[oldUser_index], self.Wanchorids_effect_embeddings_pre[oldUser_index]], axis=1)
        stack = tf.reshape(stack, [count, -1])
        Wanchorids_long_len_float = tf.reshape(tf.cast(self.Wanchorids_long_len_pre, tf.float32), [-1,1])
        Wanchorids_short_len_float = tf.reshape(tf.cast(self.Wanchorids_short_len_pre, tf.float32), [-1,1])
        Wanchorids_noclick_len_float = tf.reshape(tf.cast(self.Wanchorids_noclick_len_pre, tf.float32), [-1,1])
        Wanchorids_effect_len_float = tf.reshape(tf.cast(self.Wanchorids_effect_len_pre, tf.float32), [-1,1])
        stack = tf.concat([stack, self.redict_weights_pre[oldUser_index], self.redict_weights_pre[oldUser_index], Wanchorids_long_len_float[oldUser_index], Wanchorids_short_len_float[oldUser_index],
                           Wanchorids_noclick_len_float[oldUser_index], Wanchorids_effect_len_float[oldUser_index],  Wanchorids_long_len_float[oldUser_index], Wanchorids_short_len_float[oldUser_index],
                           Wanchorids_noclick_len_float[oldUser_index], Wanchorids_effect_len_float[oldUser_index]],axis=-1)
        rec_loss = tf.reduce_mean(tf.square(stack - rec_stack))
        return rec_loss

    def update_features(self):
        self.newUser_index = tf.reshape(self.newUser_index, [-1])

        userRecanchor_embeddings_rec_bysem = tf.where(self.newUser_index, self.userRecanchor_embeddings_rec_bysem, self.userRecanchor_embeddings_pre)
        user_stat_embedding_rec_bysem = tf.where(self.newUser_index, self.user_stat_embedding_rec_bysem, self.user_stat_embedding_pre)
        user_topHour_embedding_rec_bysem = tf.where(self.newUser_index, self.user_topHour_embedding_rec_bysem, self.user_topHour_embedding_pre)
        Wanchorids_long_embeddings_rec_bysem = tf.where(self.newUser_index, self.Wanchorids_long_embeddings_rec_bysem, self.Wanchorids_long_embeddings_pre)
        Wanchorids_short_embeddings_rec_bysem = tf.where(self.newUser_index, self.Wanchorids_short_embeddings_rec_bysem, self.Wanchorids_short_embeddings_pre)
        Wanchorids_effect_embeddings_rec_bysem = tf.where(self.newUser_index, self.Wanchorids_effect_embeddings_rec_bysem, self.Wanchorids_effect_embeddings_pre)
        Wanchorids_noclick_embeddings_rec_bysem = tf.where(self.newUser_index, self.Wanchorids_noclick_embeddings_rec_bysem, self.Wanchorids_noclick_embeddings_pre)
        redict_weights_rec_bysem = tf.where(self.newUser_index, self.redict_weights_rec_bysem, self.redict_weights_pre)

        long_log = tf.where(tf.less(self.Wanchorids_long_len_rec_bysem, 1), tf.ones_like(self.Wanchorids_long_len_rec_bysem), self.Wanchorids_long_len_rec_bysem)
        long_log = tf.where(tf.greater(long_log, 100*tf.ones_like(long_log)), 100*tf.ones_like(long_log), long_log)
        Wanchorids_long_len_rec_bysem = tf.reshape(tf.cast(long_log, tf.int64), [-1])
        Wanchorids_long_len_rec_bysem = tf.where(self.newUser_index, Wanchorids_long_len_rec_bysem, self.Wanchorids_long_len_pre)

        short_log = tf.where(tf.less(self.Wanchorids_short_len_rec_bysem, 1), tf.ones_like(self.Wanchorids_short_len_rec_bysem), self.Wanchorids_short_len_rec_bysem)
        short_log = tf.where(tf.greater(short_log, 100*tf.ones_like(short_log)), 100*tf.ones_like(short_log), short_log)
        Wanchorids_short_len_rec_bysem = tf.reshape(tf.cast(short_log, tf.int64), [-1])
        Wanchorids_short_len_rec_bysem = tf.where(self.newUser_index, Wanchorids_short_len_rec_bysem, self.Wanchorids_short_len_pre)

        effect_log = tf.where(tf.less(self.Wanchorids_effect_len_rec_bysem, 1), tf.ones_like(self.Wanchorids_effect_len_rec_bysem), self.Wanchorids_effect_len_rec_bysem)
        effect_log = tf.where(tf.greater(effect_log, 100 * tf.ones_like(effect_log)), 100 * tf.ones_like(effect_log), effect_log)
        Wanchorids_effect_len_rec_bysem = tf.reshape(tf.cast(effect_log, tf.int64), [-1])
        Wanchorids_effect_len_rec_bysem = tf.where(self.newUser_index, Wanchorids_effect_len_rec_bysem, self.Wanchorids_effect_len_pre)

        noclick_log = tf.where(tf.less(self.Wanchorids_noclick_len_rec_bysem, 1), tf.ones_like(self.Wanchorids_noclick_len_rec_bysem), self.Wanchorids_noclick_len_rec_bysem)
        noclick_log = tf.where(tf.greater(noclick_log, 100 * tf.ones_like(noclick_log)), 100 * tf.ones_like(noclick_log), noclick_log)
        Wanchorids_noclick_len_rec_bysem = tf.reshape(tf.cast(noclick_log, tf.int64), [-1])
        Wanchorids_noclick_len_rec_bysem = tf.where(self.newUser_index, Wanchorids_noclick_len_rec_bysem, self.Wanchorids_noclick_len_pre)

        self.userRecanchor_embeddings = tf.cond(self.train_phase, lambda: userRecanchor_embeddings_rec_bysem, lambda: userRecanchor_embeddings_rec_bysem )
        self.user_stat_embedding = tf.cond(self.train_phase, lambda: self.user_stat_embedding_pre, lambda: self.user_stat_embedding_pre )  #这里不需要填充
        self.user_topHour_embedding = tf.cond(self.train_phase, lambda: user_topHour_embedding_rec_bysem, lambda: user_topHour_embedding_rec_bysem )
        self.Wanchorids_long_embeddings = tf.cond(self.train_phase, lambda: Wanchorids_long_embeddings_rec_bysem, lambda: Wanchorids_long_embeddings_rec_bysem )
        self.Wanchorids_long_len = tf.cond(self.train_phase, lambda: Wanchorids_long_len_rec_bysem, lambda: Wanchorids_long_len_rec_bysem )
        self.Wanchorids_short_embeddings = tf.cond(self.train_phase, lambda: Wanchorids_short_embeddings_rec_bysem, lambda: Wanchorids_short_embeddings_rec_bysem )
        self.Wanchorids_short_len = tf.cond(self.train_phase, lambda: Wanchorids_short_len_rec_bysem, lambda: Wanchorids_short_len_rec_bysem )
        self.Wanchorids_effect_embeddings = tf.cond(self.train_phase, lambda: Wanchorids_effect_embeddings_rec_bysem, lambda: Wanchorids_effect_embeddings_rec_bysem )
        self.Wanchorids_effect_len = tf.cond(self.train_phase, lambda: Wanchorids_effect_len_rec_bysem, lambda: Wanchorids_effect_len_rec_bysem )
        self.Wanchorids_noclick_embeddings = tf.cond(self.train_phase, lambda: Wanchorids_noclick_embeddings_rec_bysem, lambda: Wanchorids_noclick_embeddings_rec_bysem )
        self.Wanchorids_noclick_len = tf.cond(self.train_phase, lambda: Wanchorids_noclick_len_rec_bysem, lambda: Wanchorids_noclick_len_rec_bysem )
        self.redict_weights = tf.cond(self.train_phase, lambda: redict_weights_rec_bysem, lambda: redict_weights_rec_bysem )



    def make_embedding(self):
        # ===========用户基础特征embedding===========
        with tf.name_scope("user_base_fea_embedding"):
            self.baseFea_W = tf.Variable(tf.random_normal([self.base_fea_size, self.embedding_size], 0, 0.01),
                                         name="baseFea_W")
            self.user_profile_basefea_embeddings = tf.nn.embedding_lookup(self.baseFea_W, self.user_profile_basefea,
                                                                          name="user_profile_basefea_embeddings")
            self.user_appcate1_embeddings = tf.nn.embedding_lookup_sparse(self.baseFea_W, self.user_appcate1, None,
                                                                          combiner="sum",
                                                                          name="user_appcate1_embeddings")
            self.user_appcate1_embeddings = tf.reshape(self.user_appcate1_embeddings,
                                                       shape=[-1, 1, self.embedding_size])
            self.user_appcate2_embeddings = tf.nn.embedding_lookup_sparse(self.baseFea_W, self.user_appcate2, None,
                                                                          combiner="sum",
                                                                          name="user_appcate2_embeddings")
            self.user_appcate2_embeddings = tf.reshape(self.user_appcate2_embeddings,
                                                       shape=[-1, 1, self.embedding_size])


        with tf.name_scope("user_action_fea_embedding"):
            self.anchorId_W = tf.Variable(tf.random_normal([self.anchorId_size, self.embedding_size], 0, 0.01), name="anchorId_W")
            userRecanchor_embeddings = tf.nn.embedding_lookup_sparse(self.anchorId_W, self.user_RecAnchor, None,
                                                                     combiner="sum", name="userRecanchor_embeddings")
            self.userRecanchor_embeddings_pre = tf.reshape(userRecanchor_embeddings, shape=[-1, 1, self.embedding_size])
            self.userStat_W = tf.Variable(tf.random_normal([self.user_stat_fea_size, self.embedding_size], 0, 0.01),
                                          name="userStat_W")
            self.user_stat_embedding_pre = tf.nn.embedding_lookup(self.userStat_W, self.user_statIds,
                                                              name="user_statid_embedding")
            self.userTophour_W = tf.Variable(tf.random_normal([self.Tophour_fea_size, self.embedding_size], 0, 0.01),
                                             name="userTophour_W")
            self.user_topHour_embedding_pre = tf.nn.embedding_lookup(self.userTophour_W, self.user_topHourIds,
                                                                 name="user_topHourIds_embedding")
            Wanchorids_long_embeddings = tf.nn.embedding_lookup(self.anchorId_W, self.Wanchorids_long_dense)
            Wanchorids_long_embeddings = tf.reshape(Wanchorids_long_embeddings,
                                                    shape=[-1, tf.shape(self.Wanchorids_long_dense)[1],
                                                           self.embedding_size])
            pad1 = tf.zeros([tf.shape(self.Wanchorids_long_dense)[0], 50 - tf.shape(self.Wanchorids_long_dense)[1],
                             self.embedding_size])
            self.Wanchorids_long_embeddings_pre = tf.cond(tf.shape(self.Wanchorids_long_dense)[1] < 50,
                                                      lambda: tf.concat([Wanchorids_long_embeddings, pad1], axis=1),
                                                      lambda: tf.identity(Wanchorids_long_embeddings))
            Wanchorids_short_embeddings = tf.nn.embedding_lookup(self.anchorId_W, self.Wanchorids_short_dense)
            Wanchorids_short_embeddings = tf.reshape(Wanchorids_short_embeddings,
                                                     shape=[-1, tf.shape(self.Wanchorids_short_dense)[1],
                                                            self.embedding_size])
            pad2 = tf.zeros([tf.shape(self.Wanchorids_short_dense)[0], 50 - tf.shape(self.Wanchorids_short_dense)[1],
                             self.embedding_size])
            self.Wanchorids_short_embeddings_pre = tf.cond(tf.shape(self.Wanchorids_short_dense)[1] < 50,
                                                       lambda: tf.concat([Wanchorids_short_embeddings, pad2], axis=1),
                                                       lambda: tf.identity(Wanchorids_short_embeddings))
            Wanchorids_effect_embeddings = tf.nn.embedding_lookup(self.anchorId_W, self.Wanchorids_effect_dense)
            Wanchorids_effect_embeddings = tf.reshape(Wanchorids_effect_embeddings,
                                                      shape=[-1, tf.shape(self.Wanchorids_effect_dense)[1],
                                                             self.embedding_size])
            pad3 = tf.zeros([tf.shape(self.Wanchorids_effect_dense)[0], 50 - tf.shape(self.Wanchorids_effect_dense)[1],
                             self.embedding_size])
            self.Wanchorids_effect_embeddings_pre = tf.cond(tf.shape(self.Wanchorids_effect_dense)[1] < 50,
                                                        lambda: tf.concat([Wanchorids_effect_embeddings, pad3], axis=1),
                                                        lambda: tf.identity(Wanchorids_effect_embeddings))
            Wanchorids_noclick_embeddings = tf.nn.embedding_lookup(self.anchorId_W, self.Wanchorids_noclick_dense)
            Wanchorids_noclick_embeddings = tf.reshape(Wanchorids_noclick_embeddings,
                                                       shape=[-1, tf.shape(self.Wanchorids_noclick_dense)[1],
                                                              self.embedding_size])
            pad4 = tf.zeros(
                [tf.shape(self.Wanchorids_noclick_dense)[0], 50 - tf.shape(self.Wanchorids_noclick_dense)[1],
                 self.embedding_size])
            self.Wanchorids_noclick_embeddings_pre = tf.cond(tf.shape(self.Wanchorids_noclick_dense)[1] < 50,
                                                         lambda: tf.concat([Wanchorids_noclick_embeddings, pad4],
                                                                           axis=1),
                                                         lambda: tf.identity(Wanchorids_noclick_embeddings))


        with tf.name_scope("anchor_base_fea_embedding"):
            self.anchor_profile_basefea_embeddings = tf.nn.embedding_lookup(self.baseFea_W, self.anchor_profile_basefea,
                                                                            name="anchor_profile_basefea_embeddings")
            self.anchor_appcate1_embeddings = tf.nn.embedding_lookup_sparse(self.baseFea_W, self.anchor_appcate1, None,
                                                                            combiner="sum",
                                                                            name="anchor_appcate1_embeddings")
            self.anchor_appcate1_embeddings = tf.reshape(self.anchor_appcate1_embeddings,
                                                         shape=[-1, 1, self.embedding_size])
            self.anchor_appcate2_embeddings = tf.nn.embedding_lookup_sparse(self.baseFea_W, self.anchor_appcate2, None,
                                                                            combiner="sum",
                                                                            name="anchor_appcate2_embeddings")
            self.anchor_appcate2_embeddings = tf.reshape(self.anchor_appcate2_embeddings,
                                                         shape=[-1, 1, self.embedding_size])
            self.anchorid_embeddings = tf.nn.embedding_lookup(self.anchorId_W, self.anchorId)

        with tf.name_scope("anchor_action_fea_embedding"):
            self.anchorStat_W = tf.Variable(tf.random_normal([self.statsFea_size, self.embedding_size], 0, 0.01),
                                            name="anchorStat_W")
            self.anchor_stat_embeddings = tf.nn.embedding_lookup(self.anchorStat_W, self.anchor_stats)
            self.dayFea_W = tf.Variable(tf.random_normal([self.day_fea_size, self.embedding_size], 0, 0.01),
                                        name="dayFea_W")

            self.day_ctrid_embeddings = tf.nn.embedding_lookup(self.dayFea_W, self.day_ctrid_seq_re)
            self.day_seqid_len = tf.shape(self.day_ctrid_embeddings)[1]
            pad_seqid = tf.cast(tf.zeros([tf.shape(self.day_ctrid_embeddings)[0], 7 - self.day_seqid_len, self.embedding_size]),tf.float32)
            self.day_ctrid_embeddings_ = tf.reshape(self.day_ctrid_embeddings, shape=[-1, self.day_seqid_len, self.embedding_size])
            self.day_ctrid_embeddings = tf.cond(self.day_seqid_len < 7,  lambda: tf.concat([self.day_ctrid_embeddings_, pad_seqid], axis=1), lambda: self.day_ctrid_embeddings_)


            self.day_cvrid_embeddings = tf.nn.embedding_lookup(self.dayFea_W, self.day_cvrid_seq_re)
            self.day_cvrid_embeddings_ = tf.reshape(self.day_cvrid_embeddings, shape=[-1, self.day_seqid_len, self.embedding_size])
            self.day_cvrid_embeddings = tf.cond(self.day_seqid_len < 7,  lambda: tf.concat([self.day_cvrid_embeddings_, pad_seqid], axis=1), lambda: self.day_cvrid_embeddings_)


            self.day_seq_len = tf.shape(self.day_ctr_seq_re)[1]
            pad_seq = tf.cast(tf.zeros([tf.shape(self.day_ctr_seq_re)[0], 7 - self.day_seq_len]),tf.float32)
            self.day_ctr_value_ = tf.cond(self.day_seq_len < 7,  lambda: tf.concat([self.day_ctr_seq_re, pad_seq], axis=1), lambda: self.day_ctr_seq_re)
            self.day_cvr_value_ = tf.cond(self.day_seq_len < 7,  lambda: tf.concat([self.day_cvr_seq_re, pad_seq], axis=1), lambda: self.day_cvr_seq_re)
            self.temp = self.day_seq_len
            self.day_seq_len = 7
            self.day_ctr_value = tf.reshape(self.day_ctr_value_, shape=[-1, self.day_seq_len, 1])
            self.day_cvr_value = tf.reshape(self.day_cvr_value_, shape=[-1, self.day_seq_len, 1])



            self.day_ctr_dense1 = tf.multiply(self.day_ctrid_embeddings, self.day_ctr_value)  # 其余需要注意的特征
            self.day_ctr_dense = self.day_ctr_dense1[:,0:self.temp,:]
            self.day_cvr_dense1 = tf.multiply(self.day_cvrid_embeddings, self.day_cvr_value)  # 其余需要注意的特征
            self.day_cvr_dense = self.day_cvr_dense1[:,0:self.temp,:]
            self.day_ctcvr = tf.concat([self.day_ctr_dense, self.day_cvr_dense], axis=-1)  # 其余需要注意的特征
            anchor_stat_value = tf.reshape(self.anchor_stat_values, shape=[-1, self.anchor_stats_size, 1])  # 其余需要注意的特征
            self.anchor_stat_dense = tf.multiply(self.anchor_stat_embeddings, anchor_stat_value)  # 其余需要注意的特征

        with tf.name_scope("fea_embedding_by_music"):
            self.songTag_W = tf.Variable(tf.random_normal([self.songtag_size, self.embedding_size], 0, 0.01),
                                         name="songTag_W")
            userSongTag_embeddings = tf.nn.embedding_lookup_sparse(self.songTag_W, sp_ids=self.user_songTagids,
                                                                   sp_weights=self.user_tagWeights, combiner="sum",
                                                                   name="userSongTag_embeddings")
            self.userSongTag_embeddings = tf.reshape(userSongTag_embeddings, shape=[-1, 1, self.embedding_size])
            anchorSongTag_embeddings = tf.nn.embedding_lookup_sparse(self.songTag_W, sp_ids=self.anchor_songTagids,
                                                                     sp_weights=self.anchor_tagWeights, combiner="sum",
                                                                     name="userSongTag_embeddings")
            self.anchorSongTag_embeddings = tf.reshape(anchorSongTag_embeddings, shape=[-1, 1, self.embedding_size])


        with tf.name_scope("online_fea_embedding"):
            self.anchorLiveFea_W = tf.Variable(
                tf.random_normal([self.anchor_livefea_size, self.embedding_size], 0, 0.01), name="anchorLiveFea_W")
            self.anchor_livebasefea_embeddings = tf.nn.embedding_lookup(self.anchorLiveFea_W, self.anchor_live_basefea)
            self.realtimeFea_W = tf.Variable(tf.random_normal([self.realtimeFea_size, self.embedding_size], 0, 0.01),
                                             name="realtimeFea_W")
            self.realtime_ids_embeddings = tf.nn.embedding_lookup(self.realtimeFea_W, self.realtime_ids)
            self.impressHour_W = tf.Variable(tf.random_normal([self.hour_size, self.embedding_size], 0, 0.01),
                                             name="impressHour_W")
            self.hour_embeddings = tf.nn.embedding_lookup(self.impressHour_W, self.hourId)
            anchor_tagids_embeddings = tf.nn.embedding_lookup_sparse(self.anchorLiveFea_W, self.anchor_tagids, None,
                                                                     combiner="sum", name="anchor_tagids_embeddings")
            self.anchor_tagids_embeddings = tf.reshape(anchor_tagids_embeddings, shape=[-1, 1, self.embedding_size])
            self.impressDay_W = tf.Variable(tf.random_normal([8, self.embedding_size], 0, 0.01), name="impressDay_W")
            self.Day_embeddings = tf.nn.embedding_lookup(self.impressDay_W, self.dayOfWeek)
            self.LivePosition_W = tf.Variable(tf.random_normal([self.LivePosition_size, self.embedding_size], 0, 0.01),
                                              name="LivePosition_W")
            self.livePosition_embeddings = tf.nn.embedding_lookup(self.LivePosition_W, self.live_position)

            realtime_value = tf.reshape(self.realtime_values, shape=[-1, self.realtime_values_size, 1])
            self.realtime_value_dense = tf.multiply(self.realtime_ids_embeddings, realtime_value)  

    def make_position_embedding(self):
        with tf.name_scope("position_embeddings"):
            self.position_his_long = tf.range(100)
            self.dm_position_his_long = tf.range(100)
            self.position_his_short = tf.range(100)
            self.dm_position_his_short = tf.range(100)
            self.position_his_effect = tf.range(100)
            self.dm_position_his_effect = tf.range(100)
            self.position_his_noclick = tf.range(100)
            self.dm_position_his_noclick = tf.range(100)

            self.ct_position_his = tf.range(tf.shape(self.day_ctr_seq_re)[1])
            self.ct_position_W = tf.Variable(tf.random_normal([7, self.embedding_size], 0, 0.01), name="ct_position_W")

            self.position_W_long = tf.Variable(tf.random_normal([100, self.embedding_size], 0, 0.01),
                                               name="position_W_long")
            self.dm_position_W_long = tf.Variable(tf.random_normal([100, self.embedding_size], 0, 0.01),
                                                  name="dm_position_W_long")
            self.position_W_short = tf.Variable(tf.random_normal([100, self.embedding_size], 0, 0.01),
                                                name="position_W_short")
            self.dm_position_W_short = tf.Variable(tf.random_normal([100, self.embedding_size], 0, 0.01),
                                                   name="dm_position_W_short")
            self.position_W_effect = tf.Variable(tf.random_normal([100, self.embedding_size], 0, 0.01),
                                                 name="position_W_effect")
            self.dm_position_W_effect = tf.Variable(tf.random_normal([100, self.embedding_size], 0, 0.01),
                                                    name="dm_position_W_effect")
            self.position_W_noclick = tf.Variable(tf.random_normal([100, self.embedding_size], 0, 0.01),
                                                  name="position_W_noclick")
            self.dm_position_W_noclick = tf.Variable(tf.random_normal([100, self.embedding_size], 0, 0.01),
                                                     name="dm_position_W_noclick")


            self.position_his_eb_long = tf.nn.embedding_lookup(self.position_W_long, self.position_his_long)  # T,E
            self.position_his_eb_long = tf.tile(self.position_his_eb_long,
                                                [tf.shape(self.anchorSongTag_embeddings)[0], 1])  # B*T,E
            self.position_his_eb_long = tf.reshape(self.position_his_eb_long,
                                                   [tf.shape(self.anchorSongTag_embeddings)[0], -1,
                                                    self.position_his_eb_long.get_shape().as_list()[1]])  # B,T,E
            self.dm_position_his_eb_long = tf.nn.embedding_lookup(self.dm_position_W_long,
                                                                  self.dm_position_his_long)  # T,E
            self.dm_position_his_eb_long = tf.tile(self.dm_position_his_eb_long,
                                                   [tf.shape(self.anchorSongTag_embeddings)[0], 1])  # B*T,E
            self.dm_position_his_eb_long = tf.reshape(self.dm_position_his_eb_long,
                                                      [tf.shape(self.anchorSongTag_embeddings)[0], -1,
                                                       self.dm_position_his_eb_long.get_shape().as_list()[1]])  # B,T,E

            self.position_his_eb_short = tf.nn.embedding_lookup(self.position_W_short, self.position_his_short)  # T,E
            self.position_his_eb_short = tf.tile(self.position_his_eb_short,
                                                 [tf.shape(self.anchorSongTag_embeddings)[0], 1])  # B*T,E
            self.position_his_eb_short = tf.reshape(self.position_his_eb_short,
                                                    [tf.shape(self.anchorSongTag_embeddings)[0], -1,
                                                     self.position_his_eb_short.get_shape().as_list()[1]])  # B,T,E
            self.dm_position_his_eb_short = tf.nn.embedding_lookup(self.dm_position_W_short,
                                                                   self.dm_position_his_short)  # T,E
            self.dm_position_his_eb_short = tf.tile(self.dm_position_his_eb_short,
                                                    [tf.shape(self.anchorSongTag_embeddings)[0], 1])  # B*T,E
            self.dm_position_his_eb_short = tf.reshape(self.dm_position_his_eb_short,
                                                       [tf.shape(self.anchorSongTag_embeddings)[0], -1,
                                                        self.dm_position_his_eb_short.get_shape().as_list()[
                                                            1]])  # B,T,E

            self.position_his_eb_effect = tf.nn.embedding_lookup(self.position_W_effect,
                                                                 self.position_his_effect)  # T,E
            self.position_his_eb_effect = tf.tile(self.position_his_eb_effect,
                                                  [tf.shape(self.anchorSongTag_embeddings)[0], 1])  # B*T,E
            self.position_his_eb_effect = tf.reshape(self.position_his_eb_effect,
                                                     [tf.shape(self.anchorSongTag_embeddings)[0], -1,
                                                      self.position_his_eb_effect.get_shape().as_list()[1]])  # B,T,E
            self.dm_position_his_eb_effect = tf.nn.embedding_lookup(self.dm_position_W_effect,
                                                                    self.dm_position_his_effect)  # T,E
            self.dm_position_his_eb_effect = tf.tile(self.dm_position_his_eb_effect,
                                                     [tf.shape(self.anchorSongTag_embeddings)[0], 1])  # B*T,E
            self.dm_position_his_eb_effect = tf.reshape(self.dm_position_his_eb_effect,
                                                        [tf.shape(self.anchorSongTag_embeddings)[0], -1,
                                                         self.dm_position_his_eb_effect.get_shape().as_list()[
                                                             1]])  # B,T,E

            self.position_his_eb_noclick = tf.nn.embedding_lookup(self.position_W_noclick,
                                                                  self.position_his_noclick)  # T,E
            self.position_his_eb_noclick = tf.tile(self.position_his_eb_noclick,
                                                   [tf.shape(self.anchorSongTag_embeddings)[0], 1])  # B*T,E
            self.position_his_eb_noclick = tf.reshape(self.position_his_eb_noclick,
                                                      [tf.shape(self.anchorSongTag_embeddings)[0], -1,
                                                       self.position_his_eb_noclick.get_shape().as_list()[1]])  # B,T,E
            self.dm_position_his_eb_noclick = tf.nn.embedding_lookup(self.dm_position_W_noclick,
                                                                     self.dm_position_his_noclick)  # T,E
            self.dm_position_his_eb_noclick = tf.tile(self.dm_position_his_eb_noclick,
                                                      [tf.shape(self.anchorSongTag_embeddings)[0], 1])  # B*T,E
            self.dm_position_his_eb_noclick = tf.reshape(self.dm_position_his_eb_noclick,
                                                         [tf.shape(self.anchorSongTag_embeddings)[0], -1,
                                                          self.dm_position_his_eb_noclick.get_shape().as_list()[
                                                              1]])  # B,T,E

            self.ct_position_his_eb = tf.nn.embedding_lookup(self.ct_position_W, self.ct_position_his)  # T,E
            self.ct_position_his_eb = tf.tile(self.ct_position_his_eb,
                                              [tf.shape(self.anchorSongTag_embeddings)[0], 1])  # B*T,E
            self.ct_position_his_eb = tf.reshape(self.ct_position_his_eb,
                                                 [tf.shape(self.anchorSongTag_embeddings)[0], -1,
                                                  self.ct_position_his_eb.get_shape().as_list()[1]])  # B,T,E



    def make_FM_Input(self):
        user_all_profile = tf.concat([self.user_profile_basefea_embeddings, self.user_appcate1_embeddings, self.user_appcate2_embeddings, self.userSongTag_embeddings],axis=1)
        anchor_all_profile = tf.concat([self.anchor_profile_basefea_embeddings, self.anchor_appcate1_embeddings,self.anchor_appcate2_embeddings, self.anchorSongTag_embeddings], axis=1)
        anchor_liveFea = tf.concat([self.anchor_livebasefea_embeddings, self.anchor_tagids_embeddings], axis=1)
        FM_Input = tf.concat([user_all_profile, anchor_all_profile, anchor_liveFea, self.user_stat_embedding,
                              self.user_topHour_embedding, self.anchor_stat_dense, self.day_ctr_dense1,
                              self.day_cvr_dense1,
                              self.userRecanchor_embeddings, self.anchorid_embeddings, self.realtime_value_dense,
                              self.hour_embeddings, self.Day_embeddings, self.livePosition_embeddings], axis=1)

        FM_size = (self.profile_size + self.profile_size - 1) + 3 * 2 + self.live_size + 1 + self.user_statIds_size + self.user_topHourIds_size + self.anchor_stats_size \
                  + 4 + self.realtime_values_size + 1 + 2 * self.day_seq_len
        return FM_Input, FM_size

    def make_DNN_Input(self, FM_Input, FM_size):

        dnn_input_temp = tf.reshape(FM_Input, shape=[-1, FM_size * self.embedding_size])
        DNN_Input = tf.concat([dnn_input_temp, self.fea_sim, self.anchor_tagidonehot_float, self.redict_weights],
                              axis=1)
        DNN_size = FM_size * self.embedding_size + 2 + 38 + 8
        return DNN_Input, DNN_size

    def make_DMR_Input(self):
        self.anchor_fea = tf.reshape(self.anchorid_embeddings, shape=[-1, self.embedding_size])
        self.anchor_his_fea_long_v2 = tf.identity(self.Wanchorids_long_embeddings)
        self.anchor_his_fea_short_v2 = tf.identity(self.Wanchorids_short_embeddings)
        self.anchor_his_fea_effect_v2 = tf.identity(self.Wanchorids_effect_embeddings)
        self.anchor_his_fea_noclick_v2 = tf.identity(self.Wanchorids_noclick_embeddings)
        self.anchor_his_fea_sum_long = tf.reduce_sum(self.Wanchorids_long_embeddings, 1)
        self.anchor_his_fea_sum_short = tf.reduce_sum(self.Wanchorids_short_embeddings, 1)
        self.anchor_his_fea_sum_effect = tf.reduce_sum(self.Wanchorids_effect_embeddings, 1)
        self.anchor_his_fea_sum_noclick = tf.reduce_sum(self.Wanchorids_noclick_embeddings, 1)
        self.ctcvr_sum = tf.reduce_sum(self.day_ctcvr, 1)

    def read_parameters(self,config):
        self.base_fea_size = config["base_fea_size"]
        self.user_stat_fea_size = config["user_stat_fea_size"]
        self.realtimeFea_size = config["realtimeFea_size"]
        self.statsFea_size = config["statsFea_size"]
        self.Tophour_fea_size = config["Tophour_fea_size"]
        self.anchor_livefea_size = config["anchor_livefea_size"]
        self.anchorId_size = config["anchorId_size"]
        self.hour_size = config["hour_size"]
        self.day_fea_size = config["day_fea_size"]
        self.songtag_size = config["songtag_size"]
        self.LivePosition_size = config["LivePosition_size"]

        self.profile_size = config["profile_size"]
        self.user_statIds_size = config["user_statIds_size"]
        self.realtime_values_size = config["realtime_values_size"]
        self.anchor_stats_size = config["anchor_stats_size"]
        self.live_size = config["live_size"]
        self.day_ctr_size = config["day_ctr_size"]
        self.day_cvr_size = config["day_cvr_size"]
        self.user_topHourIds_size = config["user_topHourIds_size"]
        self.tagidOnehot_size = config["tagidOnehot_size"]

        self.fea_sim_size = config["fea_sim_size"]
        self.embedding_size = config["embedding_size"]
        self.l2_reg_lambda = config["l2_reg_lambda"]
        self.learning_rate = config["learning_rate"]

        self.batch_norm = config["batch_norm"]
        self.batch_norm_decay = config["batch_norm_decay"]
        # category  deep FM configuration
        self.deep_layers = config["deep_layers"]
        self.deep_layers_2 = config["deep_layers_2"]
        self.ctr_task_wgt = config["ctr_task_wgt"]
        self.batch_size = config['batch_size']

    def create_placeholder(self):

        self.label_ctr = tf.placeholder(tf.int64, [None], name="labek_ctr")
        self.label_ctr = tf.cast(self.label_ctr, dtype=tf.float32)
        self.label_ctcvr = tf.placeholder(tf.int64, [None], name="label_cvr")
        self.label_ctcvr = tf.cast(self.label_ctcvr, dtype=tf.float32)


        self.user_profile_basefea = tf.placeholder(tf.int64, [None, self.profile_size], name="user_profile_basefea")
        self.user_appcate1 = tf.sparse_placeholder(tf.int64, name="user_appcate1")
        self.user_appcate2 = tf.sparse_placeholder(tf.int64, name="user_appcate2")


        self.user_RecAnchor = tf.sparse_placeholder(tf.int64, name="user_RecAnchor")
        self.user_statIds = tf.placeholder(tf.int64, [None, self.user_statIds_size], name="user_statIds")
        self.user_topHourIds = tf.placeholder(tf.int64, [None, self.user_topHourIds_size], name="user_topHourIds")

        self.Wanchorids_long = tf.sparse_placeholder(tf.int64, name="Wanchorids_long")
        self.Wanchorids_long_dense = tf.sparse_tensor_to_dense(self.Wanchorids_long)
        self.Wanchorids_long_len_pre = tf.placeholder(tf.int64, [None], name="Wanchorids_long_len")      ####

        self.Wanchorids_short = tf.sparse_placeholder(tf.int64, name="Wanchorids_short")
        self.Wanchorids_short_dense = tf.sparse_tensor_to_dense(self.Wanchorids_short)
        self.Wanchorids_short_len_pre = tf.placeholder(tf.int64, [None], name="Wanchorids_short_len")     ####

        self.Wanchorids_effect = tf.sparse_placeholder(tf.int64, name="Wanchorids_effect")
        self.Wanchorids_effect_dense = tf.sparse_tensor_to_dense(self.Wanchorids_effect)
        self.Wanchorids_effect_len_pre = tf.placeholder(tf.int64, [None], name="Wanchorids_effect_len")    ####

        self.Wanchorids_noclick = tf.sparse_placeholder(tf.int64, name="Wanchorids_noclick")
        self.Wanchorids_noclick_dense = tf.sparse_tensor_to_dense(self.Wanchorids_noclick)
        self.Wanchorids_noclick_len_pre = tf.placeholder(tf.int64, [None], name="Wanchorids_noclick_len")   ####
        self.redict_weights_pre = tf.placeholder(tf.float32, [None, 8], name="redict_weights_pre")   


        self.anchor_profile_basefea = tf.placeholder(tf.int64, [None, self.profile_size - 1],
                                                     name="anchor_profile_basefea")
        self.anchor_appcate1 = tf.sparse_placeholder(tf.int64, name="anchor_appcate1")
        self.anchor_appcate2 = tf.sparse_placeholder(tf.int64, name="anchor_appcate2")
        self.anchorId = tf.placeholder(tf.int64, [None, 1], name="anchorId")


        self.anchor_stats = tf.placeholder(tf.int64, [None, self.anchor_stats_size], name="anchor_stats")
        self.anchor_stat_values = tf.placeholder(tf.float32, [None, self.anchor_stats_size], name="anchor_stat_values")
        self.day_ctr_seq = tf.sparse_placeholder(tf.float32, name="day_ctr_seq")
        self.day_cvr_seq = tf.sparse_placeholder(tf.float32, name="day_cvr_seq")
        self.day_ctrid_seq = tf.sparse_placeholder(tf.int64, name="day_ctrid_seq")
        self.day_cvrid_seq = tf.sparse_placeholder(tf.int64, name="day_cvrid_seq")
        self.day_ctr_seq_re = tf.sparse_tensor_to_dense(self.day_ctr_seq)
        self.day_cvr_seq_re = tf.sparse_tensor_to_dense(self.day_cvr_seq)
        self.day_ctrid_seq_re = tf.sparse_tensor_to_dense(self.day_ctrid_seq)
        self.day_cvrid_seq_re = tf.sparse_tensor_to_dense(self.day_cvrid_seq)
        self.ctcvr_seq_len = tf.placeholder(tf.int64, [None], name="ctcvr_seq_len")


        self.user_songTagids = tf.sparse_placeholder(tf.int64, name="user_songTagids")
        self.user_tagWeights = tf.sparse_placeholder(tf.float32, name="user_tagWeights")
        self.anchor_songTagids = tf.sparse_placeholder(tf.int64, name="anchor_songTagids")
        self.anchor_tagWeights = tf.sparse_placeholder(tf.float32, name="anchor_tagWeights")


        self.fea_sim = tf.placeholder(tf.float32, [None, 2], name="fea_sim")
        self.anchor_live_basefea = tf.placeholder(tf.int64, [None, self.live_size], name="anchor_live_basefea")
        self.realtime_ids = tf.placeholder(tf.int64, [None, self.realtime_values_size], name="realtime_ids")
        self.realtime_values = tf.placeholder(tf.float32, [None, self.realtime_values_size], name="realtime_values")
        self.hourId = tf.placeholder(tf.int64, [None, 1], name="hourId")   
        self.anchor_tagids = tf.sparse_placeholder(tf.int64, name="anchor_tagids")
        self.dayOfWeek = tf.placeholder(tf.int64, [None, 1], name="dayOfWeek")  
        self.live_position = tf.placeholder(tf.int64, [None, 1], name="live_position") 
        self.anchor_tagidonehot = tf.placeholder(tf.int64, [None, 38], name="anchor_tagidonehot")
        self.anchor_tagidonehot_float = tf.cast(self.anchor_tagidonehot, dtype=tf.float32)


        self.keep_prob = tf.placeholder(tf.float32, name="keep_prob")
        self.train_phase = tf.placeholder(tf.bool, name="train_phase")


        self.is_newAnchor  = tf.split(self.anchor_live_basefea, self.live_size, axis=1)
        self.newAnchor_index = tf.cast(tf.subtract(self.is_newAnchor[3], 428*tf.ones_like(self.is_newAnchor[3])), tf.bool)

        self.is_newUser = tf.split(self.user_profile_basefea, self.profile_size, axis=1)
        self.newUser_index = tf.cast(tf.subtract(self.is_newUser[7], 1172 * tf.ones_like(self.is_newUser[7])),tf.bool)










