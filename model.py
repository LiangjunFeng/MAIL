import tensorflow as tf
import sys
from util import *
from util2 import *
from tensorflow.contrib.layers.python.layers import batch_norm as batch_norm

def _bulid_dnn(X_input, deep_layers,l2_reg_lambda, batch_norm, train_phase,keep_prob,batch_norm_decay, scope):
    dnn_out=X_input
    with tf.variable_scope(scope+"_dnn" ):
        for i in range(len(deep_layers)):
            dnn_out = tf.contrib.layers.fully_connected(inputs=dnn_out, num_outputs=deep_layers[i], \
                                                        weights_regularizer=tf.contrib.layers.l2_regularizer(
                                                            l2_reg_lambda), scope='mlp%d' % i)

            if batch_norm:
                dnn_out = batch_norm_layer(dnn_out, train_phase=train_phase,
                                           scope_bn='bn_%d' % i,
                                           batch_norm_decay=batch_norm_decay) 

            dnn_out = tf.nn.dropout(dnn_out,keep_prob)  # Apply Dropout after all BN layers and set dropout=0.8(drop_ratio=0.2)
    return dnn_out

def _bulid_FM(x_concat, keep_prob):
    #input: N * F * K
    #output: N * K
    ## FM 的二阶交叉
    # ---------- second order term ---------------
    # sum_square part
    summed_features_emb = tf.reduce_sum(x_concat, 1)  # None * K
    summed_features_emb_square = tf.square(summed_features_emb)  # None * K

    # square_sum part
    squared_features_emb = tf.square(x_concat)
    squared_sum_features_emb = tf.reduce_sum(squared_features_emb, 1)  # None * K

    # second order
    y_second_order = 0.5 * tf.subtract(summed_features_emb_square,
                                           squared_sum_features_emb)  # None * K
    y_second_order = tf.nn.dropout(y_second_order, keep_prob)  # None * K

    return y_second_order


class Model_DSTN_SNPSHOT(DSTNEmbeddingLayerSnpShot):
    def __init__(self, config):
        super().__init__(config)

        self.l2_reg_lambda = config["l2_reg_lambda"]
        self.learning_rate = config["learning_rate"]

        self.batch_norm = config["batch_norm"]
        self.batch_norm_decay = config["batch_norm_decay"]
        # category  deep FM configuration
        self.deep_layers = config["deep_layers"]
        self.deep_layers_2 = config["deep_layers_2"]
        self.ctr_task_wgt = config["ctr_task_wgt"]
        self.freve_layers = config["freve_layers"]
        self.batch_size = config['batch_size']

        self.is_aux = config["is_aux"]

    def bulid_graph(self):
        print("building graph")
        with tf.variable_scope("discriminator"):
            FM_Input , FM_size, DNN_Input, DNN_size, self.rec_loss = self.build()

            out_FM = _bulid_FM(FM_Input, self.keep_prob)
            #out_cin=_build_extreme_FM(FM_Input, FM_size,self.embedding_size, self.cross_layer_sizes)
            # self-attention
            x_concat_attention = attention(FM_Input, 128, return_alphas=False)
            ## 用户序列 长序列
            # Item-to-Item Network
            with tf.name_scope('i2i_net_long'):
                att_outputs_long, alphas_long, scores_unnorm_long=DMR_DinAttention(self.anchor_fea, self.anchor_his_fea_long_v2, self.Wanchorids_long_len, self.position_his_eb_long, scope_bn='Dmri2i_long' )
                rel_i2i_long = tf.expand_dims(tf.reduce_sum(scores_unnorm_long, [1, 2]), -1)
                self.rel_i2i_long = rel_i2i_long
                self.scores_long = tf.reduce_sum(alphas_long, 1)
            # User-to-Item Network
            dm_item_vectors = tf.get_variable("dm_item_vectors", [self.anchorId_size, self.embedding_size])
            dm_item_biases = tf.get_variable('dm_item_biases', [self.anchorId_size], initializer=tf.zeros_initializer(), trainable=False)
            dm_item_vec = tf.nn.embedding_lookup(dm_item_vectors, self.anchorId)  # B,E
            dm_item_vec= tf.reshape(dm_item_vec, [-1, self.embedding_size])
            with tf.name_scope('u2i_net_long'):
                # Auxiliary Match Network
                if(self.is_aux):
                    self.aux_loss_long, dm_user_vector_long, scores_long = raw_deep_match(self.anchor_his_fea_long_v2,self.dm_position_his_eb_long,self.Wanchorids_long_len, self.Wanchorids_long_dense,
                                                                                    self.embedding_size, dm_item_vectors, dm_item_biases, self.anchorId_size,  scope_bn="u2i_aux_long")
                    self.aux_loss_long *= 0.1
                    rel_u2i_long = tf.reduce_sum(dm_user_vector_long * dm_item_vec, axis=-1, keep_dims=True)  # B,1

                else:
                    dm_user_vector_long,scores_long=  deep_match(self.anchor_his_fea_long_v2,self.dm_position_his_eb_long,self.Wanchorids_long_len, self.embedding_size,scope_bn='deepmatch_u2i_long')
                    rel_u2i_long = tf.reduce_sum(dm_user_vector_long * self.anchor_fea, axis=-1, keep_dims=True)  # B,1

                self.rel_u2i_long = rel_u2i_long
            ## 短序列
            with tf.name_scope('i2i_net_short'):
                att_outputs_short, alphas_short, scores_unnorm_short = DMR_DinAttention(self.anchor_fea,self.anchor_his_fea_short_v2,self.Wanchorids_short_len, self.position_his_eb_short, scope_bn='Dmri2i_short')
                rel_i2i_short = tf.expand_dims(tf.reduce_sum(scores_unnorm_short, [1, 2]), -1)
                self.rel_i2i_short = rel_i2i_short
                self.scores_short = tf.reduce_sum(alphas_short, 1)
            # User-to-Item Network
            with tf.name_scope('u2i_net_short'):
                if (self.is_aux):

                    self.aux_loss_short, dm_user_vector_short, scores_short = raw_deep_match(self.anchor_his_fea_short_v2, self.dm_position_his_eb_short,  self.Wanchorids_short_len, self.Wanchorids_short_dense,
                                                                                         self.embedding_size, dm_item_vectors, dm_item_biases,  self.anchorId_size,scope_bn="u2i_aux_short")
                    self.aux_loss_short *= 0.1
                    rel_u2i_short = tf.reduce_sum(dm_user_vector_short * dm_item_vec, axis=-1, keep_dims=True)  # B,1

                else:
                    dm_user_vector_short, scores_short = deep_match(self.anchor_his_fea_short_v2,
                                                                    self.dm_position_his_eb_short,
                                                                    self.Wanchorids_short_len, self.embedding_size,
                                                                    scope_bn='deepmatch_u2i_short')

                    rel_u2i_short = tf.reduce_sum(dm_user_vector_short * self.anchor_fea, axis=-1, keep_dims=True)  # B,1

                self.rel_u2i_short = rel_u2i_short
            ## 用户序列 纯有效观看序列
            # Item-to-Item Network
            with tf.name_scope('i2i_net_effect'):
                att_outputs_effect, alphas_effect, scores_unnorm_effect = DMR_DinAttention(self.anchor_fea,  self.anchor_his_fea_effect_v2,  self.Wanchorids_effect_len,   self.position_his_eb_effect,   scope_bn='Dmri2i_effect')
                rel_i2i_effect = tf.expand_dims(tf.reduce_sum(scores_unnorm_effect, [1, 2]), -1)
                self.rel_i2i_effect = rel_i2i_effect
                self.scores_effect = tf.reduce_sum(alphas_effect, 1)
            # User-to-Item Network
            with tf.name_scope('u2i_net_effect'):
                if (self.is_aux):

                    self.aux_loss_effect, dm_user_vector_effect, scores_effect = raw_deep_match(self.anchor_his_fea_effect_v2, self.dm_position_his_eb_effect, self.Wanchorids_effect_len,
                        self.Wanchorids_effect_dense,self.embedding_size, dm_item_vectors, dm_item_biases, self.anchorId_size, scope_bn="u2i_aux_effect")
                    self.aux_loss_effect *= 0.1
                    rel_u2i_effect = tf.reduce_sum(dm_user_vector_effect * dm_item_vec, axis=-1, keep_dims=True)  # B,1

                else:
                    dm_user_vector_effect, scores_effect = deep_match(self.anchor_his_fea_effect_v2,
                                                                      self.dm_position_his_eb_effect,
                                                                      self.Wanchorids_effect_len, self.embedding_size,
                                                                      scope_bn='deepmatch_u2i_effect')

                    rel_u2i_effect = tf.reduce_sum(dm_user_vector_effect * self.anchor_fea, axis=-1, keep_dims=True)  # B,1

                self.rel_u2i_effect = rel_u2i_effect
            # ## 纯曝光
            with tf.name_scope('i2i_net_noclick'):
                att_outputs_noclick, alphas_noclick, scores_unnorm_noclick = DMR_DinAttention(self.anchor_fea,  self.anchor_his_fea_noclick_v2,  self.Wanchorids_noclick_len,   self.position_his_eb_noclick,   scope_bn='Dmri2i_noclick')
                rel_i2i_noclick = tf.expand_dims(tf.reduce_sum(scores_unnorm_noclick, [1, 2]), -1)
                self.rel_i2i_noclick = rel_i2i_noclick
                self.scores_noclick = tf.reduce_sum(alphas_noclick, 1)
            # User-to-Item Network
            with tf.name_scope('u2i_net_noclick'):
                dm_user_vector_noclick, scores_noclick = deep_match(self.anchor_his_fea_noclick_v2, self.dm_position_his_eb_noclick,   self.Wanchorids_noclick_len, self.embedding_size,  scope_bn='deepmatch_u2i_noclick')
                dm_item_vec_noclick = tf.reshape(self.anchorid_embeddings, shape=[-1, self.embedding_size])
                rel_u2i_noclick = tf.reduce_sum(dm_user_vector_noclick * dm_item_vec_noclick, axis=-1, keep_dims=True)  # B,1
                self.rel_u2i_noclick = rel_u2i_noclick

            # ctcvr din attention
            with tf.name_scope('u2i_net_ctcvr'):
                ctcvr_attention= deep_match_2(self.day_ctcvr, self.ct_position_his_eb, self.ctcvr_seq_len)

            self.x_dmr= tf.concat([self.anchor_his_fea_sum_long,  self.anchor_fea* self.anchor_his_fea_sum_long,rel_u2i_long, dm_user_vector_long, rel_i2i_long,att_outputs_long,
                                   self.anchor_his_fea_sum_short, self.anchor_fea * self.anchor_his_fea_sum_short, rel_u2i_short, dm_user_vector_short, rel_i2i_short, att_outputs_short,
                                   self.anchor_his_fea_sum_effect, self.anchor_fea * self.anchor_his_fea_sum_effect, rel_u2i_effect, dm_user_vector_effect, rel_i2i_effect, att_outputs_effect,
                                   self.anchor_his_fea_sum_noclick, self.anchor_fea * self.anchor_his_fea_sum_noclick,
                                   rel_u2i_noclick, dm_user_vector_noclick, rel_i2i_noclick, att_outputs_noclick,
                                   ctcvr_attention , self.ctcvr_sum ], axis=-1)

            ## 一阶特征
            x_first_order = tf.concat([x_concat_attention, self.fea_sim, self.redict_weights, self.anchor_tagidonehot_float], axis=1)
            x_for_sigmoid = tf.concat([self.realtime_value_dense, self.hour_embeddings, self.Day_embeddings], axis=1)
            x_for_sigmoid = tf.reshape(x_for_sigmoid,
                                       [-1, (self.realtime_values_size + 2) * self.embedding_size])


            DNN_Input= tf.concat([DNN_Input, self.x_dmr, self.redict_weights], axis=-1)
            x_concat_deep = tf.nn.dropout(DNN_Input, self.keep_prob)
            with tf.name_scope("CVR_Task"):
                x_cvr = x_concat_deep
                x_cvr = _bulid_dnn(x_cvr, self.deep_layers, self.l2_reg_lambda, self.batch_norm, self.train_phase,
                                   self.keep_prob, self.batch_norm_decay, scope="cvr0")

                x_cvr = tf.concat([x_cvr, out_FM, x_first_order, x_for_sigmoid, rel_i2i_long, rel_u2i_long, att_outputs_long, rel_i2i_short, rel_u2i_short, att_outputs_short,rel_i2i_effect, rel_u2i_effect, att_outputs_effect, rel_i2i_noclick, rel_u2i_noclick, att_outputs_noclick], axis=-1)

                x_cvr = _bulid_dnn(x_cvr, self.deep_layers_2, self.l2_reg_lambda, self.batch_norm, self.train_phase,
                                   self.keep_prob,self.batch_norm_decay, scope="cvr1")

                y_cvr = tf.contrib.layers.fully_connected(inputs=x_cvr, num_outputs=1, activation_fn=tf.identity, \
                                                          weights_regularizer=tf.contrib.layers.l2_regularizer(
                                                              self.l2_reg_lambda), scope='cvr_out')
                y_cvr = tf.reshape(y_cvr, shape=[-1])

            with tf.name_scope("CTR_Task"):
                x_ctr = x_concat_deep
                x_ctr = _bulid_dnn(x_ctr, self.deep_layers, self.l2_reg_lambda, self.batch_norm, self.train_phase,
                                   self.keep_prob, self.batch_norm_decay, scope="ctr0")

                x_ctr = tf.concat([x_ctr, out_FM, x_first_order, x_for_sigmoid, rel_i2i_long, rel_u2i_long, att_outputs_long, rel_i2i_short, rel_u2i_short, att_outputs_short,rel_i2i_effect, rel_u2i_effect, att_outputs_effect, rel_i2i_noclick, rel_u2i_noclick, att_outputs_noclick], axis=-1)

                x_ctr = _bulid_dnn(x_ctr, self.deep_layers_2, self.l2_reg_lambda, self.batch_norm, self.train_phase,
                                   self.keep_prob, self.batch_norm_decay,  scope="ctr1")

                y_ctr = tf.contrib.layers.fully_connected(inputs=x_ctr, num_outputs=1, activation_fn=tf.identity, \
                                                          weights_regularizer=tf.contrib.layers.l2_regularizer(
                                                              self.l2_reg_lambda), scope='ctr_out')
                y_ctr = tf.reshape(y_ctr, shape=[-1])

            with tf.variable_scope("MTL-Layer"):
                self.pctr = tf.sigmoid(y_ctr, name="p_ctr")
                self.pcvr = tf.sigmoid(y_cvr, name="p_cvr")
                self.pctcvr = self.pctr * self.pcvr

            # ------bulid loss------
            ctr_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y_ctr, labels=self.label_ctr))
            cvr_loss = tf.reduce_mean(tf.losses.log_loss(predictions=self.pctcvr, labels=self.label_ctcvr))
            if (self.is_aux):
                self.loss = self.ctr_task_wgt * ctr_loss + (1 - self.ctr_task_wgt) * cvr_loss + 0.5 * (
                            self.aux_loss_long + self.aux_loss_short) + 0.5 * self.aux_loss_effect
            else:
                self.loss = self.ctr_task_wgt * ctr_loss + (1 - self.ctr_task_wgt) * cvr_loss

        self.params1 = [param for param in tf.trainable_variables() if 'user_ZSL' not in param.name]
        d_optimizer1 = tf.train.AdamOptimizer(self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
        self.global_step1 = tf.Variable(0, name="global_step1", trainable=False)
        grads_and_vars1 = d_optimizer1.compute_gradients(self.loss, self.params1, aggregation_method=2)
        self.train_op1 = d_optimizer1.apply_gradients(grads_and_vars1, global_step=self.global_step1)

        self.params2 = [param for param in tf.trainable_variables() if 'user_ZSL' in param.name]
        d_optimizer2 = tf.train.AdamOptimizer(0.5*self.learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8)
        self.global_step2 = tf.Variable(0, name="global_step2", trainable=False)
        grads_and_vars2 = d_optimizer2.compute_gradients(self.rec_loss, self.params2, aggregation_method=2)
        self.train_op2 = d_optimizer2.apply_gradients(grads_and_vars2, global_step=self.global_step2)

        print("graph built successfully!")











































