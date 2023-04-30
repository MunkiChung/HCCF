import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from mvecf.ImplicitCF import ImplicitCF
import pandas as pd
from mvecf.load import get_data
import numpy as np
from Params import args
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
import Utils.NNLayers as NNs
from Utils.NNLayers import FC, Regularize, Activate, Dropout, Bias, getParam, defineParam, defineRandomNameParam
from DataHandler import negSamp, transpose, DataHandler, transToLsts
import tensorflow as tf
from tensorflow.core.protobuf import config_pb2
import pickle
from numba import cuda


def save_pickle(data, name):
    with open(name, "wb") as f:
        pickle.dump(data, f)
    f.close()


class Recommender:
    def __init__(self, sess, handler, save_path, data_sampler: ImplicitCF):
        self.sess = sess
        self.handler = handler
        self.save_path = save_path
        self.data_sampler = data_sampler
        print('USER', args.user, 'ITEM', args.item)
        self.metrics = dict()
        mets = ['Loss', 'preLoss', 'Recall', 'NDCG']
        for met in mets:
            self.metrics['Train' + met] = list()
            self.metrics['Test' + met] = list()
        self.list_stop_cri = []

    def makePrint(self, name, ep, reses, save):
        ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
        for metric in reses:
            val = reses[metric]
            ret += '%s = %.4f, ' % (metric, val)
            tem = name + metric
            if save and tem in self.metrics:
                self.metrics[tem].append(val)
        ret = ret[:-2] + '  '
        return ret

    def run(self):
        self.prepareModel()
        log('Model Prepared')
        ep = 0
        if args.load_model != None:
            self.loadModel()
            stloc = len(self.metrics['TrainLoss']) * args.tstEpoch - (args.tstEpoch - 1)
        else:
            stloc = 0
            init = tf.compat.v1.global_variables_initializer()
            self.sess.run(init)
            log('Variables Inited')
        for ep in range(stloc, args.epoch):
            test = (ep % args.tstEpoch == 0)
            reses = self.trainEpoch()
            log(self.makePrint('Train', ep, reses, test))
            if test:
                reses = self.testEpoch(ep)
                log(self.makePrint('Test', ep, reses, test))
            if ep % args.tstEpoch == 0:
                self.saveHistory()
            print()
            # if self.check_early_stop():
            #     break
        reses = self.testEpoch(ep)
        log(self.makePrint('Test', args.epoch, reses, True))
        self.saveHistory()

    def messagePropagate(self, lats, adj):
        return Activate(tf.compat.v1.sparse.sparse_dense_matmul(adj, lats), self.actFunc)

    def hyperPropagate(self, lats, adj):
        lat1 = Activate(tf.compat.v1.transpose(adj) @ lats, self.actFunc)
        lat2 = tf.compat.v1.transpose(FC(tf.compat.v1.transpose(lat1), args.hyperNum, activation=self.actFunc)) + lat1
        lat3 = tf.compat.v1.transpose(FC(tf.compat.v1.transpose(lat2), args.hyperNum, activation=self.actFunc)) + lat2
        lat4 = tf.compat.v1.transpose(FC(tf.compat.v1.transpose(lat3), args.hyperNum, activation=self.actFunc)) + lat3
        ret = Activate(adj @ lat4, self.actFunc)
        # ret = adj @ lat4
        return ret

    def edgeDropout(self, mat):
        def dropOneMat(mat):
            indices = mat.indices
            values = mat.values
            shape = mat.dense_shape
            # newVals = tf.compat.v1.to_float(tf.compat.v1.sign(tf.compat.v1.nn.dropout(values, self.keepRate)))
            newVals = tf.compat.v1.nn.dropout(values, self.keepRate)
            return tf.compat.v1.sparse.SparseTensor(indices, newVals, shape)

        return dropOneMat(mat)

    def ours(self):
        uEmbed0 = NNs.defineParam('uEmbed0', [args.user, args.latdim], reg=True)
        iEmbed0 = NNs.defineParam('iEmbed0', [args.item, args.latdim], reg=True)
        uhyper = NNs.defineParam('uhyper', [args.latdim, args.hyperNum], reg=True)
        ihyper = NNs.defineParam('ihyper', [args.latdim, args.hyperNum], reg=True)
        uuHyper = (uEmbed0 @ uhyper)
        iiHyper = (iEmbed0 @ ihyper)

        ulats = [uEmbed0]
        ilats = [iEmbed0]
        gnnULats = []
        gnnILats = []
        hyperULats = []
        hyperILats = []
        for i in range(args.gnn_layer):
            ulat = self.messagePropagate(ilats[-1], self.edgeDropout(self.adj))
            ilat = self.messagePropagate(ulats[-1], self.edgeDropout(self.tpAdj))
            hyperULat = self.hyperPropagate(ulats[-1], tf.compat.v1.nn.dropout(uuHyper, self.keepRate))
            hyperILat = self.hyperPropagate(ilats[-1], tf.compat.v1.nn.dropout(iiHyper, self.keepRate))
            gnnULats.append(ulat)
            gnnILats.append(ilat)
            hyperULats.append(hyperULat)
            hyperILats.append(hyperILat)
            ulats.append(ulat + hyperULat + ulats[-1])
            ilats.append(ilat + hyperILat + ilats[-1])
        ulat = tf.compat.v1.add_n(ulats)
        ilat = tf.compat.v1.add_n(ilats)

        pckUlat = tf.compat.v1.nn.embedding_lookup(ulat, self.uids)
        pckIlat = tf.compat.v1.nn.embedding_lookup(ilat, self.iids)
        preds = tf.compat.v1.reduce_sum(pckUlat * pckIlat, axis=-1)

        def calcSSL(hyperLat, gnnLat):
            posScore = tf.compat.v1.exp(tf.compat.v1.reduce_sum(hyperLat * gnnLat, axis=1) / args.temp)
            negScore = tf.compat.v1.reduce_sum(tf.compat.v1.exp(gnnLat @ tf.compat.v1.transpose(hyperLat) / args.temp),
                                               axis=1)
            uLoss = tf.compat.v1.reduce_sum(-tf.compat.v1.log(posScore / (negScore + 1e-8) + 1e-8))
            return uLoss

        sslloss = 0
        uniqUids, _ = tf.compat.v1.unique(self.uids)
        uniqIids, _ = tf.compat.v1.unique(self.iids)
        for i in range(len(hyperULats)):
            W = NNs.defineRandomNameParam([args.latdim, args.latdim])
            pckHyperULat = tf.compat.v1.nn.l2_normalize(tf.compat.v1.nn.embedding_lookup(hyperULats[i], uniqUids),
                                                        axis=1) @ W  # tf.compat.v1.nn.l2_normalize(, axis=1)
            pckGnnULat = tf.compat.v1.nn.l2_normalize(tf.compat.v1.nn.embedding_lookup(gnnULats[i], uniqUids),
                                                      axis=1)  # tf.compat.v1.nn.l2_normalize(, axis=1)
            pckhyperILat = tf.compat.v1.nn.l2_normalize(tf.compat.v1.nn.embedding_lookup(hyperILats[i], uniqIids),
                                                        axis=1) @ W  # tf.compat.v1.nn.l2_normalize(, axis=1)
            pckGNNILat = tf.compat.v1.nn.l2_normalize(tf.compat.v1.nn.embedding_lookup(gnnILats[i], uniqIids),
                                                      axis=1)  # tf.compat.v1.nn.l2_normalize(, axis=1)
            uLoss = calcSSL(pckHyperULat, pckGnnULat)
            iLoss = calcSSL(pckhyperILat, pckGNNILat)
            sslloss += uLoss + iLoss

        return preds, sslloss, ulat, ilat

    def tstPred(self, ulat, ilat):
        pckUlat = tf.compat.v1.nn.embedding_lookup(ulat, self.uids)
        allPreds = pckUlat @ tf.compat.v1.transpose(ilat)
        allPreds = allPreds * (1 - self.trnPosMask) - self.trnPosMask * 1e8
        vals, locs = tf.compat.v1.nn.top_k(allPreds, args.shoot)
        return locs, allPreds

    def prepareModel(self):
        self.keepRate = tf.compat.v1.placeholder(dtype=tf.compat.v1.float32, shape=[])
        NNs.leaky = args.leaky
        self.actFunc = 'leakyRelu'
        adj = self.handler.trnMat
        idx, data, shape = transToLsts(adj, norm=True)
        self.adj = tf.compat.v1.sparse.SparseTensor(idx, data, shape)
        idx, data, shape = transToLsts(transpose(adj), norm=True)
        self.tpAdj = tf.compat.v1.sparse.SparseTensor(idx, data, shape)
        self.uids = tf.compat.v1.placeholder(name='uids', dtype=tf.compat.v1.int32, shape=[None])
        self.iids = tf.compat.v1.placeholder(name='iids', dtype=tf.compat.v1.int32, shape=[None])
        self.trnPosMask = tf.compat.v1.placeholder(name='trnPosMask', dtype=tf.compat.v1.float32,
                                                   shape=[None, args.item])

        self.preds, sslloss, ulat, ilat = self.ours()
        self.topLocs, self.allPreds = self.tstPred(ulat, ilat)

        sampNum = tf.compat.v1.shape(self.uids)[0] // 2
        posPred = tf.compat.v1.slice(self.preds, [0], [sampNum])
        negPred = tf.compat.v1.slice(self.preds, [sampNum], [-1])
        self.preLoss = tf.compat.v1.reduce_sum(tf.compat.v1.maximum(0.0, 1.0 - (posPred - negPred))) / args.batch
        self.regLoss = args.reg * Regularize() + args.ssl_reg * sslloss
        self.loss = self.preLoss + self.regLoss

        globalStep = tf.compat.v1.Variable(0, trainable=False)
        learningRate = tf.compat.v1.train.exponential_decay(args.lr, globalStep, args.decay_step, args.decay,
                                                            staircase=True)
        self.optimizer = tf.compat.v1.train.AdamOptimizer(learningRate).minimize(self.loss, global_step=globalStep)

    def sampleTrainBatch(self, batIds, labelMat):
        temLabel = labelMat[batIds].toarray()
        batch = len(batIds)
        temlen = batch * 2 * args.sampNum
        uLocs = [None] * temlen
        iLocs = [None] * temlen
        cur = 0
        for i in range(batch):
            posset = np.reshape(np.argwhere(temLabel[i] != 0), [-1])
            sampNum = min(args.sampNum, len(posset))
            if sampNum == 0:
                poslocs = [np.random.choice(args.item)]
                neglocs = [poslocs[0]]
            else:
                _, poslocs, neglocs = self.data_sampler.data_loader(sampNum)
            for j in range(sampNum):
                posloc = poslocs[j]
                negloc = neglocs[j]
                uLocs[cur] = uLocs[cur + temlen // 2] = batIds[i]
                iLocs[cur] = posloc
                iLocs[cur + temlen // 2] = negloc
                cur += 1
        uLocs = uLocs[:cur] + uLocs[temlen // 2: temlen // 2 + cur]
        iLocs = iLocs[:cur] + iLocs[temlen // 2: temlen // 2 + cur]
        return uLocs, iLocs

    def trainEpoch(self):
        num = args.user
        sfIds = np.random.permutation(num)[:args.trnNum]
        epochLoss, epochPreLoss = [0] * 2
        num = len(sfIds)
        steps = int(np.ceil(num / args.batch))

        for i in range(steps):
            st = i * args.batch
            ed = min((i + 1) * args.batch, num)
            batIds = sfIds[st: ed]

            target = [self.optimizer, self.preLoss, self.regLoss, self.loss]
            feed_dict = {}
            uLocs, iLocs = self.sampleTrainBatch(batIds, self.handler.trnMat)
            feed_dict[self.uids] = uLocs
            feed_dict[self.iids] = iLocs
            feed_dict[self.keepRate] = args.keepRate

            res = self.sess.run(target, feed_dict=feed_dict,
                                options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))

            preLoss, regLoss, loss = res[1:]

            epochLoss += loss
            epochPreLoss += preLoss
            log('Step %d/%d: loss = %.2f, regLoss = %.2f         ' % (i, steps, loss, regLoss), save=False,
                oneline=True)
        ret = dict()
        ret['Loss'] = epochLoss / steps
        ret['preLoss'] = epochPreLoss / steps
        return ret

    def testEpoch(self, epoch):
        epochRecall, epochNdcg = [0] * 2
        ids = self.handler.tstUsrs
        num = len(ids)
        tstBat = num
        steps = int(np.ceil(num / tstBat))
        tstNum = 0
        for i in range(steps):
            st = i * tstBat
            ed = min((i + 1) * tstBat, num)
            batIds = ids[st: ed]
            feed_dict = {}

            trnPosMask = self.handler.trnMat[batIds].toarray()
            feed_dict[self.uids] = batIds
            feed_dict[self.trnPosMask] = trnPosMask
            feed_dict[self.keepRate] = 1.0
            topLocs = self.sess.run(self.topLocs, feed_dict=feed_dict,
                                    options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))
            allPreds = self.sess.run(self.allPreds, feed_dict=feed_dict,
                                     options=config_pb2.RunOptions(report_tensor_allocations_upon_oom=True))

            save_pickle(allPreds, os.path.join(self.save_path, f"epoch_{epoch}.pkl"))
            recall, ndcg = self.calcRes(topLocs, self.handler.tstLocs, batIds)
            epochRecall += recall
            epochNdcg += ndcg
            log('Steps %d/%d: recall = %.2f, ndcg = %.2f          ' % (i, steps, recall, ndcg), save=False,
                oneline=True)
        self.list_stop_cri.append(-epochRecall)
        ret = dict()
        ret['Recall'] = epochRecall / num
        ret['NDCG'] = epochNdcg / num
        return ret

    def check_early_stop(self):
        list_stop_cri = self.list_stop_cri[-10:]
        if len(list_stop_cri) >= 10 and (
                list_stop_cri[0] <= min(list_stop_cri) or np.all(max(abs(np.diff(list_stop_cri))) < 0)
        ):
            return True
        return False

    def calcRes(self, topLocs, tstLocs, batIds):
        assert topLocs.shape[0] == len(batIds)
        allRecall = allNdcg = 0
        recallBig = 0
        ndcgBig = 0
        for i in range(len(batIds)):
            temTopLocs = list(topLocs[i])
            temTstLocs = tstLocs[batIds[i]]
            tstNum = len(temTstLocs)
            maxDcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(tstNum, args.shoot))])
            recall = dcg = 0
            for val in temTstLocs:
                if val in temTopLocs:
                    recall += 1
                    dcg += np.reciprocal(np.log2(temTopLocs.index(val) + 2))
            recall = recall / tstNum
            ndcg = dcg / maxDcg
            allRecall += recall
            allNdcg += ndcg
        return allRecall, allNdcg

    def saveHistory(self):
        if args.epoch == 0:
            return
        with open('History/' + args.save_path + '.his', 'wb') as fs:
            pickle.dump(self.metrics, fs)

        saver = tf.compat.v1.train.Saver()
        saver.save(self.sess, 'Models/' + args.save_path)
        log('Model Saved: %s' % args.save_path)

    def loadModel(self):
        saver = tf.compat.v1.train.Saver()
        saver.restore(sess, 'Models/' + args.load_model)
        with open('History/' + args.load_model + '.his', 'rb') as fs:
            self.metrics = pickle.load(fs)
        log('Model Loaded')


if __name__ == '__main__':
    import time

    data_type = args.data_type
    target_year = args.target_year
    positive_score_cri = args.positive_score_cri
    save_path = f"./{data_type}/{target_year}/HCCF"
    # save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    logger.saveDefault = True
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True

    log('Start')
    handler = DataHandler(data_type, target_year)
    handler.LoadData()
    log('Load Data')

    holdings_data, factor_params = get_data(data_type, target_year)
    train = pd.DataFrame(holdings_data["train_data"].T, columns=["userID", "itemID", "rating"])
    test = pd.DataFrame(holdings_data["valid_data"].T, columns=["userID", "itemID", "rating"])
    reg_param_mv = 10
    gamma = 3
    alpha = 10

    data_sampler = ImplicitCF(train=train, test=test, alpha=alpha,
                          factor_params=factor_params, reg_param_mv=reg_param_mv, gamma=gamma,
                          positive_score_cri=positive_score_cri)
    with tf.compat.v1.Session(config=config) as sess:
        recom = Recommender(sess, handler, save_path, data_sampler=data_sampler)
        recom.run()
        sess.close()

    time.sleep(1)
    device = cuda.get_current_device()
    device.reset()
