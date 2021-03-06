#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : Classifier.py
# @Author: Sui Huafeng
# @Date  : 2018/4
# @Desc  : 从$DATA/input中所有样本文件中训练一个日志文件分类器
#

import os
import re
import shutil
import time
from collections import Counter

import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from scipy.signal import argrelextrema
from sklearn.cluster import KMeans
from sklearn.externals import joblib

from config import Workspaces as G
from utilites import Util


# 对数十、上百万个文件进行聚类，形成聚类模型
class Classifier(object):
    """
    新建对象或者调用trainModel方法，可以生成Classifier模型
    调用predict方法，可以预测新的日志文件类型及其置信度
    $DATA/models/l1file_info.csv：记录原始样本文件信息(暂时不要？）
    $DATA/l1cache/: 存储各样本文件。目录结构就是被管服务器原始结构
    """
    __corpusCacheFile = os.path.join(G.projectModelPath, 'corpuscache.1')
    l1_dbf = os.path.join(G.projectModelPath, 'metadata.1')
    __MaxLines = G.cfg.getint('Classifier', 'MaxLines')

    def __init__(self, model_file=''):
        self.model_file = model_file
        self.model_id = 0 if model_file == G.productFileClassifierModel else 1
        self.common_filenames, self.l1_structure = ([], [])
        self.ruleSet = None  # 处理文件正则表达式
        self.statsScope = None  # 样本文件字符数、字数统计值(均值、标准差、中位数)的最小-最大值范围
        self.dictionary = None  # 字典对象(Gensim Dictionary)
        self.model = None  # 聚类模型(Kmeans)
        self.categories = None  # 聚类的类型(名称，数量占比，分位点距离，边界点距离)

        if os.path.exists(model_file):  # 从模型文件装载模型
            self.ruleSet, self.dictionary, self.statsScope, self.model, self.categories = joblib.load(model_file)
        else:
            G.log.warning('No model loaded!')

    # 重新训练模型
    def reCluster(self):
        for folder in [G.l1_cache, G.l2_cache, G.outputs]:
            if os.path.exists(folder):
                shutil.rmtree(folder)
        time.sleep(3)
        for folder in [G.l1_cache, G.l2_cache, G.outputs]:
            os.mkdir(folder)

        common_files, file2merged = Util.mergeFilesByName(G.l0_inputs, G.l1_cache)
        self.__dbFilesSampled(file2merged)
        results = self.trainModel(k_=35)
        self.__saveModel()  # model saved to file

        Util.clearModel(self.model_id)
        self.dbUpdCategories()

        db = Util.dbConnect()
        if not db:
            return
        cursor = db.cursor()
        classified_common_files, unclassified_common_files = self.splitResults(results)
        Util.dbFilesMerged(cursor, file2merged, classified_common_files, unclassified_common_files)
        Util.mergeFilesByClass(cursor, G.l2_cache)  # 同类文件合并到to_目录下
        wildcard_log_files = Util.genGatherList(cursor)  # 生成采集文件列表
        Util.dbWildcardLogFiles(cursor, wildcard_log_files)
        db.commit()
        db.close()

    def __dbFilesSampled(self, file2merged):
        db = Util.dbConnect()
        if not db:
            return
        cursor = db.cursor()
        for file_fullname, anchor_name, anchor_colRange, common_file_fullname in file2merged:
            file_fullname = file_fullname.replace('\\', '/')
            host = file_fullname[len(G.l0_inputs):].strip('/')
            host, filename = host.split('/', 1)
            archive_path, filename = os.path.split(filename)
            remote_path = '/' + archive_path if archive_path[1] != '_' else archive_path[0] + ':' + archive_path[2:]
            host = '"%s"' % host.strip('/')
            archive_path = '"%s"' % archive_path
            remote_path = '"%s"' % remote_path
            file_fullname = '"%s"' % file_fullname
            filename = '"%s"' % filename
            sql = 'INSERT INTO files_sampled (file_fullname,host,archive_path,filename,remote_path) VALUES(%s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE file_fullname=%s' % (
                file_fullname, host, archive_path, filename, remote_path, file_fullname)
            cursor.execute(sql)
        db.commit()
        db.close()

    def __iter__(self):
        self.category_id = 0
        return self

    # 返回类别的(名称，数量占比，分位点到中心距离，边界到分位点距离)
    def __next__(self):
        i = self.category_id
        if i >= len(self.categories[0]):
            raise StopIteration
        self.category_id += 1
        return self.categories[0][i], self.categories[1][i], self.categories[2][i], self.categories[3][i]

    def __len__(self):
        return len(self.categories[0])

    def __getitem__(self, item):
        if item < -len(self.categories[0]) or item >= len(self.categories[0]):
            raise IndexError
        return self.categories[0][item], self.categories[1][item], self.categories[2][item], self.categories[3][item]

    def __setitem__(self, key, value):
        if key < -len(self.categories[0]) or key >= len(self.categories[0]):
            raise IndexError
        name = str(value)
        if name in self.categories[0]:
            raise ValueError
        self.categories[0][key] = name
        self.dbUpdCategories()

    def dbUpdCategories(self):
        db = Util.dbConnect()
        if db:  # Can't connect ro db, waiting and retry forever
            cursor = db.cursor()
            c = self.categories
            for category_id, (name, percent, boundary, quantile) in enumerate(zip(c[0], c[1], c[2], c[3])):
                name = '"%s"' % name
                sql = 'INSERT INTO file_class (model_id, category_id, name, quantile, boundary, percent) VALUES(%d, %d, %s, %e,%e, %f) ON DUPLICATE KEY UPDATE name=%s,quantile=%e,boundary=%e,percent=%f' % (
                    self.model_id, category_id, name, quantile, boundary, percent, name, quantile, boundary, percent)
                cursor.execute(sql)
            db.commit()
        db.close()

    # 训练、生成模型并保存在$models/xxx.mdl中,dataset:绝对/相对路径样本文件名，或者可迭代样本字符流
    def trainModel(self, dataset_path=G.l1_cache, k_=0):
        """
        Train and generate K-Means Model
        :param dataset_path: source path contains merged log files, or iterable char stream
        :param k_: K-means parameter, 0 means auto detect
        """
        rule_sets = []  # 文本处理的替换、停用词和k-shingle规则
        for ruleset_name in sorted([section for section in G.cfg.sections() if section.split('-')[0] == 'RuleSet']):
            replace_rules, stop_words, k_list = [], [], []
            for key, value in G.cfg.items(ruleset_name):
                if key == 'stopwords':
                    stop_words = value.split(',')
                elif key == 'k-shingles':
                    k_list = eval(value)
                else:
                    replace_from, replace_to = value.split('TO')
                    replace_rules.append((re.compile(replace_from.strip(), re.I), replace_to.strip()))
            rule_sets.append((ruleset_name, replace_rules, stop_words, k_list))
        # 尝试不同的向量化规则，确定聚类数量K
        for self.ruleSet in rule_sets:
            corpus_fp = self.__buildDictionary(dataset_path)  # 建立字典，返回文档结构信息
            if len(self.dictionary) < G.cfg.getint('Classifier', 'LeastTokens'):  # 字典字数太少,重新采样
                corpus_fp.close()
                self.__clearCache()
                G.log.info('Too few tokens[%d], Re-sample with next RuleSet.', len(self.dictionary))
                continue
            corpus_fp.seek(0)
            vectors = self.__buildVectors(corpus_fp, self.dictionary.num_docs)  # 建立稀疏矩阵doc*(dct + stats)
            corpus_fp.close()  # 关闭缓存文件

            #            start_k = self.__findStartK(vectors)  # 快速定位符合分布相对均衡的起点K
            #            if start_k is None:  # 聚类不均衡，换rule set重新采样
            #                continue

            start_k = min(50, int(vectors.shape[0] / 100))
            k_ = k_ if k_ else self.__pilotClustering(vectors, start_k)  # 多个K值试聚类，返回最佳K
            if k_ != 0:  # 找到合适的K，跳出循环
                break
            self.__clearCache()  # 清除缓存的ruleset
        else:
            raise UserWarning('Cannot generate qualified corpus by all RuleSets')

        # 重新聚类, 得到模型(向量数、中心点和距离）和分类(向量-所属类)
        self.model, percents, boundaries, quantiles = self.__buildModel(k_, vectors)
        names = ['fc%d' % i for i in range(len(percents))]
        self.categories = [names, percents, boundaries, quantiles]
        results = self.__getResult(vectors)
        return results

    # 建立词典，同时缓存词表文件
    def __buildDictionary(self, new_dataset_path):
        self.dictionary = Dictionary()

        # 装载处理过的缓存语料
        cache_fp = open(self.__corpusCacheFile, mode='a+t', encoding='utf-8')  # 创建或打开语料缓存文件
        if cache_fp.tell() != 0:
            if os.path.exists(self.l1_dbf):
                self.ruleSet, self.common_filenames, self.l1_structure, self.statsScope = joblib.load(self.l1_dbf)
            cache_fp.seek(0)
            cached_documents = len(self.common_filenames)
            for lines, line_ in enumerate(cache_fp):
                if lines < cached_documents:
                    self.dictionary.add_documents([line_.split()])
            G.log.info('%d cached documents loaded.', lines)

        # 继续处理新增语料
        for document in self.__buildDocument(new_dataset_path):
            self.dictionary.add_documents([document])
            cache_fp.write(' '.join([word for word in document]) + '\n')

        if self.dictionary.num_docs < G.cfg.getint('Classifier', 'LeastFiles'):  # 字典字数太少或文档数太少，没必要聚类
            cache_fp.close()
            self.__clearCache()
            raise UserWarning('Too few documents[%d] to clustering' % self.dictionary.num_docs)

        # 去掉低频词，压缩字典
        num_token = len(self.dictionary)
        no_below = int(min(G.cfg.getfloat('Classifier', 'NoBelow'), int(self.dictionary.num_docs / 50)))
        self.dictionary.filter_extremes(no_below=no_below, no_above=0.999, keep_n=G.cfg.getint('Classifier', 'KeepN'))
        self.dictionary.compactify()
        G.log.info('Dictionary built with [%s](%d tokens, reduced from %d), from %d files( %d words)',
                   self.ruleSet[0], len(self.dictionary), num_token, self.dictionary.num_docs, self.dictionary.num_pos)

        statistics = np.array(self.l1_structure)[:, 1:7]
        statistics[statistics > 500] = 500  # 防止异常大的数干扰效果
        self.statsScope = np.min(statistics, axis=0), np.max(statistics, axis=0)
        joblib.dump((self.ruleSet, self.common_filenames, self.l1_structure, self.statsScope),
                    self.l1_dbf)  # 保存模型，供后续使用

        return cache_fp

    # 预处理，迭代方式返回某个文件的词表.
    def __buildDocument(self, dataset_path):
        amount_files, failed_files, file_fullname = 0, 0, ''
        G.log.info('Start Converting documents from ' + dataset_path)

        processed_files = os.path.join(G.projectModelPath, 'buildDocument.dbf')
        processed = [] if not os.path.exists(processed_files) else joblib.load(processed_files)

        for dir_path, dir_names, file_names in os.walk(dataset_path):
            for file_name in file_names:
                try:
                    file_fullname = os.path.join(dir_path, file_name)
                    if file_fullname in processed:
                        continue
                    amount_files += 1
                    if amount_files % 50 == 0:
                        G.log.info('Converted %d[%d failed] files:\t%s', amount_files, failed_files, file_fullname)
                    processed.append(file_fullname)
                    yield self.__file2doc(file_fullname)
                except Exception as err:
                    failed_files += 1
                    G.log.warning('Failed to convert\t%s, ignored.\t%s', file_fullname, str(err))
                    continue
        joblib.dump(processed, processed_files)
        G.log.info('Converted %d files,%d failed', amount_files, failed_files)
        raise StopIteration()

    # 使用规则集匹配和转换后，转化为词表
    def __file2doc(self, file_fullname, encoding='utf-8'):
        document = []
        line_idx, lc, lw = 0, [], []

        G.log.debug('Converting ' + file_fullname)
        for line_idx, line in enumerate(open(file_fullname, 'r', encoding=encoding)):
            words = G.getWords(line, rule_set=self.ruleSet)
            document += words  # 生成词表
            lc.append(len(line))
            lw.append(len(words))
            if line_idx > self.__MaxLines:
                break
        line_idx += 1

        # 计算统计数据
        subtotal_chars = list(np.histogram(np.array(lc), bins=[0, 40, 80, 120, 160, 200, 1000])[0] / line_idx)
        subtotal_words = list(np.histogram(np.array(lw), bins=[0, 4, 8, 12, 16, 20, 100])[0] / line_idx)
        stats = [np.mean(lc), np.mean(lw), np.std(lc), np.std(lw), np.median(lc), np.median(lw)]
        doc_structure = [line_idx] + stats + subtotal_chars + subtotal_words
        # 汇总和保持元数据
        self.common_filenames.append(file_fullname)
        self.l1_structure.append(doc_structure)
        return document

    # 从词表和文档结构形成聚类向量
    def __buildVectors(self, corpus, rows):
        cols = len(self.dictionary)

        # 构造tf-idf词袋和文档向量
        tfidf_model = TfidfModel(dictionary=self.dictionary, normalize=True)
        vectors = np.zeros((rows, cols))
        for doc_idx, document in enumerate(corpus):
            if type(document) == str:
                document = document.split()
            for (word_idx, tf_idf_value) in tfidf_model[self.dictionary.doc2bow(document)]:
                vectors[doc_idx, word_idx] = tf_idf_value  # tfidf词表加入向量

        # 按每个文档的行数对tfidf向量进行标准化，保证文档之间的可比性
        l1_fd = np.array(self.l1_structure)[-rows:, :]  # [[行数, 均值/标准差/中位数,字节和字数的12个分段数量比例]]

        lines = l1_fd[:, 0:1]
        vectors /= lines

        # 文档结构数据归一化处理,并生成向量
        min_, max_ = self.statsScope
        statistics = l1_fd[:, 1:7]
        statistics[statistics > 500] = 500  # 防止异常大的数干扰效果
        statistics = (statistics - min_) / (max_ - min_) * 0.01  # 6列统计值各占1%左右权重
        subtotal = l1_fd[:, 7:] * 0.005  # subtotal 12列各占0.5%左右权重

        cols += len(self.l1_structure[0])
        if rows > 300:
            G.log.info('[%d*%d]Vectors built' % (rows, cols))

        return np.hstack((statistics, subtotal, vectors))

    # 从k=64开始，二分法确定Top5类样本量小于指定比例的K
    @staticmethod
    def __findStartK(vectors):
        k_from, k_, k_to = 5, 64, 0
        while k_ < min(G.cfg.getint('Classifier', 'MaxCategory'), len(vectors)):
            kmeans = KMeans(n_clusters=k_).fit(vectors)  # 聚类
            n = min(5, int(k_ * 0.1) + 1)
            top5_ratio = sum([v for (k, v) in Counter(kmeans.labels_).most_common(n)]) / vectors.shape[0]
            G.log.debug('locating the starter. k=%d, SSE= %e, Top%d labels=%d%%',
                        k_, kmeans.inertia_, n, top5_ratio * 100)

            if top5_ratio < G.cfg.getfloat('Classifier', 'Top5Ratio'):  # 向前找
                if k_ - k_from < 4:  # 已靠近低限，找到大致起点
                    G.log.info('start k=%d', k_from)
                    return k_from
                k_to = k_ - 1
                k_ = k_from + int((k_ - k_from) / 2)
            else:  # 向后找
                if k_ < k_to < k_ + 4:  # 已靠近高点，找到大致起点
                    G.log.info('start k=%d', k_)
                    return k_
                k_from = k_ + 1
                if k_to > 0:  # 有上限
                    k_ = k_to - int((k_to - k_) / 2)
                else:  # 无上限
                    k_ *= 2

            if kmeans.inertia_ < 1e-5:  # 已经完全分类，但仍不均衡
                break

        G.log.info('No starter found')
        return None  # No found,re-samples

    # 聚类，得到各簇SSE（sum of the squared errors)，作为手肘法评估确定ｋ的依据
    @staticmethod
    def __pilotClustering(vectors, k_from=1, k_to=G.cfg.getint('Classifier', 'MaxCategory')):
        norm_factor = vectors.shape[1] * vectors.shape[0]  # 按行/样本数和列/字典宽度标准化因子，保证不同向量的可比性
        termination_inertia = G.cfg.getfloat('Classifier', 'NormalizedTerminationInertia') * norm_factor
        cfg_q = G.cfg.getfloat('Classifier', 'Quantile')
        k_, pilot_list = 0, []  # [(k_, inertia, criterion, top5_percent, bad_percent)] criteria取inertia变化率的一阶微分的极大值
        # 从k_from到k_to聚类,得到pilot_list
        for k_ in range(k_from, k_to):
            kmeans = KMeans(n_clusters=k_, tol=1e-5).fit(vectors)  # 试聚类
            if k_ < k_from + 2:
                pilot_list.append([k_, kmeans.inertia_, 0, 0, 0])
                continue

            retry = 0  # 多聚几次,保证inertia递减
            for retry in range(5):  # 如果inertia因误差变大，重新聚几次
                inertia = kmeans.inertia_
                if inertia <= pilot_list[-1][1]:
                    break
                G.log.debug('retries=%d, inertia=%e', retry + 1, inertia)
                kmeans = KMeans(n_clusters=k_).fit(vectors)
            else:
                inertia = pilot_list[-1][1]

            pilot_list[-1][2] = pilot_list[-2][1] / pilot_list[-1][1] - pilot_list[-1][1] / inertia
            a = pilot_list[-1]
            G.log.info('pilot clustering. (k,inertia,criteria,top5,bad)=\t%d\t%e\t%.3f\t%.3f\t%.3f',
                       pilot_list[-1][0], pilot_list[-1][1], pilot_list[-1][2], pilot_list[-1][3], pilot_list[-1][4])

            top5_percent = sum([v for (k, v) in Counter(kmeans.labels_).most_common(5)]) / len(kmeans.labels_)
            # 计算距离特别远（0.8分位点2倍距离以上）的坏点比例
            v_scores = -np.array([kmeans.score([v]) for v in vectors])
            groups = pd.DataFrame({'C': kmeans.labels_, 'S': v_scores}).groupby('C')
            c_quantiles_double = 2 * np.array([groups.get_group(i)['S'].quantile(cfg_q) for i in range(k_)])
            bad_samples = 0
            for idx, score in enumerate(v_scores):
                if score > c_quantiles_double[kmeans.labels_[idx]]:
                    bad_samples += 1
            bad_percent = bad_samples / len(v_scores)

            pilot_list.append([k_, inertia, None, top5_percent, bad_percent])
            if inertia < termination_inertia:  # 已经收敛到很小且找到可选值，没必要继续增加
                break
        # 从pilot list中取极大值的top5
        pilot_list = np.array(pilot_list)[1:-1, :]  # 去掉第一个和最后一个没法计算criterion值的
        pilot_list = pilot_list[pilot_list[:, 3] < G.cfg.getfloat('Classifier', 'Top5Ratio')]  # 去掉top5占比超标的
        pilot_list = pilot_list[argrelextrema(pilot_list[:, 2], np.greater)]  # 得到极大值
        criteria = pilot_list[:, 2].tolist()
        if not criteria:  # 没有极值
            return None

        max_top_n, idx_ = [], 0
        while criteria[idx_:]:
            idx_ = criteria.index(max(criteria[idx_:]))
            max_top_n.append(pilot_list[idx_])
            idx_ += 1
        G.log.debug('topN k=\n%s',
                    '\n'.join(['%d\t%e\t%.3f\t%.3f\t%.3f' % (k, i, c, t, b) for k, i, c, t, b in max_top_n]))
        products = [k * c for k, i, c, t, b in max_top_n]
        idx_ = products.index(max(products))
        preferred = max_top_n[idx_][0]
        G.log.info('pilot-clustering[k:%d] finished. preferred k=(%d)', k_, preferred)
        return preferred

    # 重新聚类，得到各Cluster的中心点、分位点距离、边界距离以及数量占比等
    @staticmethod
    def __buildModel(k_, vectors):
        # 再次聚类并对结果分组。 Kmeans不支持余弦距离
        kmeans = KMeans(n_clusters=k_, n_init=20, max_iter=500).fit(vectors)
        norm_factor = - vectors.shape[1]  # 按字典宽度归一化
        groups = pd.DataFrame({'C': kmeans.labels_, 'S': [kmeans.score([v]) / norm_factor for v in vectors]}).groupby(
            'C')
        percents = groups.size() / len(vectors)  # 该簇向量数在聚类总向量数中的占比
        cfg_q = G.cfg.getfloat('Classifier', 'Quantile')
        quantiles = np.array([groups.get_group(i)['S'].quantile(cfg_q, interpolation='higher') for i in range(k_)])
        boundaries = groups['S'].agg('max').values  # 该簇中最远点距离

        quantiles2 = quantiles * 2
        boundaries[boundaries > quantiles2] = quantiles2[boundaries > quantiles2]  # 边界太远的话，修正一下
        boundaries[boundaries < 1e-100] = 1e-100  # 边界为零的话，修正一下
        quantiles = boundaries - quantiles
        quantiles[quantiles < 1e-100] = 1e-100  # 避免出现0/0

        G.log.info('Model(k=%d) built. inertia=%e， max proportion=%.2f%%, max quantile=%e, max border=%e',
                   k_, kmeans.inertia_, max(percents) * 100, max(quantiles), max(boundaries))
        return kmeans, percents, boundaries, quantiles

    # 保存文本格式模型
    def __saveModel(self):
        joblib.dump((self.ruleSet, self.dictionary, self.statsScope, self.model, self.categories),
                    G.projectFileClassifierModel)
        self.dictionary.save_as_text(os.path.join(G.logsPath, 'FileDictionary.csv'))
        category_names, percents, boundaries, quantiles = self.categories
        l2_fd = pd.DataFrame({'类名': category_names, '占比': percents,
                              '分位点到边界': quantiles, '边界点': boundaries})
        l2_fd.to_csv(os.path.join(G.logsPath, 'FileCategories.csv'), sep='\t', encoding='GBK')
        G.log.info(
            'Model is built and saved to %s, %s and Database: FileDictionary.csv, FileCategories.csv successful.',
            G.projectFileClassifierModel, G.logsPath)

    def splitResults(self, results):
        classified_files, unclassified_files = [], []
        for common_name, category, category_name, confidence, distance in zip(self.common_filenames, results[0],
                                                                              results[1], results[2], results[3]):
            if confidence < G.minConfidence:  # 置信度不够,未完成分类
                unclassified_files.append(common_name)
            else:
                classified_files.append([self.model_id, common_name, category, category_name, confidence, distance])
        return classified_files, unclassified_files

    # 对单个样本文件进行分类，返回文件名称、时间戳锚点位置，类别和置信度
    def predictFile(self, file_fullname, encoding='utf-8'):
        """
        :param file_fullname: log file to be predicted:
        :param encoding: encoding of the file
        :return: None if file __process errors, tuple of filename, number-of-lines, timestamp-cols, predict category
                 index, name, confidence and distance-to-center.
                 confidence > 1 means nearer than 0.8-quantile to the center, < 0 means out of boundary
        """
        if self.model is None:
            raise UserWarning('Failed to predict: Model is not exist!')

        try:
            document = self.__file2doc(file_fullname, encoding=encoding)  # 文件转为词表
            vectors = self.__buildVectors([document], 1)
            categories, names, confidences, distances = self.__getResult(vectors)  # 预测分类并计算可信度
            return categories[0], names[0], confidences[0], distances[0]

        except Exception as err:
            G.log.warning('Failed to predict\t%s, ignored.\t%s', file_fullname, str(err))
            return None

    # 对目录下多个样本文件进行分类，返回文件名称、时间戳锚点位置，类别和置信度
    def predictFiles(self, dataset_path, encoding='utf-8'):
        """
        :param dataset_path: path which contains filed to be predicted
        :param encoding: encoding of the file
        :return: list of file-names, number-of-line, timestamp-col, predict category index, name, confidence and
                 distance-center. confidence > 1 means nearer than 0.8-quantile to the center, < 0 means out of boundary
        """
        if self.model is None:
            raise UserWarning('Failed to predict: Model is not exist!')

        corpus = []
        start_ = len(self.common_filenames)
        amount_files, failed_files, file_fullname = 0, 0, ''
        G.log.info('Start __process documents from ' + dataset_path)
        for dir_path, dir_names, file_names in os.walk(dataset_path):
            try:
                for file_name in file_names:
                    file_fullname = os.path.join(dir_path, file_name)
                    amount_files += 1
                    if amount_files % 50 == 0:
                        G.log.info('Processed %d files, failed %d', amount_files, failed_files)
                    corpus.append(self.__file2doc(file_fullname, encoding=encoding))  # 文件转为词表
            except Exception as err:
                failed_files += 1
                G.log.warning('Failed to __process\t%s, ignored.\t%s', file_fullname, str(err))
                continue
        G.log.info('Converted %d files,%d(%d%%) failed', amount_files, failed_files, failed_files / amount_files * 100)

        vectors = self.__buildVectors(corpus, len(corpus))
        categories, category_names, confidences, distances = self.__getResult(vectors)  # 预测分类并计算可信度
        files = self.common_filenames[start_:]
        return files, list(categories), category_names, list(confidences), distances

    # 预测分类并计算可信度。<0 表示超出边界，完全不对，〉1完全表示比分位点还近，非常可信
    def __getResult(self, vectors):
        c_names, c_percents, c_boundaries, c_quantiles = self.categories
        norm_factor = - vectors.shape[1]  # 按字典宽度归一化

        predicted_labels = self.model.predict(vectors)  # 使用聚类模型预测记录的类别
        predicted_names = [c_names[label] for label in predicted_labels]
        confidences = []
        distances = []
        for i, v in enumerate(vectors):
            distance = self.model.score([v]) / norm_factor
            distances.append(distance)
            category = predicted_labels[i]
            confidences.append((c_boundaries[category] - distance) / c_quantiles[category])
        confidences = np.array(confidences, copy=False)
        confidences[confidences > 99.9] = 99.9
        confidences[confidences < -99.9] = -99.9

        return predicted_labels, predicted_names, confidences, distances

    # 删除缓存文件
    def __clearCache(self):
        for f in [self.__corpusCacheFile, self.l1_dbf]:
            try:
                os.remove(self.l1_dbf) if os.path.exists(self.l1_dbf) else None
                os.remove(self.__corpusCacheFile) if os.path.exists(self.__corpusCacheFile) else None
            except Exception as err:
                G.log.warning('Failed to clear %s. %s' % (f, str(err)))
                continue


if __name__ == '__main__':
    lc = Classifier()
    lc.reCluster()

    #    lc.trainModel(G.l1_cache)
    x = 1
