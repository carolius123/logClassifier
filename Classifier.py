#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : Classifier.py
# @Author: Sui Huafeng
# @Date  : 2018/4
# @Desc  : 从$DATA/input中所有样本文件中训练一个日志文件分类器
#    新建对象或者调用trainModel方法，可以生成Classifier模型
#    调用predict方法，可以预测新的日志文件类型及其置信度
#    $DATA/models/l1file_info.csv：记录原始样本文件信息(暂时不要？）
#    $DATA/l1cache/: 存储各样本文件。目录结构就是被管服务器原始结构
#

import os
import re
from collections import Counter
import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from sklearn.cluster import KMeans
from sklearn.externals import joblib
from anchor import Anchor
from config import Workspaces as G


# 对数十、上百万个文件进行聚类，形成聚类模型
class Classifier(object):
    __corpusCacheFile = os.path.join(G.models, 'corpuscache.1')
    __metaDataFile = os.path.join(G.models, 'metadata.1')
    __MaxLines = G.cfg.getint('FileCluster', 'MaxLines')

    def __init__( self, dataset_path=None, model_file=None ):
        self.ruleset = None  # 处理文件正则表达式
        self.corpusMetaData = []  # [[是否有效，文件名，时间戳锚点，文件行数]]
        self.statsRange = None  # 样本文件字符数、字数统计值(均值、标准差、中位数)的最小-最大值范围
        self.dictionary = None  # 字典对象(Gensim Dictionary)
        self.model = None  # 聚类模型(Kmeans)
        self.categories = None  # 聚类的类型(名称，数量占比，分位点距离，边界点距离)

        if model_file:  # 从模型文件装载模型
            if not os.path.exists(model_file):
                model_file = os.path.join(G.models, model_file)
            self.ruleset, self.dictionary, self.statsRange, self.model, self.categories, results = joblib.load(
                model_file)
        elif dataset_path:  # 从样本训练模型
            amount = self.preProcess(dataset_path)
            self.trainModel() if amount else None

    def __iter__( self ):
        self.category_id = 0
        return self

    # 返回类别的(名称，数量占比，分位点到中心距离，边界到分位点距离)
    def __next__( self ):
        i = self.category_id
        self.category_id += 1
        return self.categories[0][i], self.categories[1][i], self.categories[2][i], self.categories[3][i]

    def __len__( self ):
        return len(self.categories[0])

    def __getitem__( self, item ):
        if item < 0 or item > len(self):
            raise IndexError
        return self.categories[0][item], self.categories[1][item], self.categories[2][item], self.categories[3][item]

    def __setitem__( self, key, value ):
        if key < 0 or key > len(self):
            raise IndexError
        name = str(value)
        if name in self.categories[0]:
            raise ValueError
        self.categories[0][key] = value

    # 预处理，迭代方式返回某个文件的词表.
    @staticmethod
    def preProcess( from_=G.inputs, to_=G.l1_cache ):
        G.log.info('Start probing %s, merging into %s', from_, to_)
        f_descriptor = os.path.join(G.models, 'metadata.0')
        amount = [0, 0, 0]
        (qualified_files, processed_files) = joblib.load(f_descriptor) if os.path.exists(f_descriptor) else ([], [])
        for dir_path, dir_names, file_names in os.walk(from_):
            # 同目录内相似文件分为一组
            same_logs = {}  # {公共文件名:[源文件名]}
            for filename in file_names:
                file_fullname = os.path.join(dir_path, filename)
                if file_fullname in processed_files:  # 上次处理过的文件
                    continue
                amount[0] += 1
                processed_files.append(file_fullname)
                try:
                    anchor = Anchor(file_fullname)
                except Exception as err:  # 没有时间戳锚点
                    G.log.warning('Failed to process\t%s, ignored.\t%s', file_fullname, str(err))
                    continue

                common_name = re.sub('[-\.\d]', '', filename)
                common_name = common_name if common_name else 'digital1'
                value_ = same_logs.get(common_name, [])
                value_.append((filename, anchor))
                same_logs[common_name] = value_
            #
            if not same_logs:
                continue
            # 同组文件合并为1个

            path_to = dir_path.replace(from_, to_)
            os.makedirs(path_to, exist_ok=True)
            for common_name, value_ in same_logs.items():
                if amount[1] % 500 == 0:
                    G.log.info('%d files probed and merged, %s', len(qualified_files), dir_path)
                file_to = os.path.join(path_to, common_name)
                with open(file_to, 'a', encoding='utf-8') as fp:
                    amount[2] += 1
                    for filename, anchor in value_:
                        file_from = os.path.join(dir_path, filename)
                        try:
                            qualified_files.append([file_to, file_from, anchor])
                            amount[1] += 1
                            for line in open(file_from, 'r', encoding='utf-8'):
                                fp.write(line)
                        except Exception as err:
                            G.log.warning('Failed to merge %s, ignored. %s', file_from, str(err))
                            continue
        joblib.dump((qualified_files, processed_files), f_descriptor) if amount[0] else None
        G.log.info('%d files probed, %d files qualified, merged into %d files, and recorded into %s',
                   amount[0], amount[1], amount[2], f_descriptor)
        return amount[2]

    # 训练、生成模型并保存在$models/xxx.mdl中,dataset:绝对/相对路径样本文件名，或者可迭代样本字符流
    def trainModel( self, dataset_path=G.l1_cache ):
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
        for self.ruleset in rule_sets:
            corpus_fp, corpus_structure = self.__buildDictionary(dataset_path)  # 建立字典，返回文档结构信息
            if len(self.dictionary) < G.cfg.getint('FileCluster', 'LeastTokens'):  # 字典字数太少,重新采样
                corpus_fp.close()
                self.clearCache()
                G.log.info('Too few tokens[%d], Re-sample with next RuleSet.', len(self.dictionary))
                continue
            corpus_fp.seek(0)
            vectors = self.__buildVectors(corpus_fp, corpus_structure)  # 建立稀疏矩阵doc*(dct + stats)
            corpus_fp.close()  # 关闭缓存文件

            #            start_k = self.__findStartK(vectors)  # 快速定位符合分布相对均衡的起点K
            #            if start_k is None:  # 聚类不均衡，换rule set重新采样
            #                continue
            start_k = 3
            k_ = self.__pilotClustering(vectors, start_k)  # 多个K值试聚类，返回最佳K
            if k_ != 0:  # 找到合适的K，跳出循环
                break
            self.clearCache()  # 清除缓存的ruleset
        else:
            raise UserWarning('Cannot generate qualified corpus by all RuleSets')

        # 重新聚类, 得到模型(向量数、中心点和距离）和分类(向量-所属类)
        self.model, percents, boundaries, quantiles = self.__buildModel(k_, vectors)
        names = ['fc%d' % i for i in range(len(percents))]
        self.categories = [names, percents, boundaries, quantiles]
        corpus_info = [[v[1], v[2].colRange[0]] for v in self.corpusMetaData if v[0]]
        results = [(file, col, label) for (file, col), label in zip(corpus_info, self.model.labels_)]
        # 保存聚类结果
        f_model = os.path.join(G.models, 'FileClassifierModel.1')
        joblib.dump((self.ruleset, self.dictionary, self.statsRange, self.model, self.categories, results), f_model)
        G.log.info('Model is built and saved to %s successful.', f_model)
        self.saveAsCSV(vectors)  # 保存文本信息，供人工审查

    # 建立词典，同时缓存词表文件
    def __buildDictionary( self, new_dataset_path ):
        corpus_structure = []  # [[字节和字数的12个分段数量比例，均值/标准差/中位数]]
        self.dictionary = Dictionary()
        cache_fp = open(self.__corpusCacheFile, mode='a+t', encoding='utf-8')  # 创建或打开语料缓存文件
        # 装载处理过的缓存语料
        if cache_fp.tell() != 0:
            if os.path.exists(self.__metaDataFile):
                self.ruleset, self.corpusMetaData, corpus_structure, self.statsRange = joblib.load(self.__metaDataFile)
            cache_fp.seek(0)
            cached_documents = len(corpus_structure)
            for line_ in cache_fp:
                if cached_documents == 0:
                    break
                cached_documents -= 1
                self.dictionary.add_documents([line_.split()])
        G.log.info('%d cached documents loaded.', len(corpus_structure))

        # 继续处理新增语料
        for document, doc_structure in self.__buildDocument(new_dataset_path):
            self.dictionary.add_documents([document])
            corpus_structure.append(doc_structure)
            cache_fp.write(' '.join([word for word in document]) + '\n')

        if self.dictionary.num_docs < G.cfg.getint('FileCluster', 'LeastFiles'):  # 字典字数太少或文档数太少，没必要聚类
            cache_fp.close()
            self.clearCache()
            raise UserWarning('Too few documents[%d] to clustering' % self.dictionary.num_docs)

        # 去掉低频词，压缩字典
        num_token = len(self.dictionary)
        no_below = int(min(G.cfg.getfloat('FileCluster', 'NoBelow'), int(self.dictionary.num_docs / 50)))
        self.dictionary.filter_extremes(no_below=no_below, no_above=0.999, keep_n=G.cfg.getint('FileCluster', 'KeepN'))
        self.dictionary.compactify()
        G.log.info('Dictionary built with [%s](%d tokens, reduced from %d), from %d files( %d words)',
                   self.ruleset[0], len(self.dictionary), num_token, self.dictionary.num_docs, self.dictionary.num_pos)

        statistics = np.array(corpus_structure)[:, :6]
        statistics[statistics > 500] = 500  # 防止异常大的数干扰效果
        self.statsRange = np.min(statistics, axis=0), np.max(statistics, axis=0)
        joblib.dump((self.ruleset, self.corpusMetaData, corpus_structure, self.statsRange),
                    self.__metaDataFile)  # 保存模型，供后续使用

        return cache_fp, corpus_structure

    # 预处理，迭代方式返回某个文件的词表.
    def __buildDocument( self, dataset_path ):
        amount_files, failed_files, file_fullname = 0, 0, ''
        G.log.info('Start Converting documents from ' + dataset_path)
        converted = [v[1] for v in self.corpusMetaData]
        for dir_path, dir_names, file_names in os.walk(dataset_path):
            for file_name in file_names:
                try:
                    file_fullname = os.path.join(dir_path, file_name)
                    if file_fullname in converted:
                        continue
                    amount_files += 1
                    if amount_files % 50 == 0:
                        G.log.info('Converted %d[%d failed] files:\t%s', amount_files, failed_files, file_fullname)
                    yield self.__file2doc(file_fullname)
                except Exception as err:
                    failed_files += 1
                    self.corpusMetaData.append([False, file_fullname, str(err), 0])
                    G.log.warning('Failed to convert\t%s, ignored.\t%s', file_fullname, str(err))
                    continue
        G.log.info('Converted %d files,%d failed', amount_files, failed_files)
        raise StopIteration()

    # 使用规则集匹配和转换后，转化为词表
    def __file2doc( self, file_fullname, encoding='utf-8' ):
        document = []
        anchor = Anchor()  # 创建空的锚点对象
        anchor_candidates = {}  # 初始化潜在锚点字典
        line_idx, line_chars, line_words = 0, [], []

        G.log.debug('Converting ' + file_fullname)
        for line_idx, line in enumerate(open(file_fullname, 'r', encoding=encoding)):
            anchor.appendAnchors(line, anchor_candidates)  # 统计潜在时间戳锚点
            words = G.getWords(line, rule_set=self.ruleset)
            document += words  # 生成词表
            line_chars.append(len(line))
            line_words.append(len(words))
            if line_idx > self.__MaxLines:
                break

        # 汇总和保持元数据
        line_idx += 1
        anchor.setFormat(anchor.statsAnchors(anchor_candidates, line_idx), file_fullname)
        self.corpusMetaData.append([True, file_fullname, anchor, line_idx])

        # 计算统计数据
        subtotal_chars = list(np.histogram(np.array(line_chars), bins=[0, 40, 80, 120, 160, 200, 1000])[0] / line_idx)
        subtotal_words = list(np.histogram(np.array(line_words), bins=[0, 4, 8, 12, 16, 20, 100])[0] / line_idx)
        mean_chars, mean_words = np.mean(line_chars), np.mean(line_words)
        std_chars, std_words = np.std(line_chars), np.std(line_words)
        median__chars, median__words = np.median(line_chars), np.median(line_words)
        doc_structure = [std_chars, std_words, mean_chars, mean_words, median__chars,
                         median__words] + subtotal_chars + subtotal_words
        return document, doc_structure

    # 从词表和文档结构形成聚类向量
    def __buildVectors( self, corpus, corpus_structure ):
        rows, cols = len(corpus_structure), len(self.dictionary)

        # 构造tf-idf词袋和文档向量
        tfidf_model = TfidfModel(dictionary=self.dictionary, normalize=True)
        c_vectors = np.zeros((rows, cols))
        for doc_idx, document in enumerate(corpus):
            if type(document) == str:
                document = document.split()
            for (word_idx, tf_idf_value) in tfidf_model[self.dictionary.doc2bow(document)]:
                c_vectors[doc_idx, word_idx] = tf_idf_value  # tfidf词表加入向量
        # 按每个文档的行数对tfidf向量进行标准化，保证文档之间的可比性
        lines = np.array([[v[3]] for v in self.corpusMetaData if v[0]])
        c_vectors /= lines

        # 文档结构数据归一化处理,并生成向量
        s_vectors = np.array(corpus_structure)
        s_vectors[:, 6:] *= 0.005  # subtotal 12列各占0.5%左右权重
        statistics = s_vectors[:, :6]
        statistics[statistics > 500] = 500  # 防止异常大的数干扰效果
        min_, max_ = self.statsRange
        s_vectors[:, :6] = (statistics - min_) / (max_ - min_) * 0.01  # 6列统计值各占1%左右权重

        cols += len(corpus_structure[0])
        G.log.info('[%d*%d]Vectors built' % (rows, cols))

        return np.hstack((c_vectors, s_vectors))

    # 从k=64开始，二分法确定Top5类样本量小于指定比例的K
    @staticmethod
    def __findStartK( vectors ):
        k_from, k_, k_to = 5, 64, 0
        while k_ < min(G.cfg.getint('FileCluster', 'MaxCategory'), len(vectors)):
            kmeans = KMeans(n_clusters=k_).fit(vectors)  # 聚类
            n = min(5, int(k_ * 0.1) + 1)
            top5_ratio = sum([v for (k, v) in Counter(kmeans.labels_).most_common(n)]) / vectors.shape[0]
            G.log.debug('locating the starter. k=%d, SSE= %e, Top%d labels=%d%%',
                        k_, kmeans.inertia_, n, top5_ratio * 100)

            if top5_ratio < G.cfg.getfloat('FileCluster', 'Top5Ratio'):  # 向前找
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
    def __pilotClustering( vectors, k_from=1 ):
        col_norm_factor = vectors.shape[1]  # 按列/字典宽度归一化因子
        cell_norm_factor = col_norm_factor * vectors.shape[0]  # 按行/样本数和列/字典宽度归一化因子

        k_, sse_set = 0, []
        for k_ in range(k_from, k_from + 3):
            kmeans = KMeans(n_clusters=k_).fit(vectors)  # 试聚类
            sse = kmeans.inertia_ / cell_norm_factor
            G.log.debug('pilot clustering. k=%d, normSSE= %e', k_, sse)
            sse_set.append(sse)
        last_indicator = (sse_set[0] + sse_set[2]) / sse_set[1]  # 二阶微分的相对值
        last_k = k_from + 2

        maxima = None  # (k, kmeans, sse, indicator)
        prefer = (0, None, 0, 0, 0)  # (K_, kmeans, sse, indicator, ratio of top5 lables)
        last_top5_value, last_top5_idx = 100, 1
        for k_ in range(k_from + 3, G.cfg.getint('FileCluster', 'MaxCategory')):
            kmeans = KMeans(n_clusters=k_).fit(vectors)  # 试聚类
            sse = kmeans.inertia_ / cell_norm_factor
            G.log.debug('pilot clustering. k=%d, normSSE= %e', k_, sse)
            if sse >= sse_set[-1]:  # SSE变大，是误差，应滤除之
                continue

            sse_step = (sse - sse_set[-1]) / (k_ - last_k)  # 用SSE_step代替SSE，可兼容有误差情况
            sse_set.pop(0)
            sse_set.append(sse_set[-1] + sse_step)
            indicator = (sse_set[-3] + sse_set[-1]) / sse_set[-2]

            if indicator > last_indicator:  # 开始增大
                maxima = [k_, kmeans, sse, indicator]
            elif maxima is not None and indicator < last_indicator:  # 增大后开始减小，prev是极大值点
                n = min(5, int(k_ * 0.1) + 1)
                top5_ratio = sum([v for (k, v) in Counter(maxima[1].labels_).most_common(n)]) / vectors.shape[0]
                if prefer[3] < maxima[3] and top5_ratio < G.cfg.getfloat('FileCluster',
                                                                         'Top5Ratio'):  # Top5Label中样本比例有效（没有失衡)
                    prefer = maxima + [top5_ratio]
                G.log.info('Maxima point. k=(%d,%.2f) normSSE=%e, Top%d labels=%.1f%%. Preferred (%d,%.2f)',
                           maxima[0], maxima[3], maxima[2], n, top5_ratio * 100, prefer[0], prefer[3])
                maxima = None  # 变量复位，准备下一个极大值点
                if top5_ratio < last_top5_value - 0.001:
                    last_top5_value = top5_ratio
                    last_top5_idx = k_
                else:
                    if k_ - last_top5_idx > 50:  # 连续50个K_比例不降
                        break

            if sse < 1e-15:  # 已经收敛到很小且找到可选值，没必要继续增加
                break

            sse_set[-1] = sse  # 如无异常误差点，这些操作只是重复赋值，无影响。如有，更新当前点值，准备下一循环
            sse_set[-2] = sse_set[-1] - sse_step
            last_indicator = indicator
            last_k = k_

        G.log.info('pilot-clustering[k:1-%d] finished. preferred k=(%d, %.2f),normSSE=%e, TopN labels=%.1f%%'
                   % (k_, prefer[0], prefer[3], prefer[2], prefer[4] * 100))
        return prefer[0]

    # 重新聚类，得到各Cluster的中心点、分位点距离、边界距离以及数量占比等
    @staticmethod
    def __buildModel( k_, vectors ):
        # 再次聚类并对结果分组。 Kmeans不支持余弦距离
        kmeans = KMeans(n_clusters=k_, n_init=20, max_iter=500).fit(vectors)
        norm_factor = - vectors.shape[1]  # 按字典宽度归一化
        groups = pd.DataFrame({'C': kmeans.labels_, 'S': [kmeans.score([v]) / norm_factor for v in vectors]}).groupby(
            'C')
        percents = groups.size() / len(vectors)  # 该簇向量数在聚类总向量数中的占比
        quantiles = np.array([groups.get_group(i)['S'].quantile(G.cfg.getfloat('FileCluster', 'Quantile'),
                                                                interpolation='higher') for i in range(k_)])
        boundaries = groups['S'].agg('max').values  # 该簇中最远点距离
        for i in range(k_):
            if boundaries[i] > quantiles[i] * 2:  # 边界太远的话，修正一下
                boundaries[i] = quantiles[i] * 2
            elif boundaries[i] == quantiles[i]:  # 避免出现0/0
                boundaries[i] = quantiles[i] + 1e-100

        G.log.info('Model(k=%d) built. inertia=%.3f， max proportion=%.2f%%, max quantile=%.3f, max border=%.3f',
                   k_, kmeans.inertia_, max(percents) * 100, max(quantiles), max(boundaries))
        return kmeans, percents, boundaries, quantiles

    # 为人工审查保存数据
    def saveAsCSV( self, vectors ):
        self.dictionary.save_as_text(os.path.join(G.logsPath, 'FileDictionary.csv'))
        names, percents, boundaries, quantiles = self.categories
        df = pd.DataFrame({'类别': names, '占比': percents,
                           '分位点': quantiles, '边界点': boundaries})
        df.to_csv(os.path.join(G.logsPath, 'FileCategories.csv'), sep='\t')

        categories, names, confidences, distances = self.__getResult(vectors)  # 预测分类并计算可信度

        df = pd.DataFrame(columns=('类别', '文件', '行数', '远点', '可信度', '锚点', '路径', '全路径名'))
        corpus_info = [[v[1], v[3], v[2].name + str(v[2].colRange[0])] for v in self.corpusMetaData if v[0]]
        docs = 0
        for docs, ((file_fullname, lines, anchor_col), category, distance, confidence) in enumerate(
                zip(corpus_info, names, distances, confidences)):
            path_, name_ = os.path.split(file_fullname)
            df.loc[docs] = category, name_, lines, distance, confidence, anchor_col, path_, file_fullname

        for v in self.corpusMetaData:
            status, file_fullname = v[0], v[1]
            if status:
                continue
            docs += 1
            path_, name_ = os.path.split(file_fullname)
            df.loc[docs] = '', name_, '', '', '', '', path_, file_fullname

        df.to_csv(os.path.join(G.logsPath, 'FileResults.csv'), sep='\t')

    # 对单个样本文件进行分类，返回文件名称、时间戳锚点位置，类别和置信度
    # 置信度最大99.99，如>1, 表示到中心点距离小于0.8分位点，非常可信；最小-99.99，如< 0距离大于最远点，意味着不属于此类
    def predictFile( self, file_fullname, encoding='utf-8' ):
        if self.model is None:
            raise UserWarning('Failed to predict: Model is not exist!')

        try:
            document, doc_structure = self.__file2doc(file_fullname, encoding=encoding)  # 文件转为词表
            file, anchor, lines = self.corpusMetaData[-1][1], self.corpusMetaData[-1][2], self.corpusMetaData[-1][3]
            vectors = self.__buildVectors([document], [doc_structure])
            categories, names, confidences, distances = self.__getResult(vectors)  # 预测分类并计算可信度
            return file, lines, anchor.colRange[0], categories[0], names[0], confidences[0], distances[0]

        except Exception as err:
            G.log.warning('Failed to predict\t%s, ignored.\t%s', file_fullname, str(err))
            return None

    # 对目录下多个样本文件进行分类，返回文件名称、时间戳锚点位置，类别和置信度
    def predictFiles( self, dataset_path, encoding='utf-8' ):
        if self.model is None:
            raise UserWarning('Failed to predict: Model is not exist!')

        corpus = []
        corpus_structure = []  # [[字节和字数的12个分段数量比例，均值/标准差/中位数]]
        start_ = len(self.corpusMetaData)
        amount_files, failed_files, file_fullname = 0, 0, ''
        G.log.info('Start process documents from ' + dataset_path)
        for dir_path, dir_names, file_names in os.walk(dataset_path):
            try:
                for file_name in file_names:
                    file_fullname = os.path.join(dir_path, file_name)
                    amount_files += 1
                    if amount_files % 50 == 0:
                        G.log.info('Processed %d files, failed %d', amount_files, failed_files)
                    document, doc_stats = self.__file2doc(file_fullname, encoding=encoding)  # 文件转为词表
                    corpus.append(document)
                    corpus_structure.append(doc_stats)
            except Exception as err:
                failed_files += 1
                self.corpusMetaData.append([False, file_fullname, str(err), 0])
                G.log.warning('Failed to process\t%s, ignored.\t%s', file_fullname, str(err))
                continue
        G.log.info('Converted %d files,%d(%d%%) failed', amount_files, failed_files, failed_files / amount_files * 100)

        vectors = self.__buildVectors(corpus, corpus_structure)
        categories, names, confidences, distances = self.__getResult(vectors)  # 预测分类并计算可信度
        files = [v[1] for v in self.corpusMetaData[start_:] if v[0]]
        cols = [v[2].colRange[0] for v in self.corpusMetaData[start_:] if v[0]]
        return files, cols, categories, names, confidences, distances

    # 预测分类并计算可信度。<0 表示超出边界，完全不对，〉1完全表示比分位点还近，非常可信
    def __getResult( self, vectors ):
        c_names, c_percents, c_boundaries, c_quantiles = self.categories  # 模型中各类的名称、数量占比、边界距离和分位点距离
        norm_factor = - vectors.shape[1]  # 按字典宽度归一化

        predicted_labels = self.model.predict(vectors)  # 使用聚类模型预测记录的类别
        predicted_names = [c_names[label] for label in predicted_labels]
        confidences = []
        distances = []
        for i, v in enumerate(vectors):
            distance = self.model.score([v]) / norm_factor
            distances.append(distance)
            category = predicted_labels[i]
            confidences.append((c_boundaries[category] - distance) / (c_boundaries[category] - c_quantiles[category]))
        confidences = np.array(confidences, copy=False)

        return predicted_labels, predicted_names, confidences, distances

    def clearCache( self ):
        for f in [self.__corpusCacheFile, self.__metaDataFile]:
            try:
                os.remove(self.__metaDataFile) if os.path.exists(self.__metaDataFile) else None
                os.remove(self.__corpusCacheFile) if os.path.exists(self.__corpusCacheFile) else None
            except Exception as err:
                G.log.warning('Failed to clear %s. %s' % (f, str(err)))
                continue

    # 同类文件合并供后续处理，并生成采集文件列表
    def postProcess( self, from_=os.path.join(G.logsPath, 'FileResults.csv'), to_=G.l2_cache ):
        '''
        检查聚类效果：
         一类肯定同anchor？
        切分日志有几种情况： 1) 按日归档， 应该采最新文件；2) 多文件循环写或者按日写新文件， 应该当作1个数据流都采
        目标是找出切分日志中应该采集的那个，也就是1) 一个数据流对应一个正则表达式(全路径名) 2) 长期期不更新的可去掉
        因此需要找出，同类+同目录+同anchor 的命名规则：1) 有一个不含日期且最新的？ 2-3）去掉数字后都相同
        '''
        file_list = pd.DataFrame.from_csv(from_, sep='\t').dropna(axis=0)
        for (category, anchor, path_), group in file_list.groupby(['类别', '锚点', '路径']):
            file_names = [f for f in group['文件']]

        df = pd.DataFrame(columns=())


if __name__ == '__main__':
    lc = Classifier()
    lc.trainModel()
#    lc.postProcess()
# 生成采集规则
# 形成记录聚类输入文件
# 同类文件直接copy？
# anchor不同而同类者？
