#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : Categorizer.py
# @Author: Sui Huafeng
# @Date  : 2018/3
# @Desc  : 从$DATA/l2cache中每一个样本文件中训练一个日志记录分类器


import os
import re
from collections import Counter
from tempfile import NamedTemporaryFile, TemporaryFile
from time import localtime, strftime

import numpy as np
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from pandas import DataFrame
from sklearn.cluster import KMeans
from sklearn.externals import joblib

from anchor import Anchor
from config import Workspaces as G


# 对一个数千到数十万行的文件中的记录进行聚类，形成聚类模型
class Categorizer(object):
    """
    新建对象或者调用trainModel方法，可以生成Categorizer模型
    调用predict方法，可以预测新记录的类型及其置信度
    $DATA/models/l2file_info.csv：记录(\t分割)每个样本文件名称、定界符位置
    $DATA/l2cache/: 存储各样本文件。每个样本文件由同类日志原始样本合并而成
    1. 日志文件中记录的识别。每条记录大都会在第一行特定位置输出日期和时间，因此以特定位置的时间戳(hh:mm:ss)判断上一记录
       的结束和下一记录的开始
    2. 记录聚类的首要目标是把程序执行到某处输出的记录汇聚成一类。程序在某点输出日志，大致会包含几种信息：
    2.1 日期、时间等几乎所有记录都包含的共有信息：这些对聚类结果没有影响，不必单独考虑
    2.2 常数字符串和枚举型数据(如INFO、ERROR等): 这是这一类记录最典型的独有特征，应尽量突出其在聚类中的比重
    2.3 URL、IP、Java包、path等变量：应尽量识别出来并归一化成常数字符串，改善聚类效果
    2.4 字符串变量(应用系统的数据)和数字变量：上两类有效特征在每个记录中不会超过10个，字符串/数字变量可能会很多，这会严重
         干扰聚类效果、影响运算速度，应尽量减少。数字可简单滤除。字符串变量可考虑去掉dfs字典中低频词
    3. 后续处理可能有价值的目标包括：
    3.1 数字变量中可能包含错误码、代码等高值信息，可考虑提取和利用的手段
    3.2 对于记录数特别多的类别(太粗),可根据枚举型、IP、URL、错误码等进一步细化分类
    3.4 数字变量中可能包含时长、数量等指标数据，可考虑提取和利用的手段
    """
    __RuleSets = []  # 文本处理的替换、停用词和k-shingle规则
    for rule_section in sorted([section for section in G.cfg.sections() if section.split('-')[0] == 'RuleSet']):
        replace_rules, stop_words, k_list = [], [], []
        for key, value in G.cfg.items(rule_section):
            if key == 'stopwords':
                stop_words = value.split(',')
            elif key == 'k-shingles':
                k_list = eval(value)
            else:
                replace_from, replace_to = value.split('TO')
                replace_rules.append((re.compile(replace_from.strip(), re.I), replace_to.strip()))
        __RuleSets.append((rule_section, replace_rules, stop_words, k_list))

    __LeastDocuments = G.cfg.getint('RecordCluster', 'LeastRecords')  # 样本数小于该值，没有必要聚类
    __LeastTokens = G.cfg.getint('RecordCluster', 'LeastTokens')  # 字典最少词数，低于此数没必要聚类
    __KeepN = G.cfg.getint('RecordCluster', 'KeepN')  # 字典中最多词数，降低计算量
    __NoBelow = G.cfg.getfloat('RecordCluster', 'NoBelow')  # 某词出现在文档数低于该比例时，从字典去掉，以排除干扰，降低计算量

    __Top5Ratio = G.cfg.getfloat('RecordCluster', 'Top5Ratio')  # Top5类中样本数占总样本比例。大于该值时聚类结果不可接受
    __MaxCategory = G.cfg.getint('RecordCluster', 'MaxCategory')  # 尝试聚类的最大类别，以降低计算量
    __Quantile = G.cfg.getfloat('RecordCluster', 'Quantile')  # 类别的边界，该类中以该值为分位数的点，作为该类的边界

    def __init__( self, dataset=None, anchor=None, model_file=None ):
        self.__anchor = anchor  # 时间戳锚点Anchor
        self.__ruleSet = None  # 处理文件正则表达式
        self.__d = None  # 字典Dictionary
        self.__k = None  # 聚类模型 KMeans
        self.__a = None  # cluster名称
        self.__p = None  # 该类别数量占比
        self.__b = None  # 最远点到(0.8)分位点距离平方
        self.__q = None  # (0.8)分位点到中心点距离平方

        if model_file is not None:  # 从模型文件装载模型
            if not os.path.exists(model_file):
                model_file += '.mdl'
                if not os.path.exists(model_file):
                    model_file = os.path.join(G.projectModelPath, model_file)
            model = joblib.load(model_file)
            self.__anchor = model[0]  # 时间戳锚点Anchor
            self.__ruleSet = model[1]  # 处理文件正则表达式
            self.__d = model[2]  # 字典Dictionary
            self.__k = model[3]  # 聚类模型 KMeans
            self.__a = model[4]  # cluster名称
            self.__p = model[5]  # 该类别数量占比
            self.__b = model[6]  # 最远点到(0.8)分位点距离平方
            self.__q = model[7]  # (0.8)分位点到中心点距离平方
        elif dataset is not None:  # 从样本训练模型
            self.trainModel(dataset, anchor)

    # 训练、生成模型并保存在$models/xxx.mdl中,dataset:绝对/相对路径样本文件名，或者可迭代样本字符流
    def trainModel( self, dataset, anchor=None ):
        """

        :param dataset:
        :param anchor:
        """
        samples_file, temple_fp = self.__buildSamplesFile(dataset)  # 获得有效的样本文件名称
        if anchor is None:  # 从样本文件中提取时间戳锚点
            self.__anchor = Anchor(samples_file)

        for self.__ruleSet in self.__RuleSets:
            # 日志文件预处理为记录向量,并形成字典。vectors是稀疏矩阵(行-记录，列-词数)
            self.__d, vectors = self.__buildVectors(samples_file)
            if self.__d is None:  # 字典太短，换rule set重新采样
                continue
            start_k = self.__findStartK(vectors)  # 快速定位符合分布相对均衡的起点K
            if start_k is None:  # 聚类不均衡，换rule set重新采样
                continue
            preferred_k = self.__pilotClustering(vectors, start_k)  # 多个K值试聚类，返回最佳K
            if preferred_k == 0:  # 没有合适的K，换rule set重新采样
                continue

            # 重新聚类, 得到模型(向量数、中心点和距离）和分类(向量-所属类)
            self.__k, self.__a, self.__p, self.__b, self.__q = self.__buildClusterModel(preferred_k, vectors)

            joblib.dump((self.__anchor, self.__ruleSet, self.__d, self.__k, self.__a, self.__p,
                         self.__b, self.__q), os.path.join(G.projectModelPath, samples_file + '.mdl'))  # 保存模型，供后续使用

            self.__d.save_as_text(os.path.join(G.logsPath, samples_file + '.dic.csv'))  # 保存文本字典，供人工审查

            df = DataFrame({'类型': self.__a, '样本占比': self.__p,
                            '分位点距离': self.__q, '边界-分位点距离': self.__b})
            df.to_csv(os.path.join(G.logsPath, samples_file + '.mdl.csv'), sep='\t')  # 保存聚类模型，供人工审查

            a = self.predict(samples_file)  # 保存聚类结果，供人工审查
            date_time = [strftime('%Y-%m-%d %H:%M:%S', localtime(seconds)) for seconds in a[3]]
            df = DataFrame({'类型名称': a[1], '置信度': a[2], '时间': date_time, '记录内容': a[4], '记录词汇': a[5]})
            df.to_csv(os.path.join(G.logsPath, os.path.split(samples_file)[1] + '.out.csv'), index=False, sep='\t')

            G.log.info('Model saved to %s successful.' % os.path.join(G.projectModelPath, samples_file + '.mdl'))
            break
        else:
            raise UserWarning('Cannot generate qualified corpus by all RuleSets')

        if temple_fp: os.remove(samples_file)

    # 从dataset(utf-8格式。绝对/相对路径样本文件、可迭代样本字符流）生成样本文件
    @staticmethod
    def __buildSamplesFile( dataset ):
        samples_file, temple_fp = None, False
        if type(dataset) is str:  # 样本是文件名
            if os.path.exists(dataset):  # $绝对路径
                samples_file = dataset
            elif os.path.exists(os.path.join(G.l2_cache, dataset)):  # 相对路径($l2_cache目录下)
                samples_file = os.path.join(G.l2_cache, dataset)

        if samples_file is None:  # 输入不是文件名称
            temple_fp = NamedTemporaryFile(mode='w+t', delete=False, encoding='utf-8')
            size = 0
            for data in dataset:
                temple_fp.write(data)
                size += len(data)
                if size > 102400:  # 10K
                    break
            else:
                raise UserWarning('Failed to build Model: samples(%dK) is too small, or wrong filename' % (size / 1024))
            samples_file = temple_fp.name
            temple_fp.close()

        return samples_file, temple_fp

    # 文档向量化。dataset-[document:M]-[[word]]-[[token]]-[BoW:M]-corpus-tfidf-dictionary:N, [vector:M*N]
    def __buildVectors( self, dataset_file ):
        lines = 0
        dct = Dictionary()
        tmp_file = TemporaryFile(mode='w+t', encoding='utf-8')
        for doc_idx, (document, lines) in enumerate(self.__buildDocument(dataset_file)):
            dct.add_documents([document])
            tmp_file.write(' '.join(document) + '\n')
            if doc_idx % 500 == 499:
                G.log.debug('%d', doc_idx)
        if dct.num_docs < self.__LeastDocuments:  # 字典字数太少或文档数太少，没必要聚类
            tmp_file.close()
            raise UserWarning('Too few records[%d]' % dct.num_docs)

        # 去掉低频词，压缩字典
        num_token = len(dct)
        no_below = int(min(self.__NoBelow, int(dct.num_docs / 50)))
        dct.filter_extremes(no_below=no_below, no_above=0.999, keep_n=self.__KeepN)
        dct.compactify()
        G.log.info('Dictionary[%d tokens, reduced from %d] built with [%s]. '
                   '[%d]records(%d lines, %d words) in %s', len(dct), num_token, self.__ruleSet[0],
                   dct.num_docs, lines, dct.num_pos, dataset_file)
        if len(dct) < self.__LeastTokens:  # 字典字数太少,重新采样
            G.log.info('Too few tokens[%d], Re-sample with next RuleSet].' % (len(dct)))
            tmp_file.close()
            return None, None

        # 构造tf-idf词袋和文档向量
        tfidf_model = TfidfModel(dictionary=dct, normalize=False)
        vectors = np.zeros((dct.num_docs, len(dct)))
        tmp_file.seek(0)
        for doc_idx, new_line in enumerate(tmp_file):
            for (word_idx, tf_idf_value) in tfidf_model[dct.doc2bow(new_line.split())]:  # [(id,tf-idf)...], id是升序
                vectors[doc_idx, word_idx] = tf_idf_value
        G.log.info('[%d*%d]Vectors built, %.2f%% non-zeros.' %
                   (dct.num_docs, len(dct), dct.num_nnz * 100 / len(dct) / dct.num_docs))
        tmp_file.close()
        return dct, vectors

    # 预处理文件，迭代方式返回某条记录的词表.
    def __buildDocument( self, dataset_file ):
        line_idx, record = 0, ''
        with open(dataset_file, 'r', encoding='utf-8') as fp:
            for line_idx, next_line in enumerate(fp):
                try:
                    # 判断定界位置是否为恰好是时间戳，形成一条完整record
                    absent = self.__anchor.getAnchorTimeStamp(next_line) is None
                    if absent or (record == ''):
                        if absent ^ (record == ''):  # 开始行是定界，或者当前行不是定界行，表示尚未读到下一记录
                            record += next_line
                        continue

                    document = G.getWords(record, rule_set=self.__ruleSet)  # 完整记录Record(变量替换/停用词/分词/Kshingle)-〉[word]

                    # 得到词表，并准备下次循环
                    record = next_line  # 当前行存入当前记录
                    yield document, line_idx
                except (UnicodeError, UnicodeDecodeError, UnicodeEncodeError):
                    G.log.exception('Record [%s] ignored due to the following error:', record)
                    record = ''  # 清空并丢弃现有记录
                    continue
        if record != '':  # 处理最后一个记录
            try:
                # 完整记录Record(变量替换/停用词/分词/Kshingle)-〉[word]
                yield G.getWords(record, rule_set=self.__ruleSet), line_idx
            except (UnicodeError, UnicodeDecodeError, UnicodeEncodeError):
                G.log.exception('Record [%s] ignored due to the following error:', record)
        raise StopIteration()

    # 从k=64开始，二分法确定Top5类样本量小于指定比例的K
    def __findStartK( self, vectors ):
        k_from, k_, k_to = 1, 64, 0
        while k_ < min(self.__MaxCategory, len(vectors)):
            kmeans = KMeans(n_clusters=k_).fit(vectors)  # 聚类
            n = min(5, int(k_ * 0.1) + 1)
            top5_ratio = sum([v for (k, v) in Counter(kmeans.labels_).most_common(n)]) / vectors.shape[0]
            G.log.debug('locating the starter. k=%d, SSE= %f, Top%d labels=%d%%',
                        k_, kmeans.inertia_, n, top5_ratio * 100)

            if top5_ratio < self.__Top5Ratio:  # 向前找
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
                k_ = k_to - int((k_to - k_) / 2) if k_to > 0 else k_ * 2

            if kmeans.inertia_ < 1e-5:  # 已经完全分类，但仍不均衡
                break

        G.log.info('No starter found')
        return None  # No found,re-samples

    # 聚类，得到各簇SSE（sum of the squared errors)，作为手肘法评估确定ｋ的依据
    def __pilotClustering( self, vectors, k_from=1 ):
        cell_norm_factor = vectors.shape[1] * vectors.shape[0]  # 按行/样本数和列/字典宽度归一化

        k_, sse_set = 0, []
        for k_ in range(k_from, k_from + 3):
            kmeans = KMeans(n_clusters=k_).fit(vectors)  # 试聚类
            sse = kmeans.inertia_ / cell_norm_factor
            G.log.debug('pilot clustering. k=%d, normSSE= %f', k_, sse)
            sse_set.append(sse)
        last_indicator = (sse_set[0] + sse_set[2]) / sse_set[1]  # 二阶微分的相对值
        last_k = k_from + 2

        maxima = None  # (k, kmeans, sse, indicator)
        prefer = (0, None, 0, 0, 0)  # (K_, kmeans, sse, indicator, ratio of top5 lables)
        last_top5_value, last_top5_idx = 100, 1
        for k_ in range(k_from + 3, self.__MaxCategory):
            kmeans = KMeans(n_clusters=k_).fit(vectors)  # 试聚类
            sse = kmeans.inertia_ / cell_norm_factor
            G.log.debug('pilot clustering. k=%d, normSSE= %f', k_, sse)
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
                if prefer[3] < maxima[3] and top5_ratio < self.__Top5Ratio:  # Top5Label中样本比例有效（没有失衡)
                    prefer = maxima + [top5_ratio]
                G.log.info('Maxima point. k=(%d,%.2f) normSSE=%.2f, Top%d labels=%.1f%%. Preferred (%d,%.2f)',
                           maxima[0], maxima[3], maxima[2], n, top5_ratio * 100, prefer[0], prefer[3])
                maxima = None  # 变量复位，准备下一个极大值点
                if top5_ratio < last_top5_value - 0.001:
                    last_top5_value = top5_ratio
                    last_top5_idx = k_
                else:
                    if k_ - last_top5_idx > 50:  # 连续50个K_比例不降
                        break

            if sse < 1:  # 已经收敛到很小且找到可选值，没必要继续增加
                break

            sse_set[-1] = sse  # 如无异常误差点，这些操作只是重复赋值，无影响。如有，更新当前点值，准备下一循环
            sse_set[-2] = sse_set[-1] - sse_step
            last_indicator = indicator
            last_k = k_

        G.log.info('pilot-clustering[k:1-%d] finished. preferred k=(%d, %.2f),normSSE=%.2f, TopN labels=%.1f%%'
                   % (k_, prefer[0], prefer[3], prefer[2], prefer[4] * 100))
        return prefer[0]

    # 重新聚类，得到各Cluster的中心点、分位点距离、边界距离以及数量占比等
    def __buildClusterModel( self, k_, vectors ):
        # 再次聚类并对结果分组。 Kmeans不支持余弦距离
        kmeans = KMeans(n_clusters=k_, n_init=20, max_iter=500).fit(vectors)
        norm_factor = - vectors.shape[1]  # 按字典宽度归一化，保证不同模型的可比性
        groups = DataFrame({'C': kmeans.labels_, 'S': [kmeans.score([v]) / norm_factor for v in vectors]}).groupby('C')
        alias = ['Type' + str(i) for i in range(k_)]  # 簇的别名，默认为Typei，可人工命名
        proportions = groups.size() / len(vectors)  # 该簇向量数在聚类总向量数中的占比
        quantiles = np.array([groups.get_group(i)['S'].quantile(self.__Quantile, interpolation='higher')
                              for i in range(k_)])
        boundaries = groups['S'].agg('max').values - quantiles  # 该簇中最远点到分位点的距离
        for i in range(k_):
            if boundaries[i] > quantiles[i]:  # 边界太远的话，修正一下
                boundaries[i] = quantiles[i]
            elif boundaries[i] == 0:  # 避免出现0/0
                boundaries[i] = 1e-100

        G.log.info('Model(k=%d) built. inertia=%.3f， max proportion=%.2f%%, max quantile=%.3f, max border=%.3f',
                   k_, kmeans.inertia_, max(proportions) * 100, max(quantiles), max(boundaries))
        return kmeans, alias, proportions, boundaries, quantiles

    # predict from file or list of lines，返回“类别id、名称，记录的置信度、时间戳、原文和词表”的列表
    # 置信度最大99.99，如>1, 表示到中心点距离小于0.8分位点，非常可信；最小-99.99，如< 0距离大于最远点，意味着不属于此类
    def predict( self, dataset ):
        """

        :param dataset:
        :return:
        """
        if self.__k is None:
            raise UserWarning('Failed to predict: Model is not exist!')

        if type(dataset) is str:  # 如输入文件名，读取内容，得到dataset
            if not os.path.exists(dataset):  # $绝对路径
                dataset = os.path.join(G.l1_cache, dataset)
            dataset = [line for line in open(dataset, encoding='utf-8')]

        timestamps, records, documents = self.__getCorpus(dataset)
        vectors = self.__getVectors(documents)  # 计算向量[vector:Record*dictionary]
        type_ids, type_names, confidences = self.__getResult(vectors)  # 预测分类并计算可信度
        return type_ids, type_names, confidences, timestamps, records, documents

    # 预处理原始数据集，返回时间戳锚点、记录和文档的列表[timestamps] [records] [documents]
    def __getCorpus( self, dataset ):
        timestamps, records, documents = [], [], []  # 初始化返回数据
        record = ''
        timestamp = None
        for new_line in dataset:
            try:
                # 获得记录的锚点时间戳，没有返回None
                next_timestamp = self.__anchor.getAnchorTimeStamp(new_line)

                # 判断定界位置是否是时间戳，形成一条完整record
                absent = next_timestamp is None
                if absent or (record == ''):
                    if absent ^ (record == ''):  # 开始行是定界，或者当前行不是定界行，表示尚未读到下一记录
                        if not absent:
                            timestamp = next_timestamp  # 需要改！！！！
                        record += new_line
                    continue
                words = G.getWords(record, rule_set=self.__ruleSet)  # 完整记录Record(变量替换/停用词/分词/Kshingle)-〉[word]

                timestamps.append(timestamp)  # 保存时间戳
                records.append(record)  # 保存记录
                documents.append(words)  # 保存语料
                timestamp = next_timestamp  # 保存下一个时间戳
                record = new_line  # 当前行存入当前记录，准备下次循环
            except (UnicodeError, UnicodeDecodeError, UnicodeEncodeError):
                G.log.exception('Record [%s] ignored due to the following error:', record)
                record = ''  # 清空并丢弃现有记录
                continue

        if record != '':  # 处理最后一行
            try:
                words = G.getWords(record, rule_set=self.__ruleSet)  # 完整记录Record(变量替换/停用词/分词/Kshingle)-〉[word]
                timestamps.append(timestamp)  # 保存时间戳
                records.append(record)  # 保存记录
                documents.append(words)  # 保存语料
            except (UnicodeError, UnicodeDecodeError, UnicodeEncodeError):
                G.log.exception('Record [%s] ignored due to the following error:', record)

        return timestamps, records, documents

    # 构造不归一的IF_IDF词袋和文档向量
    def __getVectors( self, corpus ):
        tfidf_model = TfidfModel(dictionary=self.__d, normalize=False)
        vectors = np.zeros((len(corpus), len(self.__d)))
        for doc_idx, document in enumerate(corpus):
            for (word_idx, tf_idf_value) in tfidf_model[self.__d.doc2bow(document)]:  # [(id,tf-idf)...], id是升序
                vectors[doc_idx, word_idx] = tf_idf_value
        return vectors

    # 预测分类并计算可信度。<0 表示超出边界，完全不对，〉1完全表示比分位点还近，非常可信
    def __getResult( self, vectors ):
        type_ids = self.__k.predict(vectors)  # 使用聚类模型预测记录的类别
        type_names = [self.__a[label] for label in type_ids]
        norm_factor = - vectors.shape[1]  # 按字典宽度标准化，保证不同模型的可比性
        confidences = [1 + round((self.__k.score([v]) / norm_factor + self.__q[type_ids[i]])
                                 / (self.__q[type_ids[i]] + self.__b[type_ids[i]]), 2) for i, v in
                       enumerate(vectors)]  #####公式不对，应该改正
        confidences = np.array(confidences, copy=False)
        confidences[confidences > 99.99] = 99.99  # 太大或太小没有太多意义
        confidences[confidences < -99.99] = -99.99
        #        labels[confidences <= 0] = len(vectors) + 1  # 不可信的置入新的一类

        return type_ids, type_names, confidences


if __name__ == '__main__':
    if os.path.exists(G.l2FilesList):
        file_list = open(G.l2FilesList, 'r', encoding='utf-8').readlines()
    else:
        file_list = [file_name for file_name in os.listdir(G.l2_cache)]

    errors = 0
    filename, index = '', 0
    for index, line in enumerate(file_list):
        try:
            inputs = line.split('\t')
            filename = inputs[0].strip()
            G.log.info('[%d]%s: Record classifying...', index, filename)
            rc = Categorizer(filename)
        except Exception as err:
            errors += 1
            G.log.exception('%s ignored due to: %s', filename, str(err))
            continue
    G.log.info('%d files classified, %d failed.', index + 1 - errors, errors)
