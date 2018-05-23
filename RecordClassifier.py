#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : RecordClassifier @Author: Sui Huafeng
# @Date  : 2018/3
# @Desc  : 从$DATA/l2cache中每一个样本文件中训练一个日志记录分类器


import os
from tempfile import TemporaryFile

import numpy as np
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from pandas import DataFrame
from sklearn.cluster import KMeans
from sklearn.externals import joblib

from anchor import Anchor
from config import Workspaces as G
from utilites import FileUtil


# 对一个数千到数十万行的文件中的记录进行聚类，形成聚类模型
class RecordClassifier(object):
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
    __ClassName = 'RecordClassifier'
    __LeastDocuments = G.cfg.getint(__ClassName, 'LeastRecords')  # 样本数小于该值，没有必要聚类
    __LeastTokens = G.cfg.getint(__ClassName, 'LeastTokens')  # 字典最少词数，低于此数没必要聚类
    __KeepN = G.cfg.getint(__ClassName, 'KeepN')  # 字典中最多词数，降低计算量
    __NoBelow = G.cfg.getfloat(__ClassName, 'NoBelow')  # 某词出现在文档数低于该比例时，从字典去掉，以排除干扰，降低计算量

    __Top5Ratio = G.cfg.getfloat(__ClassName, 'Top5Ratio')  # Top5类中样本数占总样本比例。大于该值时聚类结果不可接受
    __MaxCategory = G.cfg.getint(__ClassName, 'MaxCategory')  # 尝试聚类的最大类别，以降低计算量
    __Quantile = G.cfg.getfloat(__ClassName, 'Quantile')  # 类别的边界，该类中以该值为分位数的点，作为该类的边界

    def __init__(self, model_file='', samples_file='', anchor=None):
        self.ruleSet = None  # 处理文件正则表达式
        self.dictionary = None  # 字典Dictionary
        self.models = None  # 聚类模型 KMeans
        self.alias = None  # cluster名称
        self.percent = None  # 该类别数量占比
        self.boundaries = None  # 最远点到(0.8)分位点距离平方
        self.quantiles = None  # (0.8)分位点到中心点距离平方
        self.anchor = anchor  # 时间戳锚点Anchor

        self.tmp_cache_file = None

        if os.path.exists(model_file):  # 从模型文件装载模型
            self.anchor, self.ruleSe, self.dictionary, self.models, self.alias, self.percent, self.boundaries, self.quantiles = joblib.load(
                model_file)

        if os.path.exists(samples_file):  # 从样本训练模型
            if not self.anchor:  # 从样本文件中提取时间戳锚点
                self.anchor = Anchor(samples_file, probe_date=True)
            self.buildModel(samples_file)

    def buildModel(self, samples_file):
        for self.ruleSet in FileUtil.loadRuleSets():
            # 日志文件预处理为记录向量,并形成字典。vectors是稀疏矩阵(行-记录，列-词数)
            self.dictionary, vectors = self.__buildVectors(samples_file)
            if not self.dictionary:  # 字典太短，换rule set重新采样
                continue
            start_k = int(min(len(self.dictionary) / 10, vectors.shape[0] / 100))
            k_ = self.pilotClustering(self.__ClassName, vectors, start_k)  # 多个K值试聚类，返回最佳K
            if k_:  # 找到合适的K，跳出循环, 否则换rule set重新采样
                break
        else:
            raise UserWarning('Cannot generate qualified corpus by all RuleSets')

        self.models, self.percent, self.boundaries, self.quantiles = FileUtil.buildModel(self.__ClassName, k_, vectors)
        self.alias = ['rc' + str(i) for i in range(len(self.quantiles))]  # 簇的别名，默认为rci，可人工命名

        self.__saveModel(samples_file)
        self.__saveResult(samples_file, vectors)

    # 文档向量化。dataset-[document:M]-[[word]]-[[token]]-[BoW:M]-corpus-tfidf-dictionary:N, [vector:M*N]
    def __buildVectors(self, samples_file):
        lines = 0
        dct = Dictionary()
        self.tmp_cache_file = TemporaryFile(mode='w+t', encoding='utf-8')
        for doc_idx, (document, lines) in enumerate(self.__buildDocument(samples_file)):
            dct.add_documents([document])
            self.tmp_cache_file.write(' '.join(document) + '\n')
            if doc_idx % 500 == 0: G.log.debug('Processed %d records in %s', doc_idx, samples_file)
        if dct.num_docs < self.__LeastDocuments:  # 字典字数太少或文档数太少，没必要聚类
            self.tmp_cache_file.close()
            raise UserWarning('Too few records[%d]' % dct.num_docs)

        # 去掉低频词，压缩字典
        num_token = len(dct)
        no_below = int(min(self.__NoBelow, int(dct.num_docs / 50)))
        dct.filter_extremes(no_below=no_below, no_above=0.999, keep_n=self.__KeepN)
        dct.compactify()
        G.log.info('Dictionary[%d tokens, reduced from %d] built with [%s]. [%d]records(%d lines, %d words) in %s',
                   len(dct), num_token, self.ruleSet[0], dct.num_docs, lines, dct.num_pos, samples_file)
        if len(dct) < self.__LeastTokens:  # 字典字数太少,重新采样
            G.log.info('Too few tokens[%d], Re-sample with next RuleSet].' % (len(dct)))
            self.tmp_cache_file.close()
            return None, None

        # 构造tf-idf词袋和文档向量
        tfidf_model = TfidfModel(dictionary=dct, normalize=False)
        vectors = np.zeros((dct.num_docs, len(dct)))
        self.tmp_cache_file.seek(0)
        for doc_idx, new_line in enumerate(self.tmp_cache_file):
            for (word_idx, tf_idf_value) in tfidf_model[dct.doc2bow(new_line.split())]:  # [(id,tf-idf)...], id是升序
                vectors[doc_idx, word_idx] = tf_idf_value
        G.log.info('[%d*%d]Vectors built, %.2f%% non-zeros.' % (
            dct.num_docs, len(dct), dct.num_nnz * 100 / len(dct) / dct.num_docs))
        return dct, vectors

    # 预处理文件，迭代方式返回某条记录的词表.
    def __buildDocument(self, samples_file):
        line_idx, record = 0, ''
        for line_idx, next_line in enumerate(open(samples_file, 'r', encoding='utf-8')):
            try:
                # 判断定界位置是否为恰好是时间戳，形成一条完整record
                absent = self.anchor.getTimeStamp(next_line) is None
                if absent or (record == ''):
                    if absent ^ (record == ''):  # 开始行是定界，或者当前行不是定界行，表示尚未读到下一记录
                        record += next_line
                    continue

                document = G.getWords(record, rule_set=self.ruleSet)  # 完整记录Record(变量替换/停用词/分词/Kshingle)-〉[word]
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
                yield G.getWords(record, rule_set=self.ruleSet), line_idx
            except (UnicodeError, UnicodeDecodeError, UnicodeEncodeError):
                G.log.exception('Record [%s] ignored due to the following error:', record)
        raise StopIteration()

    # 聚类，得到各簇SSE（sum of the squared errors)，作为手肘法评估确定ｋ的依据
    @staticmethod
    def pilotClustering(classifier_section_name, vectors, from_=1):
        norm_factor = vectors.shape[1] * vectors.shape[0]  # 按行/样本数和列/字典宽度标准化因子，保证不同向量的可比性
        termination_inertia = G.cfg.getfloat(classifier_section_name, 'NormalizedTerminationInertia') * norm_factor

        k_ = G.cfg.getint(classifier_section_name, 'MaxCategory')

        kmeans = KMeans(n_clusters=k_, tol=1e-5).fit(vectors)  # 试聚类
        G.log.info('pilot clustering. (k,inertia)=\t%d\t%e', k_, kmeans.inertia_)
        if kmeans.inertia_ > termination_inertia:
            return k_

        # while k_ < len(vectors):
        #     kmeans = KMeans(n_clusters=k_, tol=1e-5).fit(vectors)  # 试聚类
        #     G.log.info('pilot clustering. (k,inertia)=\t%d\t%e', k_, kmeans.inertia_)
        #     if kmeans.inertia_ < termination_inertia:
        #         break
        #     k_ *= 2

        to_ = k_
        while to_ - from_ > 1:
            k_ = int(sum([from_, to_]) / 2)
            kmeans = KMeans(n_clusters=k_, tol=1e-5).fit(vectors)  # 试聚类
            G.log.info('pilot clustering. (k,inertia)=\t%d\t%e', k_, kmeans.inertia_)
            if kmeans.inertia_ < termination_inertia:
                to_ = k_
            else:
                from_ = k_
        G.log.info('pilot-cluster finished. preferred k=%d', to_)
        return to_

    # 保存模型和结果
    def __saveModel(self, samples_file_fullname):
        samples_file = os.path.splitext(os.path.split(samples_file_fullname)[1])[0]
        joblib.dump((self.anchor, self.ruleSet, self.dictionary, self.models, self.alias, self.percent,
                     self.boundaries, self.quantiles),
                    os.path.join(G.projectModelPath, samples_file + '.mdl'))  # 保存模型，供后续使用
        self.dictionary.save_as_text(os.path.join(G.logsPath, samples_file + '.dic.csv'))  # 保存文本字典，供人工审查
        df = DataFrame({'类型': self.alias, '样本占比': self.percent, '分位点距离': self.quantiles, '边界-分位点距离': self.boundaries})
        df.to_csv(os.path.join(G.logsPath, samples_file + '.mdl.csv'), sep='\t', encoding='utf-8')  # 保存聚类模型，供人工审查
        G.log.info('Model saved to %s successful.', os.path.join(G.projectModelPath, samples_file + '.mdl'))

    def __saveResult(self, samples_file_fullname, vectors):
        c_ids, c_names, confidences, distances = FileUtil.getPredictResult(self.models, self.alias, self.boundaries,
                                                                           self.quantiles, vectors)
        self.tmp_cache_file.seek(0)
        records = [line for line in self.tmp_cache_file]
        df = DataFrame({'Category': c_names, 'Confidence': confidences, 'Record': records})
        # df = DataFrame(columns=('Category', 'Confidence', 'Record'))
        # for idx, (c_name, confidence, record) in enumerate(zip(c_names, confidences, self.tmp_cache_file)):
        #     df.loc[idx] = c_name, confidence, record
        samples_file = os.path.splitext(os.path.split(samples_file_fullname)[1])[0]
        df.to_csv(os.path.join(G.logsPath, samples_file + '.out.csv'), sep='\t', encoding='utf-8')
        self.tmp_cache_file.close()
        G.log.info('Result saved to %s successful.', os.path.join(G.logsPath, samples_file + '.out.csv'))

    # predict from txt file or txt data flow
    def predict(self, data_stream):
        """
         predict from file or list of lines，
        置信度最大99.99，如>1, 表示到中心点距离小于0.8分位点，非常可信；最小-99.99，如< 0距离大于最远点，意味着不属于此类
        :param data_stream: samples file name or data stream
        :return: records [[category_id、category_alias, confidence, timestamp, record, bag_of_word]]
        """
        if not self.models:
            raise UserWarning('Failed to predict: Model is not exist!')

        if type(data_stream) is str:  # 如输入文件名，读取内容，得到dataset
            data_stream = open(data_stream, encoding='utf-8')

        timestamp, record = None, ''
        for next_line in data_stream:
            try:
                # 获得记录的锚点时间戳，没有返回None
                next_timestamp = self.anchor.getTimeStamp(next_line)

                # 判断定界位置是否是时间戳，形成一条完整record
                absent = next_timestamp is None
                if absent or (record == ''):
                    if absent ^ (record == ''):  # 开始行是定界，或者当前行不是定界行，表示尚未读到下一记录
                        if not absent:
                            timestamp = next_timestamp
                        record += next_line
                    continue
                result = self.__predictRecord(timestamp, record)
                timestamp = next_timestamp  # 保存下一个时间戳
                record = next_line  # 当前行存入当前记录，准备下次循环
                yield result
            except (UnicodeError, UnicodeDecodeError, UnicodeEncodeError):
                G.log.exception('Record [%s] ignored due to the following error:', record)
                record = ''  # 清空并丢弃现有记录
                continue
        # 处理最后一行
        if record != '':
            try:
                result = self.__predictRecord(timestamp, record)
                yield result
            except (UnicodeError, UnicodeDecodeError, UnicodeEncodeError):
                G.log.exception('Record [%s] ignored due to the following error:', record)
        raise StopIteration

    def __predictRecord(self, timestamp, record):
        words = G.getWords(record, rule_set=self.ruleSet)  # 完整记录Record(变量替换/停用词/分词/Kshingle)-〉[word]
        vectors = self.__getVectors([words])  # 计算向量[vector:Record*dictionary]
        c_ids, c_names, confidences, distances = FileUtil.getPredictResult(self.models, self.alias, self.boundaries,
                                                                           self.quantiles, vectors)
        return timestamp, c_ids[0], c_names[0], confidences[0], distances[0], record, words

    # 构造不归一的IF_IDF词袋和文档向量
    def __getVectors( self, corpus ):
        tfidf_model = TfidfModel(dictionary=self.dictionary, normalize=False)
        vectors = np.zeros((len(corpus), len(self.dictionary)))
        for doc_idx, document in enumerate(corpus):
            for (word_idx, tf_idf_value) in tfidf_model[self.dictionary.doc2bow(document)]:  # [(id,tf-idf)...], id是升序
                vectors[doc_idx, word_idx] = tf_idf_value
        return vectors


if __name__ == '__main__':
    errors = 0
    filename, index = '', 0
    for index, filename in enumerate(os.listdir(G.l2_cache)):
        try:
            filename = os.path.join(G.l2_cache, filename)
            if os.path.isfile(filename):
                G.log.info('[%d]%s: Record classifying...', index, filename)
                rc = RecordClassifier(samples_file=filename)
        except Exception as err:
            errors += 1
            G.log.error('%s ignored due to: %s', filename, str(err))
            continue
    G.log.info('%d models built and stored in %s, %d failed.', G.projectModelPath, index + 1 - errors, errors)
