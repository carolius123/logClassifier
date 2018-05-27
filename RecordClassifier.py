#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : RecordClassifier @Author: Sui Huafeng
# @Date  : 2018/3
# @Desc  : 从$DATA/l2cache中每一个样本文件中训练一个日志记录分类器


import os
import re
import time
from tempfile import TemporaryFile

import numpy as np
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from pandas import DataFrame
from sklearn.cluster import KMeans
from sklearn.externals import joblib

from anchor import Anchor
from config import Workspaces as G
from utilites import Dbc, FileUtil


# 对一个数千到数十万行的文件中的记录进行聚类，形成聚类模型
class RecordClassifier(object):
    """
    以模型文件名为参数创建对象或者调用loadModel方法，可装载rc模型
    以样本文件名或样本文件列表为参数创建对象,或者调用buildModel方法, 以该这些文件为样本聚类生成rc模型
    以日志流未参数调用predictRecord方法, 迭代返回每条记录及其时间戳, 类型及其置信度
    3. 后续处理可能有价值的目标包括：
    3.1 数字变量中可能包含错误码、代码等高值信息，可考虑提取和利用的手段
    3.2 对于记录数特别多的类别(太粗),可根据枚举型、IP、URL、错误码等进一步细化分类
    3.4 数字变量中可能包含时长、数量等指标数据，可考虑提取和利用的手段
    """
    __ClassName = 'RecordClassifier'
    leastDocuments = G.cfg.getint(__ClassName, 'LeastRecords')  # 样本数小于该值，没有必要聚类
    __LeastTokens = G.cfg.getint(__ClassName, 'LeastTokens')  # 字典最少词数，低于此数没必要聚类
    __KeepN = G.cfg.getint(__ClassName, 'KeepN')  # 字典中最多词数，降低计算量
    __NoBelow = G.cfg.getfloat(__ClassName, 'NoBelow')  # 某词出现在文档数低于该比例时，从字典去掉，以排除干扰，降低计算量

    __Top5Ratio = G.cfg.getfloat(__ClassName, 'Top5Ratio')  # Top5类中样本数占总样本比例。大于该值时聚类结果不可接受
    __MaxCategory = G.cfg.getint(__ClassName, 'MaxCategory')  # 尝试聚类的最大类别，以降低计算量
    __Quantile = G.cfg.getfloat(__ClassName, 'Quantile')  # 类别的边界，该类中以该值为分位数的点，作为该类的边界

    def __init__(self, model_or_sample):
        self.anchor = None  # 时间戳锚点Anchor
        self.ruleSet = None  # 处理文件正则表达式
        self.dictionary = None  # 字典Dictionary
        self.model = None  # 聚类模型 KMeans
        self.categories = None

        if type(model_or_sample) is str:
            if not os.path.exists(model_or_sample):
                G.log.warning('[err]File not found. No model loaded!')
                return
            try:  # 从模型文件装载模型
                self.anchor, self.ruleSet, self.dictionary, self.model, self.categories = joblib.load(model_or_sample)
                match_ = re.match(r'fc(\d+)-(\d+)', os.path.split(model_or_sample)[1])
                if not match_:
                    raise UserWarning('Filename format err')
                self.model_id, self.fc_id = match_.groups()
                self.model_id, self.fc_id = int(self.model_id), int(self.fc_id)
                self.model_path = os.path.join(G.modelPath, str(self.model_id))
                return
            except Exception:
                sample_file_list = [model_or_sample]
        else:
            sample_file_list = model_or_sample

        if type(sample_file_list) is not list:
            raise UserWarning('parameter should be model/sample file name, sample file name list. No model loaded!')

        match_ = re.match(r'fc(\d+)-(\d+)', os.path.split(sample_file_list[0])[1])
        if not match_:
            raise UserWarning('Filename format err')
        self.model_id, self.fc_id = match_.groups()
        self.model_id, self.fc_id = int(self.model_id), int(self.fc_id)
        self.model_path = os.path.join(G.modelPath, str(self.model_id))

        self.tmp_cache_file = None  # 样本临时文件指针,需跨多层使用,参数传递麻烦
        self.buildModel(sample_file_list)

    def buildModel(self, samples_file_list):
        """
    日志文件中每条记录大都会在第一行特定位置输出日期和时间，因此以特定位置的时间戳判断上一记录的结束和下一记录的开始
    记录聚类的首要目标是把程序执行到某处输出的记录汇聚成一类。程序在某点输出日志，大致会包含几种信息：
    - 日期、时间等几乎所有记录都包含的共有信息：这些对聚类结果没有影响，不必单独考虑
    - 常数字符串和枚举型数据(如INFO、ERROR等): 这是这一类记录最典型的独有特征，应尽量突出其在聚类中的比重
    - URL、IP、Java包、path等变量：应尽量识别出来并归一化成常数字符串，改善聚类效果
    - 字符串变量(应用系统的数据)和数字变量：上两类有效特征在每个记录中不会超过10个，字符串/数字变量可能会很多，这会严重干扰聚类效果、影响运算速度，应尽量减少。数字可简单滤除。字符串变量可考虑去掉dfs字典中低频词
        :param samples_file_list: sample file or sample file list.
        """
        try:
            for self.ruleSet in FileUtil.loadRuleSets():
                # 日志文件预处理为记录向量,并形成字典。vectors是稀疏矩阵(行-记录，列-词数)
                self.dictionary, vectors = self.__buildVectors(samples_file_list)
                if not self.dictionary:  # 字典太短，换rule set重新采样
                    continue
                # start_k = int(min(len(self.dictionary) / 10, vectors.shape[0] / 100))
                k_ = self.__pilotClustering(self.__ClassName, vectors)  # 多个K值试聚类，返回最佳K
                if k_:  # 找到合适的K，跳出循环, 否则换rule set重新采样
                    break
            else:
                raise UserWarning('Cannot generate qualified corpus by all RuleSets')
        except UserWarning:
            self.__saveToDb('无模型')  # 更新本模型对应的日志文件类型的数据库记录
            raise

        self.model, self.categories = FileUtil.buildModel(self.__ClassName, k_, vectors)
        self.__saveModel()
        self.__saveResult(vectors)

    # 文档向量化。dataset-[document:M]-[[word]]-[[token]]-[BoW:M]-corpus-tfidf-dictionary:N, [vector:M*N]
    def __buildVectors(self, samples_file):
        lines = 0
        dct = Dictionary()
        self.tmp_cache_file = TemporaryFile(mode='w+t', encoding='utf-8')
        for doc_idx, (document, lines) in enumerate(self.__buildDocument(samples_file)):
            dct.add_documents([document])
            self.tmp_cache_file.write(' '.join(document) + '\n')
            if doc_idx % 1000 == 0:
                G.log.info('Processed %d records in %s', doc_idx, samples_file)
            if doc_idx > 5000:  # 防止记录太多太慢
                break
        if dct.num_docs < self.leastDocuments:  # 字典字数太少或文档数太少，没必要聚类
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
    def __buildDocument(self, samples_file_list):
        line_idx, record = 0, ''
        for samples_file in samples_file_list:
            if not self.anchor:  # 从样本文件中提取时间戳锚点
                self.anchor = Anchor(samples_file, probe_date=True)

            for line_idx, next_line in enumerate(open(samples_file, 'r', encoding='utf-8')):
                try:
                    # 判断定界位置是否为恰好是时间戳，形成一条完整record
                    absent = self.anchor.getTimeStamp(next_line) is None
                    if absent or (record == ''):
                        if absent ^ (record == ''):  # 开始行是定界，或者当前行不是定界行，表示尚未读到下一记录
                            record += next_line
                        continue

                    # 完整记录Record(变量替换/停用词/分词/Kshingle)-〉[word]
                    document = FileUtil.getWords(record, rule_set=self.ruleSet)
                    # 得到词表，并准备下次循环
                    record = next_line  # 当前行存入当前记录
                    yield document, line_idx
                except (UnicodeError, UnicodeDecodeError, UnicodeEncodeError):
                    G.log.exception('Record [%s] ignored due to the following error:', record)
                    record = ''  # 清空并丢弃现有记录
                    continue
            # 处理最后一个记录
            if record != '':
                try:
                    # 完整记录Record(变量替换/停用词/分词/Kshingle)-〉[word]
                    yield FileUtil.getWords(record, rule_set=self.ruleSet), line_idx
                except (UnicodeError, UnicodeDecodeError, UnicodeEncodeError):
                    G.log.exception('Record [%s] ignored due to the following error:', record)
        raise StopIteration()

    # 聚类，得到各簇SSE（sum of the squared errors)，作为手肘法评估确定ｋ的依据
    @staticmethod
    def __pilotClustering(classifier_section_name, vectors, from_=1):
        norm_factor = vectors.shape[1] * vectors.shape[0]  # 按行/样本数和列/字典宽度标准化因子，保证不同向量的可比性
        termination_inertia = G.cfg.getfloat(classifier_section_name, 'NormalizedTerminationInertia') * norm_factor

        k_ = min(G.cfg.getint(classifier_section_name, 'MaxCategory'), vectors.shape[0])

        kmeans = KMeans(n_clusters=k_, tol=1e-5).fit(vectors)  # 试聚类
        G.log.info('pilot clustering. (k,inertia)=\t%d\t%e', k_, kmeans.inertia_)
        if kmeans.inertia_ > termination_inertia:
            return k_

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
    def __saveModel(self):
        samples_file = 'fc%d-%d' % (self.model_id, self.fc_id)
        joblib.dump((self.anchor, self.ruleSet, self.dictionary, self.model, self.categories),
                    os.path.join(self.model_path, samples_file + '.mdl'))  # 保存模型，供后续使用
        self.__saveToDb('无指标')  # 更新本模型对应的日志文件类型的数据库记录

        self.dictionary.save_as_text(os.path.join(G.logsPath, samples_file + '.dic.csv'))  # 保存文本字典，供人工审查
        df = DataFrame({'类型': self.categories[:, 0], '样本数': self.categories[:, 3], '坏点数': self.categories[:, 4],
                        '边界': self.categories[:, 1], '分位点-边界': self.categories[:, 2]})
        df.to_csv(os.path.join(G.logsPath, samples_file + '.mdl.csv'), sep='\t', encoding='utf-8')  # 保存聚类模型，供人工审查
        G.log.info('Model saved to %s successful.', os.path.join(self.model_path, samples_file + '.mdl'))

    def __saveToDb(self, status):
        with Dbc() as cursor:
            sql = 'UPDATE file_class SET status="%s"  WHERE model_id=%d AND category_id=%d' % (
            status, self.model_id, self.fc_id)
            cursor.execute(sql)

    def __saveResult(self, vectors):
        c_ids, confidences, distances = FileUtil.predict(self.model, self.categories[:, 1:3], vectors)
        self.tmp_cache_file.seek(0)
        records = [line for line in self.tmp_cache_file]
        df = DataFrame({'Category': c_ids, 'Confidence': confidences, 'Record': records})
        samples_file = 'fc%d-%d' % (self.model_id, self.fc_id)
        df.to_csv(os.path.join(G.logsPath, samples_file + '.out.csv'), sep='\t', encoding='utf-8')
        self.tmp_cache_file.close()
        G.log.info('Result saved to %s successful.', os.path.join(G.logsPath, samples_file + '.out.csv'))

    # predict from txt file or txt data flow
    def predictRecords(self, data_stream):
        """
         predict from file or list of lines，
        置信度最大99.99，如>1, 表示到中心点距离小于0.8分位点，非常可信；最小-99.99，如< 0距离大于最远点，意味着不属于此类
        :param data_stream: samples file name or data stream
        :return: records iterator of [timestamp, record category, confidence, record content]
        """
        if not self.model:
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

                c_ids, confidences, distances = self.__predict(record)
                timestamp = next_timestamp  # 保存下一个时间戳
                record = next_line  # 当前行存入当前记录，准备下次循环
                yield timestamp, c_ids, confidences, distances, record
            except (UnicodeError, UnicodeDecodeError, UnicodeEncodeError):
                G.log.exception('Record [%s] ignored due to the following error:', record)
                record = ''  # 清空并丢弃现有记录
                continue
        # 处理最后一行
        if record != '':
            try:
                c_ids, confidences, distances = self.__predict(record)
                yield timestamp, c_ids, confidences, distances, record
            except (UnicodeError, UnicodeDecodeError, UnicodeEncodeError):
                G.log.exception('Record [%s] ignored due to the following error:', record)
        raise StopIteration

    # 构造不归一的IF_IDF词袋和文档向量
    def __buildVector(self, corpus):
        tfidf_model = TfidfModel(dictionary=self.dictionary, normalize=False)
        vectors = np.zeros((len(corpus), len(self.dictionary)))
        for doc_idx, document in enumerate(corpus):
            for (word_idx, tf_idf_value) in tfidf_model[self.dictionary.doc2bow(document)]:  # [(id,tf-idf)...], id是升序
                vectors[doc_idx, word_idx] = tf_idf_value
        return vectors

    #
    def __predict(self, record):
        words = FileUtil.getWords(record, rule_set=self.ruleSet)
        vectors = self.__buildVector([words])  # 计算向量[vector:Record*dictionary]
        c_ids, confidences, distances = FileUtil.predict(self.model, self.categories[:, 1:3], vectors)
        return c_ids, confidences, distances

    # 新的日志流连接建立后,如该日志流已有文件类型fc, 则被分发到本入口
    @classmethod
    def dispatcher(cls, flow_id, data_flow):
        status, model_id, fc_id, rc_model = cls.__getRcModel(flow_id)

        # 尚未形成记录模型
        if status == '无模型':
            status = cls.__cacheLogRecords(model_id, fc_id, flow_id, data_flow)
        # 缓存行数已经足够,本线程启动聚类
        if status == 'OK':
            prefix = 'fc%d' % fc_id
            samples = sorted([os.path.join(G.classifiedFilePath, f)
                              for f in os.listdir(G.classifiedFilePath) if f[:len(prefix)] == prefix])
            try:
                rc_model = RecordClassifier(samples)
                status = '无指标'
            except UserWarning:
                status = '无模型'

        # 别的线程正在算记录模型, 等待其完成
        while status == '计算中':
            time.sleep(60)
            status, model_id, fc_id, rc_model = cls.__getRcModel(flow_id)

        if status in ['无指标', '无基线', '已完备']:
            if rc_model:
                return

    # 把流数据缓存到文件中 ,并记录已经缓存的行数
    @classmethod
    def __cacheLogRecords(cls, model_id, fc_id, flow_id, data_flow):
        file_fullname = os.path.join(G.classifiedFilePath, 'fc%d-%d' % (fc_id, flow_id))
        received_lines, now = 0, time.time()

        with open(file_fullname, 'a', encoding='utf-8', errors='ignore') as fp:
            for line in data_flow:
                fp.write(line)
                received_lines += 1
                if time.time() - now > 30 and received_lines:
                    status = cls.__dbUpdFileClass(model_id, fc_id, received_lines)
                    if status != '无模型':
                        return status
                    received_lines, now = 0, time.time()
            else:  # 数据流关闭
                return cls.__dbUpdFileClass(model_id, fc_id, received_lines)

    @staticmethod
    def __dbUpdFileClass(model_id, fc_id, received_lines):
        with Dbc() as cursor:
            sql = 'SELECT status,total_lines from file_class WHERE model_id=%d AND category_id=%d' % (model_id, fc_id)
            cursor.execute(sql)
            status, total_lines = cursor.fetchone()
            if status != '无模型':  # 被别的同类日志线程修改了状态
                return status
            least_lines = RecordClassifier.leastDocuments * 10
            total_lines += received_lines
            status = '计算中' if total_lines > least_lines else '无模型'
            sql = 'UPDATE file_class SET total_lines=%d,status="%s" WHERE model_id=%d AND category_id=%d' % (
                total_lines, status, model_id, fc_id)
            cursor.execute(sql)
            status = 'OK' if total_lines > least_lines else '无模型'
        return status

    @staticmethod
    def __getRcModel(flow_id):
        with Dbc() as cursor:
            sql = 'SELECT f.model_id,f.category_id,f.status FROM tbl_log_flow AS t, file_class as f WHERE t.model_id=f.model_id AND t.category_id=f.category_id AND id = %d' % flow_id
            cursor.execute(sql)
            model_id, fc_id, status = cursor.fetchone()

        if status in ['无模型', '计算中']:
            rc_model = None
        else:
            rc_model_file = os.path.join(G.modelPath, str(model_id))
            rc_model_file = os.path.join(rc_model_file, 'fc%d-%d.mdl' % (model_id, fc_id))
            rc_model = RecordClassifier(rc_model_file)
        return status, model_id, fc_id, rc_model



if __name__ == '__main__':
    RecordClassifier(['D:\\home\\suihf\\data\\classified\\fc0-23'])
    # errors = 0
    # filename, index = '', 0
    # for index, filename in enumerate(os.listdir(G.classifiedFilePath)):
    #     try:
    #         filename = os.path.join(G.classifiedFilePath, filename)
    #         if os.path.isfile(filename):
    #             G.log.info('[%d]%s: Record classifying...', index, filename)
    #             rc = RecordClassifier([filename])
    # except Exception as err:
    #     errors += 1
    #     G.log.error('%s ignored due to: %s', filename, str(err))
    #     continue
    # G.log.info('%d model built, %d failed.', index + 1 - errors, errors)
