#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : FileClassifier.py
# @Author: Sui Huafeng
# @Date  : 2018/4
# @Desc  : 从$DATA/input中所有样本文件中训练一个日志文件分类器
#

import os

import numpy as np
import pandas as pd
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from sklearn.externals import joblib

from config import Workspaces as G
from utilites import Dbc, FileUtil, DbUtil


# 对数十、上百万个文件进行聚类，形成聚类模型
class FileClassifier(object):
    """
    以目录名为参数创建对象,或者调用buildModel方法, 以该目录下文件作为样本聚类生成fc模型
    以文件名为参数创建对象或者调用loadModel方法，可装载fc模型
    调用predictFile或predictFiles方法, 可以预测新的日志文件类型及其置信度
    """
    __corpusCacheFile = os.path.join(G.projectModelPath, 'corpuscache.1')
    l1_dbf = os.path.join(G.projectModelPath, 'metadata.1')

    def __init__(self, model_file_or_samples_path):
        self.ruleSet = None  # 处理文件正则表达式
        self.dictionary = None  # 字典对象(Gensim Dictionary)
        self.statsScope = None  # 样本文件字符数、字数统计值(均值、标准差、中位数)的最小-最大值范围
        self.models = None  # 聚类模型(Kmeans)
        self.categories = None  # 聚类的类型[[名称]，[数量占比]，[分位点距离]，[边界点距离]]
        self.model_id = 1
        self.common_filenames, self.l1_structure = ([], [])

        if os.path.isfile(model_file_or_samples_path):  # 从模型文件装载模型
            self.loadModel(model_file_or_samples_path)
        elif os.path.isdir(model_file_or_samples_path):
            self.buildModel()
        else:
            G.log.warning('No models loaded!')

    def loadModel(self, model_file=G.projectFileClassifierModel):
        self.ruleSet, self.dictionary, self.statsScope, self.models, self.categories = joblib.load(model_file)
        if model_file == G.productFileClassifierModel:
            self.model_id = 0
        self.__dbUpdCategories()

    def __dbUpdCategories(self):
        with Dbc() as cursor:
            c = self.categories
            for category_id, (name, percent, boundary, quantile) in enumerate(zip(c[0], c[1], c[2], c[3])):
                name = '"%s"' % name
                sql = 'INSERT INTO file_class (model_id, category_id, name, quantile, boundary, percent) VALUES(%d, %d, %s, %e,%e, %f) ON DUPLICATE KEY UPDATE name=%s,quantile=%e,boundary=%e,percent=%f' % (
                    self.model_id, category_id, name, quantile, boundary, percent, name, quantile, boundary, percent)
                cursor.execute(sql)

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
        self.__saveModel()

    # 训练、生成模型并保存在$models/xxx.mdl中,dataset:绝对/相对路径样本文件名，或者可迭代样本字符流
    def buildModel(self, from_path=G.l1_cache, k_=0):
        """
        Train and generate K-Means Model
        :param from_path: source path contains merged log files
        :param k_: K-means parameter, 0 means auto detect
        """
        # 尝试不同的向量化规则，确定聚类数量K_
        for self.ruleSet in FileUtil.loadRuleSets():
            corpus_cache_fp = self.__buildDictionary(from_path)  # 建立字典self.dictionary，返回语料缓存文件
            if len(self.dictionary) < G.cfg.getint('Classifier', 'LeastTokens'):  # 字典字数太少,重新采样
                corpus_cache_fp.close()
                self.__clearCache()
                G.log.info('Too few tokens[%d], Re-sample with next RuleSet.', len(self.dictionary))
                continue
            corpus_cache_fp.seek(0)
            vectors = self.__buildVectors(corpus_cache_fp, self.dictionary.num_docs)  # 建立稀疏矩阵doc*(dct + stats)
            corpus_cache_fp.close()  # 关闭缓存文件
            start_k = min(50, int(vectors.shape[0] / 100))
            k_ = k_ if k_ else FileUtil.pilotClustering('Classifier', vectors, start_k)  # 多个K值试聚类，返回最佳K
            if k_ != 0:  # 找到合适的K，跳出循环
                break
            self.__clearCache()  # 清除缓存的corpus_cache_file
        else:
            raise UserWarning('Cannot generate qualified corpus by all RuleSets')
        # 重新聚类, 得到模型(向量数、中心点和距离）和分类(向量-所属类)
        self.models, percents, boundaries, quantiles = FileUtil.buildModel('Classifier', k_, vectors)
        names = ['fc%d' % i for i in range(len(percents))]
        self.categories = [names, percents, boundaries, quantiles]
        results = self.__getResult(vectors)
        # 保存模型文件和数据库
        self.__saveModel()  # models saved to file
        with Dbc() as cursor:
            file_and_results = [self.common_filenames] + list(results)
            classified_common_files, unclassified_common_files = FileUtil.splitResults(self.model_id, file_and_results)
            DbUtil.dbUdFilesMerged(cursor, classified_common_files, unclassified_common_files)
            FileUtil.mergeFilesByClass(cursor, G.l2_cache)  # 同类文件合并到to_目录下
            wildcard_log_files = FileUtil.genGatherList(cursor)  # 生成采集文件列表
            DbUtil.dbWildcardLogFiles(cursor, wildcard_log_files)

    # 建立词典，同时缓存词表文件
    def __buildDictionary(self, new_dataset_path):
        self.dictionary = Dictionary()
        lines = ''
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
        joblib.dump((self.ruleSet, self.common_filenames, self.l1_structure, self.statsScope), self.l1_dbf)

        return cache_fp

    # 预处理，迭代方式返回某个文件的词表.
    def __buildDocument(self, dataset_path):
        amount_files, failed_files, file_fullname = 0, 0, ''
        G.log.info('Start Converting documents from ' + dataset_path)
        processed_files = os.path.join(G.projectModelPath, 'buildDocument.dbf')
        processed = [] if not os.path.exists(processed_files) else joblib.load(processed_files)
        db = DbUtil.dbConnect()
        cursor = db.cursor() if db else None

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
                    if self.__hasClassified(cursor, file_fullname):
                        continue

                    yield self.__file2doc(file_fullname)
                except Exception as err:
                    failed_files += 1
                    G.log.warning('Failed to convert\t%s, ignored.\t%s', file_fullname, str(err))
                    continue
        joblib.dump(processed, processed_files)
        cursor.close()
        db.close()
        G.log.info('Converted %d files,%d failed', amount_files, failed_files)
        raise StopIteration()

    # 检查是否已经分过类
    @staticmethod
    def __hasClassified(cursor, common_name):
        if not cursor:
            return None
        common_name = '"%s"' % common_name.replace('\\', '/')
        sql = 'SELECT common_name FROM files_merged WHERE common_name=%s And files_merged.category_id IS NOT NULL' % common_name
        cursor.execute(sql)
        result = cursor.fetchone()
        return result

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
            if line_idx > G.maxClassifyLines:
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

    # 保存模型到G.projectFileClassifierModel
    def __saveModel(self):
        joblib.dump((self.ruleSet, self.dictionary, self.statsScope, self.models, self.categories),
                    G.projectFileClassifierModel)
        self.__dbUpdCategories()
        self.dictionary.save_as_text(os.path.join(G.logsPath, 'FileDictionary.csv'))
        category_names, percents, boundaries, quantiles = self.categories
        l2_fd = pd.DataFrame({'类名': category_names, '占比': percents,
                              '分位点到边界': quantiles, '边界点': boundaries})
        l2_fd.to_csv(os.path.join(G.logsPath, 'FileCategories.csv'), sep='\t', encoding='GBK')
        G.log.info('Model is built and saved to %s, %s: FileDictionary.csv, FileCategories.csv successful.',
                   G.projectFileClassifierModel, G.logsPath)

    # 对单个样本文件进行分类，返回文件名称、时间戳锚点位置，类别和置信度
    def predictFile(self, file_fullname, encoding='utf-8'):
        """
        :param file_fullname: log file to be predicted:
        :param encoding: encoding of the file
        :return: None if file __process errors, tuple of filename, number-of-lines, timestamp-cols, predict category
                 index, name, confidence and distance-to-center.
                 confidence > 1 means nearer than 0.8-quantiles to the center, < 0 means out of boundaries
        """
        if self.models is None:
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
                 distance-center. confidence > 1 means nearer than 0.8-quantiles to the center, < 0 means out of boundaries
        """
        if self.models is None:
            raise UserWarning('Failed to predict: Model is not exist!')

        corpus = []
        start_ = len(self.common_filenames)
        amount_files, failed_files, file_fullname = 0, 0, ''
        G.log.info('Start process documents from ' + dataset_path)
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
        if amount_files == 0:
            return [[], [], [], [], []]
        G.log.info('Converted %d files,%d(%d%%) failed', amount_files, failed_files, failed_files / amount_files * 100)

        vectors = self.__buildVectors(corpus, len(corpus))
        categories, category_names, confidences, distances = self.__getResult(vectors)  # 预测分类并计算可信度
        files = self.common_filenames[start_:]
        return files, list(categories), category_names, list(confidences), distances

    # 预测分类并计算可信度。<0 表示超出边界，完全不对，〉1完全表示比分位点还近，非常可信
    def __getResult(self, vectors):
        c_names, _, c_boundaries, c_quantiles = self.categories
        return FileUtil.getPredictResult(self.models, c_names, c_boundaries, c_quantiles, vectors)

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
    print(FileClassifier.__name__, FileClassifier.__doc__)
    print('This program cannot run directly')
    fc2 = FileClassifier(G.l1_cache)
    fc1 = FileClassifier(G.projectFileClassifierModel)
