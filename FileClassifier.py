#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : FileClassifier.py
# @Author: Sui Huafeng
# @Date  : 2018/4
# @Desc  : 从$DATA/input中所有样本文件中训练一个日志文件分类器
#

import os
import re
import shutil

import numpy as np
from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from sklearn.externals import joblib

from RecordClassifier import RecordClassifier
from config import Workspaces as G
from utilites import Dbc, FileUtil, DbUtil


# 对数十、上百万个文件进行聚类，形成聚类模型
class FileClassifier(object):
    """
    以目录名为参数创建对象,或者调用buildModel方法, 以该目录下文件作为样本聚类生成fc模型
    以文件名为参数创建对象或者调用loadModel方法，可装载fc模型
    调用predict方法, 可以预测新的日志文件类型及其置信度
    """
    __ClassName = 'FileClassifier'
    maxClassifyLines = G.cfg.getint(__ClassName, 'MaxLines')
    maxFcModels = G.cfg.getint(__ClassName, 'MaxModels')
    minFileConfidence = G.cfg.getfloat(__ClassName, 'MinConfidence')

    def __init__(self, model_id, model_file_or_samples_path):
        self.ruleSet = None  # 处理文件正则表达式
        self.dictionary = None  # 字典对象(Gensim Dictionary)
        self.statsScope = None  # 样本文件字符数、字数统计值(均值、标准差、中位数)的最小-最大值范围
        self.model = None  # 聚类模型(Kmeans)
        self.categories = None  # 聚类的类型[[数量占比，分位点距离，边界点距离]]

        self.model_id = model_id
        self.model_path = os.path.join(G.modelPath, str(self.model_id))
        if not os.path.exists(self.model_path):
            os.mkdir(self.model_path)
        self.common_filenames, self.l1_structure = ([], [])

        if os.path.isfile(model_file_or_samples_path):  # 从模型文件装载模型
            model_file = model_file_or_samples_path
            self.ruleSet, self.dictionary, self.statsScope, self.model, self.categories = joblib.load(model_file)
            self.__dbRebuildCategories()
        elif os.path.isdir(model_file_or_samples_path):
            self.corpusCacheFile = os.path.join(self.model_path, 'corpuscache.1')
            self.corpusDescriptionFile = os.path.join(self.model_path, 'metadata.1')
            self.buildModel(model_file_or_samples_path)
        else:
            raise UserWarning('No valid model file or samples path!')

    # 更新数据库中相关记录
    def __dbRebuildCategories(self):
        with Dbc() as cursor:
            cursor.execute('DELETE FROM file_class WHERE model_id=%d' % self.model_id)
            dataset = [[self.model_id, c_id, 'fc%d' % c_id, percent, boundary, quantile] for
                       c_id, (percent, boundary, quantile) in
                       enumerate(zip(self.categories[0], self.categories[1], self.categories[2]))]
            header = 'file_class:model_id, category_id, name, quantile, boundary, percent'
            DbUtil.dbInsert(cursor, header, dataset, update=False)

    # 训练、生成模型并保存在$model/xxx.mdl中,dataset:绝对/相对路径样本文件名，或者可迭代样本字符流
    def buildModel(self, from_path, k_=0):
        """
        Train and generate K-Means Model
        :param from_path: source path contains merged log files
        :param k_: K-means parameter, 0 means auto detect
        """
        try:
            # 尝试不同的向量化规则，确定聚类数量K_
            for self.ruleSet in FileUtil.loadRuleSets():
                corpus_cache_fp = self.__buildDictionary(from_path)  # 建立字典self.dictionary，返回语料缓存文件
                if len(self.dictionary) < G.cfg.getint(self.__ClassName, 'LeastTokens'):  # 字典字数太少,重新采样
                    corpus_cache_fp.close()
                    self.__clearCache()
                    G.log.info('Too few tokens[%d], Re-sample with next RuleSet.', len(self.dictionary))
                    continue
                corpus_cache_fp.seek(0)
                vectors = self.__buildVectors(corpus_cache_fp, self.dictionary.num_docs)  # 建立稀疏矩阵doc*(dct + stats)
                corpus_cache_fp.close()  # 关闭缓存文件
                start_k = min(50, int(vectors.shape[0] / 100))
                k_ = k_ if k_ else FileUtil.pilotClustering(self.__ClassName, vectors, start_k)  # 多个K值试聚类，返回最佳K
                if k_ != 0:  # 找到合适的K，跳出循环
                    break
                self.__clearCache()  # 清除缓存的corpus_cache_file
            else:
                raise UserWarning('Cannot generate qualified corpus by all RuleSets')
        except UserWarning:
            shutil.rmtree(self.model_path)
            raise

        # 重新聚类, 得到模型(向量数、中心点和距离）和分类(向量-所属类)
        self.model, percents, boundaries, quantiles = FileUtil.buildModel(self.__ClassName, k_, vectors)
        self.categories = [percents, boundaries, quantiles]
        # 保存模型文件和数据库
        self.__saveModel()  # model saved to file
        self.__postProcess(vectors)

    # 建立词典，同时缓存词表文件
    def __buildDictionary(self, new_dataset_path):
        self.dictionary = Dictionary()
        lines = ''
        # 装载处理过的缓存语料
        cache_fp = open(self.corpusCacheFile, mode='a+t', encoding='utf-8')  # 创建或打开语料缓存文件
        if cache_fp.tell() != 0:
            if os.path.exists(self.corpusDescriptionFile):
                self.ruleSet, self.common_filenames, self.l1_structure, self.statsScope = joblib.load(
                    self.corpusDescriptionFile)
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

        if self.dictionary.num_docs < G.cfg.getint(self.__ClassName, 'LeastFiles'):  # 字典字数太少或文档数太少，没必要聚类
            cache_fp.close()
            self.__clearCache()
            raise UserWarning('Too few documents[%d] to clustering' % self.dictionary.num_docs)

        # 去掉低频词，压缩字典
        num_token = len(self.dictionary)
        no_below = int(min(G.cfg.getfloat(self.__ClassName, 'NoBelow'), int(self.dictionary.num_docs / 50)))
        self.dictionary.filter_extremes(no_below=no_below, no_above=0.999,
                                        keep_n=G.cfg.getint(self.__ClassName, 'KeepN'))
        self.dictionary.compactify()
        G.log.info('Dictionary built with [%s](%d tokens, reduced from %d), from %d files( %d words)',
                   self.ruleSet[0], len(self.dictionary), num_token, self.dictionary.num_docs, self.dictionary.num_pos)

        statistics = np.array(self.l1_structure)[:, 1:7]
        statistics[statistics > 500] = 500  # 防止异常大的数干扰效果
        self.statsScope = np.min(statistics, axis=0), np.max(statistics, axis=0)
        joblib.dump((self.ruleSet, self.common_filenames, self.l1_structure, self.statsScope),
                    self.corpusDescriptionFile)

        return cache_fp

    # 预处理，迭代方式返回某个文件的词表.
    def __buildDocument(self, dataset_path):
        amount_files, failed_files, file_fullname = 0, 0, ''
        G.log.info('Start Converting documents from ' + dataset_path)
        processed_files = os.path.join(self.model_path, 'buildDocument.dbf')
        processed = [] if not os.path.exists(processed_files) else joblib.load(processed_files)
        with Dbc() as cursor:
            # cursor = db.cursor() if db else None
            #
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
        # cursor.close()
        # db.close()
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
            words = FileUtil.getWords(line, rule_set=self.ruleSet)
            document += words  # 生成词表
            lc.append(len(line))
            lw.append(len(words))
            if line_idx > self.maxClassifyLines:
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
        if rows > 300:  # predict时经常一个文件做一次,没有必要记log
            G.log.info('[%d*%d]Vectors built' % (rows, cols))

        return np.hstack((statistics, subtotal, vectors))

    # 保存模型到G.projectFileClassifierModel
    def __saveModel(self):
        model_file = os.path.join(self.model_path, 'FileClassifier.Model')
        joblib.dump((self.ruleSet, self.dictionary, self.statsScope, self.model, self.categories), model_file)
        self.__dbRebuildCategories()
        self.dictionary.save_as_text(os.path.join(G.logsPath, 'FileDictionary.csv'))
        G.log.info('Model is built and saved to %s successful.', model_file)

    def __postProcess(self, vectors):
        results = FileUtil.predict(self.model, self.categories[1], self.categories[2], vectors)
        with Dbc() as cursor:
            self.__dbUdFilesMerged(cursor, self.common_filenames, results)
            self.__updLogFlows(cursor)  # 生成采集文件列表
            # 同类文件合并到to_目录下
            sql = 'SELECT category_id, file_fullname FROM files_classified WHERE model_id=%d AND confidence>=%f ORDER BY category_id' % (
            self.model_id, self.minFileConfidence)
            cursor.execute(sql)
            from_list = cursor.fetchall()

        FileUtil.mergeFilesByClass(self.model_id, from_list, G.classifiedFilePath)
        self.__buildRcModels(range(len(self.categories[1])))

    #  插入或更新合并文件分类表
    def __dbUdFilesMerged(self, cursor, file_fullnames, results):
        dataset = []
        for f_name, c_id, con, dis in zip(file_fullnames, results[0], results[1], results[2]):
            if con < self.minFileConfidence:  # 置信度不够,未完成分类
                model_id, c_id = None, None
            else:
                model_id = self.model_id
            f_name = f_name.replace('\\', '/')
            dataset.append([f_name, model_id, c_id, con, dis])
        DbUtil.dbInsert(cursor, 'files_merged:common_name,model_id,category_id,confidence,distance', dataset)

    # 识别切分日志, 形成应采文件的列表.
    @staticmethod
    def __updLogFlows(cursor, additional_where=''):
        """
        切分日志包括定时归档、定长循环、定时新建3种情况，定时归档日志只需采集最新文件，定长循环、定时新建日志当作1个
        数据流处理, 采集日期最新的,如一段时间未更新, 则试图切换到更新的采集.
        """
        prev_id, wildcard_log_files = '', []
        files_by_class_and_path, common_name_set = [], set()
        results = FileClassifier.__querySamples(cursor, additional_where)
        for row in results:
            model_id, category_id, host, remote_path, filename, common_name, anchor_name, anchor_start_col, anchor_end_col, last_collect, last_update = row
            next_id = '%d-%d-%s-%s' % (model_id, category_id, host, remote_path)
            seconds_ago = (last_collect - last_update).total_seconds()
            anchor = '%s:%d:%d' % (anchor_name, anchor_start_col, anchor_end_col)
            common_name = os.path.split(common_name)[1]
            files_by_class_and_path.append([filename, common_name, seconds_ago, anchor])
            common_name_set.add(common_name)
            if prev_id == '':
                prev_id = next_id
            if prev_id == next_id:
                continue
            cp_groups = FileClassifier.__splitSamples(files_by_class_and_path, common_name_set)
            FileClassifier.__getWildcard_files(cp_groups, prev_id, wildcard_log_files)
            prev_id = next_id
            files_by_class_and_path, common_name_set = [], set()

        if files_by_class_and_path:  # 最后一行
            cp_groups = FileClassifier.__splitSamples(files_by_class_and_path, common_name_set)
            FileClassifier.__getWildcard_files(cp_groups, prev_id, wildcard_log_files)

        DbUtil.dbInsertOrUpdateLogFlow(cursor, wildcard_log_files)  # 保存采集文件列表

    @staticmethod
    def __querySamples(cursor, additional_where):
        sql = 'SELECT model_id,category_id, host, remote_path, filename, common_name,anchor_name, anchor_start_col, anchor_end_col,last_collect,last_update FROM files_classified WHERE confidence >= %f %s ORDER BY model_id, category_id, host, remote_path' % (
        FileClassifier.minFileConfidence, additional_where)
        cursor.execute(sql)
        return cursor.fetchall()

    @staticmethod
    def __splitSamples(files_by_class_and_path, common_name_set):
        num_common_names = len(common_name_set)
        differences = [len(G.fileCheckPattern.findall(name)) for name, _, _, _ in files_by_class_and_path]
        special_files = differences.count(min(differences))  # 不是通常归档文件的文件数量
        if num_common_names == 2 and special_files > 1 or num_common_names > 2:  # 同目录下多组日志文件, 进一步拆分
            common_name_groups = FileUtil.groupCommonName(common_name_set)
            cp_groups = FileClassifier.__splitFurther(common_name_groups, files_by_class_and_path)
        else:
            cp_groups = {0: files_by_class_and_path}
        return cp_groups

    @staticmethod
    def __splitFurther(common_name_groups, files_by_class_and_path):
        cp_groups = {}
        for filename, common_name, seconds_ago, anchor in files_by_class_and_path:
            for idx, common_name_group in enumerate(common_name_groups):
                if common_name in common_name_group:
                    break

            v_ = cp_groups.get(idx, [])
            v_.append([filename, common_name, seconds_ago, anchor])
            cp_groups[idx] = v_
        return cp_groups

    @staticmethod
    def __getWildcard_files(cp_groups, prev_id, wildcard_log_files):
        for cp_group in cp_groups.values():
            cp_group.sort(key=lambda x: '%fA%s' % (x[2], x[0]))
            filename, common_name, seconds_ago, anchor = cp_group[0]
            if seconds_ago > G.last_update_seconds:  # 最新的文件长期未更新,不用采
                continue
            # 取hostname
            model_id, category_id, host, remote_path = prev_id.split('-', maxsplit=3)
            if len(cp_group) == 1:  # 仅有一个有效文件
                wildcard_log_files.append([host, remote_path, filename, model_id, category_id, anchor])
                continue

            differences = [len(name) for name, _, _, _ in cp_group]  # 文件名不同部分的长度
            min_len, max_len = min(differences), max(differences)
            min_files = differences.count(min_len)
            if min_files == len(differences):  # 定时新建(所有文件名同样长): 通配符采集
                filename = G.fileCheckPattern.sub('?', filename)
            elif max_len - min_len < 3:  # 定长归档(长度差不超过2).共同前缀作为通配符
                filename = FileClassifier.__getFixedSizeLogFilePrefix(cp_group)
            else:  # 其他情况为定期归档, 采集最新的特别文件
                pass

            wildcard_log_files.append([host, remote_path, filename, model_id, category_id, anchor])

    # 计算字符串数组的公共前缀
    @staticmethod
    def __getFixedSizeLogFilePrefix(cp_group):
        cp_group.sort(key=lambda x: len(x[0]))
        shortest, compared = cp_group[0][0], cp_group[1][0]
        i = 0
        for i in range(len(shortest), 0, -1):
            if shortest[:i] == compared[:i]:
                break
        return shortest[:i] + '*'

    def __buildRcModels(self, fc_list):
        errors = 0
        for fc_id in fc_list:
            try:
                prefix = 'fc%d-%d' % (self.model_id, fc_id)
                G.log.info('Record classifying [%s]...', prefix)
                files = sorted([os.path.join(G.classifiedFilePath, filename)
                                for filename in os.listdir(G.classifiedFilePath) if filename[:len(prefix)] == prefix])
                RecordClassifier(files)
            except Exception as err:
                errors += 1
                G.log.error('ignored due to: %s', str(err))
                continue
        G.log.info('rc model built, %d failed.', errors)

    # 对多个样本文件列表进行分类预测
    def predict(self, file_fullnames, encoding='utf-8'):
        """
        :param file_fullnames: file list to be predicted
        :param encoding: encoding of the file
        :return: list of file-names, number-of-line, timestamp-col, predict category index, name, confidence and
                 distance-center. confidence > 1 means nearer than 0.8-quantiles to the center, < 0 means out of boundaries
        """
        if self.model is None:
            raise UserWarning('Failed to predict: Model is not exist!')

        # 生成词库和向量
        corpus = []
        for file_fullname in file_fullnames:
            try:
                corpus.append(self.__file2doc(file_fullname, encoding=encoding))  # 文件转为词表
            except Exception as err:
                G.log.warning('Failed to __process\t%s, ignored.\t%s', file_fullname, str(err))
                continue
        if not corpus:
            return
        vectors = self.__buildVectors(corpus, len(corpus))
        # 预测类别并更新数据库
        c_ids, confidences, distances = FileUtil.predict(self.model, self.categories[1], self.categories[2], vectors)
        with Dbc() as cursor:
            self.__dbUdFilesMerged(cursor, file_fullnames, (c_ids, confidences, distances))
            # 计算采集表更更新数据库
            where_clause = ['"%s"' % common_name.replace('\\', '/') for common_name in file_fullnames]
            where_clause = ' AND common_name in (%s)' % ','.join(where_clause)
            self.__updLogFlows(cursor, where_clause)  # 生成采集文件列表并更新日志流配置
        classified = '","'.join([file_fullname.replace('\\', '/') for file_fullname, confidence
                                 in zip(file_fullnames, confidences) if confidence >= self.minFileConfidence])
        # 分类文件存储到已分类目录, 为日志记录分类储备数据
        if classified:
            with Dbc() as cursor:
                sql = 'SELECT f.id, f.category_id, f.common_name FROM files_merged AS f, file_class as c WHERE f.model_id=c.model_id AND f.category_id=c.category_id AND c.status="无模型" AND f.common_name in ("%s")' % classified
                cursor.execute(sql)
                results = cursor.fetchall()
            fc_ids = set()
            for id_, category_id, common_name in results:
                file_fullname_to = os.path.join(G.classifiedFilePath, 'fc%d-%d-%d' % (self.model_id, category_id, id_))
                shutil.copy(common_name, file_fullname_to)
                fc_ids.add(category_id)
            self.__buildRcModels(fc_ids)

        unclassified = [file_fullname for file_fullname, confidence
                        in zip(file_fullnames, confidences) if confidence < self.minFileConfidence]
        return unclassified

    # 删除缓存文件
    def __clearCache(self):
        for f in [self.corpusCacheFile, self.corpusDescriptionFile]:
            try:
                os.remove(self.corpusDescriptionFile) if os.path.exists(self.corpusDescriptionFile) else None
                os.remove(self.corpusCacheFile) if os.path.exists(self.corpusCacheFile) else None
            except Exception as err:
                G.log.warning('Failed to clear %s. %s' % (f, str(err)))
                continue

    @staticmethod
    def loadAllModels():
        models = []
        for idx, model_path in enumerate(sorted([os.path.join(G.modelPath, path) for path in os.listdir(G.modelPath)
                                                 if re.match('\d+', path) and os.path.isdir(
                os.path.join(G.modelPath, path))])):
            model_file = os.path.join(model_path, 'FileClassifier.Model')
            if os.path.exists(model_file):
                models.append(FileClassifier(idx, model_file))
        return models


if __name__ == '__main__':
    print(FileClassifier.__name__, FileClassifier.__doc__)
    print('This program cannot run directly')
