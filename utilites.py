#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : utilities.py
# @Author: Sui Huafeng
# @Date  : 2018/5/15
# @Desc  : Utilities for common usage
#

import os
import re
import shutil
import tarfile

import numpy as np
import pandas as pd
import pymysql
from sklearn.cluster import KMeans

from anchor import Anchor
from config import Workspaces as G


class Dbc(object):

    def __init__(self, autocommit=False):
        self.conn = DbUtil.dbConnect(autocommit)
        self.cursor = self.conn.cursor()

    def __enter__(self):
        return self.cursor

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.conn.commit()
        self.cursor.close()
        self.conn.close()


# Shared utilities
class FileUtil(object):
    """
    
    """

    @staticmethod
    def moveFiles(from_, to_, counter):
        for sub_path_from, dir_names, file_names in os.walk(from_):
            sub_path_to = sub_path_from.replace(from_, to_)
            os.makedirs(sub_path_to, exist_ok=True)
            for file_name in file_names:
                try:
                    shutil.move(os.path.join(sub_path_from, file_name), os.path.join(sub_path_to, file_name))
                    counter['Success'] += 1
                except Exception as err:
                    G.log.info('File move err %s', str(err))
                    counter['Failed'] += 1
                    continue


    # 解压文件
    @staticmethod
    def extractFile(tar_file, to_):
        tar = tarfile.open(tar_file)
        tar.extractall(path=to_)
        tar.close()

    # 相同目录, 相似文件名的合并
    @staticmethod
    def mergeFilesByName(from_, to_):
        G.log.info('Merging files from %s into %s', from_, to_)
        merged_files, file2merged, failed_to_merge = [], [], []
        processed = [0]
        for dir_path, dir_names, file_names in os.walk(from_):
            # 形成文件描述列表
            file_descriptors = FileUtil.__getCommonName(dir_path, file_names, processed, failed_to_merge)
            if len(file_descriptors) == 0:  # 所有文件都没有anchor
                continue
            # 按需建立目标目录
            path_to = dir_path.replace(from_, to_)
            os.makedirs(path_to, exist_ok=True)
            FileUtil.__groupAndMerge(file_descriptors, path_to, merged_files, file2merged, failed_to_merge)

        with Dbc() as cursor:
            DbUtil.dbUpdFilesSampled(cursor, file2merged)
            DbUtil.dbInsert(cursor, 'files_sampled:file_fullname, anchor_name, notes', failed_to_merge)
        G.log.info('Merged %d files from %s into %s', processed[0], from_, to_)

        return merged_files, file2merged

    @staticmethod
    def __getCommonName(dir_path, file_names, processed, failed_to_merge):
        file_descriptors = pd.DataFrame(
            columns=('filename', 'common_name', 'anchor_name', 'anchor_start', 'anchor_end'))
        for idx, filename in enumerate(file_names):
            file_fullname = os.path.join(dir_path, filename)
            processed[0] += 1
            if processed[0] % 100 == 0:
                G.log.info('Merging %d files. %s', processed[0], file_fullname)
            # 计算时间戳锚点, 滤掉没有锚点的文件
            try:
                anchor = Anchor(file_fullname)
            except UserWarning as err:
                failed_to_merge.append([file_fullname, 'NONE', str(err)[:254]])
                G.log.debug('Failed to process\t%s, ignored.\t%s', file_fullname, str(err))
                continue
            common_name = G.fileMergePattern.sub('', os.path.splitext(filename)[0])  # 计算切分日志的公共文件名
            if not common_name:  # 全数字文件名
                common_name = 'digital1'
            file_descriptors.loc[idx] = file_fullname, common_name, anchor.name, anchor.colSpan[0], anchor.colSpan[
                1]  # 添加到descriptors中
        return file_descriptors

    # 同目录内名称相似的文件分为一组
    @staticmethod
    def __groupAndMerge(file_descriptors, path_to, merged_files, file2merged, failed_to_merge):
        for k_, v_ in file_descriptors.groupby(['anchor_name', 'common_name']):
            file_descriptors.sort_values('filename')  # 按文件名排序,以便顺序合并
            # 同组文件合并为1个
            common_name = '-'.join(k_)
            common_file_fullname = os.path.join(path_to, common_name)
            success = 0
            with open(common_file_fullname, 'wb') as fp_to:
                for file_fullname, anchor_name, anchor_start, anchor_end in zip(v_['filename'], v_['anchor_name'],
                                                                                v_['anchor_start'], v_['anchor_end']):
                    try:
                        fp_from = open(file_fullname, 'rb')
                        shutil.copyfileobj(fp_from, fp_to)
                        file2merged.append([file_fullname, anchor_name, anchor_start, anchor_end, common_file_fullname])
                        success += 1
                    except Exception as err:
                        failed_to_merge.append([file_fullname, 'NONE', str(err)[:254]])
                        G.log.warning('Failed to merge %s, ignored. %s', file_fullname, str(err))
                        continue
            if success == 0:  # All files failed to merge
                os.remove(common_file_fullname)
            else:
                merged_files.append(common_file_fullname)

    @staticmethod
    def loadRuleSets():
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
        return rule_sets

    # 对一个样本(字符串)进行处理，返回词表[word]
    @staticmethod
    def getWords(document, rule_set):
        """
        split document into words by rules in
        :param document:
        :param rule_set:
        :return:
        """
        # 按照replace_rules对原文进行预处理，替换常用变量，按标点拆词、滤除数字等等
        keep_words = []
        for (replace_from, replace_to) in rule_set[1]:
            if replace_to == 'KEEP':  # 保留变量原值，防止被分词、去掉数字等后续规则破坏
                keep_word = replace_from.findall(document)
                if not keep_word:  # 未找到，无需进行后续替换
                    continue
                keep_words += [word[0] for word in keep_word]  # 保存找到的原值
                replace_to = ''  # 让后续替换在原文中去掉原值
            document = replace_from.sub(replace_to, document)
        words = [w for w in document.split() if len(w) > 1 and w.lower() not in rule_set[2]]  # 分词,滤除停用词和单字母

        # 实现k-shingle逻辑，把连续多个词合并为一个词
        k_shingles = []
        for k in rule_set[3]:
            if k == 1:
                k_shingles = words
                continue
            for i in range(len(words) - k):
                k_shingle = ''
                for j in range(k):
                    k_shingle += words[i + j]
                k_shingles += ([k_shingle])

        # 合并返回单词列表
        return keep_words + k_shingles

    # 重新聚类，得到各Cluster的中心点、分位点距离、边界距离以及数量占比等
    @staticmethod
    def buildModel(caller, k_, vectors):
        norm_factor = - vectors.shape[1]  # 按字典宽度归一化
        kmeans = KMeans(n_clusters=k_, n_init=20, max_iter=500).fit(vectors)
        scores = np.array([kmeans.score([v]) / norm_factor for v in vectors])
        groups = pd.DataFrame({'C': kmeans.labels_, 'S': scores}).groupby('C')
        # 计算结果各类的0.8分位点和边界距离
        quantiles = np.array(
            [groups.get_group(i)['S'].quantile(caller.Quantile, interpolation='higher') for i in range(k_)])
        boundaries = groups['S'].agg('max').values  # 该簇中最远点距离
        double_quantiles = quantiles * 2
        boundaries[boundaries > double_quantiles] = double_quantiles[boundaries > double_quantiles]  # 边界太远的话，修正一下
        quantiles = boundaries - quantiles
        # 计算结果各类的向量数量和坏点数量(离中心太远)
        total_points = groups.size()
        min_distances = boundaries - quantiles * caller.MinConfidence
        bad_points = []
        for label, group_ in groups:
            distances = np.array(group_['S'])
            distances[distances < min_distances[label]] = 1
            distances[distances >= min_distances[label]] = 0
            bad_points.append(np.sum(distances))
        c_ids = [i for i in range(k_)]
        categories = np.vstack((c_ids, boundaries, quantiles, total_points, bad_points)).transpose()

        G.log.info('Model(k=%d) built. inertia=%e， max proportion=%.2f%%, max quantiles=%e, max border=%e',
                   k_, kmeans.inertia_, max(total_points) / len(vectors) * 100, max(quantiles), max(boundaries))
        return kmeans, categories

    @staticmethod
    def predict(kmeans, model_quantiles, vectors):
        """
        :param kmeans: k-means model
        :param model_quantiles: k*2 array of [distance from border to center, and distance from border to quantile]
        :param vectors:
        :return: samples * 3 array of [label, confidence, distance to center]
        """
        norm_factor = - vectors.shape[1]  # 按字典宽度归一化
        predicted_labels = kmeans.predict(vectors)  # 使用聚类模型预测记录的类别
        confidences = []
        distances = []
        for i, v in enumerate(vectors):
            distance = kmeans.score([v]) / norm_factor
            distances.append(distance)
            category = predicted_labels[i]
            confidence = model_quantiles[category, 0] - distance
            if model_quantiles[category, 1]:
                confidence /= model_quantiles[category, 1]
            else:
                confidence /= 1e-100
                if confidence >= 0:
                    confidence += 1
            confidences.append(confidence)
        confidences = np.array(confidences, copy=False)
        confidences[confidences > 99.9] = 99.9
        confidences[confidences < -99.9] = -99.9

        return predicted_labels, confidences, distances

    # 同类文件合并到to_目录下，供后续记录聚类使用
    @staticmethod
    def mergeFilesByClass(model_id, list_from, path_to):
        c = [0, 0, 0]  # counter
        prev_id = -1
        file_to, fp_to = None, None
        result_files = []
        for next_id, file_fullname in list_from:
            if next_id != prev_id:
                if fp_to:
                    fp_to.close()
                    result_files.append(file_to)
                file_to = os.path.join(path_to, 'fc%d-%d.samples' % (model_id, next_id))  # 文件命名规则后续逻辑使用
                fp_to = open(file_to, 'wb')
                prev_id = next_id
                c[2] += 1
            try:
                fp_from = open(file_fullname, 'rb')
                shutil.copyfileobj(fp_from, fp_to)
                c[0] += 1
                fp_from.close()
                G.log.debug('%s merged into %s', file_fullname, file_to)
            except Exception as err:
                c[1] += 1
                G.log.warning('Failed to merge %s, ignored. %s', file_fullname, str(err))
                continue

        fp_to.close()
        G.log.info('%d files(%d errors) merged into %d sample files, stored into %s', c[0], c[1], c[2], path_to)
        return result_files

    @staticmethod
    def groupCommonName(common_name_set):
        common_names = list(common_name_set)
        common_names_group = []
        processed = []
        common_names.sort(key=lambda x: len(x))
        for idx, prev_ in enumerate(common_names):
            if prev_ in processed:
                continue
            elif idx == len(common_names) - 1:
                common_names_group.append([prev_])
                break
            for next_ in common_names[idx + 1:]:
                if re.match(prev_, next_):
                    common_names_group.append([prev_, next_])
                    processed.append(next_)
                    break
            else:
                common_names_group.append([prev_])
        return common_names_group

class DbUtil(object):
    # 连接数据库
    @classmethod
    def dbConnect(cls, autocommit=False):
        type_, host, usr, password, database = G.cfg.get('General', 'db').split(':')
        try:
            if type_ == 'mysql':
                db = pymysql.connect(host, usr, password, database, charset='utf8', autocommit=autocommit)
                return db
            else:
                return None
        except Exception as err:
            G.log.warning('Db error. %s', str(err))
            return None

    @staticmethod
    def dbInsert(cursor, header, dataset, update=True):
        if not (cursor and len(dataset)):  # 无游标或者无数据
            return
        table_name, col_names = header.split(':')
        col_name_list = [col_name.strip() for col_name in col_names.split(',')]

        sql_head = 'INSERT INTO %s (%s) VALUES(' % (table_name, col_names)
        sql_tail = ') ON DUPLICATE KEY UPDATE '

        for row in dataset:
            insert_data = ''
            update_data = ''
            for col_name, cell_data in zip(col_name_list, row):
                str_data = 'null,' if cell_data is None else '"%s",' % str(cell_data)
                str_data = str_data.replace('\\', '\\\\')
                insert_data += str_data
                update_data += col_name + '=' + str_data

            if update:
                sql = sql_head + insert_data[:-1] + sql_tail + update_data[:-1]
                cursor.execute(sql)
            else:
                sql = sql_head + insert_data[:-1] + ')'
                try:
                    cursor.execute(sql)
                except Exception as err:
                    G.log.debug('Error in dbInsert %s', str(err))
                    continue

    @staticmethod
    def dbInsertOrUpdateLogFlow(cursor, wildcard_log_files):
        for wildcard_log_file in wildcard_log_files:
            host, remote_path, filename, model_id, category_id, anchor = wildcard_log_file
            match_ = re.match('flow(\d+)\.log', filename)
            remote_path = remote_path.replace('\\', '\\\\')
            host = '"%s"' % host
            remote_path = '"%s"' % remote_path
            filename = '"%s"' % filename
            anchor = '"%s"' % anchor
            if match_:  # 是已有日志流,重新聚类后的情况
                flow_id = int(match_.group(1))
                cursor.execute('SELECT COUNT(id) from tbl_log_flow where id=%d' % flow_id)
                if cursor.fetchall()[0][0]:
                    sql = 'UPDATE tbl_log_flow SET host=%s, path=%s, wildcard_name=%s, model_id=%s, category_id=%s, anchor=%s, status="活动中" WHERE id=%d' % (
                    host, remote_path, filename, model_id, category_id, anchor, flow_id)
                    cursor.execute(sql)
                    continue

            # 常规情况
            sql = 'INSERT INTO  tbl_log_flow(host, path, wildcard_name, model_id, category_id, anchor) VALUES(%s, %s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE model_id=%s, category_id=%s, anchor=%s' % (
                host, remote_path, filename, model_id, category_id, anchor, model_id, category_id, anchor)
            cursor.execute(sql)

    @staticmethod
    def dbUpdFilesSampled(cursor, file2merged):
        dataset = []
        dataset1 = []
        for file_fullname, anchor_name, anchor_start, anchor_end, common_file_fullname in file2merged:
            file_fullname = file_fullname.replace('\\', '/')
            host = file_fullname[len(G.inboxPath):].strip('/')
            host, filename = host.split('/', 1)
            archive_path, filename = os.path.split(filename)
            remote_path = ''
            if len(archive_path) > 3:
                remote_path = '/' + archive_path if archive_path[1] != '_' else archive_path[0] + ':' + archive_path[2:]
            common_file_fullname = common_file_fullname.replace('\\', '/')
            dataset.append([file_fullname, host, archive_path, filename, remote_path, 'reverse engineering'])
            dataset1.append([file_fullname, anchor_name, anchor_start, anchor_end, common_file_fullname])

        DbUtil.dbInsert(cursor, 'files_sampled:file_fullname, host, archive_path, filename, remote_path, notes',
                        dataset, update=False)

        DbUtil.dbInsert(cursor, 'files_sampled:file_fullname,anchor_name,anchor_start_col,anchor_end_col,common_name',
                        dataset1)

    @staticmethod
    def dbUpdFlow(cursor, flow_id, col_names, data):
        update_data = ''
        for col_name, cell_data in zip(col_names, data):
            str_data = 'null,' if cell_data is None else '"%s",' % str(cell_data)
            update_data += col_name + '=' + str_data
        sql = 'UPDATE tbl_log_flow SET %s WHERE id=%d' % (update_data[:-1], flow_id)
        cursor.execute(sql)



if __name__ == '__main__':
    FileUtil()
