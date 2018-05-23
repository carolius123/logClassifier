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
from collections import Counter

import numpy as np
import pandas as pd
import pymysql
from scipy.signal import argrelextrema
from sklearn.cluster import KMeans

from anchor import Anchor
from config import Workspaces as G


class Dbc(object):
    db_type, db_addr, db_user, db_passwd = G.cfg.get('General', 'db').split(':')

    def __init__(self):
        self.conn = None
        self.cursor = None
        if self.db_type == 'mysql':
            self.conn = pymysql.connect(self.db_addr, self.db_user, self.db_passwd, 'ailog-backend', charset='utf8')
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
    # 解压文件
    @classmethod
    def extractFile(cls, tar_file, to_):
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
            DbUtil.dbInsert(cursor, 'files_sampled', ['file_fullname', 'anchor_name', 'notes'], failed_to_merge)
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

    # 聚类，得到各簇SSE（sum of the squared errors)，作为手肘法评估确定ｋ的依据
    @staticmethod
    def pilotClustering(classifier_section_name, vectors, k_from=1):
        pilot_list = []  # [(k_, inertia, criterion, top5_percent)] criteria取inertia变化率的一阶微分的极大值
        FileUtil.__pilot(classifier_section_name, k_from, vectors, pilot_list)
        return FileUtil.__getTop(classifier_section_name, pilot_list)

    # 从k_from到k_to聚类,得到pilot_list
    @staticmethod
    def __pilot(classifier_section_name, k_from, vectors, pilot_list):

        norm_factor = vectors.shape[1] * vectors.shape[0]  # 按行/样本数和列/字典宽度标准化因子，保证不同向量的可比性
        termination_inertia = G.cfg.getfloat(classifier_section_name, 'NormalizedTerminationInertia') * norm_factor
        k_to = G.cfg.getint(classifier_section_name, 'MaxCategory')
        for k_ in range(k_from, k_to):
            kmeans = KMeans(n_clusters=k_, tol=1e-5).fit(vectors)  # 试聚类
            if k_ < k_from + 2:
                pilot_list.append([k_, kmeans.inertia_, 0, 0, 0])
                continue

            for retry in range(5):  # 如果inertia因误差变大，重新聚几次
                inertia = kmeans.inertia_
                if inertia <= pilot_list[-1][1]:
                    break
                G.log.debug('retries=%d, inertia=%e', retry + 1, inertia)
                kmeans = KMeans(n_clusters=k_).fit(vectors)
            else:
                inertia = pilot_list[-1][1]

            pilot_list[-1][2] = pilot_list[-2][1] / pilot_list[-1][1] - pilot_list[-1][1] / inertia
            G.log.info('pilot clustering. (k,inertia,criteria,top5)=\t%d\t%e\t%.3f\t%.3f', pilot_list[-1][0],
                       pilot_list[-1][1], pilot_list[-1][2], pilot_list[-1][3])

            top5_percent = sum([v for (k, v) in Counter(kmeans.labels_).most_common(5)]) / len(kmeans.labels_)
            #             bad_percent = FileUtil.__getBadPointPercents(kmeans,vectors,cfg_q, k_)  # 作用不大,太慢
            bad_percent = 0
            pilot_list.append([k_, inertia, None, top5_percent, bad_percent])
            if inertia < termination_inertia:  # 已经收敛到很小且找到可选值，没必要继续增加
                break

    # 从pilot list中取极大值
    @staticmethod
    def __getTop(classifier_section_name, pilot_list):
        pilot_array = np.array(pilot_list)[1:, :]  # 去掉第一个没法计算criterion值的
        pilot_array = pilot_array[
            pilot_array[:, 3] < G.cfg.getfloat(classifier_section_name, 'Top5Ratio')]  # 去掉top5占比超标的
        pilot_array = pilot_array[argrelextrema(pilot_array[:, 2], np.greater)]  # 得到极大值
        criteria = pilot_array[:, 2].tolist()
        if not criteria:  # 没有极值
            return pilot_list[-1][0]

        max_top_n, idx_ = [], 0
        while criteria[idx_:]:
            idx_ = criteria.index(max(criteria[idx_:]))
            max_top_n.append(pilot_array[idx_])
            idx_ += 1
        G.log.debug('topN k=\n%s',
                    '\n'.join(['%d\t%e\t%.3f\t%.3f\t%.3f' % (k, i, c, t, b) for k, i, c, t, b in max_top_n]))
        products = [k * c for k, i, c, t, b in max_top_n]
        idx_ = products.index(max(products))
        preferred = max_top_n[idx_][0]
        G.log.info('pilot-cluster finished. preferred k=(%d)', preferred)
        return preferred

    @staticmethod
    # 计算距离特别远（0.8分位点2倍距离以上）的坏点比例
    def __getBadPointPercents(kmeans, vectors, cfg_q, k_):
        v_scores = -np.array([kmeans.score([v]) for v in vectors])
        groups = pd.DataFrame({'C': kmeans.labels_, 'S': v_scores}).groupby('C')
        c_quantiles_double = 2 * np.array([groups.get_group(i)['S'].quantile(cfg_q) for i in range(k_)])
        bad_samples = 0
        for idx, score in enumerate(v_scores):
            if score > c_quantiles_double[kmeans.labels_[idx]]:
                bad_samples += 1
        return bad_samples / len(v_scores)

    # 重新聚类，得到各Cluster的中心点、分位点距离、边界距离以及数量占比等
    @staticmethod
    def buildModel(classifier_section_name, k_, vectors):
        # 再次聚类并对结果分组。 Kmeans不支持余弦距离
        kmeans = KMeans(n_clusters=k_, n_init=20, max_iter=500).fit(vectors)
        norm_factor = - vectors.shape[1]  # 按字典宽度归一化
        groups = pd.DataFrame({'C': kmeans.labels_, 'S': [kmeans.score([v]) / norm_factor for v in vectors]}).groupby(
            'C')
        percents = groups.size() / len(vectors)  # 该簇向量数在聚类总向量数中的占比
        cfg_q = G.cfg.getfloat(classifier_section_name, 'Quantile')
        quantiles = np.array([groups.get_group(i)['S'].quantile(cfg_q, interpolation='higher') for i in range(k_)])
        boundaries = groups['S'].agg('max').values  # 该簇中最远点距离

        quantiles2 = quantiles * 2
        boundaries[boundaries > quantiles2] = quantiles2[boundaries > quantiles2]  # 边界太远的话，修正一下
        boundaries[boundaries < 1e-100] = 1e-100  # 边界为零的话，修正一下
        quantiles = boundaries - quantiles
        quantiles[quantiles < 1e-100] = 1e-100  # 避免出现0/0

        G.log.info('Model(k=%d) built. inertia=%e， max proportion=%.2f%%, max quantiles=%e, max border=%e',
                   k_, kmeans.inertia_, max(percents) * 100, max(quantiles), max(boundaries))
        return kmeans, percents, boundaries, quantiles

    @staticmethod
    def predictFiles(models, merged_files):
        classified_files, unclassified_files = [], []
        for common_name in merged_files:
            for model in models:
                if model:  # 模型存在
                    result = model.predictFile(common_name)
                    if not result:
                        continue
                    category, category_name, confidence, distance = result
                    if confidence < G.minConfidence:  # 置信度不够,未完成分类
                        continue
                    classified_files.append(
                        [model.model_id, common_name, category, category_name, confidence, distance])
                    break
            else:
                unclassified_files.append(common_name)
        return classified_files, unclassified_files

    @staticmethod
    def getPredictResult(model, c_names, c_boundaries, c_quantiles, vectors):
        norm_factor = - vectors.shape[1]  # 按字典宽度归一化

        predicted_labels = model.predict(vectors)  # 使用聚类模型预测记录的类别
        predicted_names = [c_names[label] for label in predicted_labels]
        confidences = []
        distances = []
        for i, v in enumerate(vectors):
            distance = model.score([v]) / norm_factor
            distances.append(distance)
            category = predicted_labels[i]
            confidences.append((c_boundaries[category] - distance) / c_quantiles[category])
        confidences = np.array(confidences, copy=False)
        confidences[confidences > 99.9] = 99.9
        confidences[confidences < -99.9] = -99.9

        return predicted_labels, predicted_names, confidences, distances

    # 同类文件合并到to_目录下，供后续记录聚类使用
    @staticmethod
    def mergeFilesByClass(cursor, to_):
        c = [0, 0, 0]  # counter
        sql = 'SELECT model_id+category_id*10,category_name, file_fullname FROM files_classified WHERE confidence >= %f ORDER BY model_id, category_id, last_collect, last_update' % G.minConfidence
        cursor.execute(sql)
        results = cursor.fetchall()
        prev_id = -1
        file_to, fp_to = None, None
        for next_id, category_name, file_fullname in results:
            if next_id != prev_id:
                fp_to.close() if fp_to else None
                file_to = os.path.join(to_, category_name)  # 以类别名称作为输出文件名称
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
        G.log.info('%d files(%d errors) merged into %d sample files, stored into %s', c[0], c[1], c[2], to_)

    # 识别切分日志, 形成应采文件的列表.
    @staticmethod
    def genGatherList(cursor, additional_where=''):
        """
        切分日志包括定时归档、定长循环、定时新建3种情况，定时归档日志只需采集最新文件，定长循环、定时新建日志当作1个
        数据流处理, 采集日期最新的,如一段时间未更新, 则试图切换到更新的采集.
        """
        prev_id, wildcard_log_files = '', []
        files_by_class_and_path, common_name_set = [], set()
        results = FileUtil.__querySamples(cursor, additional_where)
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

            cp_groups = FileUtil.__splitSamples(files_by_class_and_path, common_name_set)
            FileUtil.__getWildcard_files(cp_groups, prev_id, wildcard_log_files)

            prev_id = next_id
            files_by_class_and_path, common_name_set = [], set()

        return wildcard_log_files

    @staticmethod
    def __querySamples(cursor, additional_where):
        sql = 'SELECT model_id,category_id, host, remote_path, filename, common_name,anchor_name, anchor_start_col, anchor_end_col,last_collect,last_update FROM files_classified WHERE confidence >= %f %s ORDER BY model_id, category_id, host, remote_path' % (
            G.minConfidence, additional_where)
        cursor.execute(sql)
        return cursor.fetchall()

    @staticmethod
    def __splitSamples(files_by_class_and_path, common_name_set):
        num_common_names = len(common_name_set)
        differences = [len(G.fileCheckPattern.findall(name)) for name, _, _, _ in files_by_class_and_path]
        special_files = differences.count(min(differences))  # 不是通常归档文件的文件数量
        if num_common_names == 2 and special_files > 1 or num_common_names > 2:  # 同目录下多组日志文件, 进一步拆分
            common_name_groups = FileUtil.__groupCommonName(common_name_set)
            cp_groups = FileUtil.__splitFurther(common_name_groups, files_by_class_and_path)
        else:
            cp_groups = {0: files_by_class_and_path}
        return cp_groups

    @staticmethod
    def __groupCommonName(common_name_set):
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
                filename = FileUtil.__getFixedSizeLogFilePrefix(cp_group)
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

    @staticmethod
    def splitResults(model_id, results):
        classified_files, unclassified_files = [], []
        for common_name, category, category_name, confidence, distance in zip(results[0], results[1], results[2],
                                                                              results[3], results[4]):
            if confidence < G.minConfidence:  # 置信度不够,未完成分类
                unclassified_files.append(common_name)
            else:
                classified_files.append([model_id, common_name, category, category_name, confidence, distance])
        return classified_files, unclassified_files


class DbUtil(object):
    # 连接数据库
    @classmethod
    def dbConnect(cls):
        db_type, db_addr, db_user, db_passwd = G.cfg.get('General', 'db').split(':')
        try:
            if db_type == 'mysql':
                db = pymysql.connect(db_addr, db_user, db_passwd, 'ailog', charset='utf8')
                return db
            else:
                return None
        except Exception as err:
            G.log.warning('Db error. %s', str(err))
            return None

    @staticmethod
    def dbInsert(cursor, table_name, col_names, dataset, update=True):
        if not (cursor and len(dataset)):  # 无游标或者无数据
            return

        sql_head = 'INSERT INTO %s (%s) VALUES(' % (table_name, ','.join(col_names))
        sql_tail = ') ON DUPLICATE KEY UPDATE '

        for row in dataset:
            insert_data = ''
            update_data = ''
            for col_name, cell_data in zip(col_names, row):
                str_data = 'null,' if cell_data is None else '"%s",' % str(cell_data)
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
                    G.log.warning('Error in dbInsert %s', str(err))
                    continue

    #  插入或更新合并文件分类表
    @staticmethod
    def dbUdFilesMerged(cursor, classified_files, unclassified_files):
        for common_name in unclassified_files:
            common_name = '"%s"' % common_name.replace('\\', '/')
            sql = 'INSERT INTO files_merged (common_name) VALUES(%s) ON DUPLICATE KEY UPDATE model_id=NULL , category_id=NULL , confidence=NULL , distance=NULL ' % (
                common_name)
            cursor.execute(sql)

        for model_id, common_name, category, category_name, confidence, distance in classified_files:
            common_name = '"%s"' % common_name.replace('\\', '/')
            sql = 'INSERT INTO files_merged (common_name, model_id, category_id, confidence, distance) VALUES(%s, %d, %d, %f, %e) ON DUPLICATE KEY UPDATE model_id=%d , category_id=%d , confidence=%f , distance=%e' % (
                common_name, model_id, category, confidence, distance, model_id, category, confidence, distance)
            cursor.execute(sql)

    @staticmethod
    def dbWildcardLogFiles(cursor, wildcard_log_files):
        for wildcard_log_file in wildcard_log_files:
            host, remote_path, filename, model_id, category_id, anchor = wildcard_log_file
            remote_path = remote_path.replace('\\', '\\\\')
            host = '"%s"' % host
            remote_path = '"%s"' % remote_path
            filename = '"%s"' % filename
            anchor = '"%s"' % anchor
            sql = 'INSERT INTO  tbl_log_flow(host, path, wildcard_name, model_id, category_id, anchor) VALUES(%s, %s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE model_id=%s, category_id=%s, anchor=%s' % (
                host, remote_path, filename, model_id, category_id, anchor, model_id, category_id, anchor)
            cursor.execute(sql)

    @staticmethod
    def dbUpdFilesSampled(cursor, file2merged):
        dataset = []
        dataset1 = []
        for file_fullname, anchor_name, anchor_start, anchor_end, common_file_fullname in file2merged:
            file_fullname = file_fullname.replace('\\', '/')
            host = file_fullname[len(G.l0_inputs):].strip('/')
            host, filename = host.split('/', 1)
            archive_path, filename = os.path.split(filename)
            remote_path = ''
            if len(archive_path) > 3:
                remote_path = '/' + archive_path if archive_path[1] != '_' else archive_path[0] + ':' + archive_path[2:]
            common_file_fullname = common_file_fullname.replace('\\', '/')
            dataset.append([file_fullname, host, archive_path, filename, remote_path, 'reverse engineering'])
            dataset1.append([file_fullname, anchor_name, anchor_start, anchor_end, common_file_fullname])

        col_names = ['file_fullname', 'host', 'archive_path', 'filename', 'remote_path', 'notes']
        DbUtil.dbInsert(cursor, 'files_sampled', col_names, dataset, update=False)

        col_names = ['file_fullname', 'anchor_name', 'anchor_start_col', 'anchor_end_col', 'common_name']
        DbUtil.dbInsert(cursor, 'files_sampled', col_names, dataset1)

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
