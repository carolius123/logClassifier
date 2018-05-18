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

import pandas as pd
import pymysql

from anchor import Anchor
from config import Workspaces as G


# 
class Util(object):
    """
    
    """

    # 获取数据库游标
    @classmethod
    def dbConnect(cls):
        db_name, db_addr, db_user, db_passwd = G.cfg.get('General', 'db').split(':')
        try:
            if db_name == 'mysql':
                db = pymysql.connect(db_addr, db_user, db_passwd, 'ailog')
                db.set_charset('utf8')
                cursor = db.cursor()
                cursor.execute('SET NAMES utf8;')
                cursor.execute('SET CHARACTER SET utf8;')
                cursor.execute('SET character_set_connection=utf8;')
                return db
            else:
                return None
        except Exception as err:
            G.log.warning('Db error. %s', str(err))
            return None

    # 解压文件
    @classmethod
    def extractFile(cls, tar_file, to_):
        tar = tarfile.open(tar_file)
        tar.extractall(path=to_)
        tar.close()

    # 相同目录, 相似文件名的合并
    @classmethod
    def mergeFilesByName(cls, from_, to_):
        common_files, file2merged = [], []  # 成功合并的文件列表
        processed = 0
        for dir_path, dir_names, file_names in os.walk(from_):
            # 形成文件描述列表
            file_descriptors = pd.DataFrame(columns=('filename', 'common_name', 'anchor_name', 'anchor_cols'))
            for idx, filename in enumerate(file_names):
                file_fullname = os.path.join(dir_path, filename)
                processed += 1
                if processed % 100 == 0:
                    G.log.info('Merging %d files. %s', processed, file_fullname)
                # 计算时间戳锚点, 滤掉没有锚点的文件
                try:
                    anchor = Anchor(file_fullname)
                except UserWarning as err:
                    G.log.debug('Failed to process\t%s, ignored.\t%s', file_fullname, str(err))
                    continue

                common_name = G.fileMergePattern.sub('', os.path.splitext(filename)[0])  # 计算切分日志的公共文件名
                if not common_name:  # 全数字文件名
                    common_name = 'digital1'
                file_descriptors.loc[idx] = file_fullname, common_name, anchor.name, anchor.colRange  # 添加到descriptors中

            if not len(file_descriptors):  # 所有文件都没有anchor
                continue

            path_to = dir_path.replace(from_, to_)
            os.makedirs(path_to, exist_ok=True)  # 按需建立目标目录

            # 同目录内名称相似的文件分为一组
            for k_, v_ in file_descriptors.groupby(['anchor_name', 'common_name']):
                file_descriptors.sort_values('filename')  # 按文件名排序,以便顺序合并
                # 同组文件合并为1个
                common_name = '-'.join(k_)
                common_file_fullname = os.path.join(path_to, common_name)
                common_files.append(common_file_fullname)
                with open(common_file_fullname, 'a', encoding='utf-8') as fp:
                    for file_fullname, anchor_name, anchor_colRange in zip(v_['filename'], v_['anchor_name'],
                                                                           v_['anchor_cols']):
                        try:
                            for line in open(file_fullname, 'r', encoding='utf-8'):
                                fp.write(line)

                            file2merged.append([file_fullname, anchor_name, anchor_colRange, common_file_fullname])
                        except Exception as err:
                            G.log.warning('Failed to merge %s, ignored. %s', file_fullname, str(err))
                            continue
        G.log.info('Merged %d files from %s into %s', processed, from_, to_)
        return common_files, file2merged

    #  插入或更新合并文件分类表
    @staticmethod
    def dbFilesMerged(cursor, file2merged, classified_files, unclassified_files):
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

        for file_fullname, anchor_name, anchor_colRange, common_file_fullname in file2merged:
            file_fullname = '"%s"' % file_fullname.replace('\\', '/')
            common_file_fullname = '"%s"' % common_file_fullname.replace('\\', '/')
            anchor_name = '"%s"' % anchor_name
            sql = 'UPDATE files_sampled set common_name=%s, anchor_name=%s, anchor_start_col=%d, anchor_end_col=%d WHERE file_fullname=%s' % (
                common_file_fullname, anchor_name, anchor_colRange[0], anchor_colRange[1], file_fullname)
            cursor.execute(sql)

    # 清除处采样文件记录外所有表中数据
    @staticmethod
    def clearModel(model_id):
        db = Util.dbConnect()
        if db:
            cursor = db.cursor()
            sql = 'DELETE FROM file_class WHERE model_id = %d' % model_id
            cursor.execute(sql)
            db.commit()
            db.close()

    # 同类文件合并到to_目录下，供后续记录聚类使用
    @staticmethod
    def mergeFilesByClass(cursor, to_):
        c = [0, 0, 0]  # counter
        sql = 'SELECT model_id+category_id*10,category_name, file_fullname FROM files_classified WHERE confidence >= %f ORDER BY model_id, category_id, last_collect-files_classified.last_update' % G.minConfidence
        cursor.execute(sql)
        prev_id = -1
        for next_id, category_name, file_fullname in cursor.fetchall():
            if next_id != prev_id:
                if prev_id != -1:
                    fp_to.close()
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
    def genGatherList(cursor, addtional_where=''):
        """
        切分日志包括定时归档、定长循环、定时新建3种情况，定时归档日志只需采集最新文件，定长循环、定时新建日志当作1个
        数据流处理, 采集日期最新的,如一段时间未更新, 则试图切换到更新的采集.
        """
        prev_id, wildcard_log_files = '', []
        files_by_class_and_path, common_name_set = [], set()
        results = Util.querySamples(cursor, addtional_where)
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

            cp_groups = Util.splitSamples(files_by_class_and_path, common_name_set)
            Util.getWildcard_files(cp_groups, prev_id, wildcard_log_files)

            prev_id = next_id
            files_by_class_and_path, common_name_set = [], set()

        return wildcard_log_files

    @staticmethod
    def querySamples(cursor, addtional_where):
        sql = 'SELECT model_id,category_id, host, remote_path, filename, common_name,anchor_name, anchor_start_col, anchor_end_col,last_collect,last_update FROM files_classified WHERE confidence >= %f %s ORDER BY model_id, category_id, host, remote_path' % (
            G.minConfidence, addtional_where)
        cursor.execute(sql)
        return cursor.fetchall()

    @staticmethod
    def splitSamples(files_by_class_and_path, common_name_set):
        cp_groups = {}
        num_common_names = len(common_name_set)
        differences = [len(G.fileCheckPattern.findall(name)) for name, _, _, _ in files_by_class_and_path]
        special_files = differences.count(min(differences))  # 不是通常归档文件的文件数量
        if num_common_names == 2 and special_files > 1 or num_common_names > 2:  # 同目录下多组日志文件, 进一步拆分
            common_name_groups = Util.groupCommonName(common_name_set)
            cp_groups = Util.splitFurther(common_name_groups, files_by_class_and_path)
        else:
            cp_groups = {0: files_by_class_and_path}
        return cp_groups

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

    @staticmethod
    def splitFurther(common_name_groups, files_by_class_and_path):
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
    def getWildcard_files(cp_groups, prev_id, wildcard_log_files):
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
                filename = Util.common_file_prefix(cp_group)
            else:  # 其他情况为定期归档, 采集最新的特别文件
                pass

            wildcard_log_files.append([host, remote_path, filename, model_id, category_id, anchor])

    # 计算字符串数组的公共前缀
    @staticmethod
    def common_file_prefix(cp_group):
        cp_group.sort(key=lambda x: len(x[0]))
        shortest, compared = cp_group[0][0], cp_group[1][0]
        for i in range(len(shortest), 0, -1):
            if shortest[:i] == compared[:i]:
                break
        return shortest[:i] + '*'

    @staticmethod
    def dbWildcardLogFiles(cursor, wildcard_log_files):
        for wildcard_log_file in wildcard_log_files:
            host, remote_path, filename, model_id, category_id, anchor = wildcard_log_file
            remote_path = remote_path.replace('\\', '\\\\')
            host = '"%s"' % host
            remote_path = '"%s"' % remote_path
            filename = '"%s"' % filename
            anchor = '"%s"' % anchor
            sql = 'INSERT INTO  wildcard_logfile(host, path, wildcard_name, model_id, category_id, anchor) VALUES(%s, %s,%s,%s,%s,%s) ON DUPLICATE KEY UPDATE model_id=%s, category_id=%s, anchor=%s' % (
                host, remote_path, filename, model_id, category_id, anchor, model_id, category_id, anchor)
            cursor.execute(sql)


if __name__ == '__main__':
    Util()
