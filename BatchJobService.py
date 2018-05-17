#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : BatchJobService.py
# @Author: Sui Huafeng
# @Date  : 2018/5/10
# @Desc  : 定期扫描$data/inbox目录, 处理新上传的gz.tar样本文件

import os
import threading
import time

import pandas as pd

from Classifier import Classifier
from config import Workspaces as G
from utilites import Util


# extract and classify files from $data/inbox/*.tar.gz
class BatchJobService(object):
    """
    Extract files from $data/inbox/*.tar.gz to $data/l0inputs, merged into $data/l0cache, classify it to get wildcard_logfile, and store metadata into database.
    """

    def __init__(self):
        self.models = [Classifier(model_file=G.productFileClassifierModel),
                       Classifier(model_file=G.projectFileClassifierModel)]
        self.interval = G.cfg.getfloat('BatchJobService', 'IntervalMinutes') * 60
        self.db, self.cursor = None, None

        self.__dbUpdCategories()
        self.run()

    def __dbUpdCategories(self):
        db = Util.dbConnect()
        if db:  # Can't connect ro db, waiting and retry forever
            cursor = db.cursor()
            for model_id, model in enumerate(self.models):
                if model.model:
                    c = model.categories
                    for category_id, (name, percent, boundary, quantile) in enumerate(zip(c[0], c[1], c[2], c[3])):
                        name = '"%s"' % name
                        sql = 'INSERT INTO file_class (model_id, category_id, name, quantile, boundary, percent) VALUES(%d, %d, %s, %e,%e, %f) ON DUPLICATE KEY UPDATE name=%s,quantile=%e,boundary=%e,percent=%f' % (
                            model_id, category_id, name, quantile, boundary, percent, name, quantile, boundary, percent)
                        cursor.execute(sql)
                try:
                    db.commit()
                except Exception as err:
                    G.log.warning('can not update logfile categories into database, skiped.. %s', str(err))
                    db.rollback()
            db.close()

    def run(self):
        self.db = Util.dbConnect()
        if not self.db:  # Can't connect ro db, schedule to next
            threading.Timer(self.interval, self.run).start()
        self.cursor = self.db.cursor()
        self.processing()
        self.db.close()
        threading.Timer(self.interval, self.run).start()

    def processing(self):
        G.log.info('Extracting files from %s to %s', G.inbox, G.l0_inputs)
        for tar_file in os.listdir(G.inbox):
            if not tar_file.endswith('.tar.gz'):
                continue
            host = tar_file[:-7]
            extract_to, merge_to = os.path.join(G.l0_inputs, host), os.path.join(G.l1_cache, host)
            tar_file_fullname = os.path.join(G.inbox, tar_file)
            try:
                Util.extractFile(tar_file_fullname, extract_to)  # 解压文件到当前目录
                self.__dbFilesSampled(host, extract_to)
                os.remove(tar_file_fullname)
                common_files, file2merged = Util.mergeFiles(extract_to, merge_to)
                common_files = self.__getNewLogInstance(host, common_files)
                classified_files, unclassified_files = self.__predict(common_files)
                self.__dbFilesMerged(file2merged, classified_files, unclassified_files)
                self.__dbWildcardLogfile(classified_files)
                self.db.commit()
                G.log.debug('Processed %s successful.', tar_file)
            except Exception as err:
                G.log.warning('processing error:%s', str(err))
                self.db.rollback()
                continue

    # 滤除已成功分类的文件,返回新发现的日志文件
    def __getNewLogInstance(self, host, common_files):
        new_common_files = []
        for common_file in common_files:
            c_ = '"%s"' % common_file.replace('\\', '/')
            sql = 'SELECT COUNT(common_name) FROM files_merged WHERE  common_name=%s' % c_
            self.cursor.execute(sql)
            result = self.cursor.fetchone()
            if not result[0]:
                new_common_files.append(common_file)
        return new_common_files

    def __dbFilesSampled(self, host, extract_to):
        host = '"%s"' % host
        description_file = os.path.join(extract_to, 'descriptor.csv')
        for line in open(description_file, encoding='utf-8'):
            gather_time, last_update_time, ori_size, ori_name, archive_name = line.strip().split('\t')
            archive_name = archive_name.replace('\\', '/').strip('/')
            file_fullname = os.path.join(extract_to, archive_name).replace('\\', '/')
            if not os.path.exists(file_fullname):  # 解压出错,没有文件
                continue
            file_fullname = '"%s"' % file_fullname
            gather_time, last_update_time, ori_size = float(gather_time), float(last_update_time), int(ori_size)
            gather_time = '"%s"' % time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(gather_time))
            last_update_time = '"%s"' % time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(last_update_time))
            archive_path, filename = os.path.split(archive_name)
            archive_path = '"%s"' % archive_path
            filename = '"%s"' % filename
            remote_path, _ = os.path.split(ori_name)
            remote_path = '"%s"' % remote_path.replace('\\', '\\\\')
            sql = 'INSERT INTO files_sampled (file_fullname,host,archive_path,filename,remote_path,last_update,last_collect,size) VALUES(%s,%s,%s,%s,%s,%s,%s,%d) ON DUPLICATE KEY UPDATE last_update=%s, last_collect=%s, size=%d' % (
            file_fullname, host, archive_path, filename, remote_path, last_update_time, gather_time, ori_size,
            last_update_time, gather_time, ori_size)
            self.cursor.execute(sql)
        os.remove(description_file)

    def __dbFilesMerged(self, file2merged, classified_files, unclassified_files):
        for common_name in unclassified_files:
            common_name = '"%s"' % common_name.replace('\\', '/')
            sql = 'INSERT INTO files_merged (common_name) VALUES(%s) ON DUPLICATE KEY UPDATE common_name=%s' % (
                common_name, common_name)
            self.cursor.execute(sql)

        for model_id, cf in enumerate(classified_files):
            for common_name, category, category_name, confidence, distance in cf:
                common_name = '"%s"' % common_name.replace('\\', '/')
                sql = 'INSERT INTO files_merged (common_name, model_id, category_id, confidence, distance) VALUES(%s, %d, %d, %f, %e)' % (
                common_name, model_id, category, confidence, distance)
                self.cursor.execute(sql)

        for file_fullname, anchor_name, anchor_colRange, common_file_fullname in file2merged:
            file_fullname = '"%s"' % file_fullname.replace('\\', '/')
            common_file_fullname = '"%s"' % common_file_fullname.replace('\\', '/')
            anchor_name = '"%s"' % anchor_name
            sql = 'UPDATE files_sampled set common_name=%s, anchor_name=%s, anchor_start_col=%d, anchor_end_col=%d WHERE file_fullname=%s' % (
                common_file_fullname, anchor_name, anchor_colRange[0], anchor_colRange[1], file_fullname)
            self.cursor.execute(sql)

    def __dbWildcardLogfile(self, classified_files):
        for model_id, classified_files_by_model in enumerate(classified_files):
            if not classified_files_by_model:
                continue
            df = pd.DataFrame(columns=(
            'path', 'name', 'common_name', 'c_id', 'c_name', 'confidence', 'ori_name', 'seconds_ago', 'anchor'))
            loc_idx = 0
            for common_name, c_id, c_name, confidence, _ in classified_files_by_model:
                common_name = '"%s"' % common_name.replace('\\', '/')
                self.cursor.execute(
                    'SELECT file_fullname, remote_path, filename, last_collect, last_update, anchor_name,anchor_start_col, anchor_end_col FROM files_sampled WHERE common_name=%s' % common_name)
                for file_fullname, remote_path, filename, last_collect, last_update, anchor_name, anchor_start_col, anchor_end_col in self.cursor.fetchall():
                    path_, filename = os.path.split(file_fullname)
                    ori_name = os.path.join(remote_path, filename)
                    seconds_ago = (last_collect - last_update).total_seconds()
                    anchor = '%s:%d:%d' % (anchor_name, anchor_start_col, anchor_end_col)
                    df.loc[
                        loc_idx] = path_, filename, common_name, c_id, c_name, confidence, ori_name, seconds_ago, anchor
                    loc_idx += 1
            log_flows = self.models[model_id].genGatherList(df)
            for i in range(len(log_flows)):
                host, ori_path, filename, category_id, anchor = log_flows.loc[i]
                ori_path = ori_path.replace('\\', '\\\\')
                sql = 'INSERT INTO  wildcard_logfile(host, path, wildcard_name, model_id, category_id, anchor) VALUES("%s", "%s","%s",%d, %d, "%s")' % (
                host, ori_path, filename, model_id, category_id, anchor)
                self.cursor.execute(sql)

    def __predict(self, merged_files):
        classified_files = [[] for model in self.models]
        unclassified_files = []
        for common_name in merged_files:
            for idx, model in enumerate(self.models):
                if model.model:  # 模型存在
                    result = model.predictFile(common_name)
                    if not result:
                        continue
                    category, category_name, confidence, distance = result
                    if confidence < G.minConfidence:  # 置信度不够,未完成分类
                        continue
                    classified_files[idx].append([common_name, category, category_name, confidence, distance])
                    break
            else:
                unclassified_files.append(common_name)
        return classified_files, unclassified_files


if __name__ == '__main__':
    BatchJobService()
