#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : BatchJobService.py
# @Author: Sui Huafeng
# @Date  : 2018/5/10
# @Desc  : 定期扫描$data/inbox目录, 处理新上传的gz.tar样本文件

import os
import threading
import time

from Classifier import Classifier
from config import Workspaces as G
from utilites import Util


# extract and classify files from $data/inbox/*.tar.gz
class BatchJobService(object):
    """
    Extract files from $data/inbox/*.tar.gz to $data/l0inputs, merged into $data/l0cache, classify it to get wildcard_logfile, and store metadata into database.
    """

    def __init__(self):
        self.models = []
        for model_file in [G.productFileClassifierModel, G.projectFileClassifierModel]:
            if os.path.exists(model_file):
                model = Classifier(model_file=model_file)
                model.dbUpdCategories()
            else:
                model = None
            self.models.append(model)

        self.interval = G.cfg.getfloat('BatchJobService', 'IntervalMinutes') * 60
        self.db, self.cursor = None, None
        self.run()

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
                common_files, file2merged = Util.mergeFilesByName(extract_to, merge_to)
                common_files = self.__getNewLogInstance(host, common_files)
                classified_files, unclassified_files = self.__predict(common_files)
                Util.dbFilesMerged(self.cursor, file2merged, classified_files, unclassified_files)
                where_clause = ['"%s"' % common_name.replace('\\', '/') for _, common_name, _, _, _, _ in
                                classified_files]
                where_clause = ' AND common_name in (%s)' % ','.join(where_clause)
                wildcard_log_files = Util.genGatherList(self.cursor, where_clause)  # 生成采集文件列表
                Util.dbWildcardLogFiles(self.cursor, wildcard_log_files)
                self.db.commit()
                G.log.info('Processed %s successful.', tar_file)
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

    def __predict(self, merged_files):
        classified_files, unclassified_files = [], []
        for common_name in merged_files:
            for model_id, model in enumerate(self.models):
                if model:  # 模型存在
                    result = model.predictFile(common_name)
                    if not result:
                        continue
                    category, category_name, confidence, distance = result
                    if confidence < G.minConfidence:  # 置信度不够,未完成分类
                        continue
                    classified_files.append([model_id, common_name, category, category_name, confidence, distance])
                    break
            else:
                unclassified_files.append(common_name)
        return classified_files, unclassified_files


if __name__ == '__main__':
    BatchJobService()
