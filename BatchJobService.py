#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : BatchJobService.py
# @Author: Sui Huafeng
# @Date  : 2018/5/10
# @Desc  : 定期扫描$data/inbox目录, 处理新上传的gz.tar样本文件

import os
import shutil
import threading
import time

from FileClassifier import FileClassifier
from config import Workspaces as G
from utilites import Dbc, FileUtil, DbUtil


# extract and classify files from $data/inbox/*.tar.gz
class BatchJobService(object):
    """
    Extract files from $data/inbox/*.tar.gz to $data/l0inputs, merged into $data/l0cache, classify it to get wildcard_logfile, and store metadata into database.
    """

    def __init__(self):
        self.models = []
        for model_file in [G.productFileClassifierModel, G.projectFileClassifierModel]:
            if os.path.exists(model_file):
                model = FileClassifier(model_file=model_file)
            else:
                model = None
            self.models.append(model)

        self.reInitialization()

        self.interval = G.cfg.getfloat('BatchJobService', 'IntervalMinutes') * 60
        self.db, self.cursor = None, None
        self.run()

    def run(self):
        self.db = DbUtil.dbConnect()
        if not self.db:  # Can't connect ro db, schedule to next
            G.log.warning('Batch Service error, scheduled to next time')
            threading.Timer(self.interval, self.run).start()
        self.cursor = self.db.cursor()
        self.classifyNewLogs()
        self.db.commit()
        self.db.close()
        threading.Timer(self.interval, self.run).start()

    def classifyNewLogs(self):
        G.log.info('Extracting files from %s to %s', G.inbox, G.l0_inputs)
        for tar_file in os.listdir(G.inbox):
            if not tar_file.endswith('.tar.gz'):
                continue
            host = tar_file[:-7]
            extract_to, merge_to = os.path.join(G.l0_inputs, host), os.path.join(G.l1_cache, host)
            tar_file_fullname = os.path.join(G.inbox, tar_file)
            try:
                FileUtil.extractFile(tar_file_fullname, extract_to)  # 解压文件到当前目录
                self.__dbUpdFilesSampledFromDescriptionFile(host, extract_to)  # 更新数据库中样本文件表
                os.remove(tar_file_fullname)
                common_files, file2merged = FileUtil.mergeFilesByName(extract_to, merge_to)
                common_files = self.__getNewFiles(common_files)  # 滤除以前做过分类预测的文件
                classified_files, unclassified_files = FileUtil.predictFiles(self.models, common_files)  # 进行分类预测
                DbUtil.dbUdFilesMerged(self.cursor, classified_files, unclassified_files)  # 更新数据库中合并文件表
                where_clause = ['"%s"' % common_name.replace('\\', '/') for _, common_name, _, _, _, _ in
                                classified_files]
                where_clause = ' AND common_name in (%s)' % ','.join(where_clause)
                wildcard_log_files = FileUtil.genGatherList(self.cursor, where_clause)  # 生成采集文件列表
                DbUtil.dbWildcardLogFiles(self.cursor, wildcard_log_files)  # 更新数据库中采集文件列表
                self.db.commit()
                G.log.info('%s extracted and classified successful.', tar_file)
            except Exception as err:
                G.log.warning('%s extracted or classified error:%s', tar_file, str(err))
                self.db.rollback()
                continue

    # 滤除已成功分类的文件,返回新发现的日志文件
    def __getNewFiles(self, common_files):
        new_common_files = []
        for common_file in common_files:
            c_ = '"%s"' % common_file.replace('\\', '/')
            sql = 'SELECT COUNT(common_name) FROM files_merged WHERE  common_name=%s' % c_
            self.cursor.execute(sql)
            result = self.cursor.fetchone()
            if not result[0]:
                new_common_files.append(common_file)
        return new_common_files

    def __dbUpdFilesSampledFromDescriptionFile(self, host, extract_to):
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

    # 重新初始化模型
    def reInitialization(self):
        """
        系统积累了一定量的无法通过产品内置模型分类的样本, 或者产品模型升级后, 重新聚类形成现场模型.
        清理以前的现场数据 -> 重新合并原始样本->采用内置产品模型分类 -> 日志文件模型聚类 -> 日志记录模型聚类 -> 重新计算基线
        :return:
        """
        self.__clearModelAndConfig()
        FileUtil.mergeFilesByName(G.l0_inputs, G.l1_cache)
        product_model = FileClassifier(model_file=G.productFileClassifierModel)
        if product_model.model:
            self.__reClassifyAllSamples(product_model)
        project_model = FileClassifier()
        project_model.reCluster()

        self.models = [product_model, project_model]

    @staticmethod
    def __clearModelAndConfig():
        for folder in [G.projectModelPath, G.l2_cache, G.outputs]:
            if os.path.exists(folder):
                shutil.rmtree(folder)
        db = DbUtil.dbConnect()
        if db:
            cursor = db.cursor()
            cursor.execute('DELETE FROM file_class')
            db.commit()
            db.close()

        time.sleep(3)
        for folder in [G.projectModelPath, G.l2_cache, G.outputs]:
            os.mkdir(folder)

    @staticmethod
    def __reClassifyAllSamples(model):
        G.log.info('Classifying files in %s by model %s', G.l1_cache, G.productFileClassifierModel)
        results = model.predictFiles(G.l1_cache)
        classified_common_files, unclassified_common_files = FileUtil.splitResults(model.model_id, results)
        with Dbc() as cursor:
            DbUtil.dbUdFilesMerged(cursor, classified_common_files, unclassified_common_files)


if __name__ == '__main__':
    BatchJobService()
