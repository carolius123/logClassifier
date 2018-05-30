#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : BatchJobService.py
# @Author: Sui Huafeng
# @Date  : 2018/5/10
# @Desc  : 定期扫描$data/inbox目录, 处理新上传的gz.tar样本文件

import os
import re
import shutil
import threading
import time

from FileClassifier import FileClassifier
from config import Workspaces as G
from utilites import Dbc, FileUtil


# extract and classify files from $data/inbox/*.tar.gz
class BatchJobService(object):
    """
    Extract files from $data/inbox/*.tar.gz to $data/l0inputs, merged into $data/l0cache, classify it to get wildcard_logfile, and store metadata into database.
    """

    def __init__(self):
        self.models = FileClassifier.loadAllModels()
        self.interval = G.cfg.getfloat('BatchJobService', 'IntervalMinutes') * 60

    def run(self):
        try:
            self.classifyNewLogs()
        except Exception as err:
            G.log.warning('Batch Service error, scheduled to next time.%s', str(err))

        threading.Timer(self.interval, self.run).start()

    # 对上传的新类型文件进行分类, 生成采集列表
    def classifyNewLogs(self):
        """
        $transitPath/*.tar.gz -> 解压 -> 合并 -> 滤除已知文件 -> 新文件分类预测 -> 生成采集列表
            -> 无法分类的,积累在$mergedFilePath
            -> 分类成功,记录分类模型尚未形成的, 记录模型聚类 -> 生成指标模型
        :return: 无
        """
        G.log.info('Extracting files from %s to %s', G.transitPath, G.inboxPath)
        for tar_file in os.listdir(G.transitPath):
            if not tar_file.endswith('.tar.gz'):
                continue
            host = tar_file[:-7]
            extract_to, merge_to = os.path.join(G.inboxPath, host), os.path.join(G.mergedFilePath, host)
            tar_file_fullname = os.path.join(G.transitPath, tar_file)
            try:
                FileUtil.extractFile(tar_file_fullname, extract_to)  # 解压文件到当前目录
                with Dbc() as cursor:
                    self.__dbUpdFilesSampledFromDescriptionFile(cursor, host, extract_to)  # 更新数据库中样本文件表
                os.remove(tar_file_fullname)
                common_files, file2merged = FileUtil.mergeFilesByName(extract_to, merge_to)
                common_files = self.__getNewFiles(common_files)  # 滤除以前做过分类预测的文件
                for model in self.models:
                    common_files = model.predict(common_files)
            except Exception as err:
                G.log.warning('%s extracted or classified error:%s', tar_file, str(err))
                continue

    # 滤除已成功分类的文件,返回新发现的日志文件
    @staticmethod
    def __getNewFiles(common_files):
        new_common_files = []
        with Dbc() as  cursor:
            for common_file in common_files:
                c_ = '"%s"' % common_file.replace('\\', '/')
                sql = 'SELECT COUNT(common_name) FROM files_merged WHERE  common_name=%s' % c_
                cursor.execute(sql)
                result = cursor.fetchone()
                if not result[0]:
                    new_common_files.append(common_file)
        return new_common_files

    @staticmethod
    def __dbUpdFilesSampledFromDescriptionFile(cursor, host, extract_to):
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
            cursor.execute(sql)
        os.remove(description_file)

    # 重新初始化模型
    def buildModel(self, model_id=None, re_merge=False):
        """
        系统积累了一定量的无法通过产品内置模型分类的样本, 或者产品模型升级后, 重新聚类形成现场模型.
        清理以前数据 -> 重新合并原始样本->采用内置产品模型分类 -> 日志文件模型聚类 -> 生成采集列表 - >日志记录模型聚类
         -> 生成指标模型
        """
        if model_id is None:
            models = len([path for path in os.listdir(G.modelPath)
                          if re.match('\d+', path) and os.path.isdir(os.path.join(G.modelPath, path))])
            model_id = min(models, FileClassifier.maxFcModels - 1)

        self.__clearModelAndResult(model_id)
        if re_merge:
            shutil.rmtree(G.mergedFilePath)
            time.sleep(3)
            os.mkdir(G.mergedFilePath)
            FileUtil.mergeFilesByName(G.inboxPath, G.mergedFilePath)

        try:
            FileClassifier(model_id, G.mergedFilePath)
        except UserWarning:
            pass
        self.models = FileClassifier.loadAllModels()

    @staticmethod
    def __clearModelAndResult(model_id):
        model_ids = sorted([int(path) for path in os.listdir(G.modelPath) if re.match('\d+', path)
                            and os.path.isdir(os.path.join(G.modelPath, path)) and int(path) >= model_id])
        for id_ in model_ids:  # 清除需重建的模型和结果
            path_ = os.path.join(G.modelPath, str(id_))
            if os.path.exists(path_):
                shutil.rmtree(path_)

            # 清除原来聚类结果
            for file in os.listdir(G.classifiedFilePath):
                file_fullname = os.path.join(G.classifiedFilePath, file)
                if os.path.isfile(file_fullname) and re.match('fc%d-' % id_, file):
                    os.remove(file_fullname)
        # 清除数据库内容
        with Dbc() as cursor:
            cursor.execute('DELETE FROM file_class WHERE model_id >=%d' % model_id)


if __name__ == '__main__':
    bcs = BatchJobService()
    # bcs.buildModel()
    bcs.run()
