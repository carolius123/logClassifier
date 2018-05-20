#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : OnlineService.py
# @Author: Sui Huafeng
# @Date  : 2018/5/19
# @Desc  : 
#

import os
import shutil
import threading
import time

from Classifier import Classifier
from anchor import Anchor
from config import Workspaces as G
from utilites import Dbc, FileUtil, DbUtil


# 实时接收日志流,
class OnlineService(object):
    """

    """

    def __init__(self):
        self.service_id = G.cfg.getint('OnlineService', 'ServiceID')
        self.models = []
        for model_file in [G.productFileClassifierModel, G.projectFileClassifierModel]:
            if os.path.exists(model_file):
                model = Classifier(model_file=model_file)
            else:
                model = None
            self.models.append(model)

    # Deamon for new flow
    def deamon(self):
        x = 1

    # dispatch a new thread to a new flow
    def dispatcher(self, source_id, data_flow, host=None, path=None, wildcard_name=None):
        source_id = str(source_id)
        flow_id, flow_status = self.getFlowID(source_id, host, path, wildcard_name)
        t = threading.Thread(target=self.threadLogFlow, args=(flow_id, source_id, flow_status, data_flow))
        t.start()

    # thread to deal a data flow
    def threadLogFlow(self, flow_id, source_id, status, data_flow):
        G.log.info('Thread for log flow[%d-%d] started at %s', self.service_id, flow_id, status)
        if status == '未锚定':
            status, file_name = self.bufferFlow(flow_id, source_id, status, data_flow)
            if status == 'OK':  # 缓存行数已经足够
                status = self.predict(flow_id, file_name)
        if status == '未分类':
            status = self.bufferFlow(flow_id, source_id, status, data_flow)[0]
        if status == '活动中':
            status = self.working(flow_id, status, data_flow)
        G.log.info('Thread for log flow[%d-%d] terminated at %s', self.service_id, flow_id, status)

    # OnlineService接收到一个新连接请求时, 调用本过程获得流id
    def getFlowID(self, source_id, host=None, path=None, wildcard_name=None):
        """
        Get the flow_id for a new connection
        :param path:
        :param source_id: id from source to identify different flow
        :param host: path:wildcard_name: match key to distribute to the source
        :return: (ID, status)
        """
        with Dbc() as cursor:
            result = self.__isExistFlow(cursor, source_id)
            if not result and host and path and wildcard_name:
                result = self.__isKnowFlow(cursor, source_id, host, path, wildcard_name)
            if not result:
                result = self.__addNewFlow(cursor, source_id)
            return result

    # 是否已知flow
    def __isExistFlow(self, cursor, source_id):
        sql = "SELECT id, status FROM tbl_log_flow WHERE service_id=%d AND source_id='%s'" % (
            self.service_id, source_id)
        cursor.execute(sql)
        result = cursor.fetchone()
        if result and result[1] in ['已中断', '无锚点']:
            flow_id, status = result
            col_names = ['service_id', 'source_id', 'status', 'received_bytes', 'received_lines']
            new_status = '活动中' if status == '已中断' else '未锚定'
            row = [self.service_id, source_id, new_status, 0, 0]
            DbUtil.dbUpdFlow(cursor, flow_id, col_names, row)
        return result

    # 是否聚类时事先识别的Flow, 激活该flow
    def __isKnowFlow(self, cursor, source_id, host, path, wildcard_name):
        path = path.replace('\\', '/')
        sql = "SELECT id FROM tbl_log_flow WHERE host='%s' AND path='%s' AND wildcard_name='%s' AND status<>'活动中'" % (
            host, path, wildcard_name)
        cursor.execute(sql)
        result = cursor.fetchone()
        if result:
            result = self.__activateFlow(cursor, result[0], source_id)
        return result

    # 状态演进到'活动中'
    def __activateFlow(self, cursor, flow_id, source_id):
        sql = "UPDATE tbl_log_flow SET service_id=%d, source_id='%s', status='活动中', received_bytes=0,received_lines=0 WHERE id = %d" % (
            self.service_id, source_id, flow_id)
        cursor.execute(sql)
        return flow_id, '活动中'

    # 未知且事前未识别过的新Flow, 新增并开始积累数据
    def __addNewFlow(self, cursor, source_id):
        sql = "INSERT INTO tbl_log_flow (service_id, source_id, status) VALUES (%d,'%s','未锚定')" % (
            self.service_id, source_id)
        cursor.execute(sql)
        cursor.execute('SELECT LAST_INSERT_ID()')
        result = cursor.fetchone()
        id_ = result[0]
        return id_, '未锚定'

    # 把流数据缓存到文件中 ,并记录已经缓存的行数和字符数
    def bufferFlow(self, flow_id, source_id, flow_status, data_flow):
        filename = 'flow%d-%d-%s.log' % (flow_id, self.service_id, source_id)
        if flow_status == '未锚定':
            path = G.inbox
        else:
            path = os.path.join(G.l0_inputs, os.path.splitext(filename)[0])
        file_fullname = os.path.join(path, filename)
        now = time.time()

        with open(file_fullname, 'a', encoding='utf-8', errors='ignore') as fp, Dbc() as cursor:
            received_lines, received_bytes = self.__dbSelFlowLines(cursor, flow_id)
            for line in data_flow:
                fp.write(line)
                received_lines += 1
                received_bytes += len(line)
                if flow_status == '未锚定' and received_lines >= G.maxClassifyLines:
                    DbUtil.dbUpdFlow(cursor, flow_id, ['received_lines', 'received_bytes'],
                                     [received_lines, received_bytes])
                    return 'OK', file_fullname
                if time.time() - now > 3600:
                    now = time.time()
                    DbUtil.dbUpdFlow(cursor, flow_id, ['received_lines', 'received_bytes'],
                                     [received_lines, received_bytes])
            else:  # 数据流关闭
                DbUtil.dbUpdFlow(cursor, flow_id, ['received_lines', 'received_bytes'],
                                 [received_lines, received_bytes])
                return flow_status, file_fullname

    # 取flow的行数和字符数
    @staticmethod
    def __dbSelFlowLines(cursor, flow_id):
        cursor.execute('SELECT received_lines,received_bytes from tbl_log_flow WHERE id=%d' % flow_id)
        result = cursor.fetchone()
        return result

    # 更新flow某一行数据

    # 文件分类预测
    def predict(self, flow_id, inbox_file):
        try:  # 计算时间戳锚点, 滤掉没有锚点的文件
            anchor = Anchor(inbox_file)
        except UserWarning as err:
            G.log.debug('Failed to process\t%s, ignored.\t%s', inbox_file, str(err))
            with Dbc() as cursor:
                DbUtil.dbUpdFlow(cursor, flow_id, ['status'], ['无锚点'])
            os.remove(inbox_file)
            return '无锚点'

        filename = os.path.split(inbox_file)[1]
        path = os.path.join(G.l0_inputs, os.path.splitext(filename)[0])
        os.makedirs(path, exist_ok=True)
        file_fullname = os.path.join(path, filename)
        shutil.copy(inbox_file, file_fullname)

        path = path.replace(G.l0_inputs, G.l1_cache)
        os.makedirs(path, exist_ok=True)
        common_name = G.fileMergePattern.sub('', os.path.splitext(filename)[0])
        common_name = '%s-%s' % (anchor.name, common_name)
        common_file_fullname = os.path.join(path, common_name)
        shutil.move(inbox_file, common_file_fullname)

        classified_files, unclassified_files = FileUtil.predictFiles(self.models, [
            common_file_fullname])  # 进行分类预测. classified_files [[model_id, common_name, category, category_name, confidence, distance]], unclassified_files [common_name]

        with Dbc() as cursor:
            file2merged = [[file_fullname, anchor.name, anchor.colRange[0], anchor.colRange[1], common_file_fullname]]
            DbUtil.dbUpdFilesSampled(cursor, file2merged)
            DbUtil.dbUdFilesMerged(cursor, classified_files, unclassified_files)  # 更新数据库中合并文件表

            anchor_info = '%s:%d:%d' % (anchor.name, anchor.colRange[0], anchor.colRange[1])
            if len(classified_files):
                status = '活动中'
                model_id, _, category_id, category_name, _, _ = classified_files[0]
                G.log.info('log flow[%d-%d] classified to [%d-%s]', self.service_id, flow_id, model_id, category_name)

            else:
                status = '未分类'
                model_id, category_id = None, None
                G.log.info('log flow[%d-%d] can not to be classified', self.service_id, flow_id)
            DbUtil.dbUpdFlow(cursor, flow_id, ['status', 'model_id', 'category_id', 'anchor'],
                             [status, model_id, category_id, anchor_info])

        return status

    def working(self, flow_id, flow_status, data_flow):
        return '活动中'


if __name__ == '__main__':
    ols = OnlineService()
    ols.dispatcher('abc', open('D:\\home\\suihf\\data\\fc28', 'r', encoding='utf-8'))
    ols.dispatcher('def', open(
        'D:/home/suihf/data/l0inputs/10.20.23.139/home/cims/domains/ip_ap_dm/servers/ApServer/logs/access.log00720',
        'r', encoding='utf-8'))
    ols.dispatcher('xxx', open(
        'D:/home/suihf/data/l0inputs/10.20.23.139/home/cims/domains/ip_ap_dm/servers/ApServer/logs/ApServer.log', 'r',
        encoding='utf-8'))
    x = 1
