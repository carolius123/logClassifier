#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : FlowProcessor.py
# @Author: Sui Huafeng
# @Date  : 2018/5/23
# @Desc  : process log flow in real time to produce kpi and events, and store into database periodically
#

import re
import time

from config import Workspaces as G
from utilites import Dbc


class FlowProcessor(object):
    def __init__(self, rc_model, flow_id):
        self.model = rc_model
        self.flow_id = flow_id
        self.kpi_data = {}
        self.events = {}
        self.interval = 300

    def run(self, data_flow):
        idx = 0
        time_ = time.time()
        for rc_id, confidence, timestamp, content in self.model.predictRecords(data_flow):
            idx += 1
            None if idx % 500 else G.log.info('%d records processed for flow %d. %s', idx, self.flow_id, content[:100])
            if confidence < self.model.MinConfidence:  # 置信度不足的记录,不予处理
                continue
            converge_interval = self.model.categories[rc_id, 5]
            if converge_interval == 0:  # w为0表示偶发记录, 不统计数量, 产生时间
                self.__sendEvent(rc_id, timestamp, content)  # 形成事件
            else:
                self.__addSI(converge_interval, rc_id, timestamp)  # 频发记录, 统计kpi指标
            self.__addVI(rc_id, timestamp, content)
            if time.time() - time_ > self.interval:
                self.persist()
                time_ = time.time()
        self.persist()

    def __sendEvent(self, rc_id, time_, content):
        source = 'flow%d-%d' % (self.flow_id, rc_id)
        counts, _ = self.events.get((time_, source), (0, ''))
        self.events[(time_, source)] = (counts + 1, content[:800])

    def __addSI(self, converge_interval, rc_id, timestamp):
        time_ = int(timestamp - timestamp % converge_interval)
        self.kpi_data[(rc_id, '', time_)] = self.kpi_data.get((rc_id, time_), 0) + 1

    def __addVI(self, rc_id, timestamp, content):
        for vi_name in self.model.variables[rc_id]:
            if vi_name[:1] == '$':  # 特征变量指标
                match_ = self.model.VIRegExp[vi_name].search(content)
                if match_:
                    self.kpi_data[(rc_id, vi_name, timestamp)] = match_.group()
            else:  # 数值变量指标
                for vi_value in re.findall(r'%s(\d+)' % vi_name, content):  # 统计数字变量指标
                    self.kpi_data[(rc_id, vi_name, timestamp)] = vi_value

    def persist(self):
        with Dbc() as cursor:
            G.log.info('Store KPI and events into database')
            if self.kpi_data:
                for (rc_id, vi_name, timestamp), kpi_value in self.kpi_data.items():
                    time_ = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(timestamp))
                    sql = 'INSERT INTO kpi_data (time, flow_id, rc_id, variable_name, kpi_value) VALUES (%s, %s, %s, %s, %s) '
                    if vi_name == '':
                        sql += 'ON DUPLICATE KEY UPDATE kpi_value=kpi_value+%s'
                    else:
                        sql += 'ON DUPLICATE KEY UPDATE kpi_value=%s'
                    cursor.execute(sql, (time_, str(self.flow_id), str(rc_id), vi_name, str(kpi_value), str(kpi_value)))
                self.kpi_data = {}
            if self.events:
                for (time_, source), (counts, content) in self.events.items():
                    time_ = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time_))
                    sql = 'INSERT INTO event (arise_time, source, msg) VALUES (%s,%s,%s) ON DUPLICATE KEY UPDATE counts=counts+%s,msg=%s'
                    cursor.execute(sql, (time_, source, content, str(counts), content))
                self.events = {}
