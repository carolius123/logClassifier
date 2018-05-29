#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : KpiProcessor.py
# @Author: Sui Huafeng
# @Date  : 2018/5/23
# @Desc  : 
#

import time

from utilites import Dbc


#
class KPI(object):
    """
    
    """
    __Interval = 5 * 60  # KPI计数的汇聚周期(秒)

    def __init__(self, flow_id):
        self.flow_id = flow_id
        self.kpi = {}
        self.lastPersist = time.time()

    def count(self, rc_id, timestamp):
        time_ = int(timestamp - timestamp % self.__Interval)
        self.kpi[(rc_id, time_)] = self.kpi.get((rc_id, time_), 0) + 1
        if self.kpi and time.time() - self.lastPersist > self.__Interval:
            self.persist()

    def persist(self):
        with Dbc() as cursor:
            for (rc_id, time_), kpi_value in self.kpi.items():
                time_ = time.strftime('"%Y-%m-%d %H:%M:%S"', time.localtime(time_))
                sql = 'INSERT INTO kpi_data (time, flow_id, rc_id, variable_id, kpi_value) VALUES (%s, %d, %d, 0, %d) ON DUPLICATE KEY UPDATE kpi_value=kpi_value+%d' % (
                time_, self.flow_id, rc_id, kpi_value, kpi_value)
                cursor.execute(sql)
        self.kpi = {}
        self.lastPersist = time.time()
        return


if __name__ == '__main__':
    fp = open('d:/t.txt', 'w', encoding='gb2312')
    fp.write('一月\n')
    fp.close()

    for line in open('D:\\home\\suihf\\data\\classified\\fc0-57.samples', encoding='utf-8'):
        gbk = line.encode(encoding='gb2312')

    with Dbc() as c1:
        c1.execute('SELECT * FROM kpi')
        result = c1.fetchall()
        print(result)
        with Dbc() as c2:
            c2.execute('INSERT INTO record_class(model_id, fc_id,rc_id) VALUES (1,2,3)')
            result = c2.fetchall()
            print(result)
