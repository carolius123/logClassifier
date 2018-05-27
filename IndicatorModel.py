#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : IndicatorModel.py
# @Author: Sui Huafeng
# @Date  : 2018/5/23
# @Desc  : 
#

from utilites import Dbc


#
class IndicatorModel(object):
    """
    
    """

    def __init__(self):
        self.x = 1


if __name__ == '__main__':
    with Dbc() as c1:
        c1.execute('SELECT * FROM kpi')
        result = c1.fetchall()
        print(result)
        with Dbc() as c2:
            c2.execute('INSERT INTO record_class(model_id, fc_id,rc_id) VALUES (1,2,3)')
            result = c2.fetchall()
            print(result)

    IndicatorModel()
