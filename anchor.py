#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : anchor.py
# @Author: Sui Huafeng
# @Date  : 2018/4/9
# @Desc  : 通常情况下，日志文件每条记录都包含时间戳，可以作为区分多少行形成一条记录的定位锚点。该时间戳在每条记录第一行，
#        列的位置相对固定。本程序扫描样本文件，确定时间戳锚点的列所在范围。
#

import re
from config import Workspaces as G


class Anchor(object):
    __TimeSep = [r'(:|Ê±|时-)', r'(:|·Ö|分-)']
    __TimeSpecs = {'TIMESPAN': '(\d+)' + __TimeSep[0] + r'(\d|[0-5]\d)' + __TimeSep[1] + r'(\d|[0-5]\d)(?!\d)',
                   'TIMESTAMP': r'(?<!\d)(15\d{8})\.\d+',
                   'HHMMSS': r'(?<!\d)([0-1]\d|2[0-3])([0-5]\d){2}(?!\d)'
                   }
    __TimeRegExp = re.compile('|'.join('(?P<%s>%s)' % (k_, v_) for k_, v_ in __TimeSpecs.items()), re.I)

    # 从sample file提取时间戳锚点，或者按anchor等参数设置时间戳锚点
    def __init__( self, sample_file=None, col_range=(0, None), name='TIMESPAN' ):
        self.colRange = col_range  # (搜索的开始列，结束列)
        self.name = name  # 时间戳锚点正则表达式名称，及各值(年月日时分秒和上下午)在match对象中的位置
        self.regExp = re.compile('(%s)' % self.__TimeSpecs[name], re.I) if name else None

        self.setFormat(self.__getAnchor(sample_file), sample_file) if sample_file else None

    # 记录样本文件每行的第一和最后一个时间戳的位置和内容(最后一个时间戳记录距离尾部的距离）
    @classmethod
    def __getAnchor( cls, sample_file ):
        line_idx = 0
        anchors = {}  # 初始化潜在锚点字典
        for line_idx, line in enumerate(open(sample_file, 'r', encoding='utf-8')):
            try:
                if cls.appendAnchors(line, anchors):  # 已经找够了时间戳锚点
                    break
            except (UnicodeError, UnicodeDecodeError, UnicodeEncodeError):
                G.log.warning('Line[%d] ignored due to the following error:', line_idx)
                continue
        return Anchor.statsAnchors(anchors, line_idx)

    # 从字符串line中抽取第一和最后时间戳，追加到match_groups
    @classmethod
    def appendAnchors( cls, line, anchors ):
        first_, last_ = None, None
        matches_ = [v for v in cls.__TimeRegExp.finditer(line)]
        if not matches_:  # not found
            return False

        for i in [0, -1]:
            match_ = matches_[i]  # first or last match object
            regex_name, base_idx = match_.lastgroup, match_.lastindex
            start_ = match_.start() if i == 0 else match_.start() - len(line)  # start col position of first one
            stop_ = match_.end() if i == 0 else match_.end() - len(line)  # stop col position of first one
            counter_, stop_max = anchors.get((start_, regex_name), [0, -10000])
            anchors[(start_, regex_name)] = [counter_ + 1, max(stop_, stop_max)]
            if counter_ > 100:
                return True
        return False

    # 统计得到最佳Anchor
    @staticmethod
    def statsAnchors( anchors, lines ):
        anchors = sorted(anchors.items(), key=lambda d: d[1][0], reverse=True)
        G.log.debug('Candidate anchors detected at(col:freq) %s',
                    ', '.join(['%s(%d:%d)' % (k[1], k[0], v[0]) for k, v in anchors]))
        amount = anchors[0][1][0] if anchors != [] else 0
        if amount < max(5, lines / 50):
            raise UserWarning('No valid anchors')
        return anchors[0]

    # 跟据日志中日期变化规律(先日再月后年)，确定日期格式
    def setFormat( self, anchor, filename='' ):
        (start_, name_), (amount_, stop_) = anchor
        # 考虑时间戳前后可能有变长变量(如INFO，CRITICAL等），给起止位置留一定余量(前后5个字符）
        self.colRange = (0 if 0 <= start_ < 5 else start_ - 5, 0 if -5 <= stop_ < 0 else stop_ + 5)
        self.name = name_  # 设置正则表达式名称
        self.regExp = re.compile('(%s)' % self.__TimeSpecs[name_], re.I)  # 设置正则表达式

    # 返回锚点的时间戳值，如不存在，返回None
    def getAnchorTimeStamp( self, line ):
        if self.regExp is None:
            raise UserWarning('Failed to getAnchorTimeStamp: Anchor is not initialized!')

        anchor = self.regExp.search(line[self.colRange[0]:self.colRange[1]])
        if not anchor:  # 没有搜到有效的时间戳
            return None
        else:
            return anchor


if __name__ == '__main__':
    a = Anchor('E:\\data\\inputs\\10.21.84.7\\data\\tmp\\upload_00027669.tmp')
    b= a.name
    b = ((0, 'DATE_TIME'), [1001, {'17-12-28-0-0'}, 19])
    a.setFormat(b)
