#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : anchor.py
# @Author: Sui Huafeng
# @Date  : 2018/4/9
# @Desc  :
#
"""
设定时间戳锚点，或者扫描样本文件，确定时间戳锚点的列所在范围。
"""

import re

from config import Workspaces as G


class Anchor(object):
    """
    通常情况下，日志文件每条记录都包含时间戳，可以作为区分多少行形成一条记录的定位锚点。该时间戳在每条记录第一行，
    列的位置相对固定。
    """
    __TimeSep = [r'(:|Ê±|时|时-)', r'(:|·Ö|分|分-)']
    __PosHHMMSS = {'COLON': (1, 3, 5, 6), 'SECONDS': (None, None, 1, None), 'HHMMSS': (1, 2, 3, None)}
    __TimeSpecs = {'COLON': '(\d+)' + __TimeSep[0] + r'(\d|[0-5]\d)' + __TimeSep[1]
                            + r'(\d|[0-5]\d)([\.:,]\d{2,3})?(?!\d)',
                   'SECONDS': r'(?<!\d)(15\d{8})\.\d+',
                   'HHMMSS': r'(?<!\d)([0-1]\d|2[0-3])([0-5]\d)([0-5]\d)(?!\d)'
                   }
    __TimeRegExp = re.compile('|'.join('(?P<%s>%s)' % (k_, v_) for k_, v_ in __TimeSpecs.items()), re.I)

    # 从sample file提取时间戳锚点，或者按anchor等参数设置时间戳锚点
    def __init__(self, sample_file=None, col_range=(0, None), name='COLON'):
        self.colRange = col_range  # (搜索的开始列，结束列)
        self.name = name  # 时间戳锚点正则表达式名称，及各值(年月日时分秒和上下午)在match对象中的位置
        self.regExp = re.compile('(%s)' % self.__TimeSpecs[name], re.I) if name else None

        self.setFormat(self.__getAnchor(sample_file)) if sample_file else None

    # 记录样本文件每行的第一和最后一个时间戳的位置和内容(最后一个时间戳记录距离尾部的距离）
    @classmethod
    def __getAnchor( cls, sample_file ):
        line_idx = 0
        anchors = {}  # 初始化潜在锚点字典
        for line_idx, line in enumerate(open(sample_file, 'r', encoding='utf-8')):
            try:
                if cls.appendAnchors(line, anchors):  # 已经找够了时间戳锚点
                    break
                if line_idx > 1000 and not anchors:  # 很多行都没找到时间戳
                    break
            except (UnicodeError, UnicodeDecodeError, UnicodeEncodeError):
                G.log.warning('Line[%d] ignored due to the following error:', line_idx)
                continue
        return Anchor.statsAnchors(anchors, line_idx)

    @classmethod
    def appendAnchors( cls, line, anchors ):
        """
        从字符串line中抽取第一和最后时间戳，追加到match_groups
        :param line: 输入字符串
        :param anchors: 时间戳信息统计到字典中：｛(列开始位置，时间格式):(递增数量，递减数量，列截至位置，最后时间戳)｝
        :return: True表示已经找够，无需继续扫描
        """
        matches_ = [v for v in cls.__TimeRegExp.finditer(line)]
        if not matches_:  # not found
            return False

        for i in [0, -1]:
            match_ = matches_[i]  # first or last match object
            regex_name, base_idx = match_.lastgroup, match_.lastindex
            start_ = match_.start() if i == 0 else match_.start() - len(line)  # start col position of first one
            stop_ = match_.end() if i == 0 else match_.end() - len(line)  # stop col position of first one
            inc_counter, dec_counter, stop_max, last_time = anchors.get((start_, regex_name), [0, 0, -30000, 0])
            this_time = 0.
            for idx, hms_offset in enumerate(cls.__PosHHMMSS[regex_name]):
                if not hms_offset or not match_.group(base_idx + hms_offset):  # None不参与计算
                    value = '0'
                elif idx == 3:  # 毫秒，去掉分隔符，取10ms精度(毫秒级三位有一定比例错序的)
                    value = match_.group(base_idx + hms_offset)[1:3]
                else:
                    value = match_.group(base_idx + hms_offset)

                this_time += float(value)
                if idx < 2:  # 时分
                    this_time *= 60
                elif idx == 2:  # 秒
                    this_time *= 100
            if this_time > last_time:
                inc_counter += 1
            elif this_time < last_time:
                dec_counter += 1

            anchors[(start_, regex_name)] = [inc_counter, dec_counter, max(stop_, stop_max), this_time]
            if max(inc_counter, dec_counter) > 100:
                return True
        return False

    @staticmethod
    def statsAnchors( anchors, lines ):
        """
        统计得到最佳Anchor
        :param anchors: appendAnchors返回的字典对象组成的列表
        :param lines: 扫描过的文件行数
        :return:列表中最佳的字典对象
        """
        if not anchors:
            raise UserWarning('No valid anchors.\tanchors=0')

        anchors = sorted(anchors.items(), key=lambda d: d[1][0] - d[1][1], reverse=True)
        G.log.debug('Candidate anchors detected at(col:inc-freq,dec-freq) %s',
                    ', '.join(['(%s-%d:%d,%d)' % (k[1], k[0], v[0], v[1]) for k, v in anchors]))
        inc_anchors, dec_anchors = anchors[0][1][0], anchors[0][1][1]

        # 运行日志时间戳中时间应该绝大部分都是递增的，每天只有一次递减的。每天只有几条日志的暂不考虑
        ratio = dec_anchors / inc_anchors if inc_anchors else 999
        if inc_anchors < max(5, lines / 500) or ratio > 0.13:
            raise UserWarning('No valid anchors.\t inc anchors=%s, dec anchors=%d, lines=%s'
                              % (inc_anchors, dec_anchors, lines))
        return anchors[0]

    def setFormat(self, anchor):
        """
        跟据日志中日期变化规律(先日再月后年)，确定日期格式
        :param anchor: statsAnchors返回的字典对象
        """
        (start_, name_), (amount_, dec_counter, stop_, last_time) = anchor
        # 考虑时间戳前后可能有变长变量(如INFO，CRITICAL等），给起止位置留一定余量(前后5个字符）
        self.colRange = (0 if 0 <= start_ < G.same_anchor_width else start_ - G.same_anchor_width,
                         0 if -G.same_anchor_width <= stop_ < 0 else stop_ + G.same_anchor_width)
        self.name = name_  # 设置正则表达式名称
        self.regExp = re.compile('(%s)' % self.__TimeSpecs[name_], re.I)  # 设置正则表达式

    def getAnchorTimeStamp( self, line ):
        """
        返回锚点的时间戳值，如不存在，返回None
        :param line: 需要寻找时间戳的字符串
        :return: 时间戳值，如不存在，返回None
        """
        if self.regExp is None:
            raise UserWarning('Failed to getAnchorTimeStamp: Anchor is not initialized!')

        anchor_timestamp = self.regExp.search(line[self.colRange[0]:self.colRange[1]])
        return anchor_timestamp if anchor_timestamp else None


if __name__ == '__main__':
    import os


    def __process(file_fullname):
        filename = os.path.split(file_fullname)[1]
        try:
            a = Anchor(file_fullname)
            G.log.info('%s\t%s\t%s\t%d' % (filename, file_fullname, a.name, a.colRange[0]))
        except Exception as e:
            G.log.error('%s\t%s\tERROR\t%s' % (filename, file_fullname, str(e)))


    def __run():
        file_or_path = input('input a path or full file name:')
        if not file_or_path.strip():
            return False
        if not os.path.exists(file_or_path):
            return True
        if os.path.isfile(file_or_path):
            file_fullname = file_or_path
            __process(file_fullname)
            return True

        for dir_path, dir_names, file_names in os.walk(file_or_path):
            for filename in file_names:
                file_fullname = os.path.join(dir_path, filename)
                __process(file_fullname)
        return True


    while __run():
        pass
