#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : anchor.py
# @Author: Sui Huafeng
# @Date  : 2018/4/9
# @Desc  : 通常情况下，日志文件每条记录都包含时间戳，可以作为区分多少行形成一条记录的定位锚点。该时间戳在每条记录第一行，
#        列的位置相对固定。本程序扫描样本文件，确定时间戳锚点的列所在范围；同时利用按时间顺序记录的特点确定时间戳格式。
#

import re
import time

import numpy as np

from config import Workspaces as G


class Anchor(object):
    """

    """
    __MonthRegex = re.compile(r'(?<=\W)(Jan)|(Feb)|(Mar)|(Apr)|(May)|(Jun|June)|(Jul|July)|(Aug)|(Sep|Sept)'
                              r'|(Oct)|(Nov)|(Dec)(?=\W)', re.I)
    __time_regex = r'(\d|[01]\d|2[0-3]):(\d|[0-5]\d):(\d|[0-5]\d)(am|pm)?(?![\d:])'
    __date_regex = r'(\d{1,4})(\\|-|/|年|月|日)(\d{1,2})(\\|-|/|月|日)(\d{1,4})(年|日)?(am|pm|上午|下午)?'
    __DTSpecs = {'DATE_TIME': {'date': [1, 3, 5, 7, 11], 'time': [8, 9, 10],
                               'regex': r'(?<!\d)?' + __date_regex + r'\D?' + __time_regex},
                 'TIME_DATE': {'date': [5, 7, 9, 4, 11], 'time': [1, 2, 3],
                               'regex': r'(?<!\d)?' + __time_regex + r'\D?' + __date_regex},
                 'SHORT_DT': {'date': [1, 2, 3, 4, 9], 'time': [5, 6, 7],  # 0121/181842
                              'regex': r'(?<!\d)?(201\d|1\d)?(0[1-9]|1[0-2])([0-2]\d|3[0-1])(am|pm|上午|下午)?[/-\\]([0-1]\d|2[0-3])([0-5]\d)([0-5]\d)(am|pm|上午|下午)?(?![\d])'},
                 'TIME_STAMP': {'regex': r'15\d{8}\.\d+'},
                 'TIME_SPAN': {'time': [1, 2, 3], 'regex': r'(\d+):(\d|[0-5]\d):(\d|[0-5]\d)(?![\d:])'}}
    __DateTimeRegExp = re.compile('|'.join('(?P<%s>%s)' % (k_, v_['regex']) for k_, v_ in __DTSpecs.items()), re.I)
    __default_formatter = {'name': None, 'p': None, 'y': None, 'm': None, 'd': None, 'H': None, 'M': None, 'S': None}

    # 从sample file提取时间戳锚点，或者按anchor、format、day等参数设置时间戳锚点
    def __init__( self, sample_file=None, col_range=(0, None), formatter=__default_formatter ):
        self.colRange = col_range  # (搜索的开始列，结束列)
        self.dtFormatter = formatter  # 时间戳锚点正则表达式名称，及各值(年月日时分秒和上下午)在match对象中的位置
        self.dtRegExp = re.compile('(%s)' % self.__DTSpecs[formatter['name']]['regex'], re.I) if formatter[
            'name'] else None
        self.setFormat(self.__getAnchor(sample_file), sample_file) if sample_file else None

    # 记录样本文件每行的第一和最后一个时间戳的位置和内容(最后一个时间戳记录距离尾部的距离）
    @classmethod
    def __getAnchor( cls, sample_file ):
        line_idx = 0
        anchors = {}  # 初始化潜在锚点字典
        for line_idx, line in enumerate(open(sample_file, 'r', encoding='utf-8')):
            try:
                cls.appendAnchors(line, anchors)
            except (UnicodeError, UnicodeDecodeError, UnicodeEncodeError):
                G.log.warning('Line[%d] ignored due to the following error:', line_idx)
                continue
        return Anchor.statsAnchors(anchors, line_idx)

    # 从字符串line中抽取第一和最后时间戳，追加到match_groups
    @classmethod
    def appendAnchors( cls, line, anchors ):
        """

        :param line:
        :param anchors:
        :return:
        """
        line_ = cls.__MonthRegex.sub(lambda match_obj: str(match_obj.lastindex), line)  # 短语月份转为数字, 如Jan-〉1
        first_, last_ = None, None
        matches_ = [v for v in cls.__DateTimeRegExp.finditer(line_)]
        if not matches_:  # not found
            return

        for i in [0, -1]:
            match_ = matches_[i]  # first or last match object
            kind_, base_idx = match_.lastgroup, match_.lastindex
            start_ = match_.start() if i == 0 else match_.start() - len(line_)  # start col position of first one
            stop_ = match_.end() if i == 0 else match_.end() - len(line_)  # stop col position of first one
            counter_, kept_, stop_max = anchors.get((start_, kind_), [0, set(), -10000])
            if cls.__DTSpecs[kind_].get('date', None):  # 记录日期样本
                #            if kind_ in ['DATE_TIME', 'TIME_DATE']:  # 记录日期样本
                keep_ = [match_.group(base_idx + offset) for offset in cls.__DTSpecs[kind_]['date']]
                kept_.add('-'.join([(lambda x: '0' if x is None else x)(s) for s in keep_]))
            anchors[(start_, kind_)] = [counter_ + 1, kept_, max(stop_, stop_max)]

    # 统计得到最佳Anchor
    @staticmethod
    def statsAnchors( anchors, lines ):
        """

        :param anchors:
        :param lines:
        :return:
        """
        anchors = sorted(anchors.items(), key=lambda d: d[1][0], reverse=True)
        G.log.debug('Candidate anchors detected at(col:freq) %s',
                    ', '.join(['%s(%d:%d)' % (k[1], k[0], v[0]) for k, v in anchors]))
        if anchors == [] or anchors[0][1][0] < lines / 100.:
            raise UserWarning('No valid anchors')
        return anchors[0]

    # 跟据日志中日期变化规律(先日再月后年)，确定日期格式
    def setFormat( self, anchor, filename='' ):
        """

        :param anchor:
        :param filename:
        :return:
        """
        (start_, name_), (amount_, date_set, stop_) = anchor

        # 考虑时间戳前后可能有变长变量(如INFO，CRITICAL等），给起止位置留一定余量(前后5个字符）
        self.colRange = (0 if 0 <= start_ < 5 else start_ - 5, 0 if -5 <= stop_ < 0 else stop_ + 5)

        self.dtFormatter['name'] = name_  # 设置正则表达式名称
        regex_ = self.__DTSpecs[name_]['regex']
        self.dtRegExp = re.compile('(%s)' % regex_, re.I)  # 设置正则表达式

        if name_ == 'TIME_STAMP':  # 无需设置时间和日期
            return
        time_spec = np.array(self.__DTSpecs[name_]['time'])
        self.dtFormatter['H'], self.dtFormatter['M'], self.dtFormatter['S'] = time_spec  # 设置时间格式

        if date_set != set():  # 非空，需要设置日期的
            dates_ = np.array([[(lambda x: int(x) if x.isdigit() else -1)(v) for v in s.split('-')] for s in date_set])
            date_spec = np.array(self.__DTSpecs[name_]['date'])
            offset = date_spec[dates_[0] == -1]
            self.dtFormatter['p'] = offset[0] if offset else None  # 设置上下午格式
            unordered_date = dates_[:, :3]  # 设置日期格式
            date_mask = self.__probeDateOrder(unordered_date, filename)
            for i in range(3):
                self.dtFormatter[date_mask[i]] = date_spec[i]

    # 确定年月日的顺序, 依据包括：日期变化规律、4位是年、大于2位年的是日、唯一大于12的是年，唯一小于12的是月
    @staticmethod
    def __probeDateOrder( unordered_dates, filename='' ):

        date_value = np.max(unordered_dates, axis=0)
        date_mask = np.array(['', '', ''])

        # 先按照日期的前后变化找规律(如先后两行出现18-2-3、18-2-4)
        diff_ = (date_value - np.min(unordered_dates, axis=0))
        if np.max(diff_) > 0:  # 时间戳起码是跨天的，变化最大的一定是日
            date_mask[diff_ == np.max(diff_)] = 'd'
        else:  # 所有时间戳都是同一天的, 拿日期数字的值碰碰运气
            this_year = int(time.strftime('%y'))
            date_mask[date_value > 2000] = 'y'  # 4位肯定是年份
            date_value[date_value > 2000] = date_value[date_value > 2000] - 2000
            date_mask[date_value == 0] = 'y'  # 0是None转过来的，肯定是年
            date_value[date_value == 0] = this_year
            date_mask[date_value > this_year] = 'd'  # 大于2位年份、小于4位年份，肯定是日

        mask_r = {v: idx for idx, v in enumerate(list(date_mask))}
        if mask_r.get('d') is not None:  # 只要日找到，年月肯定可区分(年份数字一定大于月份数字,2013年以后）
            date_mask[date_mask != 'd'] = ''  # 先把可能未知的年、月清零
            date_value[date_mask == 'd'] = 0  # 比较大小前屏蔽掉已知的日
            date_mask[date_value == np.max(date_value)] = 'y'  # 大的是年
            date_mask[date_mask == ''] = 'm'  # 剩下的是月
        else:  # 未找到日期，所有时间戳都是同一天的, 接着检查日期数字的值
            less_than12_times = date_value[date_value <= 12].shape[0]  # 小于12的个数
            if less_than12_times == 1:
                date_mask[date_value <= 12] = 'm'  # 唯一不大于12的是月份
            elif less_than12_times == 2:
                date_mask[date_value > 12] = 'y'  # 唯一大于12的是年份
            elif date_value[date_value == np.max(date_value)].shape[0] == 1:  # 最大值当作年
                date_mask[date_value == np.max(date_value)] = 'y'

            # 无法完全确定的，按可能性大小设置
            mask_r = {v: idx for idx, v in enumerate(list(date_mask))}
            if mask_r.get('m') == 0 or mask_r.get('y') == 2:
                date_mask = np.array(['m', 'd', 'y'])
            elif mask_r.get('m') == 2:
                date_mask = np.array(['y', 'd', 'm'])
            elif mask_r.get('y') == 1:
                date_mask = np.array(['d', 'y', 'm'])
            else:
                date_mask = np.array(['y', 'm', 'd'])
            G.log.warning('Failed to probe date format(only one date:%d-%d-%d), using %s instead. %s'
                          % (
                              date_value[0], date_value[1], date_value[2], '-'.join([x + x for x in date_mask]),
                              filename))

        return date_mask

    # 返回锚点的时间戳值，如不存在，返回None
    def getAnchorTimeStamp( self, line ):
        """

        :param line:
        :return:
        """
        if self.dtRegExp is None:
            raise UserWarning('Failed to getAnchorTimeStamp: Anchor is not initialized!')

        line = self.__MonthRegex.sub(lambda match_obj: str(match_obj.lastindex), line)  # 英文短语月份转为数字, 如Jan-〉1
        anchor = self.dtRegExp.search(line[self.colRange[0]:self.colRange[1]])
        if not anchor:  # 没有搜到有效的时间戳
            return None
        if self.dtFormatter['name'] == 'TIME_STAMP':
            return float(anchor.group())
        groups = anchor.groups()
        seconds = int(groups[self.dtFormatter['H']]) * 3600 + int(groups[self.dtFormatter['M']]) * 60 + int(
            groups[self.dtFormatter['S']])
        if self.dtFormatter['name'] == 'TIME_SPAN':
            return float(1514736000.0 + seconds)  # 1514736000.0是2018-1-1 00:00:00
        if self.dtFormatter['p'] is not None and groups[self.dtFormatter['p']].upper() in ['PM', '下午']:
            seconds += 12 * 3600
        year_ = int(groups[self.dtFormatter['y']])
        year_ -= 2000 if year_ > 2000 else 0
        yymmdd = '%02d-%02d-%02d' % (year_, int(groups[self.dtFormatter['m']]), int(groups[self.dtFormatter['d']]))
        seconds += time.mktime(time.strptime(yymmdd, '%y-%m-%d'))
        return seconds


if __name__ == '__main__':
    a = Anchor('E:\\data\\outputs\\change.log')
    b = a.dtFormatter
    b = ((0, 'DATE_TIME'), [1001, {'17-12-28-0-0'}, 19])
    a.setFormat(b)
