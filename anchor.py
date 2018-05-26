#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : anchor.py
# @Author: Sui Huafeng
# @Date  : 2018/4/9
# @Desc  : 扫描样本文件，确定时间戳锚点的列所在范围.并可计算锚点时间值
#

import datetime
import re
import time

import numpy as np

from config import Workspaces as G


class Anchor(object):
    """
    通常情况下，日志文件每条记录都包含时间戳，可以作为区分多少行形成一条记录的定位锚点。该时间戳在每条记录第一行，
    列的位置相对固定。
    """
    __TimeSep = [r'(:|Ê±|时|时-)', r'(:|·Ö|分|分-)']
    __TimeIdx = {'COLON': (1, 3, 5, 6), 'SECONDS': (None, None, 1, None), 'HHMMSS': (1, 2, 3, None)}
    __TimeSpecs = {'COLON': '(\d+)' + __TimeSep[0] + r'(\d|[0-5]\d)' + __TimeSep[1]
                            + r'(\d|[0-5]\d)([\.:,]\d{2,3})?(?!\d)',
                   'SECONDS': r'(?<!\d)(15\d{8})\.\d+',
                   'HHMMSS': r'(?<!\d)([0-1]\d|2[0-3])([0-5]\d)([0-5]\d)(?!\d)'
                   }
    __TimeRegExp = re.compile('|'.join('(?P<%s>%s)' % (k_, v_) for k_, v_ in __TimeSpecs.items()), re.I)
    __MonthRegex = re.compile(r'(?<=\W)(Jan)|(Feb)|(Mar)|(Apr)|(May)|(Jun|June)|(Jul|July)|(Aug)|(Sep|Sept)'
                              r'|(Oct)|(Nov)|(Dec)(?=\W)', re.I)
    __DateSep = r'[\\/-]'
    __HyphenDateIdx = {'YYMMDD': (0, 2, 3), 'MMDD': (0, 1, None)}
    __HyphenDateSpecs = {'YYMMDD': r'(?<!\d)(\d{1,4})(?P<sep>' + __DateSep + ')(\d{1,4})(?P=sep)(\d{1,4})(?!\d)',
                         'MMDD': r'(?<!\d)([1-9]|[0-3]\d)' + __DateSep + '([1-9]|[0-3]\d)(?!\d)'
                         }
    __HyphenDateRegex = re.compile('|'.join('(?P<%s>%s)' % (k_, v_) for k_, v_ in __HyphenDateSpecs.items()), re.I)

    __DigitalDateRegex = re.compile(r'(?<!\d)((20)?([12]\d))?(0\d|1[012])([012]\d|3[01])(?!\d)')
    __PmRegex = re.compile(r'(?<=[\d\s])(pm\s|下午)(?=[\d\s])', re.I)

    # 从sample file提取时间戳锚点，或者按anchor等参数设置时间戳锚点
    def __init__(self, sample_file=None, col_span=(0, None), name='COLON', probe_date=False):
        self.colSpan = col_span  # (搜索的开始列，结束列)
        self.name = name  # 时间戳锚点正则表达式名称，及各值(年月日时分秒和上下午)在match对象中的位置
        self.timeRegExp = re.compile('(%s)' % self.__TimeSpecs[name], re.I) if name else None
        self.datePattern = {}  # {'left':True, 'name':'YYHHDD', 'y':0, 'm':2, 'd':4
        self.dateRegExp = None

        self.setTimeFormat(self.__getTimeData(sample_file)) if sample_file else None
        self.probeDatePattern(sample_file) if probe_date and self.name == 'COLON' else None


    # 记录样本文件每行的第一和最后一个时间戳的位置和内容(最后一个时间戳记录距离尾部的距离）
    @classmethod
    def __getTimeData(cls, sample_file):
        line_idx = 0
        time_info = {}  # 初始化潜在锚点字典
        try:
            for line_idx, line in enumerate(open(sample_file, 'r', encoding='utf-8')):
                if cls.stackTimeData(line, time_info):  # 已经找够了时间戳锚点
                    break
                if line_idx > 1000 and not time_info:  # 很多行都没找到时间戳
                    break
        except Exception as err:
            G.log.warning('Line[%d] in %s ignored due to the following error:%s', line_idx, sample_file, str(err))

        return Anchor.statsTimeData(time_info, line_idx)

    #  从字符串line中抽取第一和最后时间戳，追加到match_groups
    @classmethod
    def stackTimeData(cls, line, anchors):
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
            for idx, hms_offset in enumerate(cls.__TimeIdx[regex_name]):
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

    #  统计得到最佳Anchor
    @staticmethod
    def statsTimeData(anchors, lines):
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

    #  设置Anchor的属性
    def setTimeFormat(self, anchor):
        """
        :param anchor: statsAnchors返回的字典对象
        """
        (start_, name_), (amount_, dec_counter, stop_, last_time) = anchor
        # 考虑时间戳前后可能有变长变量(如INFO，CRITICAL等），给起止位置留一定余量(前后5个字符）
        self.colSpan = (0 if 0 <= start_ < G.timeMargin else start_ - G.timeMargin,
                        0 if -G.timeMargin <= stop_ < 0 else stop_ + G.timeMargin)
        self.name = name_  # 设置正则表达式名称
        self.timeRegExp = re.compile('(%s)' % self.__TimeSpecs[name_], re.I)  # 设置正则表达式

    # 跟据日志中日期变化规律(先日再月后年)，找出并确定日期格式
    def probeDatePattern(self, sample_file):
        counter, results = 0, set()
        left = {True: 0, False: 0}
        try:
            for line in open(sample_file, 'r', encoding='utf-8'):
                time_ = self.timeRegExp.search(line[self.colSpan[0]:self.colSpan[1]])
                if not time_:  # 没有搜到有效的时间戳
                    continue
                counter += 1
                time_span = time_.span()
                self.__probeHyphenDate(time_span, line, results, left)
                if len(results) >= 3 or (counter > 5 and not results):  # 找够2个日期, 5个时间仍没有日期就退出
                    break
            self.__setDatePattern(results, left)
        except Exception as err:
            G.log.warning('%s ignored due to the following error:%s', sample_file, str(err))

    # 找带分隔符的日期
    def __probeHyphenDate(self, time_span, line, results, left):
        base_col = self.colSpan[0] if self.colSpan[0] >= 0 else self.colSpan[0] + len(line)
        line = self.__MonthRegex.sub(lambda match_obj: str(match_obj.lastindex), line)
        line = re.sub(r'[年月日]', '-', line)

        start_col = base_col + time_span[0] - G.dateMargin
        start_col = 0 if start_col < 0 else start_col

        match_ = self.__HyphenDateRegex.search(line[start_col:start_col + G.dateMargin])

        if match_:
            left[True] += 1
        else:
            start_col = base_col + time_span[1]
            match_ = self.__HyphenDateRegex.search(line[start_col:start_col + G.dateMargin])
            if match_:
                left[False] += 1

        if match_:
            disordered_date = match_.group()
            disordered_date = re.sub(self.__DateSep, '/', disordered_date)
            results.add(disordered_date)

    def __setDatePattern(self, results, left):
        if not results:  # 没有搜到日期数据
            self.datePattern['name'] = None
            return

        unordered_dates = np.array([[int(cell) for cell in row.split('/')] for row in list(results)])
        ymd_mask = self.__probeDateOrder(unordered_dates)

        if len(ymd_mask) == 2:  # MMDD
            ymd_mask[ymd_mask != 'd'] = 'm'  # 唯一不大于12的是月份
            date_name = 'MMDD'
        else:
            date_name = 'YYMMDD'
        date_pattern = {v: Anchor.__HyphenDateIdx[date_name][idx] for idx, v in enumerate(ymd_mask)}
        date_pattern['name'] = date_name

        self.datePattern.update(date_pattern)
        self.dateRegExp = re.compile(self.__HyphenDateSpecs[self.datePattern['name']])
        self.datePattern['left'] = True if left[True] > left[False] else False

    # 确定年月日的顺序, 依据包括：日期变化规律、4位是年、大于2位年的是日、唯一大于12的是年，唯一小于12的是月
    @staticmethod
    def __probeDateOrder(unordered_dates, filename=''):
        date_value = np.max(unordered_dates, axis=0)
        date_len = unordered_dates.shape[1]
        ymd_mask = np.array(['', '', ''])[:date_len]

        # 先拿最大的年月日数字的值碰碰运气
        this_year = int(time.strftime('%y'))
        ymd_mask[date_value > 2000] = 'y'  # 4位肯定是年份
        ymd_mask[date_value == 0] = 'y'  # 0是None转过来的，肯定是年
        date_value[date_value == 0] = this_year
        date_value[date_value > 2000] = date_value[date_value > 2000] - 2000  # 统一成两位年份
        ymd_mask[date_value > this_year] = 'd'  # 大于2位年份、小于4位年份，肯定是日
        ymd_order = {v: idx for idx, v in enumerate(list(ymd_mask))}

        # 未找到日，按照日期的前后变化找规律(如先后两行出现18-2-3、18-2-4)
        if ymd_order.get('d') is None:
            diff_ = (date_value - np.min(unordered_dates, axis=0))
            if np.max(diff_) > 0:  # 时间戳起码是跨天的，变化最大的一定是日
                ymd_mask[diff_ == np.max(diff_)] = 'd'

        # 只要日找到，年月肯定可区分(年份数字一定大于月份数字,2013年以后）
        if ymd_order.get('d') is not None:
            ymd_mask[ymd_mask != 'd'] = ''  # 先把可能未知的年、月清零
            date_value[ymd_mask == 'd'] = 0  # 比较大小前屏蔽掉已知的日
            ymd_mask[date_value == np.max(date_value)] = 'y'  # 大的是年
            ymd_mask[ymd_mask == ''] = 'm'  # 剩下的是月
            return ymd_mask

        less_than12_times = date_value[date_value <= 12].shape[0]  # 小于12的个数
        if less_than12_times == 1:
            ymd_mask[date_value <= 12] = 'm'  # 唯一不大于12的是月份
        elif less_than12_times == 2:
            ymd_mask[date_value > 12] = 'y'  # 唯一大于12的是年份
        elif date_value[date_value == np.max(date_value)].shape[0] == 1:  # 最大值当作年
            ymd_mask[date_value == np.max(date_value)] = 'y'

        # MMDD无法完全确定的，按可能性大小设置
        ymd_order = {v: idx for idx, v in enumerate(list(ymd_mask))}
        if date_len == 2:  # MMDD
            if less_than12_times == 1:
                ymd_mask[date_value > 12] = 'd'  # 唯一不大于12的是月份
            else:
                ymd_mask = np.array(['m', 'd'])
                G.log.warning('Failed to probe date format for the single date:%d-%d), using %s instead. %s'
                              % (date_value[0], date_value[1], '-'.join([x + x for x in ymd_mask]), filename))
            return ymd_mask

        # YYMMDD无法完全确定的，按可能性大小设置
        if ymd_order.get('m') == 0 or ymd_order.get('y') == 2:
            ymd_mask = np.array(['m', 'd', 'y'])
        elif ymd_order.get('m') == 2:
            ymd_mask = np.array(['y', 'd', 'm'])
        elif ymd_order.get('y') == 1:
            ymd_mask = np.array(['d', 'y', 'm'])
        else:
            ymd_mask = np.array(['y', 'm', 'd'])
        G.log.warning('Failed to probe date format for the single:%d-%d-%d), using %s instead. %s'
                      % (date_value[0], date_value[1], date_value[2], '-'.join([x + x for x in ymd_mask]), filename))
        return ymd_mask

    # 返回锚点的日期时间值
    def getTimeStamp(self, line):
        """
        返回锚点的时间戳值，如不存在，返回None
        :param line: 需要寻找时间戳的字符串
        :return: 时间戳值，如不存在，返回None
        """
        if not self.timeRegExp:
            raise UserWarning('Failed to getAnchorTimeStamp: Anchor is not initialized!')
        time_ = self.timeRegExp.search(line[self.colSpan[0]:self.colSpan[1]])
        if not time_:  # 没有搜到有效的时间戳
            return None

        try:
            if self.name == 'SECONDS':
                anchor_timestamp = float(time_.group())
                return anchor_timestamp

            matches = time_.groups()
            time_idx = self.__TimeIdx[self.name]
            anchor_time = datetime.time(int(matches[time_idx[0]]), int(matches[time_idx[1]]), int(matches[time_idx[2]]))

            time_span = time_.span()
            if self.name == 'COLON':
                anchor_date = self.__getHyphenDate(time_span, line) if self.datePattern else self.__recent_day(
                    anchor_time)
                if not anchor_date:
                    return None
                if self.__isAfternoon(time_span, line):
                    anchor_date += datetime.timedelta(hours=12)
            elif self.name == 'HHMMSS':
                anchor_date = self.__getDigitDate(time_span, line)
                if not anchor_date:
                    return None
            else:
                anchor_date = self.__recent_day(anchor_time)
        except ValueError as err:
            G.log.warning('No valid Datetime detected in %s. %s', line, str(err))
            return None

        anchor_timestamp = time.mktime(datetime.datetime.combine(anchor_date, anchor_time).timetuple())
        return anchor_timestamp

    # 在时间左侧或者右侧搜索并提取以/\-等分隔的日期
    def __getHyphenDate(self, time_span, line):
        base_col = self.colSpan[0] if self.colSpan[0] >= 0 else self.colSpan[0] + len(line)
        line = self.__MonthRegex.sub(lambda match_obj: str(match_obj.lastindex), line)
        line = re.sub(r'[年月日]', '-', line)

        start_col = base_col + time_span[0] - G.dateMargin if self.datePattern['left'] else base_col + time_span[1]
        start_col = 0 if start_col < 0 else start_col
        match_ = self.dateRegExp.search(line[start_col:start_col + G.dateMargin])
        if not match_:
            return None
        matches = match_.groups()
        mm, dd = int(matches[self.datePattern['m']]), int(matches[self.datePattern['d']])
        y = self.datePattern.get('y', None)
        if y is None:
            today = datetime.date.today()
            yy = today.year
            if today.month * 100 + today.day < mm * 100 + dd:
                yy -= 1
        else:
            yy = int(matches[y])
            yy += 2000 if yy < 100 else 0

        anchor_date = datetime.date(yy, mm, dd)
        return anchor_date

    # 取距离time_最近今天或昨天
    @staticmethod
    def __recent_day(time_):
        date_ = datetime.date.today()
        if datetime.datetime.today().time() < time_:
            date_ -= datetime.timedelta(days=1)
        return date_

    # 在时间左侧及右侧搜索是否存在'pm/下午'等字符
    def __isAfternoon(self, time_span, line):
        base_col = self.colSpan[0] if self.colSpan[0] >= 0 else self.colSpan[0] + len(line)
        start_col = base_col + time_span[0] - 5
        start_col = 0 if start_col < 0 else start_col
        match_ = self.__PmRegex.search(line[start_col:start_col + 5])
        if not match_:
            start_col = base_col + time_span[1]
            match_ = self.__PmRegex.search(line[start_col:start_col + 5])
        if match_:
            return True
        return False

    # 在时间左侧搜索并提取YYYYMMDD/YYMMDD/MMDD格式日期
    def __getDigitDate(self, time_span, line):
        base_col = self.colSpan[0] if self.colSpan[0] >= 0 else self.colSpan[0] + len(line)
        start_col = base_col + time_span[0] - 10
        start_col = 0 if start_col < 0 else start_col
        match_ = self.__DigitalDateRegex.search(line[start_col:start_col + 10])
        if not match_:
            return None

        matches = match_.groups()
        return self.__getDateFrom(matches[1], matches[2], matches[3], matches[4])

    # 兼容无年份\两位年份\4位年份等情况,形成完整日期
    @staticmethod
    def __getDateFrom(century, yy, mm, dd):
        mm, dd = int(mm), int(dd)
        if not yy:  # MMDD格式
            today = datetime.date.today()
            yy = today.year
            if today.month * 100 + today.day < mm * 100 + dd:
                yy -= 1
        elif not century:  # YYMMDD格式
            yy = 2000 + int(yy)
        else:  # YYYYMMDD格式
            yy = int(century) * 100 + yy
        anchor_date = datetime.date(yy, mm, dd)
        return anchor_date


if __name__ == '__main__':
    import os


    def run():
        file_or_path = 'D:\\home\\t.txt'
        #            input('input a path or full file name:')
        if not file_or_path.strip():
            return False
        if not os.path.exists(file_or_path):
            return True
        if os.path.isfile(file_or_path):
            file_fullname = file_or_path
            process(file_fullname)
            return True

        for dir_path, dir_names, file_names in os.walk(file_or_path):
            for filename in file_names:
                file_fullname = os.path.join(dir_path, filename)
                process(file_fullname)
        return True


    def process(file_fullname):
        filename = os.path.split(file_fullname)[1]
        try:
            a = Anchor(file_fullname, probe_date=True)
            for line in open(file_fullname, 'r', encoding='utf-8'):
                a.getTimeStamp(line)
            G.log.info('%s\t%s\t%s\t%d' % (filename, file_fullname, a.name, a.colSpan[0]))
        except Exception as e:
            G.log.error('%s\t%s\tERROR\t%s' % (filename, file_fullname, str(e)))


    while run():
        pass
