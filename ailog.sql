/*
Navicat MySQL Data Transfer

Source Server         : mysql
Source Server Version : 80011
Source Host           : 127.0.0.1:3306
Source Database       : ailog

Target Server Type    : MYSQL
Target Server Version : 80011
File Encoding         : 65001

Date: 2018-05-30 22:19:02
*/

SET FOREIGN_KEY_CHECKS = 0;

-- ----------------------------
-- Table structure for event
-- ----------------------------
DROP TABLE IF EXISTS `event`;
CREATE TABLE `event` (
  `id`          INT(11)   NOT NULL AUTO_INCREMENT,
  `source`      VARCHAR(100) CHARACTER SET utf8
  COLLATE utf8_bin        NOT NULL,
  `arise_time`  TIMESTAMP NOT NULL,
  `msg`         VARCHAR(800) CHARACTER SET utf8
  COLLATE utf8_bin        NOT NULL DEFAULT '',
  `priority`    ENUM ('信息', '轻微', '警告', '严重') CHARACTER SET utf8
  COLLATE utf8_bin        NOT NULL DEFAULT '信息',
  `status`      ENUM ('未派发', '已派发', '已确认', '已恢复', '已清除') CHARACTER SET utf8
  COLLATE utf8_bin        NOT NULL DEFAULT '未派发',
  `counts`      INT(11)   NOT NULL DEFAULT '1',
  `create_time` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `update_time` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`, `arise_time`),
  UNIQUE KEY `source` (`source`) USING BTREE,
  KEY `status` (`status`)
)
  ENGINE = InnoDB
  AUTO_INCREMENT = 57760
  DEFAULT CHARSET = utf8
  COLLATE = utf8_bin;

-- ----------------------------
-- Table structure for files_merged
-- ----------------------------
DROP TABLE IF EXISTS `files_merged`;
CREATE TABLE `files_merged` (
  `id`          INT(11)   NOT NULL AUTO_INCREMENT,
  `common_name` VARCHAR(1000) CHARACTER SET utf8
  COLLATE utf8_bin        NOT NULL,
  `model_id`    INT(11)            DEFAULT NULL,
  `category_id` INT(11)            DEFAULT NULL,
  `confidence`  FLOAT(11, 0)       DEFAULT NULL,
  `distance`    FLOAT              DEFAULT NULL,
  `create_time` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `update_time` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`, `common_name`),
  UNIQUE KEY `common_filename` (`common_name`) USING BTREE,
  KEY `model_id` (`model_id`, `category_id`),
  CONSTRAINT `files_merged_ibfk_1` FOREIGN KEY (`model_id`, `category_id`) REFERENCES `file_class` (`model_id`, `category_id`)
    ON DELETE SET NULL
    ON UPDATE CASCADE
)
  ENGINE = InnoDB
  AUTO_INCREMENT = 3456
  DEFAULT CHARSET = utf8
  COLLATE = utf8_bin;

-- ----------------------------
-- Table structure for files_sampled
-- ----------------------------
DROP TABLE IF EXISTS `files_sampled`;
CREATE TABLE `files_sampled` (
  `file_fullname`    VARCHAR(1000) CHARACTER SET utf8
  COLLATE utf8_bin             NOT NULL,
  `host`             VARCHAR(100) CHARACTER SET utf8
  COLLATE utf8_bin                             DEFAULT ''
  COMMENT 'domian name or ip address',
  `archive_path`     VARCHAR(800) CHARACTER SET utf8
  COLLATE utf8_bin                             DEFAULT ''
  COMMENT 'file path in archive file, : and \\ replaced with _ /',
  `filename`         VARCHAR(100) CHARACTER SET utf8
  COLLATE utf8_bin                             DEFAULT '',
  `remote_path`      VARCHAR(800) CHARACTER SET utf8
  COLLATE utf8_bin                             DEFAULT '',
  `last_update`      TIMESTAMP NULL            DEFAULT CURRENT_TIMESTAMP
  COMMENT 'original file last update time',
  `last_collect`     TIMESTAMP NULL            DEFAULT CURRENT_TIMESTAMP
  COMMENT 'last collect time',
  `size`             INT(11) UNSIGNED ZEROFILL DEFAULT '00000000000'
  COMMENT 'original file size is byte',
  `anchor_name`      VARCHAR(10) CHARACTER SET utf8
  COLLATE utf8_bin                             DEFAULT '',
  `anchor_start_col` INT(4)                    DEFAULT NULL,
  `anchor_end_col`   INT(4)                    DEFAULT NULL,
  `common_name`      VARCHAR(1000) CHARACTER SET utf8
  COLLATE utf8_bin                             DEFAULT NULL
  COMMENT 'filename removed .-_space and number',
  `create_time`      TIMESTAMP NOT NULL        DEFAULT CURRENT_TIMESTAMP,
  `update_time`      TIMESTAMP NOT NULL        DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  `notes`            VARCHAR(255) CHARACTER SET utf8
  COLLATE utf8_bin                             DEFAULT NULL,
  PRIMARY KEY (`file_fullname`),
  UNIQUE KEY `file_fullname` (`file_fullname`) USING BTREE,
  KEY `common_name` (`common_name`)
)
  ENGINE = InnoDB
  DEFAULT CHARSET = utf8
  COLLATE = utf8_bin;

-- ----------------------------
-- Table structure for file_class
-- ----------------------------
DROP TABLE IF EXISTS `file_class`;
CREATE TABLE `file_class` (
  `model_id`    INT(11)   NOT NULL,
  `category_id` INT(11)   NOT NULL,
  `name`        VARCHAR(50) CHARACTER SET utf8
  COLLATE utf8_bin        NOT NULL,
  `quantile`    DOUBLE    NOT NULL,
  `boundary`    DOUBLE    NOT NULL,
  `total_docs`  INT(11)   NOT NULL DEFAULT '0',
  `bad_docs`    INT(11)   NOT NULL,
  `status`      ENUM ('无模型', '无基线', '已完备', '计算中') CHARACTER SET utf8
  COLLATE utf8_bin        NOT NULL DEFAULT '无模型',
  `total_lines` INT(11)   NOT NULL DEFAULT '0',
  `create_time` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `update_time` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  UNIQUE KEY `category_idx` (`model_id`, `category_id`) USING BTREE,
  UNIQUE KEY `model_id` (`model_id`, `name`)
)
  ENGINE = InnoDB
  DEFAULT CHARSET = utf8
  COLLATE = utf8_bin;

-- ----------------------------
-- Table structure for kpi
-- ----------------------------
DROP TABLE IF EXISTS `kpi`;
CREATE TABLE `kpi` (
  `id`          INT(11)   NOT NULL AUTO_INCREMENT,
  `model_id`    INT(11)            DEFAULT NULL,
  `lc_id`       INT(11)            DEFAULT NULL,
  `rc_id`       INT(11)            DEFAULT NULL,
  `period`      INT(10) UNSIGNED   DEFAULT NULL
  COMMENT 'in Second',
  `create_time` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `last_update` TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`id`),
  UNIQUE KEY `model_id` (`model_id`, `lc_id`, `rc_id`)
)
  ENGINE = InnoDB
  DEFAULT CHARSET = utf8
  COLLATE = utf8_bin;

-- ----------------------------
-- Table structure for kpi_data
-- ----------------------------
DROP TABLE IF EXISTS `kpi_data`;
CREATE TABLE `kpi_data` (
  `time`          TIMESTAMP NOT NULL,
  `flow_id`       INT(11)                       DEFAULT NULL,
  `rc_id`         INT(11)                       DEFAULT NULL,
  `variable_name` VARCHAR(100) COLLATE utf8_bin DEFAULT '',
  `kpi_value`     VARCHAR(100) COLLATE utf8_bin DEFAULT '0',
  `create_time`   TIMESTAMP NOT NULL            DEFAULT CURRENT_TIMESTAMP,
  `update_time`   TIMESTAMP NOT NULL            DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  UNIQUE KEY `flow_id` (`flow_id`, `rc_id`, `variable_name`, `time`) USING BTREE,
  CONSTRAINT `kpi_data_ibfk_1` FOREIGN KEY (`flow_id`) REFERENCES `tbl_log_flow` (`id`)
    ON UPDATE CASCADE
)
  ENGINE = InnoDB
  DEFAULT CHARSET = utf8
  COLLATE = utf8_bin;

-- ----------------------------
-- Table structure for record_class
-- ----------------------------
DROP TABLE IF EXISTS `record_class`;
CREATE TABLE `record_class` (
  `model_id`    INT(11)   NOT NULL,
  `fc_id`       INT(11)            DEFAULT NULL,
  `rc_id`       INT(11)            DEFAULT NULL,
  `confidence`  FLOAT(255, 0)      DEFAULT NULL,
  `percent`     FLOAT              DEFAULT NULL,
  `period`      INT(11)            DEFAULT NULL,
  `crate_time`  TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `update_time` TIMESTAMP NULL     DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`model_id`),
  UNIQUE KEY `model_id` (`model_id`, `fc_id`, `rc_id`)
)
  ENGINE = InnoDB
  DEFAULT CHARSET = utf8
  COLLATE = utf8_bin;

-- ----------------------------
-- Table structure for tbl_log_flow
-- ----------------------------
DROP TABLE IF EXISTS `tbl_log_flow`;
CREATE TABLE `tbl_log_flow` (
  `id`             INT(11)    NOT NULL AUTO_INCREMENT,
  `host`           VARCHAR(100) CHARACTER SET utf8
  COLLATE utf8_bin                     DEFAULT NULL,
  `path`           VARCHAR(800) CHARACTER SET utf8
  COLLATE utf8_bin                     DEFAULT NULL,
  `wildcard_name`  VARCHAR(100) CHARACTER SET utf8
  COLLATE utf8_bin                     DEFAULT NULL,
  `model_id`       INT(50)             DEFAULT NULL,
  `category_id`    INT(11)             DEFAULT NULL,
  `anchor`         VARCHAR(30) CHARACTER SET utf8
  COLLATE utf8_bin                     DEFAULT NULL,
  `service_id`     INT(11)             DEFAULT NULL
  COMMENT '标识和区分本端不同的OnlineService',
  `source_id`      VARCHAR(1000) CHARACTER SET utf8
  COLLATE utf8_bin                     DEFAULT NULL
  COMMENT '对方系统给出的区分不同流的ID',
  `status`         ENUM ('未分配', '未激活', '未锚定', '未分类', '活动中', '已中断', '无锚点') CHARACTER SET utf8
  COLLATE utf8_bin            NOT NULL DEFAULT '未分配'
  COMMENT '状态转换图: 未分配 - 未激活 - 活动中 - 已中断 - 已废弃; 未锚定 - 未分类 - 待聚类 - 活动中',
  `received_bytes` BIGINT(20) NOT NULL DEFAULT '0',
  `received_lines` BIGINT(20) NOT NULL DEFAULT '0',
  `create_time`    TIMESTAMP  NOT NULL DEFAULT CURRENT_TIMESTAMP,
  `update_time`    TIMESTAMP  NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  UNIQUE KEY `id` (`id`) USING BTREE,
  UNIQUE KEY `host` (`host`, `path`, `wildcard_name`) USING BTREE,
  UNIQUE KEY `service_id` (`service_id`, `source_id`) USING BTREE,
  KEY `wildcard_logfile_ibfk_1` (`model_id`, `category_id`) USING BTREE,
  KEY `status` (`status`) USING BTREE,
  CONSTRAINT `tbl_log_flow_ibfk_1` FOREIGN KEY (`model_id`, `category_id`) REFERENCES `file_class` (`model_id`, `category_id`)
    ON DELETE SET NULL
    ON UPDATE CASCADE
)
  ENGINE = InnoDB
  AUTO_INCREMENT = 6614
  DEFAULT CHARSET = utf8
  COLLATE = utf8_bin;

-- ----------------------------
-- View structure for files_classified
-- ----------------------------
DROP VIEW IF EXISTS `files_classified`;
CREATE ALGORITHM = UNDEFINED
  DEFINER =`root`@`%`
  SQL SECURITY DEFINER VIEW `files_classified` AS
  SELECT
    `files_sampled`.`common_name`      AS `common_name`,
    `files_sampled`.`file_fullname`    AS `file_fullname`,
    `files_sampled`.`host`             AS `host`,
    `files_sampled`.`remote_path`      AS `remote_path`,
    `files_sampled`.`filename`         AS `filename`,
    `files_merged`.`model_id`          AS `model_id`,
    `files_merged`.`category_id`       AS `category_id`,
    `files_merged`.`confidence`        AS `confidence`,
    `file_class`.`name`                AS `category_name`,
    `files_sampled`.`last_update`      AS `last_update`,
    `files_sampled`.`last_collect`     AS `last_collect`,
    `files_sampled`.`anchor_name`      AS `anchor_name`,
    `files_sampled`.`anchor_start_col` AS `anchor_start_col`,
    `files_sampled`.`anchor_end_col`   AS `anchor_end_col`
  FROM (`files_sampled`
    JOIN (`files_merged`
      JOIN `file_class` ON (((`files_merged`.`model_id` = `file_class`.`model_id`) AND
                             (`files_merged`.`category_id` = `file_class`.`category_id`)))))
  WHERE (`files_merged`.`common_name` = `files_sampled`.`common_name`);
