/*
SQLyog Ultimate v11.01 (64 bit)
MySQL - 5.5.28 : Database - nornenjs
*********************************************************************
*/

/*!40101 SET NAMES utf8 */;

/*!40101 SET SQL_MODE=''*/;

/*!40014 SET @OLD_UNIQUE_CHECKS=@@UNIQUE_CHECKS, UNIQUE_CHECKS=0 */;
/*!40014 SET @OLD_FOREIGN_KEY_CHECKS=@@FOREIGN_KEY_CHECKS, FOREIGN_KEY_CHECKS=0 */;
/*!40101 SET @OLD_SQL_MODE=@@SQL_MODE, SQL_MODE='NO_AUTO_VALUE_ON_ZERO' */;
/*!40111 SET @OLD_SQL_NOTES=@@SQL_NOTES, SQL_NOTES=0 */;
CREATE DATABASE /*!32312 IF NOT EXISTS*/`nornenjs` /*!40100 DEFAULT CHARACTER SET utf8 */;

USE `nornenjs`;

/*Table structure for table `actor` */

DROP TABLE IF EXISTS `actor`;

CREATE TABLE `actor` (
  `pn` bigint(20) NOT NULL AUTO_INCREMENT,
  `username` varchar(50) NOT NULL,
  `password` varchar(50) NOT NULL,
  `enabled` tinyint(1) NOT NULL,
  PRIMARY KEY (`pn`),
  UNIQUE KEY `username` (`username`)
) ENGINE=InnoDB AUTO_INCREMENT=81 DEFAULT CHARSET=utf8;

/*Table structure for table `actor_info` */

DROP TABLE IF EXISTS `actor_info`;

CREATE TABLE `actor_info` (
  `actorPn` bigint(20) NOT NULL,
  `email` varchar(100) NOT NULL,
  `firstName` varchar(20) NOT NULL,
  `lastName` varchar(20) NOT NULL,
  `inputDate` datetime NOT NULL,
  `updateDate` datetime NOT NULL,
  PRIMARY KEY (`actorPn`),
  CONSTRAINT `actor_info_ibfk_1` FOREIGN KEY (`actorPn`) REFERENCES `actor` (`pn`) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

/*Table structure for table `authorities` */

DROP TABLE IF EXISTS `authorities`;

CREATE TABLE `authorities` (
  `username` varchar(50) NOT NULL,
  `authority` varchar(50) NOT NULL,
  UNIQUE KEY `ix_auth_username` (`username`,`authority`),
  CONSTRAINT `fk_authorities_users` FOREIGN KEY (`username`) REFERENCES `actor` (`username`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

/*Table structure for table `board` */

DROP TABLE IF EXISTS `board`;

CREATE TABLE `board` (
  `pn` int(11) NOT NULL AUTO_INCREMENT COMMENT 'Board primary key',
  `title` varchar(100) NOT NULL COMMENT 'Board title',
  `content` longtext NOT NULL COMMENT 'Board content',
  `insertDate` datetime NOT NULL COMMENT 'Board insert date',
  `updateDate` datetime NOT NULL COMMENT 'Board update date',
  PRIMARY KEY (`pn`)
) ENGINE=InnoDB AUTO_INCREMENT=4 DEFAULT CHARSET=utf8;

/*Table structure for table `data` */

DROP TABLE IF EXISTS `data`;

CREATE TABLE `data` (
  `pn` bigint(20) NOT NULL AUTO_INCREMENT,
  `username` varchar(50) NOT NULL,
  `type` tinyint(4) NOT NULL,
  `name` varchar(50) NOT NULL,
  `savePath` varchar(100) NOT NULL,
  `inputDate` datetime NOT NULL,
  PRIMARY KEY (`pn`),
  KEY `username` (`username`),
  CONSTRAINT `data_ibfk_1` FOREIGN KEY (`username`) REFERENCES `actor` (`username`) ON DELETE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=230 DEFAULT CHARSET=utf8;

/*Table structure for table `group_authorities` */

DROP TABLE IF EXISTS `group_authorities`;

CREATE TABLE `group_authorities` (
  `groupPn` bigint(20) unsigned NOT NULL,
  `authority` varchar(50) NOT NULL,
  KEY `groupPn` (`groupPn`),
  CONSTRAINT `group_authorities_ibfk_1` FOREIGN KEY (`groupPn`) REFERENCES `groups` (`pn`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

/*Table structure for table `group_members` */

DROP TABLE IF EXISTS `group_members`;

CREATE TABLE `group_members` (
  `pn` bigint(20) unsigned NOT NULL AUTO_INCREMENT,
  `username` varchar(50) NOT NULL,
  `groupPn` bigint(20) unsigned NOT NULL,
  PRIMARY KEY (`pn`),
  KEY `username` (`username`),
  KEY `group_members_ibfk_1` (`groupPn`),
  CONSTRAINT `group_members_ibfk_1` FOREIGN KEY (`groupPn`) REFERENCES `groups` (`pn`) ON DELETE CASCADE ON UPDATE CASCADE,
  CONSTRAINT `group_members_ibfk_2` FOREIGN KEY (`username`) REFERENCES `actor` (`username`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=8 DEFAULT CHARSET=utf8;

/*Table structure for table `groups` */

DROP TABLE IF EXISTS `groups`;

CREATE TABLE `groups` (
  `pn` bigint(20) unsigned NOT NULL AUTO_INCREMENT,
  `groupName` varchar(50) NOT NULL,
  PRIMARY KEY (`pn`)
) ENGINE=InnoDB AUTO_INCREMENT=21 DEFAULT CHARSET=utf8;

/*Table structure for table `persistent_logins` */

DROP TABLE IF EXISTS `persistent_logins`;

CREATE TABLE `persistent_logins` (
  `username` varchar(64) NOT NULL,
  `series` varchar(64) NOT NULL,
  `token` varchar(64) NOT NULL,
  `last_used` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`series`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

/*Table structure for table `volume` */

DROP TABLE IF EXISTS `volume`;

CREATE TABLE `volume` (
  `pn` bigint(20) NOT NULL AUTO_INCREMENT,
  `username` varchar(50) NOT NULL,
  `volumeDataPn` bigint(20) NOT NULL,
  `title` varchar(100) NOT NULL,
  `width` int(5) NOT NULL,
  `height` int(5) NOT NULL,
  `depth` int(5) NOT NULL,
  `inputDate` datetime NOT NULL,
  PRIMARY KEY (`pn`),
  UNIQUE KEY `volumeDataPn` (`volumeDataPn`),
  KEY `username` (`username`),
  CONSTRAINT `volume_ibfk_1` FOREIGN KEY (`username`) REFERENCES `actor` (`username`) ON DELETE CASCADE
) ENGINE=InnoDB AUTO_INCREMENT=185 DEFAULT CHARSET=utf8;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;
