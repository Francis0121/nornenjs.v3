/*
SQLyog Ultimate v11.01 (64 bit)
MySQL - 5.5.41-0ubuntu0.12.04.1 : Database - nornenjs
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
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

/*Data for the table `actor` */

LOCK TABLES `actor` WRITE;

UNLOCK TABLES;

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

/*Data for the table `actor_info` */

LOCK TABLES `actor_info` WRITE;

UNLOCK TABLES;

/*Table structure for table `authorities` */

DROP TABLE IF EXISTS `authorities`;

CREATE TABLE `authorities` (
  `username` varchar(50) NOT NULL,
  `authority` varchar(50) NOT NULL,
  UNIQUE KEY `ix_auth_username` (`username`,`authority`),
  CONSTRAINT `fk_authorities_users` FOREIGN KEY (`username`) REFERENCES `actor` (`username`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

/*Data for the table `authorities` */

LOCK TABLES `authorities` WRITE;

UNLOCK TABLES;

/*Table structure for table `board` */

DROP TABLE IF EXISTS `board`;

CREATE TABLE `board` (
  `pn` int(11) NOT NULL AUTO_INCREMENT COMMENT 'Board primary key',
  `title` varchar(100) NOT NULL COMMENT 'Board title',
  `content` longtext NOT NULL COMMENT 'Board content',
  `insertDate` datetime NOT NULL COMMENT 'Board insert date',
  `updateDate` datetime NOT NULL COMMENT 'Board update date',
  PRIMARY KEY (`pn`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

/*Data for the table `board` */

LOCK TABLES `board` WRITE;

UNLOCK TABLES;

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
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

/*Data for the table `data` */

LOCK TABLES `data` WRITE;

UNLOCK TABLES;

/*Table structure for table `group_authorities` */

DROP TABLE IF EXISTS `group_authorities`;

CREATE TABLE `group_authorities` (
  `groupPn` bigint(20) unsigned NOT NULL,
  `authority` varchar(50) NOT NULL,
  KEY `groupPn` (`groupPn`),
  CONSTRAINT `group_authorities_ibfk_1` FOREIGN KEY (`groupPn`) REFERENCES `groups` (`pn`) ON DELETE CASCADE ON UPDATE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

/*Data for the table `group_authorities` */

LOCK TABLES `group_authorities` WRITE;

insert  into `group_authorities`(`groupPn`,`authority`) values (1,'ROLE_ADMIN'),(1,'ROLE_DOCTOR'),(2,'ROLE_DOCTOR');

UNLOCK TABLES;

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
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

/*Data for the table `group_members` */

LOCK TABLES `group_members` WRITE;

UNLOCK TABLES;

/*Table structure for table `groups` */

DROP TABLE IF EXISTS `groups`;

CREATE TABLE `groups` (
  `pn` bigint(20) unsigned NOT NULL AUTO_INCREMENT,
  `groupName` varchar(50) NOT NULL,
  PRIMARY KEY (`pn`)
) ENGINE=InnoDB AUTO_INCREMENT=3 DEFAULT CHARSET=utf8;

/*Data for the table `groups` */

LOCK TABLES `groups` WRITE;

insert  into `groups`(`pn`,`groupName`) values (1,'Administartor'),(2,'Doctor');

UNLOCK TABLES;

/*Table structure for table `persistent_logins` */

DROP TABLE IF EXISTS `persistent_logins`;

CREATE TABLE `persistent_logins` (
  `username` varchar(64) NOT NULL,
  `series` varchar(64) NOT NULL,
  `token` varchar(64) NOT NULL,
  `last_used` timestamp NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (`series`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

/*Data for the table `persistent_logins` */

LOCK TABLES `persistent_logins` WRITE;

UNLOCK TABLES;

/*Table structure for table `thumbnail` */

DROP TABLE IF EXISTS `thumbnail`;

CREATE TABLE `thumbnail` (
  `dataPn` bigint(20) NOT NULL,
  `thumbnailPn` bigint(20) NOT NULL,
  PRIMARY KEY (`dataPn`,`thumbnailPn`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

/*Data for the table `thumbnail` */

LOCK TABLES `thumbnail` WRITE;

UNLOCK TABLES;

/*Table structure for table `thumbnail_option` */

DROP TABLE IF EXISTS `thumbnail_option`;

CREATE TABLE `thumbnail_option` (
  `pn` int(11) NOT NULL AUTO_INCREMENT,
  `width` int(11) NOT NULL,
  `height` int(11) NOT NULL,
  `type` tinyint(1) NOT NULL,
  `mprType` tinyint(1) NOT NULL COMMENT 'MPR Type',
  `quality` tinyint(1) NOT NULL COMMENT 'Quality',
  `brightness` float NOT NULL,
  `density` float NOT NULL,
  `transferOffset` float NOT NULL,
  `transferScaleX` float NOT NULL,
  `transferScaleY` float NOT NULL,
  `transferScaleZ` float NOT NULL,
  `positionZ` float NOT NULL,
  `rotationX` float NOT NULL,
  `rotationY` float NOT NULL,
  PRIMARY KEY (`pn`)
) ENGINE=InnoDB AUTO_INCREMENT=5 DEFAULT CHARSET=utf8;

/*Data for the table `thumbnail_option` */

LOCK TABLES `thumbnail_option` WRITE;

insert  into `thumbnail_option`(`pn`,`width`,`height`,`type`,`mprType`,`quality`,`brightness`,`density`,`transferOffset`,`transferScaleX`,`transferScaleY`,`transferScaleZ`,`positionZ`,`rotationX`,`rotationY`) values (1,512,512,1,0,0,1,0.05,0,0,0,0,3,0,0),(2,512,512,3,1,0,1,0.05,0,0.5,0,0,3,0,0),(3,512,512,3,2,0,1,0.05,0,0,0.5,0,3,0,0),(4,512,512,3,3,0,1,0.05,0,0,0,0.5,3,0,0);

UNLOCK TABLES;

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
) ENGINE=InnoDB DEFAULT CHARSET=utf8;

/*Data for the table `volume` */

LOCK TABLES `volume` WRITE;

UNLOCK TABLES;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;
