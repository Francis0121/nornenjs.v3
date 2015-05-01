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

/*Table structure for table `thumbnail` */

DROP TABLE IF EXISTS `thumbnail`;

CREATE TABLE `thumbnail` (
  `dataPn` bigint(20) NOT NULL,
  `thumbnailPn` tinyint(11) NOT NULL,
  PRIMARY KEY (`dataPn`,`thumbnailPn`),
  CONSTRAINT `thumbnail_ibfk_2` FOREIGN KEY (`dataPn`) REFERENCES `data` (`pn`)
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

insert  into `thumbnail_option`(`pn`,`width`,`height`,`type`,`mprType`,`quality`,`brightness`,`density`,`transferOffset`,`transferScaleX`,`transferScaleY`,`transferScaleZ`,`positionZ`,`rotationX`,`rotationY`) values (1,512,512,1,0,0,1,0.05,0,0,0,0,3,0,0),(2,512,512,3,1,0,1,0.05,0,0,0,0,3,0,0),(3,512,512,3,2,0,1,0.05,0,0,0,0,3,0,0),(4,512,512,3,3,0,1,0.05,0,0,0,0,3,0,0);

UNLOCK TABLES;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;
