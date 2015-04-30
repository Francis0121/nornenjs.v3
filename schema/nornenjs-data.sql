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

/*Data for the table `actor` */

LOCK TABLES `actor` WRITE;

insert  into `actor`(`pn`,`username`,`password`,`enabled`) values (8,'nornenjs','f981fbf4314e1c077fc9f559c60c6272a1f95b13',1),(9,'teriusbin','6b783b11b13bced02be9fa442c045975c75c746b',1),(10,'eikids','63f5d6d8974334bd465e06899433eeffb6e85043',1),(11,'dsa12','a6e714a58c5f18e91d0043ebff1c152c0a405d48',1);

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

insert  into `actor_info`(`actorPn`,`email`,`firstName`,`lastName`,`inputDate`,`updateDate`) values (8,'nornenjs@naver.com','Francis','Kim','2015-04-29 22:24:58','2015-04-30 03:10:45'),(9,'teriusbin@naver.com','Woo','Lee','2015-04-29 23:07:11','2015-04-29 23:07:11'),(10,'eikids@naver.com','Francis','Kim','2015-04-29 23:11:13','2015-04-30 02:47:11'),(11,'ds@naver.com','sdasda','kims','2015-04-29 23:13:07','2015-04-29 23:13:07');

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
) ENGINE=InnoDB AUTO_INCREMENT=4 DEFAULT CHARSET=utf8;

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
) ENGINE=InnoDB AUTO_INCREMENT=230 DEFAULT CHARSET=utf8;

/*Data for the table `data` */

LOCK TABLES `data` WRITE;

insert  into `data`(`pn`,`username`,`type`,`name`,`savePath`,`inputDate`) values (71,'nornenjs',1,'temp.den','${data.rootUploadPath\\nornenjs\\908ba9e8-0dbd-42ec-9cf9-5bc301721045','2015-04-30 08:57:15'),(72,'nornenjs',1,'temp.den','\"E:\\nornenjs\"\\nornenjs\\62207e57-4d0b-4d3c-8999-771ed6af672a','2015-04-30 08:58:24'),(73,'nornenjs',1,'temp.den','\"E:\\nornenjs\"\\nornenjs\\19a0dd0a-bae0-4379-b43b-3485f435e019','2015-04-30 08:58:58'),(74,'nornenjs',1,'temp.den','E:\\nornenjs\\data\\51a564ea-89aa-4330-9ed8-c1cb741f6146','2015-04-30 09:01:02'),(75,'nornenjs',1,'temp.den','E:\\nornenjs\\data\\9bd002a3-3be9-4b09-b946-928752b01d63','2015-04-30 09:03:23'),(76,'nornenjs',1,'temp.den','E:\\nornenjs\\data\\3b1be3e1-a5d3-4969-88bc-7d8b68506be7','2015-04-30 09:08:15'),(77,'nornenjs',1,'temp.den','E:\\nornenjs\\data\\4a25b46d-873f-4211-a1b0-c92515506688','2015-04-30 09:08:46'),(78,'nornenjs',1,'temp.den','E:\\nornenjs\\data\\a8ed6d5d-cae0-4428-9a16-5af3bf1bb52c','2015-04-30 09:12:52'),(79,'nornenjs',1,'temp.den','E:\\nornenjs\\data\\f80b8089-ff09-4219-96e0-5442ffbda2a4','2015-04-30 09:13:57'),(80,'nornenjs',1,'temp.den','E:\\nornenjs\\data\\4809fa55-a4f3-418e-b9bd-71b965d92b22','2015-04-30 09:14:43'),(81,'nornenjs',1,'temp.den','E:\\nornenjs\\data\\4a7b3b63-9bdb-426c-ab4d-b41efa2146b3','2015-04-30 09:15:46'),(82,'nornenjs',1,'temp.den','E:\\nornenjs\\data\\d3bab178-20f4-42f8-94e7-ccc26392ff00','2015-04-30 09:16:03'),(83,'nornenjs',1,'temp.den','E:\\nornenjs\\data\\86a4b8ae-0a33-49f4-a4bc-68666bbd883b','2015-04-30 09:16:13'),(84,'nornenjs',1,'temp.den','E:\\nornenjs\\data\\edc3dd00-8241-4c15-ac21-5bcdece60ece','2015-04-30 09:16:41'),(85,'nornenjs',1,'temp.den','E:\\nornenjs\\data\\9aba8adf-b74e-4209-85d6-7e052ef89b86','2015-04-30 09:17:24'),(86,'nornenjs',1,'temp.den','E:\\nornenjs\\data\\a4e628bc-60da-441c-b934-e5f77cafc2b5','2015-04-30 09:17:33'),(87,'nornenjs',1,'temp.den','E:\\nornenjs\\data\\e1d835a9-a053-4edd-96ac-a9dead82fe25','2015-04-30 09:17:40'),(88,'nornenjs',1,'temp.den','E:\\nornenjs\\data\\21021af0-f431-4e03-91fe-0080d316849c','2015-04-30 09:18:02'),(89,'nornenjs',1,'temp.den','E:\\nornenjs\\data\\d9898024-1578-45aa-a96e-976e3a34937d','2015-04-30 09:18:34'),(90,'nornenjs',1,'temp.den','E:\\nornenjs\\data\\9f48d924-9e64-4c67-9e7f-d43c5676f3db','2015-04-30 09:19:26'),(219,'nornenjs',1,'temp.den','E:\\nornenjs\\data\\61ac7271-ec4f-4d48-b4c5-d50cb57d1efc','2015-04-30 09:31:49'),(220,'nornenjs',1,'temp.den','E:\\nornenjs\\data\\09c914fa-1e92-48bb-9917-2a233a7c408e','2015-04-30 09:36:29'),(221,'nornenjs',1,'temp.den','E:\\nornenjs\\data\\dcf55612-2835-498d-b04c-889561494d5d','2015-04-30 09:36:54'),(222,'nornenjs',1,'temp.den','E:\\nornenjs\\data\\498991b8-cba1-49eb-a732-510b34b0ec0e','2015-04-30 09:37:05'),(223,'nornenjs',1,'temp.den','E:\\nornenjs\\data\\1d6f5223-b49a-4e11-a9f6-11df55207b21','2015-04-30 09:38:06'),(224,'nornenjs',1,'temp.den','E:\\nornenjs\\data\\eecd5b43-3f64-49b1-ad66-dd15ae3b1ac1','2015-04-30 09:38:40'),(225,'nornenjs',1,'temp.den','E:\\nornenjs\\data\\205756ec-5ab0-483d-8e7d-6f53eee960ce','2015-04-30 09:39:10'),(226,'nornenjs',1,'temp.den','E:\\nornenjs\\data\\4b6ea09d-4294-47da-8437-ce726ee93ae9','2015-04-30 09:39:23'),(227,'nornenjs',1,'temp.den','E:\\nornenjs\\data\\1dcb2b0e-9be3-4ea7-a372-d03c9b2ed7cc','2015-04-30 09:40:16'),(228,'nornenjs',1,'temp.den','E:\\nornenjs\\data\\2a3b42db-508c-4149-bcfe-45277dccd680','2015-04-30 09:41:59'),(229,'nornenjs',1,'temp.den','E:\\nornenjs\\data\\b012a05e-3970-4fee-ad66-f516df05def6','2015-04-30 09:49:16');

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
) ENGINE=InnoDB AUTO_INCREMENT=8 DEFAULT CHARSET=utf8;

/*Data for the table `group_members` */

LOCK TABLES `group_members` WRITE;

insert  into `group_members`(`pn`,`username`,`groupPn`) values (1,'nornenjs',1),(2,'teriusbin',2),(3,'eikids',2),(4,'dsa12',2);

UNLOCK TABLES;

/*Table structure for table `groups` */

DROP TABLE IF EXISTS `groups`;

CREATE TABLE `groups` (
  `pn` bigint(20) unsigned NOT NULL AUTO_INCREMENT,
  `groupName` varchar(50) NOT NULL,
  PRIMARY KEY (`pn`)
) ENGINE=InnoDB AUTO_INCREMENT=21 DEFAULT CHARSET=utf8;

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

/*Data for the table `volume` */

LOCK TABLES `volume` WRITE;

insert  into `volume`(`pn`,`username`,`volumeDataPn`,`title`,`width`,`height`,`depth`,`inputDate`) values (181,'nornenjs',227,'25',100,100,100,'2015-04-30 09:40:30'),(182,'nornenjs',228,'설명',120,120,100,'2015-04-30 09:42:07'),(183,'nornenjs',229,'설명23',202,202,102,'2015-04-30 09:50:41');

UNLOCK TABLES;

/*!40101 SET SQL_MODE=@OLD_SQL_MODE */;
/*!40014 SET FOREIGN_KEY_CHECKS=@OLD_FOREIGN_KEY_CHECKS */;
/*!40014 SET UNIQUE_CHECKS=@OLD_UNIQUE_CHECKS */;
/*!40111 SET SQL_NOTES=@OLD_SQL_NOTES */;
