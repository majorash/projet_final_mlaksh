DROP DATABASE IF EXISTS RadioApp;
create database RadioApp;
use RadioApp;
CREATE TABLE RadioApp.app_user (
	user_id BIGINT NOT NULL AUTO_INCREMENT,
	user_email VARCHAR(45) NULL,
	user_username VARCHAR(45) NULL,
	user_password VARCHAR(150) NULL,
	PRIMARY KEY (user_id));
CREATE TABLE RadioApp.images (
	img_id BIGINT NOT NULL AUTO_INCREMENT,
	view_pos VARCHAR(20) NULL,
	user_id BIGINT NOT NULL,
	img_path VARCHAR(300) NULL,
    label_id VARCHAR(100) NULL,
    img_date datetime NULL,
    latent_vector BLOB NULL,
	PRIMARY KEY (img_id),
    FOREIGN KEY (user_id) REFERENCES app_user(user_id));

CREATE TABLE RadioApp.posts (
	post_id BIGINT NOT NULL AUTO_INCREMENT,
	post_content VARCHAR(10000) NULL,
	post_usr_id BIGINT NULL,
	post_date datetime DEFAULT NULL,
    post_img_id BIGINT NULL,
    img_label_val bool,
    corr_label varchar(45) NULL,
	PRIMARY KEY (post_id),
    FOREIGN KEY (post_usr_id) REFERENCES app_user(user_id),
	FOREIGN KEY (post_img_id) REFERENCES images(img_id));

USE RadioApp;
DROP procedure IF EXISTS sp_createUser;

DELIMITER $$
USE RadioApp $$
CREATE DEFINER=root@localhost PROCEDURE sp_createUser(
    IN p_username VARCHAR(20),
    IN p_email varchar(45),
    IN p_password CHAR(150)
)
BEGIN
    IF ( select exists (select 1 from app_user where user_username = p_username) ) THEN     
        select 'Username already exists';  
	ELSEIF ( select exists (select 1 from app_user where user_email = p_email) )  THEN
		select 'An account with this email already exists.';
    ELSE     
        insert into app_user
        (
			user_username,
            user_email,
            user_password
        )
        values
        (
			p_username,
            p_email,
            p_password
        );
    END IF;
END$$
DELIMITER ;

USE RadioApp;
DROP procedure IF EXISTS sp_validateLogin;


DELIMITER $$
USE RadioApp $$
CREATE DEFINER=root@localhost PROCEDURE sp_validateLogin(
IN p_username VARCHAR(20)
)
BEGIN
	select * from app_user where user_username = p_username;
END$$

DELIMITER ;

DROP procedure IF EXISTS sp_addImage;
insert into app_user(user_id,user_email,user_username,user_password) values (1,"test","test","test");

DELIMITER $$
USE RadioApp $$
CREATE DEFINER=root@localhost PROCEDURE sp_addImage(
IN p_img_path VARCHAR(200),
IN p_username VARCHAR(45),
IN p_label_id VARCHAR(45),
IN p_latent_vector BLOB
)
BEGIN
	SELECT @p_user_id := user_id FROM app_user WHERE app_user.user_username = p_username;
    
	INSERT INTO images(img_path, img_date, user_id, label_id,latent_vector) values (p_img_path, NOW(), @p_user_id ,p_label_id, p_latent_vector);
END$$

DELIMITER ;
USE RadioApp;
DROP procedure IF EXISTS sp_GetImgByUser;
 
DELIMITER $$
USE RadioApp$$
CREATE DEFINER=root@localhost PROCEDURE sp_GetImgsByUser (
IN p_user_id bigint
)
BEGIN
    select * from images where user_id = p_user_id;
END$$
 
DELIMITER ;
USE RadioApp;
DROP procedure IF EXISTS sp_GetVecsByPath;
 
DELIMITER $$
USE RadioApp$$
CREATE DEFINER=root@localhost PROCEDURE sp_GetVecsByPath (
IN p_path bigint
)
BEGIN
    select latent_vector from images where img_path = p_path;
END$$
 
DELIMITER ;

USE RadioApp;
DROP procedure IF EXISTS sp_GetAllVecs;
 
DELIMITER $$
USE RadioApp$$
CREATE DEFINER=root@localhost PROCEDURE sp_GetAllVecs (
)
BEGIN
    select latent_vector, img_path from images;
END$$
 
DELIMITER ;

DROP procedure IF EXISTS sp_addBlog;

DELIMITER $$
USE RadioApp $$
CREATE DEFINER=root@localhost PROCEDURE sp_addBlog(
    IN p_content varchar(1000),
    IN p_user_id bigint,
    IN p_post_img_id bigint
)
BEGIN
    insert into posts(
        post_content,
        post_usr_id,
        post_date,
        post_img_id
    )
    values
    (
        p_content,
        p_user_id,
        NOW(),
        p_post_img_id
    );
END$$
DELIMITER ;

DROP procedure IF EXISTS sp_addLabel;


DELIMITER $$
USE RadioApp $$
CREATE DEFINER=root@localhost PROCEDURE sp_addLabel(
IN p_label INT
)
BEGIN
	INSERT INTO images(label) values (p_label);
END$$

DELIMITER ;

USE RadioApp;
DROP procedure IF EXISTS sp_deletePost;

DELIMITER $$
USE RadioApp$$
CREATE DEFINER=root@localhost PROCEDURE sp_deletePost (
IN p_blog_id bigint,
IN p_user_id bigint
)
BEGIN
delete from post where post_id = p_blog_id and post_user_id = p_user_id;
END$$
 
DELIMITER ;

SHOW PROCEDURE STATUS WHERE db = 'RadioApp';
select* from app_user;
Select * from images;