import pymysql
from PIL import Image, ImageEnhance

conn = pymysql.connect(
    host="127.0.0.1",
    port=3306,
    user="root",
    passwd="root",
    database="highmyopiasystem_db"
)


# 用作测试
def contest():
    cursor = conn.cursor()
    sql_inset = "insert into examdetail(case_id, exam_id, path, type, downfile, localpath, dev) values(%s, %s, %s, %s, %s, %s, %s)"
    content = ["1908", "190727ME1266201-1",
               "D:\\eye\\exam_data\\EA_OPT2\\ExecutorID_1\\190727ME1266201-1\\resourceFiles\\000805-20190727@090826-L1-S.jpg",
               "JPG", "1", "/img/PNG/000805-20190727@090508-R1-S-P.jpg", "opt"]
    cursor.executemany(sql_inset, [content])
    conn.commit()


def commit_to_mysql(case_id, exam_id, path, type, downfile, dev, localpath):
    cursor = conn.cursor()
    sql_inset = "insert into examdetail(case_id, exam_id, path, type, downfile, localpath, dev, iolread) values(%s, %s, %s, %s, %s, %s, %s, %s)"
    content = [case_id, exam_id, path, type, downfile, localpath, dev, 1]
    cursor.executemany(sql_inset, [content])
    conn.commit()


def name_func(img_path, path):
    dir_path, img_name = img_path_split(img_path, '\\')
    dir_sec_path = img_path_split(path, '/')[0]
    img_name, img_name_2 = img_name_change(img_name)
    return dir_path + img_name, dir_sec_path + img_name, dir_path + img_name_2, dir_sec_path + img_name_2


def img_path_split(img_path, pattern):
    res = -1
    for index in range(len(img_path)):
        if (img_path[index] == pattern):
            res = index
    return img_path[:res + 1:], img_path[res + 1::]


def img_name_change(img_name):
    name = img_name[:len(img_name) - 4:]
    suffix = img_name[len(img_name) - 4::]
    return name + "-E" + suffix, name + '-W' + suffix


def img_file_name(img_name):
    dir_path, img_name = img_path_split(img_name, '\\')
    return img_name[:len(img_name) - 4:]


if __name__ == '__main__':
    contest()
    # print(img_name_change("000805-20190727@090508-R1-S.jpg"))
    # print(img_path_split("D:\\eye\\exam_data\\EA_OPT2\\ExecutorID_1\\190727ME1266201-1\\resourceFiles\\000805-20190727@090508-R1-S.jpg"))
    # print(img_path_split("/img/PNG/000805-20190727@090508-L2-S.jpg", '/'))
    # print(img_path_split("/img/PNG/000805-20190727@090508-R1-S.png", '/'))
