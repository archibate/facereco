1. 首先你把全体学生照片保存到 ``dataset`` 文件夹下面，格式如下：

```
dataset/
- youyangyu/
-- 001.jpg
-- 002.jpg
-- 003.jpg
-- 004.jpg
- pengyubin/
-- 001.jpg
-- 002.jpg
-- 003.jpg
-- 004.jpg
```

（为避免 locales 鬼畜，建议全部采用英文名）

2. 然后对这些图片进行训练，结果保存到 pickle 文件：

```bash
$ python train_model.py -d dataset/ -m model.pickle
training 1/8: dataset/youyangyu/001.jpg
training 2/8: dataset/youyangyu/002.jpg
training 3/8: dataset/youyangyu/003.jpg
...
```

执行完毕后你将得到 `model.pickle` 。

3. 现在你有一张新的学生照片，你想知道他是谁（叫什么名字）：

```bash
$ python get_names_in.py -i images/YYY.jpg -m model.pickle
youyangyu
```

（注：如果一张图片里有多位学生，则会依次换行打印）
（注：如果出现陌生人面孔，则显示 unknown）

4. 现在你有一张新的学生照片，你想知道彭于斌在不在这张图片里面：

```bash
$ python is_name_in.py -n pengyubin -i images/YYY.jpg -m model.pickle
NO
$ python is_name_in.py -n youyangyu -i images/YYY.jpg -m model.pickle
YES
```

（`is_name_in.py` 比 `get_names_in.py` 更高效，因为他只要检索一个人的数据）

5. 还可以启动图形界面显示出头像方框和姓名：

```bash
$ python draw_box.py -i images/YYY.jpg -m model.pickle
```
