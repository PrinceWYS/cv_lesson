# Scene Recognition with Bag of Words
## 项目结构

```shell
.
├── data            # 数据集
│   ├── test        # 测试数据集
│   └── train       # 训练数据集
├── BagOfSIFT.py    # Bag of SIFT representation
├── TinyImage.py    # Tiny images representation
├── classifier.py   # knn分类器
├── main.py         # 程序入口
├── util.py         # 工具包
└── pic             # 工具图片
```

## 运行方式
安装所需的依赖环境: (测试环境: `python=3.9`)
```shell
pip install -r requirements.txt
```

运行程序:
```shell
python main.py [-h] [-r REPRESENTATION] [-s SIZE] [-c CREATE] [-v VISUAL]

optional arguments:
  -h, --help            show this help message and exit
  -r REPRESENTATION, --representation REPRESENTATION    method of image representation
  -s SIZE, --size SIZE  size of vocabulary
  -c CREATE, --create CREATE    create new pretrain model
  -v VISUAL, --visualization VISUAL   visualize the result
```
### notice
在使用`Bag of SIFT representation`时, 从头建立模型的时间较长, 可以使用预先建立的模型`.pm`(词袋大小为200); 如果需要建立新的模型, 可以修改词袋大小`-s`以及`-c True`.