# Panorama Stitching
## 项目结构
```
.
├── README.md
├── data1               # 测试数据1
│   ├── 112_1298.JPG
│   ├── 112_1299.JPG
│   ├── 112_1300.JPG
│   ├── 113_1301.JPG
│   ├── 113_1302.JPG
│   └── 113_1303.JPG
├── data2               # 测试数据2
│   ├── IMG_0488.JPG
│   ├── IMG_0489.JPG
│   ├── IMG_0490.JPG
│   └── IMG_0491.JPG
├── data3               # 测试数据3
│   ├── IMG_0675.JPG
│   ├── IMG_0676.JPG
│   └── IMG_0677.JPG
├── data4               # 测试数据4
│   ├── IMG_7355.JPG
│   ├── IMG_7356.JPG
│   ├── IMG_7357.JPG
│   └── IMG_7358.JPG
├── feature.py          # 特征检测、提取与匹配
├── homography.py       # 计算单应性矩阵
├── main.py             # 程序入口
└── stich.py            # 图像拼接
```
## 运行方式
安装所需的依赖环境: (测试环境: `python=3.9`)
```shell
pip install -r requirements.txt
```

运行程序
```shell
python main.py [-h] [-img1 IMG1] [-img2 IMG2] [-d DIR] [-m METHOD] [-a IFALL] [-g GUI]

    -h, --help            show this help message and exit
    -img1 IMG1, --img1 IMG1
                        image 1
    -img2 IMG2, --img2 IMG2
                        image 2
    -d DIR, --dir DIR     dir of pics need to stich
    -m METHOD, --method METHOD
                        feature detector method (sift by default)
    -a IFALL, --all IFALL
                        whether stitch all pics (False by default)
    -g GUI, --gui GUI     use gui mode          (False by default)
    -i ITER, --iter ITER  max iteration
```

运行示例
```shell
# stitch two pictures using orb without gui
python main.py -img1 ./data1/112_1298.JPG -img2 ./data2/112_1299.JPG -m orb
# stitch all pictures in one directory using sift with gui
python main.py -d ./data3 -a True -m sift -g True
python main.py -d ./data3 -a True -m mysift -i 5000
```