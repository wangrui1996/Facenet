# 树莓派上移植FaceNet

--------------------
FaceNet网路可以参考具体的论文["FaceNet: A Unified Embedding for Face Recognition and Clustering"](http://arxiv.org/abs/1503.03832)了解熟悉，源码主要参考了 [Facenet](https://github.com/davidsandberg/facenet),以及[MTCNN
](https://github.com/davidsandberg/facenet/tree/master/src/align).自己对源码整体做了一些改动以及增加特征提取后的欧氏距离与数据库中对比的结果。

<video controls="" preload="none" poster="http://media.w3.org/2010/05/sintel/poster.png"><source src="https://github.com/wangrui1996/facerecognitionRaspberry/blob/master/data/result/record/f.avi" type="video/mp4"><source src="https://github.com/wangrui1996/facerecognitionRaspberry/blob/master/data/result/record/f.avi" type="video/webm"><source src="https://github.com/wangrui1996/facerecognitionRaspberry/blob/master/data/result/record/f.avi" type="video/ogg"><p>Your user agent does not support the HTML5 Video element.</p></video>







![Crates.io](https://img.shields.io/crates/l/rustc-serialize.svg)
