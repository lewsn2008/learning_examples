# Text classification examples
> 本例使用gensim的做文本分类，分为两种：
> * `text\_classification.py`实现句子级别的分类，即对一个句子或者一段文本进行分类；
> * `sense\_classification.py`实现对一段文本中的某个词进行分类，使用场景是WSD（词义消歧）或者NED（实体消歧），即一个词有多个含义（类型），需要根据上下文判断其真是的含义，本例中将消歧进行了简化，简单转换为分类问题，实际应用过程中要复杂。

## 主要步骤
##### 分词
做文本分类大部分是基于词语来抽取特征的，也有一些工作是基于字符的，本例不进行讨论，只讨论基于词语的方式，所以第一步需要对待分类文本进行分词（对中文而言），本例不讨论分词的技术问题，该例程序中的训练和测试文本都是已经分好词的.

##### 构造特征词典
使用gensim.corpora.Dictionary处理词典，该package可以很方便的实现词典创建、特征词过滤、词序列转数字特征、词典保存于加载等功能，简单列出几个：
* `add_documents`: 添加已分词文本，自动扩充词典;
* `filter_extremes`: 特征词过滤，例如过滤低频词，过滤文档频率太高的词（即在大部分文本中出现）;
* `doc2bow`: 将词序列转换为特征（数字）序列，用于模型训练和推理.

##### 特征抽取
* 最基本的是词特征；
* `ngram特征`: ngram是NLP中经常使用的技术，对于文本分类任务，某些时候单个词可能特征不是很明显，但是前后词拼接为bigram或者trigram后可能特征会较明显，`_get_ngram_terms`函数用于抽取ngram的特征；
* `positional term特征`: 对于词义分类任务，是有中心词，需要判断句子中某个词的类别，所以需要考虑词语的位置信息，即中心词的左右附近窗口内的词语，`_get_posi_terms`函数抽取positional词特征.

##### 模型
上述抽取完特征后，doc2bow将文本特征转换为数字特征，即可以用于不同的机器学习模型，没有限制，本例使用LR模型.

## 训练数据
##### 文本分类(text_classification.py)
格式: LABEL`<TAB>`word`<空格>`word`<空格>`...
##### 词义分类(sense_classification.py)
格式: LABEL`<TAB>`focus_word`<TAB>`word`<TAB>`word`<TAB>`...

## 运行
##### 文本分类(text_classification.py)
* 训练:  
    (参数dict_file和model_file用于保存词典和模型)  
    ```shell
    ./text_classifier.py --train_file data/text_classification.txt --dict_file model/text_clf.dict --model_file model/text_clf.model
    ```
* 评估:  
    (参数dict_file和model_file用于加载词典和模型)  
    ```shell
    ./text_classifier.py --test_file data/text_classification_test.txt --dict_file model/text_clf.dict --model_file model/text_clf.model
    ```
##### 词义分类(sense_classification.py)  
* 训练:  
    (参数dict_file和model_file用于保存词典和模型)  
    ```shell
    ./sense_classifier.py --train_file data/sense_classification_train.txt --dict_file model/sense_clf.dict --model_file model/sense_clf.model 
    ```
* 评估:  
    (参数dict_file和model_file用于加载词典和模型)  
    ```shell
    ./sense_classifier.py --test_file data/sense_classification_test.txt --dict_file model/sense_clf.dict --model_file model/sense_clf.model 
    ```

