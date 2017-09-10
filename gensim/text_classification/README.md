# Text classification examples
> 本例使用gensim的做文本分类，分为两种：
* text\_classification.py实现句子级别的分类，即对一个句子或者一段文本进行分类；
* sense\_classification.py实现对一段文本中的某个词进行分类，使用场景是WSD（词义消歧）或者NED（实体消歧），即一个词有多个含义（类型），需要根据上下文判断其真是的含义，本例中将消歧进行了简化，简单转换为分类问题，实际应用过程中要复杂。
