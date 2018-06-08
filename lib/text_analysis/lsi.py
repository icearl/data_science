# 得到文档矩阵
apps_list = df_train['installed_apk'].tolist()
apps_whole_list = df_merge['installed_apk'].tolist()
texts = [every_user.split(' ') for every_user in apps_list]

tpoic_num = 10
# 词频矩阵
# 字典
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
# tfidf
tfidf_model = models.TfidfModel(corpus)
corpus_tfidf = tfidf_model[corpus]
# lsi
lsi_model = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=tpoic_num)
corpus_lsi = lsi_model[corpus_tfidf]
nodes = list(corpus_lsi_whole)