from lib import *
def lr_cross_val(x_train, x_test, y_train, y_test, x_new, y_new):
    # 模型训练
    lr = LogisticRegression(C=1)
    lr.fit(x_train, y_train)
    pred_proba_test = lr.predict_proba(x_test)
    pred_proba_new = lr.predict_proba(x_new)
    fpr_test, tpr_test, thresholds_test = roc_curve(y_test, pred_proba_test[:, 1])
    fpr_new, tpr_new, thresholds_new = roc_curve(y_new, pred_proba_new[:, 1])
    # print('x_test', x_test, type(x_test))
    # print('pred_proba', pred_proba, len(pred_proba))
    auc_score_test = auc(fpr_test, tpr_test)
    auc_score_new = auc(fpr_new, tpr_new)
    return auc_score_new, pred_proba_test, pred_proba_new


def sk_xgb_cross_val(x_train, x_test, y_train, y_test, feature_name_list):
    model = XGBClassifier(
        learning_rate=0.01,         # Boosting learning rate (xgb’s “eta”)
        n_estimators=1800,          # Number of boosted trees to fit.
        max_depth=5,                # 基学习器的树的最大深度，越大越容易过拟合
        objective='binary:logistic',
        min_child_weight=30,
        subsample=0.7,              # 行抽样
        colsample_bytree=0.7,       # 列抽样
        reg_alpha=0,                # l1 正则
        reg_lambda=1,               # l2 正则
        silent=1,                   # Whether to print messages while running boosting. 0 for print
    )
    # x_train = pd.DataFrame(x_train, columns=feature_name_list)
    # x_test = pd.DataFrame(x_test, columns=feature_name_list)
    model.fit(x_train, y_train, eval_metric='auc')
    # print('x_train', x_train, type(x_train))
    # print('y_train', y_train, type(y_train))
    pred_proba = model.predict_proba(x_test)
    # print('x_test', x_test, type(x_test))
    # print('pred_proba', pred_proba, len(pred_proba))
    fpr, tpr, thresholds = roc_curve(y_test, pred_proba[:, 1])
    auc_score = auc(fpr, tpr)

    feature_importance = pd.DataFrame({'feature': feature_name_list,
                                       'importance': list(model.feature_importances_)}).sort_values(by='importance',
                                                                                                    ascending=False)
    feature_importance = feature_importance[feature_importance['importance'] > 0]
    print('sk xgb feature_importance', feature_importance)

    # ks值
    ks = max(tpr - fpr)
    print("sk xgb ks value:%0.3f" % ks)
    # draw(fpr, tpr, auc_score)
    # depart_score(pred_proba, y_test)
    return auc_score, pred_proba


def lgb_cross_train(x_train, x_test, y_train, y_test):
    model = lgb.LGBMClassifier(
            class_weight='balanced',
            boosting_type='gbdt',
            num_leaves=20,
            max_depth=5,                # 构建树的深度，越大越容易过拟合
            learning_rate=0.01,
            n_estimators=1800,          # 讯息
            # max_bin=255,
            # subsample_for_bin=20,
            # objective=None,
            min_split_gain=1,
            # min_child_weight=30,
            # min_child_samples=20,
            subsample=0.7,              # 行抽样
            # subsample_freq=1,
            colsample_bytree=0.7,       # 列抽样
            reg_alpha=0,                # l1 正则
            reg_lambda=1,               # l2 正则
            silent=0
    )
    model.fit(x_train, y_train)
    # 测试数据评价
    pred_proba = model.predict_proba(x_test)
    fpr, tpr, thresholds_1 = roc_curve(y_test, pred_proba[:, 1])
    # pred_proba = model.predict(dtest,ntree_limit=model.best_ntree_limit)
    # fpr, tpr, thresholds = roc_curve(y_test, pred_proba)
    auc_score = auc(fpr, tpr)
    return auc_score


def xgb_cross_val(x_train, x_test, y_train, y_test):
    xg_train = xgb.DMatrix(x_train, label=y_train)
    xg_test = xgb.DMatrix(x_test, label=y_test)
    params = {
        #     'booster':'gbtree',
        #     'objective': 'multi:softmax', #多分类的问题
        #     'num_class':10, # 类别数，与 multisoftmax 并用
        'gamma': 0.1,  # 用于控制是否后剪枝的参数,越大越保守，一般0.1、0.2这样子。
        'max_depth': 12,  # 构建树的深度，越大越容易过拟合
        'lambda': 2,  # 控制模型复杂度的权重值的L2正则化项参数，参数越大，模型越不容易过拟合。
        'subsample': 0.7,  # 随机采样训练样本
        #     'colsample_bytree':0.7, # 生成树时进行的列采样
        'min_child_weight': 3,
        # 这个参数默认是 1，是每个叶子里面 h 的和至少是多少，对正负样本不均衡时的 0-1 分类而言
        # ，假设 h 在 0.01 附近，min_child_weight 为 1 意味着叶子节点中最少需要包含 100 个样本。
        # 这个参数非常影响结果，控制叶子节点中二阶导的和的最小值，该参数值越小，越容易 overfitting。
        'silent': 0,  # 设置成1则没有运行信息输出，最好是设置为0.
        'eta': 0.007,  # 如同学习率
        'seed': 1000,
        #     'nthread':7,# cpu 线程数
        'eval_metric': 'auc',
    }
    param = {'max_depth': 2, 'eta': 1, 'silent': 1, 'objective': 'binary:logistic'}
    num_rounds = 1800
    #     bst = xgb.train(param, dtrain, num_round)
    model = xgb.train(param, xg_train, num_rounds)
    pred_proba = model.predict(xg_test, ntree_limit=model.best_ntree_limit)
    fpr, tpr, thresholds = roc_curve(y_test, pred_proba)
    auc_score = auc(fpr, tpr)
    # feature_importance = pd.DataFrame({'feature': x_train.columns.tolist(),
    #                                    'importance': list(model.feature_importances_)}).sort_values(by='importance',
    #                                                                                                 ascending=False)
    # feature_importance = feature_importance[feature_importance['importance'] > 0]
    # print('xgb feature_importance', feature_importance)

    return auc_score, pred_proba


def pred_cross_val(df, label_name, all_feature_df, uiq_str):
    data_x = all_feature_df
    data_y = df[label_name]
    print('len(data_x)', len(data_x))
    print('len(data_y)', len(data_y))
    x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2)
    # auc_score_lr, pred_ = lr_cross_val(x_train, x_test, y_train, y_test)
    # auc_score_xgb, pred_ = xgb_cross_val(x_train, x_test, y_train, y_test)
    auc_score_sk_xgb, pred_ = sk_xgb_cross_val(x_train, x_test, y_train, y_test, x_train.columns)
    # auc_score_lgb = lgb_cross_train(x_train, x_test, y_train, y_test)

    # print(uiq_str + ' lr auc_score', auc_score_lr)
    # print(uiq_str + ' xgb auc_score', auc_score_xgb)
    print(uiq_str + ' sk xgb auc_score', auc_score_sk_xgb)
    # print(uiq_str + 'lgb auc_score', auc_score_lgb)

def pred(test_df, test_all_feature_df, new_df, new_all_feature_df, label_name,  uiq_str):
    x_train = test_all_feature_df
    y_train = test_df[label_name]
    x_test = new_all_feature_df
    y_test = new_df[label_name]
    print('len(x_train)', len(x_train))
    print('len(y_train)', len(y_train))
    print('len(x_test)', len(x_test))
    print('len(y_test)', len(y_test))
    # print('len(y_train)', len(data_y))
    # x_train, x_test, y_train, y_test = train_test_split(data_x, data_y, test_size=0.2)
    # auc_score_lr, pred_ = lr_cross_val(x_train, x_test, y_train, y_test)
    # auc_score_xgb, pred_ = xgb_cross_val(x_train, x_test, y_train, y_test)
    auc_score_sk_xgb, pred_ = sk_xgb_cross_val(x_train, x_test, y_train, y_test, x_train.columns)
    # auc_score_lgb = lgb_cross_train(x_train, x_test, y_train, y_test)

    # print(uiq_str + ' lr auc_score', auc_score_lr)
    # print(uiq_str + ' xgb auc_score', auc_score_xgb)
    print(uiq_str + ' sk xgb auc_score', auc_score_sk_xgb)
    # print(uiq_str + 'lgb auc_score', auc_score_lgb)