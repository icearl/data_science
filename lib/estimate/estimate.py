def depart_score(pred_proba, y_test):
    # 模型得分分布
    from IPython.display import display
    y_test_val = pd.Series(pred_proba[:, 1])

    df = pd.concat([y_test_val, y_test.reset_index().drop('index', axis=1)], axis=1)
    # y_test_label = pd.Series(y_test)
    # df = pd.concat([y_test_val,y_test_label],axis=1)

    df.columns = ['prob', 'label']

    bins = [i / 20 for i in range(21)]

    score_index = ['000-050',
                   '050-100',
                   '100-150',
                   '150-200',
                   '200-250',
                   '250-300',
                   '300-350',
                   '350-400',
                   '400-450',
                   '450-500',
                   '500-550',
                   '550-600',
                   '600-650',
                   '650-700',
                   '700-750',
                   '750-800',
                   '800-850',
                   '850-900',
                   '900-950',
                   '950-1000']

    df['score'] = pd.cut(df['prob'].astype(float), bins, labels=score_index)

    sum_0 = df[['score', 'label']][df['label'] == 0].groupby('score').count()
    sum_1 = df[['score', 'label']][df['label'] == 1].groupby('score').count()

    result = pd.concat([sum_0, sum_1], axis=1)
    result.columns = ['未逾期', '逾期']
    result['放款'] = result['未逾期'] + result['逾期']
    result['区间逾期率'] = round(result['逾期'] / result['放款'] * 100, 2)
    result['累计通过率'] = round(result['放款'].cumsum() / result['放款'].sum() * 100, 2)
    result['累计逾期率'] = round(result['逾期'].cumsum() / result['放款'].cumsum() * 100, 2)

    display(result)


def draw(fpr, tpr, auc_score):
    # 画ROC曲线
    plt.plot(fpr, tpr, '--', color=(0.1, 0.1, 0.1), label='Test ROC (area = %0.3f)' % auc_score, lw=2)
    # plt.plot(fpr_train, tpr_train, '--', color=(0.1, 0.1, 0.9), label='Train ROC (area = %0.3f)' % roc_auc_train, lw=2)
    plt.plot([0, 1], [0, 1], '--', color=(0.6, 0.6, 0.6), label='Random Chance')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Model ROC curve and ROC area')
    plt.legend(loc="lower right")
    plt.show()

