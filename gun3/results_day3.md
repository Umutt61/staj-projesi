| Model              |   Accuracy |   Precision |   Recall |       F1 | Confusion Matrix      |
|:-------------------|-----------:|------------:|---------:|---------:|:----------------------|
| LogisticRegression |   0.964217 |    0.960784 | 0.748092 | 0.841202 | [[899, 4], [33, 98]]  |
| MultinomialNB      |   0.952611 |    1        | 0.625954 | 0.769953 | [[903, 0], [49, 82]]  |
| RandomForest       |   0.975822 |    1        | 0.80916  | 0.894515 | [[903, 0], [25, 106]] |

----------------------------------------------------------------------------------------------------

##  Kısa Yorum:

- **Accuracy (doğruluk)** açısından en iyi model: **RandomForest (0.975822)**  
- **Precision (isabet oranı)** en yüksek: **MultinomialNB (1.0)** ama Recall düşük.
- **Recall (yakalama oranı)** önemli çünkü spam kaçırmak istemeyiz, burada da **RandomForest** önde.
- **F1 skoru** genel dengeyi gösterir: yine **RandomForest** en yüksek.

----------------------------------------------------------------------------------------------------

##  En İyi Model: **RandomForestClassifier(n_estimators=100)**

- Hem **accuracy**, hem **recall**, hem **F1 skoru** açısından en başarılı.
- `n_estimators=100` (ağaç sayısı) ile varsayılan parametreyle çalıştı.

----------------------------------------------------------------------------------------------------

## Not:

- Naive Bayes hızlı ama Recall düşük.
- Logistic Regression daha dengeli ama RandomForest daha iyi yakalıyor.