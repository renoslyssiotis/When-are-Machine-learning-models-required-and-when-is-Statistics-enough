from sklearn.model_selection import GridSearchCV
#from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

class classifier(object):
    
    def __init__(self, X_train, X_test, y_train, y_test):
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
#=========================LOGISTIC REGRESSION==================================
    def logisticReg(self,):
    
        param_grid_logit = dict(penalty = ['none', 'l2'],
                                dual=[False],
                                C=[1.0,1.5,2.0],
                                solver = ['lbfgs'],
                                max_iter=[100,110,120]
                                )
    
        logistic_regression = LogisticRegression()
    
        grid_logit = GridSearchCV(estimator=logistic_regression, 
                                  param_grid=param_grid_logit, 
                                  cv = 3, 
                                  n_jobs=-1,
                                  error_score=0.0,
                                  iid=False) 
    
        grid_logit.fit(self.X_train, self.y_train)
    
        #Predict the test set results 
        y_pred = grid_logit.predict(self.X_test)
    
        #Performance: ROC-AUC
        roc_auc_logit = roc_auc_score(self.y_test, y_pred)
        print('Logistic Regression completed: ROC-AUC: {}'.format(roc_auc_logit)+'\n' + '------------------------------------------------------------------')
    
        #Performance: ROC-PRC
        auc_prc_logit = average_precision_score(self.y_test, y_pred, average='weighted')
        print('Logistic Regression completed: AUC-PRC: {}'.format(auc_prc_logit)+'\n' + '------------------------------------------------------------------')
        
        #Performance: F1 Metric
        f1_logit = f1_score(self.y_test, y_pred, average='weighted')
        print('Logistic Regression completed: F1 metric: {}'.format(f1_logit)+'\n' + '------------------------------------------------------------------')
        
        return roc_auc_logit, auc_prc_logit,f1_logit
    
        #Check for overfitting - check ROC-AUC on training data
        #y_pred_train = grid_logit.predict(X_train)
        #roc_auc_logit_train = roc_auc_score(y_train, y_pred_train)
    
#=========================K-NEAREST NEIGHBOURS=================================
    # def KNN(self,):
    
    #     param_grid_knn = dict(n_neighbors = [1,3,5,7],
    #                           weights = ['uniform', 'distance'],
    #                           algorithm = ['auto']
    #                           )
    
    #     knn = KNeighborsClassifier()
    
    #     grid_knn = GridSearchCV(estimator=knn, 
    #                             param_grid=param_grid_knn, 
    #                             cv = 3, 
    #                             n_jobs=-1,
    #                             error_score=0.0,
    #                             iid=False)
    
    #     grid_knn.fit(self.X_train, self.y_train)
    
    #     #Predict the test set results 
    #     y_pred = grid_knn.predict(self.X_test)
    
    #     #Performance
    #     roc_auc_knn = roc_auc_score(self.y_test, y_pred)
    #     print('KNN completed: ROC-AUC: {}'.format(roc_auc_knn)+'\n' + '------------------------------------------------------------------')

    #     auc_prc_knn = average_precision_score(self.y_test, y_pred, average='weighted')
    #     print('KNN completed: AUC-PRC: {}'.format(auc_prc_knn)+'\n' + '------------------------------------------------------------------')
    #     return roc_auc_knn , auc_prc_knn
    
        #Check for overfitting - check ROC-AUC on training data
        #y_pred_train = grid_knn.predict(X_train)
        #roc_auc_knn_train = roc_auc_score(y_train, y_pred_train)
    
#==========================DECISION TREE=======================================
    def decisionTree(self,):
    
        param_grid_dtree = dict(criterion = ['gini','entropy'],
                                splitter = ['random', 'best']
                                )
        
        decision_tree = DecisionTreeClassifier()
        
        grid_dtree = GridSearchCV(estimator=decision_tree, 
                                  param_grid=param_grid_dtree, 
                                  cv = 3, 
                                  n_jobs=-1,
                                  error_score=0.0,
                                  iid=False)
        
        grid_dtree.fit(self.X_train, self.y_train)
        
        #Predict the test set results 
        y_pred = grid_dtree.predict(self.X_test)
        
        #Performance: AUC-ROC
        roc_auc_dtree = roc_auc_score(self.y_test, y_pred)
        print('Decision Tree completed: ROC-AUC: {}'.format(roc_auc_dtree)+'\n' + '------------------------------------------------------------------')
        
        #Performance: AUC-PRC        
        auc_prc_dtree = average_precision_score(self.y_test, y_pred, average='weighted')
        print('Decision Tree completed: AUC-PRC: {}'.format(auc_prc_dtree)+'\n' + '------------------------------------------------------------------')
        
        #Performance: F1 Metric
        f1_dtree = f1_score(self.y_test, y_pred, average='weighted')
        print('Decision Tree completed: F1 metric: {}'.format(f1_dtree)+'\n' + '------------------------------------------------------------------')
        
        
        return roc_auc_dtree, auc_prc_dtree, f1_dtree
        
        #Check for overfitting - check ROC-AUC on training data
        #y_pred_train = grid_dtree.predict(X_train)
        #roc_auc_dtree_train = roc_auc_score(y_train, y_pred_train)
        
#==========================RANDOM FOREST=======================================
    def randomForest(self,):
        
        param_grid_rforest = dict(n_estimators = [10,20,30,40,50,60,70,80,90,100], #Number of trees
                                  criterion = ['gini','entropy']
                                  )
        
        random_forest = RandomForestClassifier()
        
        grid_rforest = GridSearchCV(estimator=random_forest, 
                                    param_grid=param_grid_rforest, 
                                    cv = 3, 
                                    n_jobs=-1,
                                    error_score=0.0,
                                    iid=False)
        
        grid_rforest.fit(self.X_train, self.y_train)
        
        #Predict the test set results 
        y_pred = grid_rforest.predict(self.X_test)
        
        #Performance: AUC-ROC
        roc_auc_rforest = roc_auc_score(self.y_test, y_pred)
        print('Random Forest completed: ROC-AUC: {}'.format(roc_auc_rforest)+'\n' + '------------------------------------------------------------------')

        #Performance: AUC-PRC
        auc_prc_rforest = average_precision_score(self.y_test, y_pred, average='weighted')
        print('Random Forest completed: AUC-PRC: {}'.format(auc_prc_rforest)+'\n' + '------------------------------------------------------------------')
        
        #Performance: F1 Metric
        f1_rforest = f1_score(self.y_test, y_pred, average='weighted')
        print('Random Forest completed: F1 metric: {}'.format(f1_rforest)+'\n' + '------------------------------------------------------------------')
        
        
        return roc_auc_rforest, auc_prc_rforest, f1_rforest
        
        #Check for overfitting - check ROC-AUC on training data
        #y_pred_train = grid_rforest.predict(X_train)
        #roc_auc_rforest_train = roc_auc_score(y_train, y_pred_train)
    
#==========================PERCEPTRON==========================================
    def perceptron(self,):
        
        param_grid_perceptron = dict(penalty = [None, 'l1', 'l2'],
                                     shuffle = [True, False])
        
        perceptron = Perceptron()
        
        grid_perceptron = GridSearchCV(estimator=perceptron, 
                                        param_grid=param_grid_perceptron, 
                                        cv = 3, 
                                        n_jobs=-1,
                                        error_score=0.0, 
                                        iid=False)
        
        grid_perceptron.fit(self.X_train,self.y_train)
        
        #Predict the test set results 
        y_pred = grid_perceptron.predict(self.X_test)
        
        #Performance: AUC-ROC
        roc_auc_perceptron = roc_auc_score(self.y_test, y_pred)
        print('Perceptron completed: ROC-AUC: {}'.format(roc_auc_perceptron)+'\n' + '------------------------------------------------------------------')

        #Performance: AUC-PRC
        auc_prc_perceptron = average_precision_score(self.y_test, y_pred, average='weighted')
        print('Perceptron completed: AUC-PRC: {}'.format(auc_prc_perceptron)+'\n' + '------------------------------------------------------------------')
        
        #Performance: F1 Metric
        f1_perceptron = f1_score(self.y_test, y_pred, average='weighted')
        print('Perceptron completed: F1 metric: {}'.format(f1_perceptron)+'\n' + '------------------------------------------------------------------')
    
        
        return roc_auc_perceptron, auc_prc_perceptron, f1_perceptron
        
        #Check for overfitting - check ROC-AUC on training data
        #y_pred_train = grid_perceptron.predict(X_train)
        #roc_auc_perceptron_train = roc_auc_score(y_train, y_pred_train)
    
#==========================Multi-layer perceptron==============================
    def MLP(self,):
        MLP = MLPClassifier(max_iter = 1000)
        
        MLP.fit(self.X_train, self.y_train)
        
        #Predict the test set results 
        y_pred = MLP.predict(self.X_test)
        
        #Performance: AUC-ROC
        roc_auc_MLP = roc_auc_score(self.y_test, y_pred)
        print('MLP completed: ROC-AUC: {}'.format(roc_auc_MLP)+'\n' + '------------------------------------------------------------------')

        #Performance: AUC-PRC
        auc_prc_MLP = average_precision_score(self.y_test, y_pred, average='weighted')
        print('MLP completed: AUC-PRC: {}'.format(auc_prc_MLP)+'\n' + '------------------------------------------------------------------')
        
        #Performance: F1 Metric
        f1_MLP= f1_score(self.y_test, y_pred, average='weighted')
        print('MLP completed: F1 metric: {}'.format(f1_MLP)+'\n' + '------------------------------------------------------------------')
        
        return roc_auc_MLP, auc_prc_MLP, f1_MLP
        
        #Check for overfitting - check ROC-AUC on training data
        #y_pred_train = MLP.predict(X_train)
        #roc_auc_MLP_train = roc_auc_score(y_train, y_pred_train)

#===========================ADA BOOST==========================================
    def ADAboost(self, ):
        
        param_grid_ada = dict(n_estimators = [50,75,100])
        
        ada_boost = AdaBoostClassifier()
        
        grid_ada = GridSearchCV(estimator=ada_boost, 
                                param_grid=param_grid_ada, 
                                cv = 3, 
                                n_jobs=-1,iid=False)
        
        grid_ada.fit(self.X_train,self.y_train)
        
        #Predict the test set results 
        y_pred = grid_ada.predict(self.X_test)
        
        #Performance: AUC-ROC
        roc_auc_ada = roc_auc_score(self.y_test, y_pred)
        print('ADA Boost completed: ROC-AUC: {}'.format(roc_auc_ada)+'\n' + '------------------------------------------------------------------')

        #Performance: AUC-PRC
        auc_prc_ada = average_precision_score(self.y_test, y_pred, average='weighted')
        print('ADA Boost completed: AUC-PRC: {}'.format(auc_prc_ada)+'\n' + '------------------------------------------------------------------')
        
        #Performance: F1 Metric
        f1_ada= f1_score(self.y_test, y_pred, average='weighted')
        print('ADA Boost completed: F1 metric: {}'.format(f1_ada)+'\n' + '------------------------------------------------------------------')
        
        return roc_auc_ada, auc_prc_ada, f1_ada
        
        #Check for overfitting - check ROC-AUC on training data
        #y_pred_train = grid_ada.predict(X_train)
        #roc_auc_ada_train = roc_auc_score(y_train, y_pred_train)
    
#===========================GRADIENT BOOSTING==================================
#    def gradientBoost(self, ):
#        
#        param_grid_gradientboost = dict(n_estimators = [50,75,100],
#                                        criterion = ['riedman_mse', 'mse', 'mae']
#                                        )
#        
#        gradient_boosting = GradientBoostingClassifier()
#        
#        grid_gradientboost = GridSearchCV(estimator=gradient_boosting, 
#                                        param_grid=param_grid_gradientboost, 
#                                        cv = 3, 
#                                        n_jobs=-1,
#                                        error_score=0.0
#                                        )
#        
#        grid_gradientboost.fit(self.X_train, self.y_train)
#        
#        #Predict the test set results 
#        y_pred = grid_gradientboost.predict(self.X_test)
#        
#        #Performance
#        roc_auc_gradientboosting = roc_auc_score(self.y_test, y_pred)
#        print('Gradient Boosting completed: ROC-AUC: {}'.format(roc_auc_gradientboosting)+'\n' + '------------------------------------------------------------------')
#        return roc_auc_gradientboosting
    
        #Check for overfitting - check ROC-AUC on training data
#        y_pred_train = grid_gradientboost.predict(X_train)
#        roc_auc_gboost_train = roc_auc_score(y_train, y_pred_train)
    
#===========================SVM: Linear Support Vector=========================
    def linearSVC(self, ):
    
        param_grid_linearsvc = dict(penalty = ['l1', 'l2'],
                                    loss  = ['hinge', 'squared_hinge'],
                                    dual = [False]     #      for n_samples > n_features
                                    )     
        
        linearsvc = LinearSVC()
        
        grid_linearsvc = GridSearchCV(estimator=linearsvc, 
                                    param_grid=param_grid_linearsvc, 
                                    cv = 3, 
                                    n_jobs=-1,
                                    error_score=0.0,
                                    iid=False)
        
        grid_linearsvc.fit(self.X_train, self.y_train)
        
        #Predict the test set results 
        y_pred = grid_linearsvc.predict(self.X_test)
        
        #Performance: AUC-ROC
        roc_auc_linearsvc = roc_auc_score(self.y_test, y_pred)
        print('Linear SVC completed: ROC-AUC: {}'.format(roc_auc_linearsvc)+'\n' + '------------------------------------------------------------------')

        #Performance: AUC-PRC
        auc_prc_linearsvc = average_precision_score(self.y_test, y_pred, average='weighted')
        print('Linear SVC completed: AUC-PRC: {}'.format(auc_prc_linearsvc)+'\n' + '------------------------------------------------------------------')
        
        #Performance: F1 Metric
        f1_linearsvc= f1_score(self.y_test, y_pred, average='weighted')
        print('Linear SVC completed: F1 metric: {}'.format(f1_linearsvc)+'\n' + '------------------------------------------------------------------')

        
        return roc_auc_linearsvc, auc_prc_linearsvc, f1_linearsvc
        
        #Check for overfitting - check ROC-AUC on training data
        #y_pred_train = grid_linearsvc.predict(X_train)
        #roc_auc_linearsvc_train = roc_auc_score(y_train, y_pred_train)
    
#======================Linear Discriminant Analysis============================
    def LDA(self,):   
        
        param_grid_LDA = dict(solver = ['svd', 'lsqr', 'eigen'])     
        
        LDA = LinearDiscriminantAnalysis()
        
        grid_LDA = GridSearchCV(estimator=LDA, 
                                param_grid=param_grid_LDA, 
                                cv = 3, 
                                n_jobs=-1,
                                error_score=0.0, 
                                iid=False)
        
        grid_LDA.fit(self.X_train,self.y_train)
        
        #Predict the test set results 
        y_pred = grid_LDA.predict(self.X_test)
        
        #Performance: AUC-ROC
        roc_auc_LDA = roc_auc_score(self.y_test, y_pred)
        print('LDA completed: ROC-AUC: {}'.format(roc_auc_LDA)+'\n' + '------------------------------------------------------------------')

        #Performance: AUC-PRC
        auc_prc_LDA = average_precision_score(self.y_test, y_pred, average='weighted')
        print('LDA completed: AUC-PRC: {}'.format(auc_prc_LDA)+'\n' + '------------------------------------------------------------------')
        
        #Performance: F1 Metric
        f1_LDA = f1_score(self.y_test, y_pred, average='weighted')
        print('LDA completed: F1 metric: {}'.format(f1_LDA)+'\n' + '------------------------------------------------------------------')

        return roc_auc_LDA, auc_prc_LDA, f1_LDA
        
        #Check for overfitting - check ROC-AUC on training data
        #y_pred_train = grid_LDA.predict(X_train)
        #roc_auc_LDA_train = roc_auc_score(y_train, y_pred_train)
    
#===================Quadratic Discriminant Analysis============================
    def QDA(self, ):    
        
        QDA = QuadraticDiscriminantAnalysis()
        
        QDA.fit(self.X_train, self.y_train)
        
        #Predict the test set results 
        y_pred = QDA.predict(self.X_test)
        
        #Performance: AUC-ROC
        roc_auc_QDA = roc_auc_score(self.y_test, y_pred)
        print('QDA completed: ROC-AUC: {}'.format(roc_auc_QDA)+'\n' + '------------------------------------------------------------------')

        #Performance: AUC-PRC
        auc_prc_QDA = average_precision_score(self.y_test, y_pred, average='weighted')
        print('QDA completed: AUC-PRC: {}'.format(auc_prc_QDA)+'\n' + '------------------------------------------------------------------')
        
        #Performance: F1 Metric
        f1_QDA = f1_score(self.y_test, y_pred, average='weighted')
        print('QDA completed: F1 metric: {}'.format(f1_QDA)+'\n' + '------------------------------------------------------------------')
        
        return roc_auc_QDA, auc_prc_QDA, f1_QDA
        
        #Check for overfitting - check ROC-AUC on training data
        #y_pred_train = QDA.predict(X_train)
        #roc_auc_QDA_train = roc_auc_score(y_train, y_pred_train)
    
#======================Gaussian Naive Bayes====================================
    def gaussianNB(self, ):
    
        gaussianNB = GaussianNB()
        
        gaussianNB.fit(self.X_train, self.y_train)
        
        #Predict the test set results 
        y_pred = gaussianNB.predict(self.X_test)
        
        #Performance: AUC-ROC
        roc_auc_gaussianNB = roc_auc_score(self.y_test, y_pred)
        print('GaussianNB completed: ROC-AUC: {}'.format(roc_auc_gaussianNB)+'\n' + '------------------------------------------------------------------')

        #Performance: AUC-PRC
        auc_prc_gaussianNB = average_precision_score(self.y_test, y_pred, average='weighted')
        print('GaussianNB completed: AUC-PRC: {}'.format(auc_prc_gaussianNB)+'\n' + '------------------------------------------------------------------')
        
        #Performance: F1 Metric
        f1_gaussianNB = f1_score(self.y_test, y_pred, average='weighted')
        print('QDA completed: F1 metric: {}'.format(f1_gaussianNB)+'\n' + '------------------------------------------------------------------')
        
        return roc_auc_gaussianNB, auc_prc_gaussianNB, f1_gaussianNB
        
        #Check for overfitting - check ROC-AUC on training data
        #y_pred_train = gaussianNB.predict(X_train)
        #roc_auc_gaussianNB_train = roc_auc_score(y_train, y_pred_train)
    
#======================Bernoulli Naive Bayes===================================
    def bernoulliNB(self, ):
        
        #Designed for binary features (not the case for this dataset)
        bernoulliNB = BernoulliNB() 
        
        bernoulliNB.fit(self.X_train, self.y_train)
        
        #Predict the test set results 
        y_pred = bernoulliNB.predict(self.X_test)
        
        #Performance: AUC-ROC
        roc_auc_bernoulliNB = roc_auc_score(self.y_test, y_pred)
        print('BernoulliNB completed: ROC-AUC: {}'.format(roc_auc_bernoulliNB)+'\n' + '------------------------------------------------------------------')

        #Performance: AUC-PRC
        auc_prc_bernoulliNB = average_precision_score(self.y_test, y_pred, average='weighted')
        print('BernoulliNB completed: AUC-PRC: {}'.format(auc_prc_bernoulliNB)+'\n' + '------------------------------------------------------------------')
        
        #Performance: F1 Metric
        f1_bernoulliNB = f1_score(self.y_test, y_pred, average='weighted')
        print('BernoulliNB completed: F1 metric: {}'.format(f1_bernoulliNB)+'\n' + '------------------------------------------------------------------')
        
        return roc_auc_bernoulliNB, auc_prc_bernoulliNB, f1_bernoulliNB
        
        #Check for overfitting - check ROC-AUC on training data
        #y_pred_train = bernoulliNB.predict(X_train)
        #roc_auc_bernoulliNB_train = roc_auc_score(y_train, y_pred_train)
    
#======================Multinomial Naive Bayes=================================
#    def multinomialNB(self, ): 
#        
#        #Suitable for classification with discrete features 
#        multinomialNB = MultinomialNB() 
#        
#        multinomialNB.fit(self.X_train, self.y_train)
#        
#        #Predict the test set results 
#        y_pred = multinomialNB.predict(self.X_test)
#        
#        #Performance
#        roc_auc_multiNB = roc_auc_score(self.y_test, y_pred)
#        print('MultinomialNB completed: ROC-AUC: {}'.format(roc_auc_multiNB)+'\n' + '------------------------------------------------------------------')
#        return roc_auc_multiNB
#        
#        #Check for overfitting - check ROC-AUC on training data
#        #y_pred_train = multinomialNB.predict(X_train)
#        #roc_auc_multinomialNB_train = roc_auc_score(y_train, y_pred_train)
    
    
#================================Bagging=======================================
    def bagging(self, ):
        
        param_grid_bagging = dict(n_estimators = [10,20,30,40,50])     
        
        bagging = BaggingClassifier()
        
        grid_bagging = GridSearchCV(estimator=bagging, 
                                    param_grid=param_grid_bagging, 
                                    cv = 3, 
                                    n_jobs=-1,
                                    error_score=0.0,
                                    iid=False)
        
        grid_bagging.fit(self.X_train, self.y_train)
        
        #Predict the test set results 
        y_pred = grid_bagging.predict(self.X_test)
        
        #Performance: AUC-ROC
        roc_auc_bagging = roc_auc_score(self.y_test, y_pred)
        print('Bagging completed: ROC-AUC: {}'.format(roc_auc_bagging)+'\n' + '------------------------------------------------------------------')


        #Performance: AUC-PRC
        auc_prc_bagging = average_precision_score(self.y_test, y_pred, average='weighted')
        print('Bagging completed: AUC-PRC: {}'.format(auc_prc_bagging)+'\n' + '------------------------------------------------------------------')
        
        #Performance: F1 Metric
        f1_bagging = f1_score(self.y_test, y_pred, average='weighted')
        print('Bagging completed: F1 metric: {}'.format(f1_bagging)+'\n' + '------------------------------------------------------------------')
        
        return roc_auc_bagging, auc_prc_bagging, f1_bagging
        
        #Check for overfitting - check ROC-AUC on training data
        #y_pred_train = grid_bagging.predict(X_train)
        #roc_auc_bagging_train = roc_auc_score(y_train, y_pred_train)

#================================EVALUATION====================================  
    def evaluate_metrics(self):
        
        logisticReg = self.logisticReg()
        decisionTree = self.decisionTree()
        randomForest = self.randomForest()
        perceptron = self.perceptron()
        MLP = self.MLP()
        ADAboost = self.ADAboost()
        linearSVC = self.linearSVC()
        LDA = self.LDA()
        QDA = self.QDA()
        gaussianNB = self.gaussianNB()
        bernoulliNB = self.bernoulliNB()
        bagging = self.bagging()
        
        results_roc = {'auc_roc_logit' : logisticReg[0],
                       'auc_roc_dtree' : decisionTree[0],
                       'auc_roc_rforest' : randomForest[0],
                       'auc_roc_perceptron' : perceptron[0],
                       'auc_roc_MLP' : MLP[0],
                       'auc_roc_ADAboost' : ADAboost[0],
                       #'auc_roc_Gradientboost': self.gradientBoost(),
                       'auc_roc_linearSVC' : linearSVC[0],
                       'auc_roc_LDA' : LDA[0],
                       'auc_roc_QDA' : QDA[0],
                       'auc_roc_gaussianNB' : gaussianNB[0],
                       'auc_roc_bernoulliNB' : bernoulliNB[0],
                       'auc_roc_bagging' : bagging[0]
                       }
        
        results_prc = {'auc_prc_logit' : logisticReg[1],
                       'auc_prc_dtree' : decisionTree[1],
                       'auc_prc_rforest' : randomForest[1],
                       'auc_prc_perceptron' : perceptron[1],
                       'auc_prc_MLP' : MLP[1],
                       'auc_prc_ADAboost' : ADAboost[1],
                       'auc_prc_linearSVC' : linearSVC[1],
                       'auc_prc_LDA' : LDA[1],
                       'auc_prc_QDA' : QDA[1],
                       'auc_prc_gaussianNB' : gaussianNB[1],
                       'auc_prc_bernoulliNB' : bernoulliNB[1],
                       'auc_prc_bagging' : bagging[1]
                       }
        
        results_f1 = {'auc_f1_logit' : logisticReg[2],
                       'auc_f1_dtree' : decisionTree[2],
                       'auc_f1_rforest' : randomForest[2],
                       'auc_f1_perceptron' : perceptron[2],
                       'auc_f1_MLP' : MLP[2],
                       'auc_f1_ADAboost' : ADAboost[2],
                       'auc_f1_linearSVC' : linearSVC[2],
                       'auc_f1_LDA' : LDA[2],
                       'auc_f1_QDA' : QDA[2],
                       'auc_f1_gaussianNB' : gaussianNB[2],
                       'auc_f1_bernoulliNB' : bernoulliNB[2],
                       'auc_f1_bagging' : bagging[2]
                       }
        
        return results_roc, results_prc, results_f1 



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

