from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.svm import LinearSVC
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier
from sklearn.metrics import f1_score, accuracy_score

class classifier_multiclass(object):
    
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
                                max_iter=[100,110,120])
    
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
    
        #Performance: Average accuracy
        accuracy_logit = accuracy_score(self.y_test, y_pred)
        print('Logistic Regression completed: Average accuracy: {}'.format(accuracy_logit)+'\n' + '------------------------------------------------------------------')
        
        #Performance: F1 Metric
        f1_logit = f1_score(self.y_test, y_pred, average='weighted')
        print('Logistic Regression completed: F1 metric: {}'.format(f1_logit)+'\n' + '------------------------------------------------------------------' +'\n' + '------------------------------------------------------------------')
        
        return accuracy_logit,f1_logit
    
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
        
        #Performance: Average accuracy
        accuracy_dtree = accuracy_score(self.y_test, y_pred)
        print('Decision tree completed: Average accuracy: {}'.format(accuracy_dtree)+'\n' + '------------------------------------------------------------------')
        
        #Performance: F1 Metric
        f1_dtree = f1_score(self.y_test, y_pred, average='weighted')
        print('Decision tree completed: F1 metric: {}'.format(f1_dtree)+'\n' + '------------------------------------------------------------------')
        
        return accuracy_dtree, f1_dtree
        
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
        
        #Performance: Average accuracy
        accuracy_rforest = accuracy_score(self.y_test, y_pred)
        print('Random Forest completed: Average accuracy: {}'.format(accuracy_rforest)+'\n' + '------------------------------------------------------------------')

        #Performance: F1 Metric
        f1_rforest = f1_score(self.y_test, y_pred, average='weighted')
        print('Random Forest completed: F1 metric: {}'.format(f1_rforest)+'\n' + '------------------------------------------------------------------')
        
        return accuracy_rforest, f1_rforest
    
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
        
        #Performance: Average accuracy
        accuracy_perceptron = accuracy_score(self.y_test, y_pred)
        print('Perceptron completed: Average accuracy: {}'.format(accuracy_perceptron)+'\n' + '------------------------------------------------------------------')
        
        #Performance: F1 Metric
        f1_perceptron = f1_score(self.y_test, y_pred, average='weighted')
        print('Perceptron completed: F1 metric: {}'.format(f1_perceptron)+'\n' + '------------------------------------------------------------------')
    
        return accuracy_perceptron, f1_perceptron
    
#==========================Multi-layer perceptron==============================
    def MLP(self,):
        MLP = MLPClassifier(max_iter = 1000)
        
        MLP.fit(self.X_train, self.y_train)
        
        #Predict the test set results 
        y_pred = MLP.predict(self.X_test)
        
        #Performance: Average accuracy
        accuracy_MLP = accuracy_score(self.y_test, y_pred)
        print('MLP completed: Average accuracy: {}'.format(accuracy_MLP)+'\n' + '------------------------------------------------------------------')
        
        #Performance: F1 Metric
        f1_MLP= f1_score(self.y_test, y_pred, average='weighted')
        print('MLP completed: F1 metric: {}'.format(f1_MLP)+'\n' + '------------------------------------------------------------------')
        
        return accuracy_MLP, f1_MLP

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
        
        #Performance: Average accuracy
        accuracy_ada = accuracy_score(self.y_test, y_pred)
        print('ADA Boost completed: Average accuracy: {}'.format(accuracy_ada)+'\n' + '------------------------------------------------------------------')
        
        #Performance: F1 Metric
        f1_ada= f1_score(self.y_test, y_pred, average='weighted')
        print('ADA Boost completed: F1 metric: {}'.format(f1_ada)+'\n' + '------------------------------------------------------------------')
        
        return accuracy_ada, f1_ada
    
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
        
        #Performance: Average accuracy
        accuracy_linearsvc = accuracy_score(self.y_test, y_pred)
        print('Linear SVC completed: Average accuracy: {}'.format(accuracy_linearsvc)+'\n' + '------------------------------------------------------------------')
        
        #Performance: F1 Metric
        f1_linearsvc= f1_score(self.y_test, y_pred, average='weighted')
        print('Linear SVC completed: F1 metric: {}'.format(f1_linearsvc)+'\n' + '------------------------------------------------------------------')
        
        return accuracy_linearsvc, f1_linearsvc
    
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
        
        #Performance: Average accuracy
        accuracy_LDA = accuracy_score(self.y_test, y_pred)
        print('LDA completed: Average accuracy: {}'.format(accuracy_LDA)+'\n' + '------------------------------------------------------------------')
        
        #Performance: F1 Metric
        f1_LDA = f1_score(self.y_test, y_pred, average='weighted')
        print('LDA completed: F1 metric: {}'.format(f1_LDA)+'\n' + '------------------------------------------------------------------')

        return accuracy_LDA, f1_LDA
    
#===================Quadratic Discriminant Analysis============================
    def QDA(self, ):    
        
        QDA = QuadraticDiscriminantAnalysis()
        
        QDA.fit(self.X_train, self.y_train)
        
        #Predict the test set results 
        y_pred = QDA.predict(self.X_test)
        
        #Performance: Average accuracy
        accuracy_QDA = accuracy_score(self.y_test, y_pred)
        print('QDA completed: Average accuracy: {}'.format(accuracy_QDA)+'\n' + '------------------------------------------------------------------')

        #Performance: F1 Metric
        f1_QDA = f1_score(self.y_test, y_pred, average='weighted')
        print('QDA completed: F1 metric: {}'.format(f1_QDA)+'\n' + '------------------------------------------------------------------')
        
        return accuracy_QDA, f1_QDA
    
#======================Gaussian Naive Bayes====================================
    def gaussianNB(self, ):
    
        gaussianNB = GaussianNB()
        
        gaussianNB.fit(self.X_train, self.y_train)
        
        #Predict the test set results 
        y_pred = gaussianNB.predict(self.X_test)
        
        #Performance: Average accuracy
        accuracy_gaussianNB = accuracy_score(self.y_test, y_pred)
        print('GaussianNB completed: Average accuracy: {}'.format(accuracy_gaussianNB)+'\n' + '------------------------------------------------------------------')

        #Performance: F1 Metric
        f1_gaussianNB = f1_score(self.y_test, y_pred, average='weighted')
        print('QDA completed: F1 metric: {}'.format(f1_gaussianNB)+'\n' + '------------------------------------------------------------------')
        
        return accuracy_gaussianNB, f1_gaussianNB
    
#======================Bernoulli Naive Bayes===================================
    def bernoulliNB(self, ):
        
        #Designed for binary features (not the case for this dataset)
        bernoulliNB = BernoulliNB() 
        
        bernoulliNB.fit(self.X_train, self.y_train)
        
        #Predict the test set results 
        y_pred = bernoulliNB.predict(self.X_test)
        
        #Performance: Average accuracy
        accuracy_bernoulliNB = accuracy_score(self.y_test, y_pred)
        print('BernoulliNB completed: Average accuracy: {}'.format(accuracy_bernoulliNB)+'\n' + '------------------------------------------------------------------')
        
        #Performance: F1 Metric
        f1_bernoulliNB = f1_score(self.y_test, y_pred, average='weighted')
        print('BernoulliNB completed: F1 metric: {}'.format(f1_bernoulliNB)+'\n' + '------------------------------------------------------------------')
        
        return accuracy_bernoulliNB, f1_bernoulliNB
    
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
        
        #Performance: Average accuracy
        accuracy_bagging = accuracy_score(self.y_test, y_pred)
        print('Bagging completed: Average accuracy: {}'.format(accuracy_bagging)+'\n' + '------------------------------------------------------------------')
        
        #Performance: F1 Metric
        f1_bagging = f1_score(self.y_test, y_pred, average='weighted')
        print('Bagging completed: F1 metric: {}'.format(f1_bagging)+'\n' + '------------------------------------------------------------------')
        
        return accuracy_bagging, f1_bagging

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
        
        results_accuracy = {'accuracy_logit' : logisticReg[0],
                           'accuracy_dtree' : decisionTree[0],
                           'accuracy_rforest' : randomForest[0],
                           'accuracy_perceptron' : perceptron[0],
                           'accuracy_MLP' : MLP[0],
                           'accuracy_ADAboost' : ADAboost[0],
                           'accuracy_linearSVC' : linearSVC[0],
                           'accuracy_LDA' : LDA[0],
                           'accuracy_QDA' : QDA[0],
                           'accuracy_gaussianNB' : gaussianNB[0],
                           'accuracy_bernoulliNB' : bernoulliNB[0],
                           'accuracy_bagging' : bagging[0]
                       }
        
        results_f1 = {'f1_logit' : logisticReg[1],
                       'f1_dtree' : decisionTree[1],
                       'f1_rforest' : randomForest[1],
                       'f1_perceptron' : perceptron[1],
                       'f1_MLP' : MLP[1],
                       'f1_ADAboost' : ADAboost[1],
                       'f1_linearSVC' : linearSVC[1],
                       'f1_LDA' : LDA[1],
                       'f1_QDA' : QDA[1],
                       'f1_gaussianNB' : gaussianNB[1],
                       'f1_bernoulliNB' : bernoulliNB[1],
                       'f1_bagging' : bagging[1]
                       }
        
        return results_accuracy, results_f1 



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

