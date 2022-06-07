import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from collections import Counter

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        # When an instance is created this initializes it
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    
    def is_leaf(self):
        # function that determines if it is a leaf node
        return self.value is not None
    

class DecisionTreeModel:

    def __init__(self, max_depth=100, criterion = 'gini', min_samples_split=2, impurity_stopping_threshold = 1):
        # When an instance is created this initializes it
        self.max_depth = max_depth
        self.criterion = criterion
        self.min_samples_split = min_samples_split
        self.impurity_stopping_threshold = impurity_stopping_threshold
        self.root = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        # Converts X and Y to numpy arrays
        # calls _fit
        X = X.to_numpy()
        y = y.to_numpy() 
        return self._fit(X, y)
        


    def predict(self, X: pd.DataFrame):
        # Converts X to a numpy array
        # calls _predict 
        X = X.to_numpy()
        return self._predict(X)
        
        

        
    
        
    def _fit(self, X, y):
        # This function is called to train the decision tree model.
        self.root = self._build_tree(X, y)
        
    def _predict(self, X):
        # This function returns the predictions in an array
        predictions = [self._traverse_tree(x, self.root) for x in X]
        return np.array(predictions)    
        
    def _is_finished(self, depth):
        # Checks if the stopping criteria (in this case max depth) has been reached.
        if (depth >= self.max_depth
            or self.n_class_labels == 1
            or self.n_samples < self.min_samples_split):
            return True
        # end TODO
        return False
    
    def _is_homogenous_enough(self):
        # TODO: for graduate students only
        result = False
        # end TODO
        return result
                              
    def _build_tree(self, X, y, depth=0):
        # This function builds the tree with data provided
        self.n_samples, self.n_features = X.shape
        self.n_class_labels = len(np.unique(y))

        # stopping criteria
        if self._is_finished(depth):
            most_common_Label = np.argmax(np.bincount(y))
            return Node(value=most_common_Label)

        # get best split
        rnd_feats = np.random.choice(self.n_features, self.n_features, replace=False)
        best_feat, best_thresh = self._best_split(X, y, rnd_feats)
        # grow children recursively
        left_idx, right_idx = self._create_split(X[:, best_feat], best_thresh)
        left_child = self._build_tree(X[left_idx, :], y[left_idx], depth + 1)
        right_child = self._build_tree(X[right_idx, :], y[right_idx], depth + 1)

        return Node(best_feat, best_thresh, left_child, right_child)
    

    def _gini(self, y):
        '''
        Calculates the gini index which is defined as summation(p(i) * (1 - p(i))
        Measurement of how pure information in a data set is
        '''
        proportions = np.bincount(y) / len(y)
        gini = np.sum([p * (1 - p) for p in proportions if p > 0])
        return gini
    
    def _entropy(self, y):
        '''
        Calculates the entropy or average level of uncertainty or information.
        Formula is sum(p(i)log(p(i)))
        '''

        proportions = np.bincount(pd.factorize(y)[0]) / len(y)
        entropy = -np.sum([p * np.log2(p) for p in proportions if p > 0])

        # end TODO
        return entropy
        
    def _create_split(self, X, thresh):
        left_idx = np.argwhere(X <= thresh).flatten()
        right_idx = np.argwhere(X > thresh).flatten()
        return left_idx, right_idx

    def _information_gain(self, X, y, thresh):
        # TODO: fix the code so it can switch between the two criterion: gini and entropy 

        if self.criterion == 'gini':
            # Information gain here is the difference in impurity before and after the split.
            # We want to minimize the gini impurity at the leaf nodes    
            parent_loss = self._gini(y)
            left_idx, right_idx = self._create_split(X, thresh)
            n, n_left, n_right = len(y), len(left_idx), len(right_idx)

            if n_left == 0 or n_right == 0: 
                return 0
            
            child_loss = (n_left / n) * self._gini(y[left_idx]) + (n_right / n) * self._gini(y[right_idx])
            result = parent_loss - child_loss
        else:
            # Information gain here is the difference in entropy before and after the split.
            parent_loss = self._entropy(y)
            left_idx, right_idx = self._create_split(X, thresh)
            n, n_left, n_right = len(y), len(left_idx), len(right_idx)
            
            if n_left == 0 or n_right == 0: 
                return 0
            
            child_loss = (n_left / n) * self._entropy(y[left_idx]) + (n_right / n) * self._entropy(y[right_idx])
            result = parent_loss - child_loss
        # end TODO
        return result
       
    def _best_split(self, X, y, features):
        '''
        Figures out which feature and threshold value produces the split with 
        the highest information gain by looping through every unique threshold
        for each feature. The best combination is returned as a tuple
        '''
        split = {'score':- 1, 'feat': None, 'thresh': None}

        for feat in features:
            X_feat = X[:, feat]
            thresholds = np.unique(X_feat)
            for thresh in thresholds:
                score = self._information_gain(X_feat, y, thresh)

                if score > split['score']:
                    split['score'] = score
                    split['feat'] = feat
                    split['thresh'] = thresh

        return split['feat'], split['thresh']
    
    def _traverse_tree(self, x, node):
        '''
        Recursion function that when called will end up 
        returning a prediction after a tree traversal.
        Compares the node feature and threshold values to the current sample's values to dermine if
        it should travel down the righ or left node.
        When a leaf node is reached the most common class label is returned.

        '''
        if node.is_leaf():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


class RandomForestModel(object):

    def __init__(self, n_estimators):
        # When an instance is created this initializes it
        self.n_estimators = n_estimators
        self.trees = []
        




    def fit(self, X: pd.DataFrame, y: pd.Series):
        '''
        This function is called to train the random forest model. It creates n_estimator
        decision trees and trains them. Then it saves them to an array for later.
        ''' 
        X = X.to_numpy()
        y = y.to_numpy()
        counter = 0
        while counter < self.n_estimators:
            
            clf = DecisionTreeModel(max_depth=10)
            # Create random samples with replacement for bagging
            n_samples = X.shape[0]
            random_samples = np.random.choice(n_samples, n_samples, replace=True)
            
            tempx, tempy = X[random_samples], y[random_samples]
            
            # train current decision tree
            clf._fit(tempx, tempy)
            # save to array trees 
            self.trees.append(clf)
            counter += 1
             

        


    def predict(self, X: pd.DataFrame):
        # goes through the predictions for all the trees in the forest and then predicts 
        # the most common class label among them
        predictions = []
        for tree in self.trees:
            predictions.append(tree.predict(X))

        predictions = np.swapaxes(predictions, 0, 1)
        y_preds = []

        for pred in predictions:
            c = Counter(pred)
            y_preds.append(c.most_common(1)[0][0])
        return y_preds
       

    

def accuracy_score(y_true, y_pred):
    # calculates accuracy scores
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    
    return accuracy

def classification_report(y_test, y_pred):
    # Calculates precision, recall, f1-score
    cm = confusion_matrix(y_test, y_pred)
    tp = cm[0][0]
    fp = cm[0][1]
    fn = cm[1][0]
 
    result = {}
    
    # Precision: tp / (tp + fp)
    prec = tp / (tp + fp)
    result["Precision"] = prec
    # Recall: tp / (tp + fn)
    recall = tp / (tp + fn)
    result["Recall"] = recall
    # F1-Score 2((precision * Recall)/(Precision + Recall))
    f1 = (2 * (prec * recall)/(prec + recall))
    result["F1-Score"] = f1


    # end TODO
    return (result)

def confusion_matrix(y_test, y_pred):
    # returns the 2x2 confusion matrix
    '''
    ---------
    0,0 both yes   0,1 predicted yes, actual no
    1,0 predicted no, actual yes   1,1 both no
    0 means malignent. 1 means benign
    ---------
    '''
    result = np.array([[0, 0], [0, 0]])
    y_test = y_test.to_numpy()

    for i in range(len(y_test)):
        if y_test[i] == 0 and y_pred[i] == 0:
            result[0][0] += 1
        elif y_test[i] == 1 and y_pred[i] == 1:
            result[1][1] += 1
        elif y_test[i] == 1 and y_pred[i] == 0:
            result[0][1] += 1
        elif y_test[i] == 0 and y_pred[i] == 1:
            result[1][0] += 1
    
    
    return(result)


def _test():
    
    df = pd.read_csv('breast_cancer.csv')
    
    #X = df.drop(['diagnosis'], axis=1).to_numpy()
    #y = df['diagnosis'].apply(lambda x: 0 if x == 'M' else 1).to_numpy()

    X = df.drop(['diagnosis'], axis=1)
    y = df['diagnosis'].apply(lambda x: 0 if x == 'M' else 1)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=1
    )

    clf = DecisionTreeModel(max_depth=10, criterion='entroy')
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    

    print("Accuracy:", acc)
    print("Confusion Matrix:", cm)


    model = RandomForestModel(n_estimators=100)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(acc)
    
if __name__ == "__main__":
    _test()
