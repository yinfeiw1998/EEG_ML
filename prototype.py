import features
import numpy as np
from sklearn import svm
from sklearn.svm import SVR
from sklearn import metrics
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

from sklearn import preprocessing
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge


#Open the file that contains feature data and read it
data = []

f = open("features(with ica).txt", "r")

while True:
    line = f.readline()
    row = []
    
    if not line:
        break
    
    if(line == "id | rms | ApprEnt | LZComp | mpf | sef\n"):
        #print("found")
        continue
    else:
        for word in line.split():
            try:
                row.append(float(word))
            except ValueError:
                pass

        #print(row)
        data.append(row)
    

f.close()


#Split feature data into features and identity. The identity numbers are our desired results BASED off of the eeg features.
data_np = np.array(data)
cols = len(data_np[0])
rows = len(data_np)

result = []

#A user logs in by submitting both an id and password. The same logic applies here, except the password is a user's eeg features
#The program looks through stored data and gathers multiple eeg features of both those that are and are not associated with
#the id specified by the user. This will allow the linear regression model to have several examples of what IS and ISN'T the desired
#user account. To simplify things for the linear regression model, positive match examples are temporarily given ids of 100 while negative ones
#are given ids of 0.
desired_id = 8
submitted_data_index = 54

for i in range(0, rows):
    
    if(data_np[i][0] != desired_id): 
        data_np[i][0] = 0   #negative matching eeg features must be given low ID values.
    else:
        data_np[i][0] = 100 #positive matching eeg features must be given high ID values.(explained on line 94 comments)
    
    result.append(data_np[i][0])


train = data_np[0:rows,1:31]

#For the sake of testing this program, the "train_test_split" function is used so that only SOME of
#the eeg features stored within "features(with_ica).txt" are used to train the linear regression model
#That way, the unused features can be used to test the model to see what it predicts.
X_train, X_test, y_train, y_test = train_test_split(train, result, test_size = 0.2, random_state=2, stratify=result)

knn = linear_model.LinearRegression()

# Fit the classifier to the training data
knn.fit(X_train, y_train)

# Print the accuracy
print(knn.score(X_test, y_test))
#print(data_np)

#knn.predict only works with 2d arrays. In this case, the test array is TECHNICALLY a 2d array
#because it is an array that contains an array.
test = []
test.append(data_np[submitted_data_index][1:31])
prediction = knn.predict(test)

#print out prediction and correct result. Although the accuracy on its own is not very high,
#notice how all features associated with the desired user are predicted to have MUCH higher ID values
#compared to the predicted ID values of features NOT associated with the desired ID.
#This means a threshold can be established, wherein as long as the linear regression
#model predicts a high enough ID value from a given set of eeg features, then those features
#are a POSITIVE match with the ID specified by the user. Hence, this is why high ID values are used,
#so that it is easy to establish a threshold
threshold = 40

print(prediction)
print(data_np[submitted_data_index][0])
if(prediction[0] > threshold):
    print("The \"password\" provided was CORRECT (the provided eeg features match with the specified ID)")
else:
    print("The \"password\" provided was INCORRECT (the provided eeg features match with the specified ID)")
    
#This is the accuracy test redone when factoring in the threshold.
prediction = knn.predict(X_test)
correct_predictions = 0
length = len(prediction)
for i in range(0, length):
    if(prediction[i] > threshold):
        prediction[i] = 1
    else:
        prediction[i] = 0
        
    if(y_test[i] != 0):
        y_test[i] = 1
    
    if(prediction[i] == y_test[i]):
        correct_predictions = correct_predictions+1
        
print(correct_predictions/length)

'''
Judge which features seemed to contribute the most
'''

'''
chi-squared statistical test
'''
print("--------------------------------------\n")
# Feature extraction
test = SelectKBest(score_func=chi2, k=3)
fit = test.fit(train, result)

# Summarize scores
np.set_printoptions(precision=3)
print(fit.scores_)

'''
RFE with logistic regression
'''
print("--------------------------------------\n")
model = LogisticRegression()
rfe = RFE(model, 3)
fit = rfe.fit(preprocessing.scale(train), result)
print("Num Features: %s" % (fit.n_features_))
print("Selected Features: %s" % (fit.support_))
print("Feature Ranking: %s" % (fit.ranking_))

'''
Ridge Regression
'''
print("--------------------------------------\n")
ridge = Ridge(alpha=1.0)
ridge.fit(train, result)
# A helper method for pretty-printing the coefficients
def pretty_print_coefs(coefs, names = None, sort = False):
    if names == None:
        names = ["X%s" % x for x in range(len(coefs))]
    lst = zip(coefs, names)
    if sort:
        lst = sorted(lst,  key = lambda x:-np.abs(x[0]))
    return " + ".join("%s * %s" % (round(coef, 3), name)
                                   for coef, name in lst)
print ("Ridge model:", pretty_print_coefs(ridge.coef_))


    

'''


#This code was used to first initialize the eeg data (ica preprocessing and feature extraction) and then store it in features.txt

learned_data1 = features.features('101/101')
learned_data2 = features.features('102/102')
learned_data3 = features.features('153/153')
learned_data4 = features.features('208/208')
learned_data5 = features.features('244/244')
learned_data6 = features.features('245/245')
learned_data7 = features.features('340/340')
learned_data8 = features.features('399/399')
learned_data9 = features.features('424/424')
learned_data10 = features.features('440/440')
learned_data11 = features.features('457/457')
learned_data12 = features.features('488/488')
learned_data13 = features.features('495/495')
learned_data14 = features.features('510/510')
learned_data15 = features.features('556_1/556_1')

learned_data1.extract_features(1, 0, 1000)    #arguments are the id under which to store features under, starting time stamp, and ending time stamp
learned_data1.extract_features(1, 500, 1500)
learned_data1.extract_features(1, 1000, 2000)
learned_data1.extract_features(1, 1500, 2500)
learned_data1.extract_features(1, 2000, 3000)
learned_data1.extract_features(1, 2500, 3500)
learned_data1.extract_features(1, 4000, 5000)

learned_data2.extract_features(2, 0, 1000)
learned_data2.extract_features(2, 500, 1500)
learned_data2.extract_features(2, 1000, 2000)
learned_data2.extract_features(2, 1500, 2500)
learned_data2.extract_features(2, 2000, 3000)
learned_data2.extract_features(2, 2500, 3500)
learned_data2.extract_features(2, 4000, 5000)

learned_data3.extract_features(3, 0, 1000)
learned_data3.extract_features(3, 500, 1500)
learned_data3.extract_features(3, 1000, 2000)
learned_data3.extract_features(3, 1500, 2500)
learned_data3.extract_features(3, 2000, 3000)
learned_data3.extract_features(3, 2500, 3500)
learned_data3.extract_features(3, 4000, 5000)

learned_data4.extract_features(4, 0, 1000)
learned_data4.extract_features(4, 500, 1500)
learned_data4.extract_features(4, 1000, 2000)
learned_data4.extract_features(4, 1500, 2500)
learned_data4.extract_features(4, 2000, 3000)
learned_data4.extract_features(4, 2500, 3500)
learned_data4.extract_features(4, 4000, 5000)

learned_data5.extract_features(5, 0, 1000)
learned_data5.extract_features(5, 500, 1500)
learned_data5.extract_features(5, 1000, 2000)
learned_data5.extract_features(5, 1500, 2500)
learned_data5.extract_features(5, 2000, 3000)
learned_data5.extract_features(5, 2500, 3500)
learned_data5.extract_features(5, 4000, 5000)

learned_data6.extract_features(6, 0, 1000)
learned_data6.extract_features(6, 500, 1500)
learned_data6.extract_features(6, 1000, 2000)
learned_data6.extract_features(6, 1500, 2500)
learned_data6.extract_features(6, 2000, 3000)
learned_data6.extract_features(6, 2500, 3500)
learned_data6.extract_features(6, 4000, 5000)

learned_data7.extract_features(7, 0, 1000)
learned_data7.extract_features(7, 500, 1500)
learned_data7.extract_features(7, 1000, 2000)
learned_data7.extract_features(7, 1500, 2500)
learned_data7.extract_features(7, 2000, 3000)
learned_data7.extract_features(7, 2500, 3500)
learned_data7.extract_features(7, 4000, 5000)

learned_data8.extract_features(8, 0, 1000)
learned_data8.extract_features(8, 500, 1500)
learned_data8.extract_features(8, 1000, 2000)
learned_data8.extract_features(8, 1500, 2500)
learned_data8.extract_features(8, 2000, 3000)
learned_data8.extract_features(8, 2500, 3500)
learned_data8.extract_features(8, 4000, 5000)

learned_data9.extract_features(9, 0, 1000)
learned_data9.extract_features(9, 500, 1500)
learned_data9.extract_features(9, 1000, 2000)
learned_data9.extract_features(9, 1500, 2500)
learned_data9.extract_features(9, 2000, 3000)
learned_data9.extract_features(9, 2500, 3500)
learned_data9.extract_features(9, 4000, 5000)

learned_data10.extract_features(10, 0, 1000)
learned_data10.extract_features(10, 500, 1500)
learned_data10.extract_features(10, 1000, 2000)
learned_data10.extract_features(10, 1500, 2500)
learned_data10.extract_features(10, 2000, 3000)
learned_data10.extract_features(10, 2500, 3500)
learned_data10.extract_features(10, 4000, 5000)

learned_data11.extract_features(11, 0, 1000)
learned_data11.extract_features(11, 500, 1500)
learned_data11.extract_features(11, 1000, 2000)
learned_data11.extract_features(11, 1500, 2500)
learned_data11.extract_features(11, 2000, 3000)
learned_data11.extract_features(11, 2500, 3500)
learned_data11.extract_features(11, 4000, 5000)

learned_data12.extract_features(12, 0, 1000)
learned_data12.extract_features(12, 500, 1500)
learned_data12.extract_features(12, 1000, 2000)
learned_data12.extract_features(12, 1500, 2500)
learned_data12.extract_features(12, 2000, 3000)
learned_data12.extract_features(12, 2500, 3500)
learned_data12.extract_features(12, 4000, 5000)

learned_data13.extract_features(13, 0, 1000)
learned_data13.extract_features(13, 500, 1500)
learned_data13.extract_features(13, 1000, 2000)
learned_data13.extract_features(13, 1500, 2500)
learned_data13.extract_features(13, 2000, 3000)
learned_data13.extract_features(13, 2500, 3500)
learned_data13.extract_features(13, 4000, 5000)

learned_data14.extract_features(14, 0, 1000)
learned_data14.extract_features(14, 500, 1500)
learned_data14.extract_features(14, 1000, 2000)
learned_data14.extract_features(14, 1500, 2500)
learned_data14.extract_features(14, 2000, 3000)
learned_data14.extract_features(14, 2500, 3500)
learned_data14.extract_features(14, 4000, 5000)

learned_data15.extract_features(15, 0, 1000)
learned_data15.extract_features(15, 500, 1500)
learned_data15.extract_features(15, 1000, 2000)
learned_data15.extract_features(15, 1500, 2500)
learned_data15.extract_features(15, 2000, 3000)
learned_data15.extract_features(15, 2500, 3500)
learned_data15.extract_features(15, 4000, 5000)


'''
