import math
from operator import add

#It will build a decision stump based on Information gain.
class DecisionStump:
    def __init__(self, traindata, w):
        self.traindata = traindata
        self.w = w
        self.createDict()
        self.entropyparent = self.entropy(self.positivecount, self.negativcount, self.positivecount + self.negativcount)
        self.makeDecision()

    def createDict(self):
        self.dictp = {}
        self.dictn = {}
        self.dict = {}
        self.totalcount = 0
        self.positivecount = 0
        self.negativcount = 0
        self.probailtyprediction = []

        for i in range(8):
            self.dictp[i] = {}
            self.dictn[i] = {}
            self.dict[i] = []

        for k in self.traindata:
            #I am using game_codedata set, if you want to use game_attrdata set,
            # then change k[-1] == 1 to k[-1] == 'SetterofCatan'
            if k[-1] == 1:
                self.positivecount += self.w[self.totalcount]
                self.addToDict(self.dictp, k, self.w[self.totalcount])
            else:
                self.negativcount += self.w[self.totalcount]
                self.addToDict(self.dictn, k, self.w[self.totalcount])
            self.totalcount += 1

        for i in range(8):
            if set(self.dictp[i].keys()) == set(self.dictn[i].keys()):
                self.dict[i] = self.dictp[i].keys()

            else:
                self.dict[i] = list(set(self.dictp[i].keys()) | set(self.dictn[i].keys()))

    def addToDict(self, dict, value, weight):
        for i in range(8):
            if value[i] in dict[i]:
                dict[i][value[i]] += weight
            else:
                dict[i][value[i]] = weight

    def entropy(self, pos, neg, total):
        if pos == 0:
            return neg / float(total) * math.log(total / float(neg), 2)
        elif neg == 0:
            return pos / float(total) * math.log(total / float(pos), 2)
        else:
            return (pos / float(total) * math.log(total / float(pos), 2)) + (
            neg / float(total) * math.log(total / float(neg), 2))

    def makeDecision(self):
        maxig = 0
        self.node = 0

        for i in range(8):
            ig = self.entropyparent
            for j in self.dict[i]:
                if j not in self.dictp[i]:
                    dp = 0
                else:
                    dp = self.dictp[i][j]
                if j not in self.dictn[i]:
                    dn = 0
                else:
                    dn = self.dictn[i][j]
                ig -= (dp + dn) * self.entropy(dp, dn, dp + dn)
            if ig > maxig:
                maxig = ig
                self.node = i
        self.makePred()

    def makePred(self):
        self.pred = []

        for i in self.dict[self.node]:
            if i not in self.dictp[self.node]:
                dp = 0
            else:
                dp = self.dictp[self.node][i]

            if i not in self.dictn[self.node]:
                dn = 0
            else:
                dn = self.dictn[self.node][i]
            if dp > dn:
                # I am using game_codedata set, if you want to use game_attrdata set,
                # then change self.pred.append(1) toself.pred.append('SetterofCatan')
                self.pred.append(1)
                self.probailtyprediction.append(dp/(dp+dn))
            else:
                # I am using game_codedata set, if you want to use game_attrdata set,
                # then change self.pred.append(-1) toself.pred.append('AppleTOApples')
                self.pred.append(-1)
                self.probailtyprediction.append(dn/(dp+dn))

    def test(self, testdata):
        prediction = []
        for k in testdata:
            prediction.append(self.pred[k[self.node]])

        return prediction

#It will create the decision tree based on information gain and decsion stump algorithm above.
#I will recursivel call the decision stump till there is no information gain to build the decision tree.
class DecisionTree:
    def __init__(self, traindata, w):
        self.traindata = traindata
        self.w = w
        self.createDecisionTree(traindata, w)

    def createDecisionTree(self, traindata, w):
        self.decisionTree = {}
        ds = DecisionStump(traindata, w)
        self.splitDataset(traindata, ds.node, ds.dict[ds.node])
        self.decisionTree[ds.node] = {}
        for i in range(len(ds.dict[ds.node])):
            if len(self.c[i].keys()) > 1:
                self.decisionTree[ds.node][i] = DecisionTree(self.splitTrainData[i], w).decisionTree
            else:
                self.decisionTree[ds.node][i] = self.splitTrainData[i][0][-1]

    def splitDataset(self, traindata, node, list):
        self.splitTrainData = [[] for i in range(len(list))]
        self.c = [{} for i in range(len(list))]
        for i in traindata:
            self.splitTrainData[i[node]].append(i)
            if i[-1] in self.c[i[node]]:
                self.c[i[node]][i[-1]] += 1
            else:
                self.c[i[node]][i[-1]] = 1

    def test(self, testdata):
        prediction = []
        for i in testdata:
            key = 0
            dt = self.decisionTree
            while (isinstance(dt, dict)):
                key = dt.keys()[0]
                dt = dt[key][i[key]]

            prediction.append(dt)

        return prediction


class AdaBoost:
    def __init__(self, traindata, learner, k):
        self.traindata = traindata
        self.learner = learner
        self.k = k
        self.N = len(traindata)
        self.w = [(1 / float(self.N)) for i in range(self.N)]
        self.h = []
        self.z = []
        self.train()

    #Training algo for Adaboost, based on the algorithm given.
    #If error for any iteration is 0, then I will stop the
    # adaboost algorithm assign weight to that hypothesis as 1.
    def train(self):
        for i in range(self.k):
            if self.learner == 'ds':
                ds = DecisionStump(self.traindata, self.w)
                self.h.append(ds)
            elif self.learner == 'dt':
                dt = DecisionTree(self.traindata, self.w)
                self.h.append(dt)

            error = 0
            g = self.h[i].test(traindata)
            for j in range(self.N):
                if g[j] != self.traindata[j][-1]:
                    error += self.w[j]

            if error == 0:
                self.z.append(1)
                return
            for j in range(self.N):
                if g[j] == self.traindata[j][-1]:
                    self.w[j] *= error / (1 - error)
            s = sum(self.w)
            self.w = [float(i) / s for i in self.w]
            self.z.append(math.log((1 - error) / error))

    def test(self, testdata):
        d = [0 for i in range(len(testdata))]
        for i in range(len(self.h)):
            def mul(x): return self.z[i] * x

            d = map(add, d, map(mul, self.h[i].test(testdata)))

        prediction = []
        for i in d:
            if i > 0:
                prediction.append(1)
            else:
                prediction.append(-1)

        return prediction


if __name__ == "__main__":
    train = open("game_codedata_train.dat", "r")
    test = open("game_codedata_test.dat", "r")

    traindata = []
    testdata = []
    for line in train:
        k = map(int, line.strip().split(","))
        traindata.append(k)

    for line in test:
        k = map(int, line.strip().split(","))
        testdata.append(k)

    N = len(traindata)
    w = [(1 / float(N)) for i in range(N)]

    #Select the classifier you want, decision stump, decsion tree or adaboost.
    #For adaboost you can give 'ds' to use decision stump as learner or dt to use decision tree as learner.
    #The third variable is the number of ensembles you want.
    '''ds = DecisionStump(traindata,w)
    prediction = ds.test(testdata)
    accuracy = 0
    totacount = 0
    avg = 0
    for i in testdata:
        if i[-1] == prediction[totacount]:
            accuracy += 1
            avg += ds.probailtyprediction[i[ds.node]]
        else:
            avg += (1-ds.probailtyprediction[i[ds.node]])
        totacount += 1

    print accuracy
    print "Average probabilty assigned to the correct class across the test set: ",avg/len(testdata)

    dt = DecisionTree(traindata,w)
    print dt.decisionTree'''

    ad = AdaBoost(traindata, 'ds', 2)
    prediction = ad.test(testdata)

    accuracy = 0
    totacount = 0
    for i in testdata:
        if i[-1] == prediction[totacount]:
            accuracy += 1
        totacount += 1

    print accuracy/len(testdata)
