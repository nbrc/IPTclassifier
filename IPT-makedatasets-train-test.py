import os, math, fnmatch, librosa, glob, csv, pickle
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import MinMaxScaler
from pysndfx import AudioEffectsChain

sampling = 24000

ACC_ModelMLP= []
ACC_ModelIsolationForest= []
ACC_ModelBagging = []
ACC_ModelLightGBM = []
ACC_ModelAdaBoost = []
ACC_ModelRandomForest = []
ACC_ModelkNN = []
ACC_ModelTree = []
ACC_ModelSVCrbf = []
ACC_ModelSVCpoly = []
ACC_ModelSVClinear = []

# ========================================================================
# 音響分析関数

def get_spectrogram(audioSamples):
    melSpec = librosa.feature.melspectrogram(y=audioSamples, sr=sampling) #valeurs par défaut: n_fft=2048, hop_length=512
    # melSpec = np.log10(melSpec + 1e-9)
    melSpec = librosa.power_to_db(melSpec)

    spectrogram = scale_minmax(melSpec, 0, 1.).astype("float32")

    return spectrogram

# 規格化
def scale_minmax(X, min=0.0, max=1.0):
    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled

# data augmentationするために音響変調
def pitch_shift_sound(x):
	random_number = np.random.random_sample()
	random_number = (random_number * 40) - 20
	random_tuning = 440 + random_number
	change_tuning = librosa.A4_to_tuning(random_tuning)
	return librosa.effects.pitch_shift(x, sr=sampling, n_steps=change_tuning, bins_per_octave=12)

reverb = (
    AudioEffectsChain()
    .reverb()
)

def add_noise(x):
	return x+0.002*np.random.randn(len(x))

# ========================================================================
#　アルゴリズム

def ModelSVCrbf():
	#Create a svm Classifier
	clf = svm.SVC(kernel='rbf') # Linear Kernel

	#Train the model using the training sets
	clf.fit(X_train_2dim, y_train)

	#Predict the response for test dataset
	y_pred = clf.predict(X_test_2dim)

	#Import scikit-learn metrics module for accuracy calculation
	from sklearn import metrics

	# Model Accuracy: how often is the classifier correct?
	print("SVC rbf Accuracy:", metrics.accuracy_score(y_test, y_pred))
	ACC_ModelSVCrbf.append(metrics.accuracy_score(y_test, y_pred))

def ModelSVCpoly():
	#Create a svm Classifier
	clf = svm.SVC(kernel='poly') # Linear Kernel

	#Train the model using the training sets
	clf.fit(X_train_2dim, y_train)

	#Predict the response for test dataset
	y_pred = clf.predict(X_test_2dim)

	#Import scikit-learn metrics module for accuracy calculation
	from sklearn import metrics

	# Model Accuracy: how often is the classifier correct?
	print("SVC poly Accuracy:", metrics.accuracy_score(y_test, y_pred))
	ACC_ModelSVCpoly.append(metrics.accuracy_score(y_test, y_pred))


def ModelSVClinear():
	#Create a svm Classifier
	clf = svm.SVC(kernel='linear') # Linear Kernel

	#Train the model using the training sets
	clf.fit(X_train_2dim, y_train)

	#Predict the response for test dataset
	y_pred = clf.predict(X_test_2dim)

	#Import scikit-learn metrics module for accuracy calculation
	from sklearn import metrics

	# Model Accuracy: how often is the classifier correct?
	print("SVC linear Accuracy:", metrics.accuracy_score(y_test, y_pred))
	ACC_ModelSVClinear.append(metrics.accuracy_score(y_test, y_pred))

def ModelTree():
	#Create a svm Classifier
	clf = tree.DecisionTreeClassifier()

	#Train the model using the training sets
	clf.fit(X_train_2dim, y_train)

	#Predict the response for test dataset
	y_pred = clf.predict(X_test_2dim)

	#Import scikit-learn metrics module for accuracy calculation
	from sklearn import metrics

	# Model Accuracy: how often is the classifier correct?
	print("Decision Tree Accuracy:", metrics.accuracy_score(y_test, y_pred))
	ACC_ModelTree.append(metrics.accuracy_score(y_test, y_pred))

def ModelkNN():
	#Create a svm Classifier
	clf = KNeighborsClassifier(n_neighbors=10)

	#Train the model using the training sets
	clf.fit(X_train_2dim, y_train)

	#Predict the response for test dataset
	y_pred = clf.predict(X_test_2dim)

	#Import scikit-learn metrics module for accuracy calculation
	from sklearn import metrics

	# Model Accuracy: how often is the classifier correct?
	print("kNN Accuracy:", metrics.accuracy_score(y_test, y_pred))
	ACC_ModelkNN.append(metrics.accuracy_score(y_test, y_pred))

def ModelRandomForest():
	#Create a svm Classifier
	clf = RandomForestClassifier(max_depth=22, random_state=0)

	#Train the model using the training sets
	clf.fit(X_train_2dim, y_train)

	#Predict the response for test dataset
	y_pred = clf.predict(X_test_2dim)

	#Import scikit-learn metrics module for accuracy calculation
	from sklearn import metrics

	# Model Accuracy: how often is the classifier correct?
	print("Random Forest Accuracy:", metrics.accuracy_score(y_test, y_pred))
	ACC_ModelRandomForest.append(metrics.accuracy_score(y_test, y_pred))

def ModelAdaBoost():
	#Create a svm Classifier
	clf = AdaBoostClassifier(n_estimators=100)

	#Train the model using the training sets
	clf.fit(X_train_2dim, y_train)

	#Predict the response for test dataset
	y_pred = clf.predict(X_test_2dim)

	#Import scikit-learn metrics module for accuracy calculation
	from sklearn import metrics

	# Model Accuracy: how often is the classifier correct?
	print("AdaBoost Accuracy:", metrics.accuracy_score(y_test, y_pred))
	ACC_ModelAdaBoost.append(metrics.accuracy_score(y_test, y_pred))

def ModelLightGBM():
	#Create a svm Classifier
	clf = HistGradientBoostingClassifier()

	#Train the model using the training sets
	clf.fit(X_train_2dim, y_train)

	#Predict the response for test dataset
	y_pred = clf.predict(X_test_2dim)

	#Import scikit-learn metrics module for accuracy calculation
	from sklearn import metrics

	# Model Accuracy: how often is the classifier correct?
	print("LightGBM Accuracy:", metrics.accuracy_score(y_test, y_pred))
	ACC_ModelLightGBM.append(metrics.accuracy_score(y_test, y_pred))

def ModelBagging():
	#Create a svm Classifier
	clf = BaggingClassifier()

	#Train the model using the training sets
	clf.fit(X_train_2dim, y_train)

	#Predict the response for test dataset
	y_pred = clf.predict(X_test_2dim)

	#Import scikit-learn metrics module for accuracy calculation
	from sklearn import metrics

	# Model Accuracy: how often is the classifier correct?
	print("Bagging Accuracy:", metrics.accuracy_score(y_test, y_pred))
	ACC_ModelBagging.append(metrics.accuracy_score(y_test, y_pred))


def ModelIsolationForest():
	#Create a svm Classifier
	clf = IsolationForest()

	#Train the model using the training sets
	clf.fit(X_train_2dim, y_train)

	#Predict the response for test dataset
	y_pred = clf.predict(X_test_2dim)

	#Import scikit-learn metrics module for accuracy calculation
	from sklearn import metrics

	# Model Accuracy: how often is the classifier correct?
	print("Isolation Forest Accuracy:", metrics.accuracy_score(y_test, y_pred))
	ACC_ModelIsolationForest.append(metrics.accuracy_score(y_test, y_pred))

def ModelMLP():
	#Create a svm Classifier
	clf = MLPClassifier(random_state=0, max_iter=1000)

	#Train the model using the training sets
	clf.fit(X_train_2dim, y_train)

	#Predict the response for test dataset
	y_pred = clf.predict(X_test_2dim)

	#Import scikit-learn metrics module for accuracy calculation
	from sklearn import metrics

	# Model Accuracy: how often is the classifier correct?
	print("MLP Accuracy:", metrics.accuracy_score(y_test, y_pred))
	ACC_ModelMLP.append(metrics.accuracy_score(y_test, y_pred))


def findTechniques(soundFileName):
	for j in range(len(allTechniquesNames)):
	# Vérification si le type existe déjà dans la liste
	#　楽器技術タイプのリストにはタイプ名前を調べる
		if fnmatch.fnmatch(soundFileName, ('*-'+allTechniquesNames[j]+'-*-*')):
			return 0

	# Création d'un nouveau type dans la liste	
	# 楽器技術のリストには新しいタイプを生成する
	if fnmatch.fnmatch(soundFileName, '*-*-*'):
		positions = []
		techniqueName = ''
		# Trouve les bornes où est inscrit le mode de jeu
		for pos, char in enumerate(soundFileName):
			if (char == '-'):
				positions.append(pos)
		a = positions[0]+1
		b = positions[1]
		# copie les charactères entre les bornes définies
		for k in range(a, b):
			techniqueName += soundFileName[k]
			# techniqueName = techniqueName[1:]
		# ajout du type dans la liste des types 
		allTechniquesNames.append(techniqueName)
		print('New technique created:', techniqueName)
	else:
		print('Error occured!')


# ========================================================================
# 実験を開始する

for w in range(10):

	# ========================================================================
	# PRÉPARATION
	# 準備
	print('TURN ', w+1)

	print('... load database ...')
	path = os.path.dirname(__file__)+'/database/'
	allSoundFilesPath = glob.glob(path+'/*/*.wav')
	allSoundFilesName = []

	print('... get file path ...')
	for path in range(len(allSoundFilesPath)):
		positions = []
		soundFileName = ''
		soundFilePath = allSoundFilesPath[path]
		for pos, char in enumerate(soundFilePath):
			if (char == '/'):
				positions.append(pos)
		a = positions[-1]+1
		for i in range(a, len(soundFilePath)):
			soundFileName += soundFilePath[i]
		allSoundFilesName.append(soundFileName)


	# 25% des fichiers seront tirés au sort pour être les données de test
	# テストのデータはデータベースの25パーセントから、ランドムで選択された
	testDataPath = []
	testTechniqueData = []
	print('... separate test/train samples ...')
	for i in range(len(allSoundFilesPath)//4):
		picktestDataPath = np.random.randint(len(allSoundFilesPath))
		testDataPath.append(allSoundFilesPath[picktestDataPath])
		allSoundFilesPath.pop(picktestDataPath)
		testTechniqueData.append(allSoundFilesName[picktestDataPath])
		allSoundFilesName.pop(picktestDataPath)
	print(len(testDataPath), ' samples have been chosen to be test samples.')


	# ========================================================================
	# 学習データ
	# データセットを作る
	# メルスペクトログラムでサンプルを分析する
	samplesArray = []
	print('... get train spectrogram ...')
	for i in range(len(allSoundFilesPath)):
		print(i+1, "/", len(allSoundFilesPath))
		y, srate = librosa.load(allSoundFilesPath[i], sr=sampling, mono=True)
		# supprime silence
		yt, index = librosa.effects.trim(y)
		# prend la première seconde
		# pour avoir 60 frames d'analyses
		nbr_de_samples = math.floor((60/46.875) * sampling)-1
		y1s = yt[0:nbr_de_samples]

		# remplissage par des zéros à la fin
		if len(y1s)!=nbr_de_samples:
			N = (nbr_de_samples-len(y1s))
			y1s = np.concatenate([y1s, np.zeros(N)])
			# print('concat!')
			# np.pad(y1s, (0,N), 'constant', constant_values=(0,0))

		specArray = get_spectrogram(y1s)
		samplesArray.append(specArray)

		pitch_shift_sound(y1s)
		specArray = get_spectrogram(y1s)
		samplesArray.append(specArray)

		reverb(y1s)
		specArray = get_spectrogram(y1s)
		samplesArray.append(specArray)

		add_noise(y1s)
		specArray = get_spectrogram(y1s)
		samplesArray.append(specArray)


	samplesArray = np.asarray(samplesArray)
	print(samplesArray.shape)


	testArray = []
	allTechniquesNames = []
	# 奏法の名前を探す
	print('... get train techniques ...')

	for i in range(len(allSoundFilesName)):
		findTechniques(allSoundFilesName[i])

	# ========================================================================
	# テストデータ
	# data augmentationなし
	testArray = []
	print('... get test spectrogram ...')
	for i in range(len(testDataPath)):
		print(i+1, "/", len(testDataPath))
		y, srate = librosa.load(testDataPath[i], sr=sampling, mono=True)
		# supprime silence
		yt, index = librosa.effects.trim(y)
		# prend la première seconde
		# pour avoir 60 frames d'analyses
		nbr_de_samples = math.floor((60/46.875) * sampling)-1
		y1s = yt[0:nbr_de_samples]

		# remplissage par des zéros à la fin
		if len(y1s)!=nbr_de_samples:
			N = (nbr_de_samples-len(y1s))
			y1s = np.concatenate([y1s, np.zeros(N)])
			# print('concat!')
			# np.pad(y1s, (0,N), 'constant', constant_values=(0,0))

		specTestArray = get_spectrogram(y1s)
		testArray.append(specTestArray)

	samplesArray = np.asarray(samplesArray)
	print(samplesArray.shape)

	print('... get test techniques ...')
	for i in range(len(testTechniqueData)):
		findTechniques(testTechniqueData[i])

	# ========================================================================
	# LABELISATION DES TYPES ET DES INSTRUMENTS
	# 奏法の名前からラベルを作る

	# Créer un dictionaire du format x: i avec une boucle d'énumération (pour garder une trace de l'indice)
	techniquesMapping = {x: i for i, x in enumerate(allTechniquesNames)}
	# Encodage binaire : on créer un array à partir du integer_mapping pour rassembler les nombres entiers
	# On associe une chiffre binaire à chaque nom de types
	techniquesEncoding = [techniquesMapping[techniques] for techniques in allTechniquesNames]
	# On utilise la fonctionnalité du tensorflow.keras qui fait l'encodage one-hot
	# ONE-HOTエンコーディング
	techniqueLabels = to_categorical(techniquesEncoding)

	trainLabels = []
	testLabels = []

	print('... generate train labels ...')
	for i in range(len(allSoundFilesName)):
		for j in range(len(allTechniquesNames)):
			if fnmatch.fnmatch(allSoundFilesName[i], ('*-'+allTechniquesNames[j]+'-*-*')):
				for k in range(4): #　natural spectrogram, 
					trainLabels.append(techniqueLabels[j])

	print('... generate test labels ...')
	for i in range(len(testTechniqueData)):
		for j in range(len(allTechniquesNames)):
			if fnmatch.fnmatch(testTechniqueData[i], ('*-'+allTechniquesNames[j]+'-*-*')):
					testLabels.append(techniqueLabels[j])

	# ========================================================================

	from sklearn.preprocessing import StandardScaler
	from sklearn import svm, tree, ensemble
	from sklearn.svm import SVC
	from sklearn.tree import DecisionTreeClassifier
	from sklearn.neighbors import KNeighborsClassifier
	from sklearn.ensemble import RandomForestClassifier
	from sklearn.ensemble import AdaBoostClassifier
	from sklearn.ensemble import HistGradientBoostingClassifier
	from sklearn.ensemble import BaggingClassifier
	from sklearn.ensemble import IsolationForest
	from sklearn.neural_network import MLPClassifier

	# ------------------------------------------------------------------------
	# PREPROCESSING

	# sequence = int(samples.shape[1])

	# samples = samples.reshape(len(samples),128,44)

	trainLabels = np.argmax(trainLabels, axis=1)
	testLabels = np.argmax(testLabels, axis=1)

	X_train_2dim = np.asarray(samplesArray).reshape(len(samplesArray),-1)
	y_train = np.asarray(trainLabels)
	X_test_2dim = np.asarray(testArray).reshape(len(testArray),-1)
	y_test = np.asarray(testLabels)

	# ------------------------------------------------------------------------
	# CONSTRUCTION DU RÉSEAU
	#Import svm model
	print('... test models ...')
	ModelSVCrbf()
	ModelSVCpoly()
	ModelSVClinear()
	ModelTree()
	ModelkNN()
	ModelRandomForest()

ACC_ModelSVCrbf = np.asarray(ACC_ModelSVCrbf).sum() / len(ACC_ModelSVCrbf)
ACC_ModelSVCpoly = np.asarray(ACC_ModelSVCpoly).sum() / len(ACC_ModelSVCpoly)
ACC_ModelSVClinear = np.asarray(ACC_ModelSVClinear).sum() / len(ACC_ModelSVClinear)
ACC_ModelTree = np.asarray(ACC_ModelTree).sum() / len(ACC_ModelTree)
ACC_ModelkNN = np.asarray(ACC_ModelkNN).sum() / len(ACC_ModelkNN)
ACC_ModelRandomForest = np.asarray(ACC_ModelRandomForest).sum() / len(ACC_ModelRandomForest)
ACC_ModelAdaBoost = np.asarray(ACC_ModelAdaBoost).sum() / len(ACC_ModelAdaBoost)
ACC_ModelLightGBM = np.asarray(ACC_ModelLightGBM).sum() / len(ACC_ModelLightGBM)
# ACC_ModelBagging = np.asarray(ACC_ModelBagging).sum() / len(ACC_ModelBagging)
# ACC_ModelIsolationForest = np.asarray(ACC_ModelIsolationForest).sum() / len(ACC_ModelIsolationForest)
# ACC_ModelMLP = np.asarray(ACC_ModelMLP).sum() / len(ACC_ModelMLP)
# ACC_Dense = np.asarray(ACC_Dense).sum() / len(ACC_Dense)

print("... final accuracy results ...")
print("ACC_ModelSVCrbf: ",ACC_ModelSVCrbf)
print("ACC_ModelSVCpoly: ",ACC_ModelSVCpoly)
print("ACC_ModelSVClinear: ",ACC_ModelSVClinear)
print("ACC_ModelDecisionTree: ",ACC_ModelTree)
print("ACC_ModelkNN: ",ACC_ModelkNN)
print("ACC_ModelRandomForest: ",ACC_ModelRandomForest)
print("ACC_ModelAdaBoost: ",ACC_ModelAdaBoost)
print("ACC_ModelLightGBM: ",ACC_ModelLightGBM)
# print("ACC_ModelBagging: ",ACC_ModelBagging)
# print("ACC_ModelIsolationForest: ",ACC_ModelIsolationForest)
# print("ACC_ModelMLP: ",ACC_ModelMLP)
# print("ACC_Dense: ",ACC_Dense)