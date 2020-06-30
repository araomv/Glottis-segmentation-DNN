from keras.models import Sequential
import scipy.io
import numpy
import sys
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Nadam
import h5py
#from  normalize_data import normalize,denormalize
from sklearn.cross_validation import train_test_split
from keras.utils.generic_utils import Progbar
from keras.callbacks import EarlyStopping
import os
from keras.callbacks import TensorBoard
import ntpath
import logging
from keras.callbacks import ModelCheckpoint
import glob, os
cnfg = sys.argv[1];
datafile = sys.argv[2];
mdlfile = sys.argv[3];
pred_path = sys.argv[4];
test_path = sys.argv[5];
#train_flag = int(sys.argv[4]);
newDict = {}

with open(cnfg) as f:
  for line in f:
    line=line.rstrip('\n')
    print line
    splitLine = line.split(':')
    if((splitLine[0]=='indim') | (splitLine[0]=='outdim')|(splitLine[0]=='nbepochs')|(splitLine[0]=='batchsize')| (splitLine[0]=='normalizex')|(splitLine[0]=='normalizey')):
      newDict[(splitLine[0])] = int(splitLine[1].rstrip()) 
    elif(splitLine[0]=='loss'):
      newDict[(splitLine[0])] = (splitLine[1].strip()) 	
    else:
      line = splitLine[1];
      line = line.split(' ')
      newDict[(splitLine[0])] = line
    
#    if(splitLine[0]!='nbepochs'):
	#mdlfile = mdlfile+splitLine[0]+'_'+ str(newDict[(splitLine[0])])

#name=os.path.splitext(datafile)[0]
#C= name.split('/')
#print C[-1],C[-2]
#mdlfile = C[-1]+'_'+C[-2]+'_'+xname+'_'+yname;    
directory='./models/'+mdlfile;    
logging.basicConfig(level=logging.DEBUG)

print directory

os.system('cp '+cnfg+' '+directory+'/nnet_config.txt')

  
print newDict
#nins = newDict['indim']
#nouts = newDict['outdim']
activations = newDict['activation']

hiddenlayers = newDict['hiddenlayers']
hiddenlayers=[x for x in hiddenlayers if(x!='')]


data = scipy.io.loadmat(datafile);
Xtrain = data['train_data']
Ytrain = data['train_lab']

Xval = data['val_data']
Yval = data['val_lab']

Xtest = data['test_data']
Ytest = data['test_lab']


#Ytrain = numpy.log(Ytrain)
print Xtrain.shape
if(newDict['normalizex']==1):
 print "normalizing input data"
 mux = numpy.mean(Xtrain,axis=0);	      
 sigmax= numpy.std(Xtrain,axis=0);
 print mux.shape,sigmax.shape
 sigmax[sigmax<1e-5]=1;
 numpy.savetxt(directory+'/mux.txt',mux);
 numpy.savetxt(directory+'/sigmax.txt',sigmax);
 Xtrain=(Xtrain-mux)/sigmax;
 Xval=(Xval-mux)/sigmax;
 Xtest=(Xtest-mux)/sigmax;

	
print 'number of infinite values:' + str(sum(sum(~numpy.isfinite(Xtrain))))
xtrain_size=Xtrain.shape
ytrain_size=Ytrain.shape
print 'input number of greater than +-1:' + str(sum(sum(abs(Xtrain)>1))*100.00/(xtrain_size[0]*xtrain_size[1]))
print 'input data dimention is '+str(Xtrain.shape)        
model = Sequential()
print '..building model'
print('creating the layer: Input {} -> Output {} with activation {}'.format(Xtrain.shape[1], int(hiddenlayers[1]), activations[0]))
model.add(Dense(output_dim=int(hiddenlayers[1]), input_dim=Xtrain.shape[1],activation=activations[0]))    
for k in xrange(2,len(hiddenlayers)-1):
  print('creating the layer: Input {} -> Output {} with activation {}'.format(int(hiddenlayers[k-1]), int(hiddenlayers[k]), activations[k]))
  model.add(Dense(output_dim=int(hiddenlayers[k]),activation=activations[k]))    

print('creating the layer: Input {} -> Output {} with activation {}'.format(int(hiddenlayers[len(hiddenlayers)-2]),Ytrain.shape[1], activations[-1]))  
model.add(Dense(output_dim=int(Ytrain.shape[1]),activation=activations[-1]))   

  
print '..compiling model'  
model.compile(loss=newDict['loss'], optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08),metrics=['accuracy'])


print '..fitting model'
batch_size=int(newDict['batchsize'])
early_stopping = EarlyStopping(monitor='val_loss', patience=2)
checkpointer = ModelCheckpoint(filepath='saved_weight.hdf5', verbose=1, save_best_only=True)
history2=model.fit(Xtrain,Ytrain, nb_epoch=int(newDict['nbepochs']), batch_size=int(newDict['batchsize']), validation_data=(Xval,Yval), callbacks=[early_stopping,checkpointer])
model.load_weights('saved_weight.hdf5')
list1=glob.glob(test_path+"*.mat");

for k in list1:
   print 'processing '+k
   st=k.split('/');
   name=st[-1]
   data=scipy.io.loadmat(k)
   Xtest=data['all_data'].transpose();
   Xtest=(Xtest-mux)/sigmax;
   pred=model.predict(Xtest,verbose=1);
   scipy.io.savemat(pred_path+name,{'pred':pred})
