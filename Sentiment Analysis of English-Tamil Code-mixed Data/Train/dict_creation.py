import json
dictio={}
f=open('train_sentence.txt','r',errors='ignore').read().split('\n')
count=7 #check dict_assumtions.txt for one-hot assumptions
for i in range(len(f)):
    tokens=f[i].split()
    for j in tokens:
        if j not in dictio.keys():
            dictio[j]=count
            count+=1

#print ('len of dict',len(dictio))
#print ('max length',ln)

json.dump(dictio,open('word_dict.json','w'))
