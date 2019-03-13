import ast
import pandas as pd

folder = r"FILES/Datasets/"
movieConversationsFile = r"movie_conversations.txt"
movieLinesFile = r"movie_lines.txt"
l = list()
with open(folder+movieConversationsFile,'r') as f:
    g = f.read().split('\n')
    for line in g:
        try:
            l.append(ast.literal_eval(line.split("+++$+++")[-1].strip(' ')))
        except:
            pass

questionsAnswers = list()

for line in l:
    for idx in range(0,len(line),2):
        try:
            questionsAnswers.append([line[idx].strip(' '),line[idx+1].strip(' ')])
        except:
            continue

del l

lineCodes = dict()

with open(folder+movieLinesFile,'r') as f:
    g = f.read().split('\n')
    for line in g:
        try:
            aux = line.split("+++$+++")
            lineCodes[aux[0].strip(' ')] = aux[-1]
        except:
            pass

converdations = list()

for line in questionsAnswers:
    converdations.append([lineCodes[line[0]].strip(' '), lineCodes[line[1]].strip(' ')])


dfConversations = pd.DataFrame(data=converdations, columns=['questions', 'answers']).to_csv(folder+"DialogueFrame.csv")