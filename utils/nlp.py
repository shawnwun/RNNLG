######################################################################
######################################################################
#  Copyright Tsung-Hsien Wen, Cambridge Dialogue Systems Group, 2016 #
######################################################################
######################################################################
import os
import json
import sys
import operator
import re

fin = file('utils/nlp/mapping.pair')
replacements = []
for line in fin.readlines():
    tok_from, tok_to = line.replace('\n','').split('\t')
    replacements.append((' '+tok_from+' ',' '+tok_to+' '))

def insertSpace(token,text):
    sidx = 0
    while True:
        sidx = text.find(token,sidx)
        if sidx==-1:
            break
        if sidx+1<len(text) and re.match('[0-9]',text[sidx-1]) and \
                re.match('[0-9]',text[sidx+1]):
            sidx += 1
            continue
        if text[sidx-1]!=' ':
            text = text[:sidx] + ' ' + text[sidx:]
            sidx +=1
        if sidx+len(token)<len(text) and text[sidx+len(token)]!=' ':
            text = text[:sidx+1] + ' ' + text[sidx+1:]
        sidx+=1
    return text

def normalize(text):

    # lower case every word
    text = text.lower()

    # replace white spaces in front and end
    text = re.sub(r'^\s*|\s*$','',text)
    
    # normalize phone number
    ms = re.findall('\(?(\d{3})\)?[-.\s]?(\d{3})[-.\s]?(\d{4})',text)
    if ms:
        sidx = 0
        for m in ms:
            sidx = text.find(m[0],sidx)
            if text[sidx-1]=='(':
                sidx -= 1
            eidx = text.find(m[-1],sidx)+len(m[-1])
            text = text.replace(text[sidx:eidx],''.join(m))
    
    # replace st.
    text = text.replace(';',',')
    text = re.sub('$\/','',text)
    text = text.replace('/',' and ')

    # replace other special characters
    text = re.sub('[\":\<>@]','',text)
    #text = re.sub('[\":\<>@\(\)]','',text)
    text = text.replace(' - ','')

    # insert white space before and after tokens:
    for token in ['?','.',',','!']:
        text = insertSpace(token,text)
          
    # replace it's, does't, you'd ... etc
    text = re.sub('^\'','',text)
    text = re.sub('\'$','',text)
    text = re.sub('\'\s',' ',text)
    text = re.sub('\s\'',' ',text)
    for fromx, tox in replacements:
		text = ' '+text+' '
		text = text.replace(fromx,tox)[1:-1]

    # insert white space for 's
    text = insertSpace('\'s',text)

    # remove multiple spaces
    text = re.sub(' +',' ',text)
    
    # concatenate numbers
    tmp = text
    tokens = text.split()
    i = 1
    while i<len(tokens):
        if re.match(u'^\d+$',tokens[i]) and \
                re.match(u'\d+$',tokens[i-1]):
            tokens[i-1]+=tokens[i]
            del tokens[i]
        else:
            i += 1
    text = ' '.join(tokens)
    
    return text

