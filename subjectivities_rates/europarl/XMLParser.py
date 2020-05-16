import re
import glob
import numpy as np
import os
import re


langs = ["pt","de","en","es","it"]
path_template = "./txt/{}/{}"

#returns the dictionary with all files for a specific language
def list_of_files(path,languages_list):
    lang_files = {}
    for lang in languages_list:
        files = [f.split("/")[len(f.split("/"))-1] for f in glob.glob(path+lang+"/*")]
        lang_files[lang] = files
    return lang_files



#outputs a list of file names that exist in all specified languages.
def get_common_files(path,languages_list):
    all_files = list_of_files(path,languages_list)
    pt = all_files.get("pt")
    de = all_files.get("de")
    en = all_files.get("en")
    es = all_files.get("es")
    it = all_files.get("it")
    common_files = np.intersect1d(en, np.intersect1d(de, np.intersect1d(es, np.intersect1d(it, pt))))
    return common_files



#Parse txt to doctionary of form document = {chapterid:{talks}}, talks = {speaker id+ name:{talk}}, talk ={speaker:speaker id+name,text:raw text}
def parse_file(path_to_file):
    document = {}
    speaker =""
    cep_id =None
    talks = {}
    talk = {}
    counter = 1
    skip_next = False
    id = None
    name = None
    for line in open(path_to_file):
        if line.startswith('<CHAPTER ID'):
            if cep_id:
                if talk:
                    talks[id+":"+name] = talk
                    talk = {}
                document[cep_id]=talks
                counter=1
                talks ={}
            cep_id = line[1:len(line)-2]
            cep_id = re.sub('"', '', cep_id)
            skip_next = True
            continue
        if line.startswith('<SPEAKER ID=') and ("NAME=" in line):
            if talk:
                talks[id+":"+name]=talk
                counter+=1
            id, name = get_id_name(line)
            #talk = {'speaker':re.sub('"', '',line[1:len(line)-1]),'text':""}
            talk = {'speaker':id+":"+name,'text':""}
            skip_next=False
            continue
        if skip_next:
            skip_next = False
            continue
        if "<P>" in line:
            continue
        else:
            line = re.sub('\n', '', line)
            if talk:
                talk['text']=talk['text']+" "+line
    if talks:
        talks[id+":"+name]=talk
    document[cep_id]=talks
    return document





#parse ID and name of a speaker
def get_id_name(str_to_parse):
    str_to_parse = re.sub('"', '', str_to_parse)
    id = str_to_parse[str_to_parse.find("ID=")+3:str_to_parse.find(" ",str_to_parse.find("ID="))]
    name = str_to_parse[str_to_parse.find("NAME=")+5:str_to_parse.find(">",str_to_parse.find("NAME="))]
    return id, name





#Compare the same file in certain languages to find missing or wrong chapters and speakers.
#Returns a filtered and aligned dictionary of a specific file in a specific language.
def compare_docs(docs):
    removed_chapter_num = 0
    removed_speaker_num = 0
    same_chap_num =True
    size = len(docs['en'])
    common = set(docs['en'].keys())
    #Test if the file contains the same number of chapters in all languages
    for key in docs:
        if len(docs[key])!=size:
            same_chap_num = False
    #If the number of chapters is different, calculate the set of common chapters
    if not same_chap_num:
        for key in docs:
            common = common.intersection(set(docs[key].keys()))
        #remove chapters that are not presented in every language
        for key in list(docs.keys()):
            docum = docs[key]
            docum_chapters = list(docum.keys())
            for chapter in docum_chapters:
                if not chapter in common:
                    del docum[chapter]
                    removed_chapter_num+=1
    #Go through all chapters and check that all speakers match in each language
    to_remove = common
    for chapter in to_remove:
        #Calculate the set of common speakers for a given chapter
        common_speaker = set(docs['en'][chapter].keys())
        for document in docs.keys():
            common_speaker = common_speaker.intersection(set(docs[document][chapter].keys()))
        #If the speaker is not presented in all languages in this chapter, remove it
        for document in list(docs.keys()):
            for speaker in list(docs[document][chapter].keys()):
                if speaker not in common_speaker:
                    removed_speaker_num+=1
                    del (docs[document][chapter])[speaker]
    return docs,removed_chapter_num, removed_speaker_num



#Creates or opens a file and writes the document in each language without tags
def write_to_file(documents,file_name,output_path):
    chapter_to_write = list(documents['en'].keys())
    for l in langs:
        if not os.path.exists(os.path.join(output_path,l)):
            os.makedirs(os.path.join(output_path,l))
    for chapt in chapter_to_write:
        speaker_to_write = list(documents['en'][chapt].keys())
        for spk in speaker_to_write:
            for lang in langs:
                lan_doc=documents[lang][chapt]
                path = os.path.join(output_path,lang,lang+"_"+file_name)
                file = open(path, 'a+')
                file.write(lan_doc[spk]['text'].lstrip(' ')+"\n")



if __name__ == '__main__':
    #use the correct path to the dataset
    path = './txt/'
    files_to_parse = get_common_files(path,langs)
    total_removed_chapters = 0
    total_removed_speakers = 0
    counter = 0
    #for each file, retrieve its language versions, align them and write them to the file
    print("Start")
    for f in files_to_parse:
        files = {}
        for lang in langs:
            path = path_template.format(lang, f)
            files[lang] = parse_file(path)
        docs, removed_ch, removed_sp = compare_docs(files)
        total_removed_chapters += removed_ch
        total_removed_speakers += removed_sp
        write_to_file(docs, f, './output')
        counter+=1
        if counter%10==0:
            print("Already ",counter," files are processed")
    print("Finished")
    print("Removed ",total_removed_chapters," chapters")
    print("Removed ", total_removed_speakers, " speakers")
    print("Total information loss approx: chapters",total_removed_chapters/len(langs)," speakers: ",total_removed_speakers/len(langs))







