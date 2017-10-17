

fsenone = open("clean_labels_dev.txt")
fnoisy = open("feats_dev.scp")
fclean = open("feats_dev_clean.scp")

fnoisy_orig = open("feats_dev_noisy_orig.scp","w")
#fsenone_orig = open("clean_labels_dev_mod.txt","w")
fclean_orig = open("feats_dev_clean_orig.scp","w")

dictnoisy={}
senonedict={}
dictclean = {}

for line in fclean:
	utt_id, path = line.split()
	dictclean[utt_id] = path


for line in fsenone:
    utt_id = line.split()[0]
    senones = line.split()[1:]
    senonedict[utt_id] = senones


for line in fnoisy:
    utt_id, path = line.split()
    dictnoisy[utt_id] = path

line = 0
for utt_id in sorted(dictnoisy.keys()):
    search_key = utt_id[:-1]+'0'
    search_path = dictnoisy[utt_id]
    search_path_clean = dictclean[search_key]
    senone = senonedict[search_key]
    fnoisy_orig.write(search_key+"_"+str(line)+" "+search_path+"\n")
    fclean_orig.write(search_key+"_"+str(line)+" "+search_path_clean+"\n")
    #fsenone_orig.write(search_key+"_"+str(line)+" "+" ".join(senone)+"\n")
    line = (line + 1)%6

