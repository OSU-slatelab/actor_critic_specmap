

fnoisy = open("feats_dev.scp")
fnoisy_mod = open("feats_dev_noisy_orig.scp","w")
lnum=0
for line in fnoisy:
	utt_id, path = line.split()
	search_key = utt_id[:-1]+'0'
	fnoisy_mod.write(search_key+"_"+str(lnum)+" "+path+"\n")
	lnum = (lnum + 1)%6
	

