import os


labels = set()
counter = 0
with open('labels.csv', 'r') as csvfile:
	for line in csvfile:
		counter += 1
		if (counter % 3) == 2:
			current = line.strip().split(',')
			#print(len(current))
			#print(current)
			for x in current:
				labels.add(x.lower())

images = {}
counter = 0
with open('labels.csv', 'r') as csvfile:
	name = ''
	lab = []
	perc = []
	for line in csvfile:
		counter += 1
		if(counter % 3) == 1:
			name = line.strip()
			#if counter == 1: print(name)
		if(counter % 3) == 2:
			lab = line.strip().split(',')
		if(counter % 3) == 0:
			perc = line.strip().split(',')
			images[name] = {}
			for i in range(len(lab)):
				images[name][lab[i]] = perc[i]


#print(len(labels))
#print(labels)
#print(counter//3)
#print(len(images))
#print(images['2133115688.1.JPG'])
#print(images['2133115688.1.JPG']['cassette_player'])

with open('xylabels.csv', 'w') as newfile:
	newfile.write(',')
	newfile.write(','.join(str(l) for l in labels))
	newfile.write('\n')
	for key, value in images.items():
		newfile.write(key)
		newfile.write(',')
		li = []
		for label in labels:
			#print(label)
			if label in value:
				li.append(value[label])
			else:
				li.append(0)
		newfile.write(','.join(str(n) for n in li))
		newfile.write('\n')	
		
