import sys

print(sys.argv[0])

discard = ['recorded', 'moment', 'holds', 'equality', 'conclude', 'case', 'proceeded', 'denominator', 'measures']
total   = []
filter  =[",", ".", ":", ";", "(", ")", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "\\", "{", "}", "<", ">", '"', '[', ']']
count = 0
common = []

with open("common.txt", 'r') as f:
    for line in f:
        word= line.strip().rstrip().lower()
        for c in filter:    
            word = word.replace(c, '')
        if len(word) > 0 and word[-1] == 's':
            word = word[:-1]
        common.append(word)
with open("junk.txt", 'r') as f:
    for line in f:
        word= line.strip().rstrip().lower()
        for c in filter:    
            word = word.replace(c, '')
        if len(word) > 0 and word[-1] == 's':
            word = word[:-1]
        common.append(word)
        
with open("non_physics.txt", 'r') as f:
    for line in f:
        word= line.strip().rstrip().lower()
        for c in filter:    
            word = word.replace(c, '')
        if len(word) > 0 and word[-1] == 's':
            word = word[:-1]
        common.append(word)
        
with open(sys.argv[1], 'r', encoding="utf-8") as f:
    for line in f:
        words = line.split()
        for word in words:
            word = word.strip().rstrip().lower()
           
            for c in filter:    
                word = word.replace(c, '')
            if len(word) > 0 and word[-1] == 's':
                word = word[:-1]
            word = "".join(k for k in word if (ord(k) >= ord('-') and ord(k) <= ord('z')) )
            if word:
                if word[0] == "-":
                    continue
                if word[-1] == "-":
                    continue
                if -1 != word.find("http"):
                    continue
                if word not in total and word not in discard and word not in common:
                    total.append(word)
                    print(word)
                    count = count + 1
print(count)
       